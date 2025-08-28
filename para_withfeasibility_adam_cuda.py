# -*- coding: utf-8 -*-
import os
import math
import time
import random
import numbers
import numpy as np
import torch
import torch.nn as nn

# --- MPS robustness knobs (safe no-ops on CUDA/CPU) ---
try:
    import torch._dynamo as dynamo  # type: ignore
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dynamo.disable()  # avoid MPS placeholder issues during optimizer init/probes
except Exception:
    pass

if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# ---------- utilities ----------

def linear_map_deterministic_biased(dim, p=0.7):
    if dim <= 0:
        raise ValueError("dim must be a positive integer")
    if dim == 1:
        return np.array([[1.0]], dtype=float)
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1]")
    M = np.full((dim, dim), (1.0 - p) / (dim - 1), dtype=float)
    np.fill_diagonal(M, p)
    return M

def to_torch_tensor(x, device, dtype=None):
    if dtype is None:
        dtype = torch.get_default_dtype()
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=dtype)
    dev_str = device if isinstance(device, str) else device.type
    if t.dtype != dtype:
        t = t.to(dtype)
    if t.device.type != dev_str:
        t = t.to(dev_str)
    return t

def W_from_vectors(x, w):
    # x: (Nx,), w: (Nw,) → returns (Nx, Nw) via broadcasting
    return torch.exp(x.view(-1,1) - 1) * torch.exp(w.view(1,-1) - 1)

def softmin(a, b, device, t=0.8):
    device = torch.device(device if isinstance(device, str) else device.type)
    dtype = torch.get_default_dtype()
    a_t = torch.as_tensor(a, dtype=dtype, device=device)
    b_t = torch.as_tensor(b, dtype=dtype, device=device)
    diff = (b_t - a_t) / t
    w = torch.sigmoid(-diff)
    return a_t + (b_t - a_t) * w

def eval_f(X, U, T, rmax, Cp: float = 1.0, Cr: float = 1.0) -> float:
    sqrtX = math.sqrt(float(X))
    term1 = T * (T + 1) * rmax * ((Cp + 1) ** (T + 1) - 1.0)
    term2 = 2.0 * Cr * ((Cp + 1) ** (T + 2) - (T + 2) * Cp - 1.0) / (Cp ** 2)
    term3 = (float(X) ** 1.5) * U * ((T + 2) ** 3) * rmax
    term4 = sqrtX * U * (T + 1)
    term5 = math.sqrt(float(T))
    fval = sqrtX * term1 + sqrtX * term2 + term3 + term4 + term5
    return float(fval)

# ---------- device picker (CUDA → MPS → CPU) ----------

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch.backends.mps as mps
        if mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")

# ---------- feature maps & parameter makers ----------

def features_power_basis(zz, q):
    # zz: (k,), return Φ: (q, k) with Φ[m,:] = zz**m
    k = zz.shape[-1]
    exps = torch.arange(q, device=zz.device, dtype=zz.dtype).view(q, 1)
    return zz.view(1, k) ** exps

def make_params_new(X, U, T, q, dtype, device, init):
    # We create leaf nn.Parameters and initialize in-place under no_grad()
    def init_param(shape):
        p = nn.Parameter(torch.empty(shape, dtype=dtype, device=device))
        with torch.no_grad():
            if isinstance(init, numbers.Real):
                p.fill_(float(init))
                p.add_(0.01 * torch.randn_like(p))  # tiny noise; still a leaf
            else:
                p.uniform_(0.0, 1.0)
        return p

    D     = init_param((T + 1, X, U, q))  # weights for d
    Y     = init_param((T + 1, X,     q)) # weights for y
    Z     = init_param((T + 1, X, U, q))  # weights for z
    ZGATE = init_param((T + 1,     q))    # gate weights for z
    YGATE = init_param((T + 1,     q))    # gate weights for y
    return D, Y, Z, ZGATE, YGATE

# ---------- parametrizations (vectorized; einsum) ----------

def d_para_by_simplex(X, U, T, D, zz, tau_d=2.0):
    """
    D: (T+1, X, U, q)
    zz: (k,) α-grid
    return d: (T+1, X, U, k) with softmax across (X*U) per (t, α)
    """
    q = D.shape[-1]
    Φ = features_power_basis(zz, q)                 # (q,k)
    logits = torch.einsum('txuq,qk->txuk', D, Φ)    # (T+1, X, U, k)
    T1, X1, U1, k = logits.shape
    logits_flat = logits.view(T1, X1*U1, k)         # (T+1, XU, k)
    probs_flat = torch.softmax(logits_flat / tau_d, dim=1)
    d = probs_flat.view(T1, X1, U1, k)
    return d

def z_para_by_capped_simplex(X, U, T, Z, ZGATE, zz, Cz, tau_z=2.0, beta_z=2.0):
    """
    Z: (T+1, X, U, q), ZGATE: (T+1, q)
    return z: (T+1, X, U, k)
    """
    q = Z.shape[-1]
    Φ = features_power_basis(zz, q)                       # (q,k)
    u = torch.einsum('txuq,qk->txuk', Z, Φ)               # (T+1, X, U, k)
    shp = torch.nn.functional.softplus(u, beta=beta_z)    # (T+1, X, U, k)
    tiny = 1e-12
    shp_sum = shp.sum(dim=(1,2), keepdim=True) + tiny     # (T+1,1,1,k)
    tilde = shp / shp_sum                                 # (T+1, X, U, k)
    gate_raw = torch.einsum('tq,qk->tk', ZGATE, Φ)        # (T+1, k)
    gate = torch.sigmoid(gate_raw / tau_z)                # (T+1, k)
    z = Cz * tilde * gate.unsqueeze(1).unsqueeze(1)       # (T+1, X, U, k)
    return z

def y_para_by_l2ball(X, T, Y, YGATE, zz, Cy, tau_y=2.0, eps=1e-8):
    """
    Y: (T+1, X, q), YGATE: (T+1, q)
    return y: (T+1, X, k)
    """
    q = Y.shape[-1]
    Φ = features_power_basis(zz, q)                       # (q,k)
    raw = torch.einsum('txq,qk->txk', Y, Φ)               # (T+1, X, k)
    norm = torch.linalg.vector_norm(raw, dim=1, keepdim=True)  # (T+1, 1, k)
    direction = raw / (norm + eps)
    gate_raw = torch.einsum('tq,qk->tk', YGATE, Φ)        # (T+1, k)
    gate = torch.sigmoid(gate_raw / tau_y)                # (T+1, k)
    y = Cy * direction * gate.unsqueeze(1)                # (T+1, X, k)
    return y

# ---------- helpers ----------

def sum_over_U(M):
    # M: (T+1, X, U, k) → (T+1, X, k)
    return M.sum(dim=2)

def compute_G_trapz(sumd, x, w):
    """
    sumd: (T+1, X, Nw)  — ∑_a d(t,s,a; w)
    x: (Nx,), w: (Nw,)
    returns G: (T+1, X, Nx), where
        G[t,s,i] = ∫ W(x_i, w) * sumd[t,s,w] dw
    """
    Wmat = W_from_vectors(x, w)                       # (Nx, Nw)
    prod = sumd.unsqueeze(2) * Wmat.unsqueeze(0).unsqueeze(0)  # (T+1, X, Nx, Nw)
    G = torch.trapz(prod, w, dim=-1)                  # (T+1, X, Nx)
    return G

# ---------- P and R ----------

def _simplex_table_all_sa(X, U, t, device, dtype):
    """
    S[t]: (X, U, X) where S[s,a,:] is the simplex over s' for given (s,a,t).
    """
    i = torch.arange(X, device=device, dtype=dtype).view(1,1,X)       # s' index 0..X-1
    s_idx = torch.arange(1, X+1, device=device, dtype=dtype).view(X,1,1)
    a_idx = torch.arange(1, U+1, device=device, dtype=dtype).view(1,U,1)
    denom = 2.0 * (t**2) + 1.0
    phi = -((i - s_idx)**2) / denom                                   # (X,U,X)
    w = torch.exp(a_idx * phi)                                        # (X,U,X)
    out = w / (w.sum(dim=-1, keepdim=True) + 1e-12)
    return out                                                        # (X,U,X)

def make_P(X, U, T, G, para, device):  # -> (T+1, X, U, X, Nx)
    """
    G: (T+1, X, Nx), para: (X,X)
    """
    para = to_torch_tensor(para, device)
    if not (isinstance(para, torch.Tensor) and para.ndim == 2 and para.shape[0] == para.shape[1] == X):
        raise ValueError("para must be a square torch matrix aligned with X")

    T1, X1, Nx = G.shape
    dtype = G.dtype
    P_out = []
    for t in range(T1):
        base = torch.matmul(para, G[t])                               # (X, Nx)
        q_t = 1.0 / (float(t)**2 + 2.0)
        S = _simplex_table_all_sa(X, U, t, G.device, dtype)           # (X, U, X)
        base_exp = (1.0 - q_t) * base.unsqueeze(0).unsqueeze(0)       # (1,1,X,Nx) → (X,U,X,Nx)
        simp_exp = q_t * S.unsqueeze(-1)                               # (X,U,X,1) → (X,U,X,Nx)
        P_t = base_exp + simp_exp
        P_out.append(P_t)
    return torch.stack(P_out, dim=0)                                   # (T+1, X, U, X, Nx)

def make_R(X, U, T, G, rmax, para, device):  # -> (T+1, X, U)
    dtype = torch.get_default_dtype()
    para = to_torch_tensor(para, device)
    rmax = to_torch_tensor(rmax, device, dtype=dtype)

    T1, X1, Nx = G.shape
    R_out = []
    s_idx = torch.arange(1, X+1, device=G.device, dtype=dtype).view(X,1)
    a_idx = torch.arange(1, U+1, device=G.device, dtype=dtype).view(1,U)

    for t in range(T1):
        M = torch.matmul(para, G[t])                                  # (X,Nx)
        S_t = torch.sum(M**2)                                         # scalar
        coeff = 10.0 * torch.exp(-(t + 1.0) * ((s_idx - a_idx)**2 + 2.0))  # (X,U)
        b = coeff * S_t                                               # (X,U)
        R_t = softmin(rmax, b, device)                                # (X,U)
        R_out.append(R_t)
    return torch.stack(R_out, dim=0)                                  # (T+1, X, U)

# ---------- main (Adam + trapz integration, no minibatch) ----------

def main():
    dtype = torch.float
    device = pick_device()
    print(f"Using device: {device}")

    # problem sizes
    rmax = 10
    X = 3
    U = 3
    T = 3
    q = 16
    mu0 = torch.full((X,), 1.0 / X, dtype=dtype, device=device)  # (X,)

    # grids (full, no random sampling)
    Nx, Nw = 110, 110
    x = torch.linspace(0, 1, Nx, dtype=dtype, device=device)  # (Nx,)
    w = torch.linspace(0, 1, Nw, dtype=dtype, device=device)  # (Nw,)

    # report f
    f_value = eval_f(X, U, T, rmax)
    print(f"f(|X|,|U|,T,Cp,Cr,rmax) = {f_value:.6f}")

    # params (dense leaf nn.Parameters on device)
    D, Y, Z, ZGATE, YGATE = make_params_new(X, U, T, q, dtype, device, init=0.01)
    with torch.no_grad(): # cautious... may affect the autograd feature or leag/non-leaf feature
        ZGATE[:, 0].fill_(-5.0)
        YGATE[:, 0].fill_(-5.0)

    # linear maps
    para_in_p = linear_map_deterministic_biased(X, 0.4)
    para_in_r = linear_map_deterministic_biased(X, 0.65)

    # caps & knobs
    lambda_z, lambda_y = 0.1, 0.1
    Cz = lambda_z * (X * U * (T ** 2 + T + 2) * rmax)
    Cy = lambda_y * ((X * (T + 1) * (T + 2) / 2) * rmax)
    tau_d, tau_z, tau_y = 2.0, 2.0, 2.0
    beta_z, eps = 2.0, 1e-8

    # optimizer (+ optional scheduler)
    learning_rate = 1e-2 *6
    params = [D, Y, Z, ZGATE, YGATE]
    optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

    print_every = 5
    iters = 10000  # full-grid trapz is heavier; adjust as you like
    time1 = time.perf_counter()

    init_obj = None
    final_obj = None

    for iter in range(iters):
        # evaluate parameters on full grids (no random batch)
        d_x = d_para_by_simplex(X, U, T, D, x, tau_d)                  # (T+1, X, U, Nx)
        z_x = z_para_by_capped_simplex(X, U, T, Z, ZGATE, x, Cz, tau_z, beta_z)  # (T+1, X, U, Nx)
        y_x = y_para_by_l2ball(X, T, Y, YGATE, x, Cy, tau_y, eps)      # (T+1, X, Nx)

        d_w = d_para_by_simplex(X, U, T, D, w, tau_d)                  # (T+1, X, U, Nw)
        Sum_d_w = sum_over_U(d_w)                                      # (T+1, X, Nw)

        # inner integral over w: G(t,s,x) = ∫ W(x,w) * sum_a d(t,s,a;w) dw
        G = compute_G_trapz(Sum_d_w, x, w)                             # (T+1, X, Nx)

        # kernels P and R
        P = make_P(X, U, T, G, para_in_p, device)                      # (T+1, X, U, X, Nx)
        R = make_R(X, U, T, G, rmax, para_in_r, device)                # (T+1, X, U)

        # objective (vector over x-grid, length Nx)
        Nxg = x.shape[0]
        zero = torch.zeros(Nxg, dtype=dtype, device=device)

        # obj_part1_1
        obj_part1_1 = zero.clone()
        for t in range(T):  # 0..T-1
            term1 = (d_x[t].unsqueeze(2) * P[t]).sum(dim=(0,1))       # (X,Nx) over s,a
            term2 = d_x[t+1].sum(dim=1)                                # (X,Nx) over a
            resid = (term1 - term2)**2                                 # (X,Nx)
            obj_part1_1 = obj_part1_1 + resid.sum(dim=0)               # (Nx,)

        # obj_part1_2
        term_mu = d_x[0].sum(dim=1) - mu0.view(X,1)                    # (X,Nx)
        obj_part1_2 = (term_mu**2).sum(dim=0)                          # (Nx,)
        obj_part1 = obj_part1_1 + obj_part1_2

        # obj_part2_1 for t=1..T-1
        obj_part2_1 = zero.clone()
        for t in range(1, T):
            PvY = (P[t] * y_x[t].unsqueeze(0).unsqueeze(0)).sum(dim=2) # (X,U,Nx)
            resid = PvY - y_x[t-1].unsqueeze(1) + z_x[t] - R[t].unsqueeze(-1)  # (X,U,Nx)
            obj_part2_1 = obj_part2_1 + (resid**2).sum(dim=(0,1))      # (Nx,)

        # obj_part2_2 at t=0 boundary
        PvY0 = (P[0] * y_x[0].unsqueeze(0).unsqueeze(0)).sum(dim=2)    # (X,U,Nx)
        resid0 = PvY0 + y_x[T].unsqueeze(1) + z_x[0] - R[0].unsqueeze(-1)
        obj_part2_2 = (resid0**2).sum(dim=(0,1))                       # (Nx,)

        # obj_part2_3 at t=T boundary
        residT = -y_x[T-1].unsqueeze(1) + z_x[T] - R[T].unsqueeze(-1)  # (X,U,Nx)
        obj_part2_3 = (residT**2).sum(dim=(0,1))                       # (Nx,)

        obj_part2 = obj_part2_1 + obj_part2_2 + obj_part2_3

        # obj_part3 = sum_{t,s,a} z * d
        obj_part3 = (z_x * d_x).sum(dim=(0,1,2))                       # (Nx,)

        integrand = obj_part1 + obj_part2 + obj_part3                  # (Nx,)
        integral = torch.trapz(integrand, x)                           # scalar

        if iter % print_every == 0:
            print(f"iter {iter:03d} | objective = {integral.item():.6f}")

        if init_obj is None:
            init_obj = integral.item()

        optimizer.zero_grad(set_to_none=True)
        integral.backward()
        optimizer.step()
        scheduler.step(integral.item())

    final_obj = integral.item()

    print("improment ratio: " , final_obj/init_obj)
    print("Final objective:", float(integral.detach().cpu().item()))
    time2 = time.perf_counter()
    print("time used ", time2 - time1)

if __name__ == "__main__":
    main()
