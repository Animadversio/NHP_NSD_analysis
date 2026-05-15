"""
Piecewise Linear Dynamical Systems for regression weight trajectory analysis.

The regression weight matrix W[t, u] encodes the coding direction in PCA feature
space that best predicts unit u at time t.  Projecting the population trajectory
into a k-dimensional subspace gives z_t ∈ R^k.  This module fits:

  Global LDS (one-step):        z[t+1] = A z[t]               (OLS one-step)
  Global LDS (open-loop/AR):    min_A  Σ_t ||z_t - A^t z_0||²  (gradient descent)
  Piecewise LDS:                z[t+1] = A_k z[t]  for t ∈ segment k

and provides data-driven changepoint detection via Dynp (ruptures library).
"""

import numpy as np
from numpy.linalg import svd, lstsq


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------

def get_latent_trajectory(W: np.ndarray, k: int = 10):
    """
    Extract top-k latent trajectory from weight matrix W via SVD.

    Parameters
    ----------
    W : (n_t, n_units, n_pca)
    k : number of latent dimensions

    Returns
    -------
    Z    : (k, n_t)   — latent trajectory (U * S for top-k components)
    var  : (n_k,)     — fraction of variance explained per component
    Vt_k : (k, n_pca*n_units) — top-k right singular vectors (decoder)
    """
    W_flat = W.reshape(W.shape[0], -1).astype(np.float64)
    W_flat -= W_flat.mean(axis=0, keepdims=True)
    U, sv, Vt = svd(W_flat, full_matrices=False)
    Z = (U[:, :k] * sv[:k]).T          # (k, n_t)
    var = sv[:k] ** 2 / (sv ** 2).sum()
    return Z, var, Vt[:k]


def _r2(true: np.ndarray, pred: np.ndarray) -> float:
    """Scalar R² for matrices of any shape (flatten internally)."""
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean(axis=-1, keepdims=True)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def diagnose_lds_residuals(W: np.ndarray, k: int = 10) -> dict:
    """
    Fit a global LDS and return per-timestep one-step residuals.

    Useful for identifying *where* in time a single A matrix fails.

    Returns
    -------
    dict:
        'residuals'  : (n_t-1,) — ||z[t+1] - A z[t]||^2 per step
        'Z'          : (k, n_t)
        'A'          : (k, k) — global transition matrix
        'r2_onestep' : float
        'r2_openloop': float
    """
    Z, var, _ = get_latent_trajectory(W, k)
    A, _, _, _ = lstsq(Z[:, :-1].T, Z[:, 1:].T, rcond=None)
    A = A.T                                         # (k, k)

    pred_onestep = A @ Z[:, :-1]
    residuals = np.sum((Z[:, 1:] - pred_onestep) ** 2, axis=0)   # (n_t-1,)

    # Open-loop
    Z_ol = np.zeros_like(Z)
    Z_ol[:, 0] = Z[:, 0]
    for t in range(1, Z.shape[1]):
        Z_ol[:, t] = A @ Z_ol[:, t - 1]

    return {
        'residuals':   residuals,
        'Z':           Z,
        'Z_ol':        Z_ol,
        'A':           A,
        'r2_onestep':  _r2(Z[:, 1:], pred_onestep),
        'r2_openloop': _r2(Z, Z_ol),
        'var_explained': np.cumsum(var),
    }


# ---------------------------------------------------------------------------
# Segment fitting
# ---------------------------------------------------------------------------

def fit_segment(Z: np.ndarray, t_start: int, t_end: int, alpha: float = 0.0):
    """
    Fit A for the slice Z[:, t_start:t_end] → z[t+1] = A z[t].

    Parameters
    ----------
    Z       : (k, n_t) latent trajectory
    t_start : first time index of segment (inclusive)
    t_end   : last time index of segment (exclusive)
    alpha   : ridge regularisation for A (0 = OLS, >0 = ridge).
              Recommended ≥ 1e-2 for short segments to keep |λ| ≤ 1.

    Returns
    -------
    A  : (k, k) transition matrix
    r2 : float one-step R² on this segment (NaN if < 2 timepoints)
    """
    seg = Z[:, t_start:t_end]           # (k, seg_len)
    if seg.shape[1] < 2:
        return np.eye(Z.shape[0]), float('nan')

    X = seg[:, :-1].T   # (n_pairs, k)
    Y = seg[:, 1:].T    # (n_pairs, k)
    k = Z.shape[0]

    if alpha > 0:
        # Ridge: solve (X'X + α I) A' = X'Y  →  A = Y'X (X'X + α I)^{-1}
        XtX = X.T @ X + alpha * np.eye(k)
        A_T = np.linalg.solve(XtX, X.T @ Y)  # (k, k)
        A = A_T.T
    else:
        A, _, _, _ = lstsq(X, Y, rcond=None)
        A = A.T

    pred = A @ seg[:, :-1]
    r2 = _r2(seg[:, 1:], pred)
    return A, r2


# ---------------------------------------------------------------------------
# Piecewise LDS
# ---------------------------------------------------------------------------

def fit_piecewise_lds(
    W: np.ndarray,
    breakpoints: list[int],
    k: int = 10,
    reset_at_boundaries: bool = True,
    alpha: float = 1.0,
) -> dict:
    """
    Fit independent A_k per temporal segment.

    Parameters
    ----------
    W                   : (n_t, n_units, n_pca)
    breakpoints         : list of time indices where new segments begin
                          e.g. [15, 30] → segments [0,15), [15,30), [30,n_t)
                          indices are into the time axis of W.
    k                   : LDS state dimension
    reset_at_boundaries : if True (default), open-loop rollout resets z to the
                          true value at each segment boundary.  If False, error
                          accumulates across boundaries (full open-loop).
    alpha               : ridge regularisation for each A_k fit (default 1.0).
                          Keeps eigenvalues near the unit circle for short segs.

    Returns
    -------
    dict:
        'Z'           : (k, n_t)   actual latent trajectory
        'Z_pred'      : (k, n_t)   open-loop predicted trajectory
        'A_list'      : list[(k,k)] — one matrix per segment
        'eigs_list'   : list[ndarray(k,complex)] — eigenvalues per segment
        'r2_onestep'  : list[float] — one-step R² per segment
        'r2_openloop' : float — overall open-loop R²
        'segments'    : list[(t_start, t_end)]
        'breakpoints' : list[int]
        'var_explained': (k,) cumulative variance explained in latent space
    """
    Z, var, _ = get_latent_trajectory(W, k)
    n_t = Z.shape[1]

    bkps = sorted(set([0] + list(breakpoints) + [n_t]))
    segments = [(bkps[i], bkps[i + 1]) for i in range(len(bkps) - 1)]

    A_list, eigs_list, r2_onestep = [], [], []
    for t_start, t_end in segments:
        A_seg, r2_seg = fit_segment(Z, t_start, t_end, alpha=alpha)
        A_list.append(A_seg)
        eigs_list.append(np.linalg.eigvals(A_seg))
        r2_onestep.append(r2_seg)

    # Open-loop rollout
    Z_pred = np.zeros_like(Z)
    Z_pred[:, 0] = Z[:, 0]
    for (t_start, t_end), A_seg in zip(segments, A_list):
        if reset_at_boundaries and t_start > 0:
            Z_pred[:, t_start] = Z[:, t_start]
        for t in range(t_start, t_end - 1):
            Z_pred[:, t + 1] = A_seg @ Z_pred[:, t]

    return {
        'Z':            Z,
        'Z_pred':       Z_pred,
        'A_list':       A_list,
        'eigs_list':    eigs_list,
        'r2_onestep':   r2_onestep,
        'r2_openloop':  _r2(Z, Z_pred),
        'segments':     segments,
        'breakpoints':  list(breakpoints),
        'var_explained': np.cumsum(var),
    }


# ---------------------------------------------------------------------------
# Changepoint detection
# ---------------------------------------------------------------------------

def find_changepoints(
    W: np.ndarray,
    k: int = 10,
    n_bkps: int = 3,
    model: str = 'rbf',
    min_size: int = 3,
) -> list[int]:
    """
    Data-driven changepoint detection via Dynp (dynamic programming) with
    a fixed number of breakpoints.

    Parameters
    ----------
    W       : (n_t, n_units, n_pca)
    k       : latent dimension
    n_bkps  : number of changepoints to detect
    model   : ruptures cost model ('rbf', 'l2', 'l1', 'normal')
    min_size: minimum segment length in time bins

    Returns
    -------
    breakpoints : list[int] — detected breakpoints (excluding final n_t)
    """
    import ruptures as rpt
    Z, _, _ = get_latent_trajectory(W, k)
    signal = Z.T.astype(np.float64)             # (n_t, k)
    algo = rpt.Dynp(model=model, min_size=min_size).fit(signal)
    result = algo.predict(n_bkps=n_bkps)
    return result[:-1]                           # drop trailing n_t


def find_changepoints_bic(
    W: np.ndarray,
    k: int = 10,
    max_bkps: int = 6,
    model: str = 'rbf',
    min_size: int = 3,
) -> dict:
    """
    Automatic changepoint detection with BIC-based model selection.

    Uses Dynp (exact dynamic programming) to find the globally optimal
    segmentation for each K, then selects K via BIC:

        BIC(K) = cost(K) + (K+1) * k² * log(n_t)

    where cost is the ruptures reconstruction cost and (K+1)*k² counts
    the free parameters in K+1 transition matrices A_k ∈ R^{k×k}.

    Returns
    -------
    dict:
        'breakpoints' : list[int] — optimal breakpoints
        'best_n'      : int — optimal number of breakpoints
        'bic_scores'  : list[float] — BIC per n_bkps (index = n_bkps)
        'costs'       : list[float] — raw ruptures cost per n_bkps
        'all_results' : list[list[int]] — breakpoints for each K tried
    """
    import ruptures as rpt
    Z, _, _ = get_latent_trajectory(W, k)
    n_t = Z.shape[1]
    signal = Z.T.astype(np.float64)

    algo = rpt.Dynp(model=model, min_size=min_size).fit(signal)

    bic_scores, costs, results = [], [], []
    for n in range(0, max_bkps + 1):
        try:
            bkps = algo.predict(n_bkps=n)
            cost = algo.cost.sum_of_costs(bkps)
            n_params = (n + 1) * k ** 2          # one A matrix per segment
            bic = cost + n_params * np.log(n_t)
            bic_scores.append(float(bic))
            costs.append(float(cost))
            results.append(bkps[:-1])
        except Exception:
            bic_scores.append(float('inf'))
            costs.append(float('inf'))
            results.append([])

    best_n = int(np.argmin(bic_scores))
    return {
        'breakpoints': results[best_n],
        'best_n':      best_n,
        'bic_scores':  bic_scores,
        'costs':       costs,
        'all_results': results,
    }


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------

def fit_lds_openloop(
    W: np.ndarray,
    k: int = 10,
    n_iter: int = 2000,
    lr: float = 1e-3,
    reg: float = 1e-4,
    verbose: bool = False,
) -> dict:
    """
    Fit a global LDS by minimising the *open-loop* (autoregressive) prediction
    error rather than the standard one-step squared error.

    Background
    ----------
    The standard OLS approach fits A by minimising one-step residuals:
        A_OLS = argmin_A  Σ_{t=0}^{T-1} ||z_{t+1} - A z_t||²
    This has a closed-form solution but A_OLS is poorly suited for open-loop
    rollout: it averages across all temporal phases, producing a single damped
    mode that misses the fast onset rise and the slow sustained decay.

    The open-loop (autoregressive) objective is instead:
        L(A) = Σ_{t=1}^{T} ||z_t - A^t z_0||²_F  +  reg * ||A||²_F

    where A^t z_0 = A(A(...(A z_0)...)) is the t-step prediction from the
    initial state z_0.  This loss is non-linear in A (matrix power) so there
    is no closed form, but gradient descent via auto-differentiation is
    straightforward.

    Algorithm
    ---------
    1. Extract latent trajectory Z ∈ R^{k × T} via top-k SVD of the
       mean-subtracted weight matrix W_flat = W.reshape(T, -1).
    2. Initialise A from the OLS one-step estimate (warm start for speed).
    3. For each gradient step:
         a. Roll out  z_{pred}(t) = A^t z_0  sequentially for t = 1..T.
         b. Compute L(A) = Σ_t ||z_{pred}(t) - Z[:,t]||² + reg·||A||²_F.
         c. Back-propagate through the sequential rollout (PyTorch autograd).
         d. Update A with Adam.
    4. After convergence, report A_ol, Z_ol (open-loop trajectory), R², and
       eigenvalues of A_ol.

    In practice (NSD_N3, DINOv2 CLS block 7, k=10, T≈80):
        OLS one-step:   OL R² ≈ 0.17–0.40  (too damped for onset)
        AR open-loop:   OL R² ≈ 0.80–0.91  (tracks full trajectory)
        1-step R² drops ≈ 0.99 → 0.93      (expected trade-off)
    All eigenvalues remain inside the unit circle (max|λ| ≈ 0.98–0.99).

    Parameters
    ----------
    W       : (n_t, n_units, n_pca)
    k       : latent dimension (10 recommended for NSD_N3)
    n_iter  : Adam iterations (3000 is typically sufficient)
    lr      : Adam learning rate (1e-3 works well)
    reg     : Frobenius regularisation weight — small values (1e-4) suffice
              to keep eigenvalues bounded without sacrificing fit quality
    verbose : print loss every 200 iterations

    Returns
    -------
    dict:
        'Z'            : (k, n_t) — true latent trajectory (top-k SVD)
        'Z_ol'         : (k, n_t) — open-loop rollout from A_ol
        'A_ol'         : (k, k)   — open-loop fitted transition matrix
        'A_onestep'    : (k, k)   — one-step OLS A (used for initialisation)
        'eigs_ol'      : (k,)     — eigenvalues of A_ol (complex)
        'eigs_onestep' : (k,)     — eigenvalues of A_onestep (complex)
        'r2_openloop'  : float    — open-loop R² with A_ol
        'r2_onestep'   : float    — one-step R² with A_ol (reference)
        'loss_curve'   : list[float] — Adam loss per iteration
        'var_explained': (k,)     — cumulative SVD variance fraction
    """
    import torch

    Z, var, _ = get_latent_trajectory(W, k)
    n_t = Z.shape[1]

    # Initialise from one-step OLS
    A_init, _, _, _ = lstsq(Z[:, :-1].T, Z[:, 1:].T, rcond=None)
    A_init = A_init.T                                    # (k, k)

    Z_t = torch.tensor(Z, dtype=torch.float64)
    A   = torch.tensor(A_init.copy(), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([A], lr=lr)

    loss_curve = []
    for it in range(n_iter):
        # Roll out open-loop from z_0
        z = Z_t[:, 0]
        loss = torch.zeros(1, dtype=torch.float64)
        for t in range(1, n_t):
            z = A @ z
            loss = loss + ((z - Z_t[:, t]) ** 2).sum()
        if reg > 0:
            loss = loss + reg * (A ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_curve.append(float(loss.item()))
        if verbose and it % 200 == 0:
            print(f'  iter {it:4d}  loss={loss.item():.4f}')

    A_ol = A.detach().numpy()

    # Evaluate open-loop trajectory
    Z_ol = np.zeros_like(Z)
    Z_ol[:, 0] = Z[:, 0]
    for t in range(1, n_t):
        Z_ol[:, t] = A_ol @ Z_ol[:, t - 1]

    return {
        'Z':            Z,
        'Z_ol':         Z_ol,
        'A_ol':         A_ol,
        'A_onestep':    A_init,
        'eigs_ol':      np.linalg.eigvals(A_ol),
        'eigs_onestep': np.linalg.eigvals(A_init),
        'r2_openloop':  _r2(Z, Z_ol),
        'r2_onestep':   _r2(Z[:, 1:], A_ol @ Z[:, :-1]),
        'loss_curve':   loss_curve,
        'var_explained': np.cumsum(var),
    }


def compare_lds_vs_piecewise(
    W: np.ndarray,
    breakpoints: list[int],
    k: int = 10,
) -> dict:
    """
    Side-by-side comparison of global LDS vs piecewise LDS.

    Returns a dict suitable for display:
        'global'    : diagnose_lds_residuals result
        'piecewise' : fit_piecewise_lds result
        'improvement_openloop' : float — Δ R² open-loop
    """
    global_res = diagnose_lds_residuals(W, k)
    pw_res = fit_piecewise_lds(W, breakpoints, k)
    return {
        'global':               global_res,
        'piecewise':            pw_res,
        'improvement_openloop': pw_res['r2_openloop'] - global_res['r2_openloop'],
    }
