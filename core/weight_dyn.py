"""Regression weight dynamics: cosine similarity, subspace angles, LDS."""

import numpy as np
from numpy.linalg import svd
from scipy.linalg import subspace_angles as _subspace_angles


def cosine_similarity_over_time(
    W: np.ndarray,
    ref_idx: int,
) -> np.ndarray:
    """
    Per-unit cosine similarity of weight vectors relative to a reference time bin.

    Parameters
    ----------
    W       : (n_t, n_units, n_pca)
    ref_idx : time bin index to use as reference

    Returns
    -------
    cos_t : (n_t, n_units) — cosine similarity to W[ref_idx]
    """
    W_ref = W[ref_idx]                               # (n_units, n_pca)
    cos_t = np.zeros((W.shape[0], W.shape[1]))
    for ti in range(W.shape[0]):
        num = np.einsum('ij,ij->i', W[ti], W_ref)
        den = np.linalg.norm(W[ti], axis=1) * np.linalg.norm(W_ref, axis=1) + 1e-12
        cos_t[ti] = num / den
    return cos_t.astype(np.float32)


def canonical_angles_over_time(
    W: np.ndarray,
    ref_idx: int,
    top_k: int = 20,
) -> np.ndarray:
    """
    Canonical angles between population weight subspace at each time and ref_idx.

    Parameters
    ----------
    W      : (n_t, n_units, n_pca)
    ref_idx: reference time bin
    top_k  : number of leading singular vectors to compare

    Returns
    -------
    angles : (n_t, top_k) in degrees
    """
    U_ref, _, _ = svd(W[ref_idx], full_matrices=False)
    Q_ref = U_ref[:, :top_k]
    angles = []
    for ti in range(W.shape[0]):
        U_t, _, _ = svd(W[ti], full_matrices=False)
        Q_t = U_t[:, :top_k]
        angs = np.degrees(_subspace_angles(Q_ref, Q_t))
        angles.append(angs)
    return np.array(angles)              # (n_t, top_k)


def trajectory_pca(W: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    SVD of the time × (units*pca) weight trajectory matrix.

    Returns
    -------
    coords     : (n_t, n_t)    — time coordinates in SVD space (U * S)
    var_explained : (n_t,)     — fraction variance per PC
    PR         : float         — participation ratio
    """
    W_flat = W.reshape(W.shape[0], -1).astype(np.float64)
    W_flat -= W_flat.mean(axis=0, keepdims=True)
    U, sv, _ = svd(W_flat, full_matrices=False)
    coords = U * sv                               # (n_t, n_t)
    var    = sv ** 2 / (sv ** 2).sum()
    PR     = float((sv ** 2).sum() ** 2 / (sv ** 4).sum())
    return coords, var, PR


def fit_lds(
    W: np.ndarray,
    k: int = 10,
) -> dict:
    """
    Fit autonomous linear state-space model z[t+1] = A z[t] in the top-k
    PCA subspace of the weight trajectory.

    Parameters
    ----------
    W : (n_t, n_units, n_pca)
    k : dimensionality of LDS state

    Returns
    -------
    dict:
        'Z'          : (k, n_t)   — actual trajectory in latent space
        'Z_pred'     : (k, n_t)   — open-loop LDS prediction
        'A'          : (k, k)     — fitted transition matrix
        'eigs'       : (k,)       — eigenvalues of A
        'r2_onestep' : float      — one-step prediction R²
        'r2_openloop': float      — open-loop prediction R²
    """
    coords, var, PR = trajectory_pca(W)
    sv = np.sqrt(np.sum(W.reshape(W.shape[0], -1) ** 2, axis=1))   # rough scale
    # Re-derive properly: need U and sv separately
    W_flat = W.reshape(W.shape[0], -1).astype(np.float64)
    W_flat -= W_flat.mean(axis=0, keepdims=True)
    U, sv_arr, _ = svd(W_flat, full_matrices=False)
    Z = (U[:, :k] * sv_arr[:k]).T                 # (k, n_t)

    # Fit A
    A, _, _, _ = np.linalg.lstsq(Z[:, :-1].T, Z[:, 1:].T, rcond=None)
    A = A.T                                         # (k, k)
    eigs = np.linalg.eigvals(A)

    # Open-loop simulation
    Z_pred = np.zeros_like(Z)
    Z_pred[:, 0] = Z[:, 0]
    for t in range(1, Z.shape[1]):
        Z_pred[:, t] = A @ Z_pred[:, t - 1]

    def r2_matrix(true, pred):
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - true.mean(axis=1, keepdims=True)) ** 2)
        return float(1 - ss_res / ss_tot)

    return {
        'Z':           Z,
        'Z_pred':      Z_pred,
        'A':           A,
        'eigs':        eigs,
        'r2_onestep':  r2_matrix(Z[:, 1:], A @ Z[:, :-1]),
        'r2_openloop': r2_matrix(Z, Z_pred),
        'var_explained': np.cumsum(var),
        'PR':          PR,
    }
