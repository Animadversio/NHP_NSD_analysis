"""Time-resolved multi-output ridge regression pipeline."""

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA

MIN_VAR = 1e-6
ALPHAS  = np.logspace(-2, 6, 25)
N_PCA   = 200


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Per-target R², clipped to [-1, 1]. Returns NaN for near-constant targets.

    Parameters
    ----------
    y_true, y_pred : (n_samples, n_targets)
    """
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    r2 = np.where(ss_tot > MIN_VAR, 1.0 - ss_res / ss_tot, np.nan)
    return np.clip(r2, -1.0, 1.0).astype(np.float32)


def fit_pca(features: np.ndarray, train_idx: np.ndarray,
            n_components: int = N_PCA) -> tuple[np.ndarray, np.ndarray, PCA]:
    """
    Fit PCA on train split, return (X_train, X_test, pca).

    Parameters
    ----------
    features  : (n_images, n_feat)
    train_idx : indices of training images
    """
    test_idx = np.setdiff1d(np.arange(len(features)), train_idx)
    pca = PCA(n_components=min(n_components, features.shape[1]))
    X_train = pca.fit_transform(features[train_idx])
    X_test  = pca.transform(features[test_idx])
    return X_train, X_test, pca


def time_resolved_regression(
    response: np.ndarray,
    X_train: np.ndarray,
    X_test:  np.ndarray,
    train_idx: np.ndarray,
    time_indices: np.ndarray,
    save_coefs: bool = False,
    alphas: np.ndarray = ALPHAS,
) -> dict:
    """
    Run per-time-bin multi-output RidgeCV regression.

    Parameters
    ----------
    response    : (n_units, n_time, n_images)
    X_train     : (n_train, n_pca)
    X_test      : (n_test,  n_pca)
    train_idx   : indices of training images (into n_images axis)
    time_indices: which time bins to evaluate
    save_coefs  : if True, also return weight matrix (n_t, n_units, n_pca)

    Returns
    -------
    dict:
        'r2'      : (n_t, n_units) float32    — test R²
        'r2_train': (n_t, n_units) float32    — train R²
        'coefs'   : (n_t, n_units, n_pca) float32  — only if save_coefs=True
    """
    n_units = response.shape[0]
    n_t     = len(time_indices)
    test_idx = np.setdiff1d(np.arange(response.shape[2]), train_idx)

    r2       = np.full((n_t, n_units), np.nan, dtype=np.float32)
    r2_train = np.full((n_t, n_units), np.nan, dtype=np.float32)
    coefs    = np.zeros((n_t, n_units, X_train.shape[1]), dtype=np.float32) if save_coefs else None

    for ti, tidx in enumerate(time_indices):
        y = response[:, tidx, :].T                    # (n_images, n_units)
        clf = RidgeCV(alphas=alphas, alpha_per_target=True)
        clf.fit(X_train, y[train_idx])
        yhat       = clf.predict(X_test)
        yhat_train = clf.predict(X_train)
        r2[ti]       = safe_r2(y[test_idx],  yhat)
        r2_train[ti] = safe_r2(y[train_idx], yhat_train)
        if save_coefs:
            coefs[ti] = clf.coef_                     # (n_units, n_pca)

    out = {'r2': r2, 'r2_train': r2_train}
    if save_coefs:
        out['coefs'] = coefs
    return out


def weighted_layer_depth(
    r2_perunit: np.ndarray,
    min_sig: float = 0.02,
) -> np.ndarray:
    """
    Weighted center-of-mass layer depth per unit per time bin.

    Parameters
    ----------
    r2_perunit : (n_layers, n_time, n_units) float32
    min_sig    : minimum total positive R² to report a depth (else NaN)

    Returns
    -------
    depth : (n_time, n_units) float32 — NaN where not significant
    """
    n_layers = r2_perunit.shape[0]
    r2_pos = np.clip(r2_perunit, 0, None)            # (L, T, U)
    total  = np.nansum(r2_pos, axis=0)               # (T, U)
    depths = np.arange(n_layers)[:, None, None]
    com    = np.nansum(r2_pos * depths, axis=0) / np.where(total > 0, total, np.nan)
    com    = np.where(total >= min_sig, com, np.nan)
    return com.astype(np.float32)
