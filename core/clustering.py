"""Neuron clustering by time-depth profile."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer


def build_cluster_features(
    r2_perunit: np.ndarray,
    depth_curve: np.ndarray,
    t_ms: np.ndarray,
    early_window: tuple = (80, 150),
    late_window:  tuple = (180, 350),
) -> np.ndarray:
    """
    Build 137-dim feature vector per unit for clustering.

    Parameters
    ----------
    r2_perunit  : (n_layers, n_time, n_units)
    depth_curve : (n_time, n_units)  — weighted CoM depth (NaN where not sig)
    t_ms        : (n_time,)

    Returns
    -------
    features : (n_units, 137)
        = normalized depth curve (n_t=66) + R² envelope (n_t=66) + 5 summary stats
    """
    n_units = r2_perunit.shape[2]
    n_t     = len(t_ms)

    # Normalize depth curve to [0,1] per unit
    dc = depth_curve.T.copy()                        # (n_units, n_t)
    dc_norm = np.zeros_like(dc)
    for u in range(n_units):
        row = dc[u]
        valid = np.isfinite(row)
        if valid.sum() > 1:
            mn, mx = row[valid].min(), row[valid].max()
            dc_norm[u] = np.where(valid, (row - mn) / (mx - mn + 1e-8), 0.0)

    # Max R² across layers per time bin
    r2_env = np.nanmax(r2_perunit, axis=0).T         # (n_units, n_t)

    # Summary statistics
    early = (t_ms >= early_window[0]) & (t_ms <= early_window[1])
    late  = (t_ms >= late_window[0])  & (t_ms <= late_window[1])

    early_depth = np.nanmean(depth_curve[early, :], axis=0)   # (n_units,)
    late_depth  = np.nanmean(depth_curve[late,  :], axis=0)
    delta_depth = late_depth - early_depth
    peak_r2     = np.nanmax(r2_env, axis=1)
    peak_t      = t_ms[np.nanargmax(r2_env, axis=1)]

    summary = np.column_stack([
        early_depth, late_depth, delta_depth, peak_r2, peak_t
    ])                                                # (n_units, 5)

    return np.concatenate([dc_norm, r2_env, summary], axis=1)  # (n_units, 137)


def cluster_units(
    features: np.ndarray,
    k_range: range = range(2, 9),
    pca_var: float = 0.80,
    random_state: int = 42,
) -> tuple[np.ndarray, int, np.ndarray]:
    """
    PCA → k-means, k selected by silhouette score.

    Returns
    -------
    labels    : (n_units,) int cluster assignments (best k)
    best_k    : int
    sil_scores: (len(k_range),) silhouette score per k
    """
    imputer = SimpleImputer(strategy='median')
    scaler  = StandardScaler()
    pca     = PCA(n_components=pca_var)

    X = imputer.fit_transform(features)
    X = scaler.fit_transform(X)
    X = pca.fit_transform(X)

    sil_scores = []
    all_labels = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        lbl = km.fit_predict(X)
        sil_scores.append(silhouette_score(X, lbl))
        all_labels.append(lbl)

    best_idx  = int(np.argmax(sil_scores))
    best_k    = list(k_range)[best_idx]
    labels    = all_labels[best_idx]
    return labels, best_k, np.array(sil_scores)
