"""
Triple-N: Regression Weight Dynamics Across V1, V4, IT
=======================================================
For each brain area, extract the time-resolved regression weight vector
w[t, u] ∈ R^n_pca — the "coding axis" that maps DINOv2 features to unit u
at time t — then analyse how this axis rotates over time.

Analyses:
  A. Per-unit cosine similarity vs reference time (peak R² time per area)
  B. Population subspace canonical angles
  C. Trajectory PCA + participation ratio
  D. Linear state-space model (LDS): z[t+1] = A z[t]

Best DINOv2 block per area (from tripleN_layer_time_regression.py report):
  V1 → blocks.1_cls (B2)
  V4 → blocks.4_cls (B5)
  IT → blocks.8_cls (B9)

Sessions (same as layer-time analysis):
  V1: ses72, ses75, ses77
  V4: ses79, ses80, ses83
  IT: ses1,  ses2,  ses3

Usage:
    python notebooks/tripleN_weight_dynamics.py

Output (figures/tripleN_weight_dynamics/):
    fig_wdyn_cosine.png
    fig_wdyn_canonical_angles.png
    fig_wdyn_trajectory_pca.png
    fig_wdyn_lds.png
"""
import os, sys, glob, pickle
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
from numpy.linalg import svd as npsvd
from scipy.linalg import subspace_angles

import h5py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.tripleN import TRIPLE_N_ROOT, area_metadata, get_area_mask
from NSD_utils.h5_dataset_utils import load_data_from_GoodUnitStrc

# ── paths & parameters ────────────────────────────────────────────────────────
STORE_DIR  = os.environ.get('STORE_DIR',
    '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang')
WCOEF_DIR  = os.path.join(STORE_DIR, 'weight_coefs', 'tripleN')
FEAT_CACHE = os.path.join(os.path.dirname(__file__), 'cache', 'dinov2_nsd_features.pkl')
FIGDIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', 'figures', 'tripleN_weight_dynamics')

os.makedirs(WCOEF_DIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

# Best DINOv2 block per area (0-indexed)
BEST_BLOCK  = {'V1': 1, 'V4': 4, 'IT': 8}
AREA_COLORS = {'V1': '#2166ac', 'V4': '#1a9641', 'IT': '#d73027'}
# Sessions to use per area (same as layer-time analysis)
AREA_SESSIONS = {
    'V1': [72, 75, 77],
    'V4': [79, 80, 83],
    'IT': [1,  2,  3],
}
# Approximate peak time per area (ms) — reference for cosine similarity
AREA_PEAK_MS = {'V1': 71, 'V4': 61, 'IT': 96}

N_PCA      = 100
ALPHAS     = np.logspace(-2, 6, 20)
REL_THRESH = 0.2
STRIDE     = 5
MIN_VAR    = 1e-6
K_LDS      = 10
TOP_K_ANG  = 10


# ── helpers ───────────────────────────────────────────────────────────────────

def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    r2 = np.where(ss_tot > MIN_VAR, 1 - ss_res / ss_tot, np.nan)
    return np.clip(r2, -1, 1).astype(np.float32)


def load_goodunit_for_session(ses_idx: int, root: str = TRIPLE_N_ROOT):
    """Return (h5py.File, filename) for GoodUnit matching ses_idx, or (None, None)."""
    proc_files = glob.glob(f'{root}/Processed/Processed_ses{ses_idx:02d}_*.mat')
    if not proc_files:
        return None, None
    parts      = os.path.basename(proc_files[0]).replace('.mat', '').split('_')
    date, probe = parts[2], parts[4]
    gu = glob.glob(f'{root}/GoodUnit/GoodUnit_{date}_*_NSD1000_LOC_g{probe}.mat')
    if not gu:
        return None, None
    return h5py.File(gu[0], 'r'), os.path.basename(gu[0])


def extract_weight_matrix(psth_sel: np.ndarray,
                           X_tr: np.ndarray, X_te: np.ndarray,
                           t_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit RidgeCV at each time bin and return weight matrix.

    Parameters
    ----------
    psth_sel : (n_units, n_time, n_images)
    X_tr     : (n_train, n_pca)
    X_te     : (n_test,  n_pca)
    t_indices: 1-D array of time bin indices to use

    Returns
    -------
    W    : (n_t, n_units, n_pca) float32
    r2   : (n_t, n_units) float32 — test R²
    """
    n_t     = len(t_indices)
    n_units = psth_sel.shape[0]
    n_img   = psth_sel.shape[2]
    train_idx = np.arange(len(X_tr))
    test_idx  = np.arange(len(X_te))

    W  = np.zeros((n_t, n_units, X_tr.shape[1]), dtype=np.float32)
    r2 = np.full((n_t, n_units), np.nan, dtype=np.float32)

    for ti, tidx in enumerate(t_indices):
        if ti % 10 == 0:
            print(f"    t {ti+1}/{n_t}", end='\r')
        y = psth_sel[:, tidx, :].T    # (n_images, n_units)
        clf = RidgeCV(alphas=ALPHAS, alpha_per_target=True)
        clf.fit(X_tr, y[:len(X_tr)])
        W[ti]  = clf.coef_             # (n_units, n_pca)
        r2[ti] = safe_r2(y[len(X_tr):len(X_tr)+len(X_te)], clf.predict(X_te))

    return W, r2


# ── weight dynamics functions ─────────────────────────────────────────────────

def cosine_similarity_matrix(W: np.ndarray, ref_idx: int) -> np.ndarray:
    """(n_t, n_units) cosine similarity of w[t] vs w[ref_idx]."""
    W_ref = W[ref_idx]
    cos_t = np.zeros((W.shape[0], W.shape[1]), dtype=np.float32)
    for ti in range(W.shape[0]):
        num = np.einsum('ij,ij->i', W[ti], W_ref)
        den = np.linalg.norm(W[ti], axis=1) * np.linalg.norm(W_ref, axis=1) + 1e-12
        cos_t[ti] = num / den
    return cos_t


def canonical_angles_vs_ref(W: np.ndarray, ref_idx: int,
                             top_k: int = TOP_K_ANG) -> np.ndarray:
    """(n_t, top_k) canonical angles in degrees between W[t] and W[ref_idx]."""
    U_ref, _, _ = npsvd(W[ref_idx].astype(np.float64), full_matrices=False)
    Q_ref = U_ref[:, :top_k]
    angles = []
    for ti in range(W.shape[0]):
        U_t, _, _ = npsvd(W[ti].astype(np.float64), full_matrices=False)
        angles.append(np.degrees(subspace_angles(Q_ref, U_t[:, :top_k])))
    return np.array(angles, dtype=np.float32)


def trajectory_pca(W: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """SVD of the time × (units*pca) weight matrix."""
    W_flat = W.reshape(W.shape[0], -1).astype(np.float64)
    W_flat -= W_flat.mean(axis=0, keepdims=True)
    U, sv, _ = npsvd(W_flat, full_matrices=False)
    var = sv ** 2 / (sv ** 2).sum()
    PR  = float((sv ** 2).sum() ** 2 / (sv ** 4).sum())
    coords = U * sv
    return coords, var, PR


def fit_lds(W: np.ndarray, k: int = K_LDS) -> dict:
    """Fit z[t+1] = A z[t] in the top-k PCA subspace."""
    W_flat = W.reshape(W.shape[0], -1).astype(np.float64)
    W_flat -= W_flat.mean(axis=0, keepdims=True)
    U, sv, _ = npsvd(W_flat, full_matrices=False)
    var   = sv ** 2 / (sv ** 2).sum()
    Z     = (U[:, :k] * sv[:k]).T         # (k, n_t)
    A, _, _, _ = np.linalg.lstsq(Z[:, :-1].T, Z[:, 1:].T, rcond=None)
    A = A.T
    eigs = np.linalg.eigvals(A)
    Z_pred = np.zeros_like(Z); Z_pred[:, 0] = Z[:, 0]
    for t in range(1, Z.shape[1]):
        Z_pred[:, t] = A @ Z_pred[:, t - 1]
    def r2_mat(true, pred):
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - true.mean(axis=1, keepdims=True)) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-12))
    return {
        'Z': Z, 'Z_pred': Z_pred, 'A': A, 'eigs': eigs,
        'r2_onestep':  r2_mat(Z[:, 1:], A @ Z[:, :-1]),
        'r2_openloop': r2_mat(Z, Z_pred),
        'var_explained': np.cumsum(var),
        'PR': float((sv ** 2).sum() ** 2 / (sv ** 4).sum()),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # Load feature cache
    print("Loading DINOv2 features …")
    with open(FEAT_CACHE, 'rb') as f:
        feat_cache = pickle.load(f)

    n_img   = feat_cache['blocks.0_cls'].shape[0]
    rng     = np.random.default_rng(42)
    idx_all = np.arange(n_img)
    tr_idx  = rng.choice(n_img, int(0.8 * n_img), replace=False)
    te_idx  = np.setdiff1d(idx_all, tr_idx)

    results = {}    # area → dict of analysis results

    for area in ['V1', 'V4', 'IT']:
        block_key = f'blocks.{BEST_BLOCK[area]}_cls'
        print(f"\n=== {area}  (DINOv2 {block_key}) ===")

        feat = feat_cache[block_key]
        pca  = PCA(n_components=N_PCA)
        X_tr = pca.fit_transform(feat[tr_idx])
        X_te = pca.transform(feat[te_idx])

        W_list, r2_list = [], []
        t_ms_ref = None

        for ses_idx in AREA_SESSIONS[area]:
            fh, fname = load_goodunit_for_session(ses_idx)
            if fh is None:
                print(f"  ses{ses_idx}: GoodUnit not found, skipping")
                continue

            d     = load_data_from_GoodUnitStrc(fh)
            psth  = d['response_matrix_img'].astype(np.float32)  # (n_units, n_time, n_images)
            t_full = d['PsthRange']
            fh.close()

            # Time indices (every STRIDE bins, from -49 ms)
            t_indices = np.where(
                (t_full >= -49) & (np.arange(len(t_full)) % STRIDE == 0)
            )[0]
            t_ms = t_full[t_indices]
            if t_ms_ref is None:
                t_ms_ref = t_ms

            # Area + reliability mask from Processed file
            proc = glob.glob(f'{TRIPLE_N_ROOT}/Processed/Processed_ses{ses_idx:02d}_*.mat')
            pd_  = sio.loadmat(proc[0])
            rel  = np.array(pd_['reliability_best']).ravel().astype(np.float32)
            pos  = np.array(pd_['pos']).ravel().astype(np.float32)

            area_mask = get_area_mask(ses_idx, pos, area)
            sel       = area_mask & (rel >= REL_THRESH)
            n_sel     = sel.sum()
            print(f"  ses{ses_idx}: {area_mask.sum()} {area} units, {n_sel} reliable")
            if n_sel == 0:
                continue

            # Map psth to Processed unit ordering: psth rows ↔ Processed rows
            # (GoodUnit and Processed share the same unit ordering per session)
            psth_sel = psth[sel]    # (n_sel, n_time, n_images)

            # Check or verify psth shape vs tr/te split
            assert psth_sel.shape[2] == n_img, \
                f"Image count mismatch: psth {psth_sel.shape[2]} vs feat {n_img}"

            # Build tr/te sub-arrays aligned to the original tr_idx / te_idx
            # psth_sel[:, tidx, :] shape (n_img,) → index with tr_idx / te_idx
            X_tr_ordered = pca.transform(feat[tr_idx])
            X_te_ordered = pca.transform(feat[te_idx])

            # Weight extraction
            out_path = os.path.join(WCOEF_DIR, f'{area}_ses{ses_idx:02d}_coefs.npy')
            if os.path.exists(out_path):
                W_ses = np.load(out_path)
                r2_ses = None    # not cached separately — skip
                print(f"    loaded from cache: {W_ses.shape}")
            else:
                W_ses, r2_ses = extract_weight_matrix(
                    psth_sel, X_tr_ordered, X_te_ordered, t_indices
                )
                np.save(out_path, W_ses)
                print(f"\n    saved: {W_ses.shape}")

            W_list.append(W_ses)

        if not W_list:
            print(f"  No data for {area}, skipping.")
            continue

        # Concatenate units across sessions: (n_t, N_total, n_pca)
        W_area = np.concatenate(W_list, axis=1)
        N_units = W_area.shape[1]
        print(f"\n  {area} total: W {W_area.shape}")

        # Reference time index = closest to AREA_PEAK_MS[area]
        t_ref_idx = int(np.argmin(np.abs(t_ms_ref - AREA_PEAK_MS[area])))
        t_ref_ms  = t_ms_ref[t_ref_idx]
        print(f"  ref time: {t_ref_ms:.0f} ms (idx {t_ref_idx})")

        # A: cosine similarity
        cos_t = cosine_similarity_matrix(W_area, t_ref_idx)

        # B: canonical angles
        print("  computing canonical angles …")
        cang  = canonical_angles_vs_ref(W_area, t_ref_idx)

        # C: trajectory PCA
        coords, var_exp, PR = trajectory_pca(W_area)

        # D: LDS
        print("  fitting LDS …")
        lds = fit_lds(W_area, k=K_LDS)

        results[area] = {
            't_ms': t_ms_ref,
            't_ref_idx': t_ref_idx,
            'cos_t': cos_t,
            'cang':  cang,
            'coords': coords, 'var_exp': var_exp, 'PR': PR,
            'lds': lds,
            'N_units': N_units,
            'block': BEST_BLOCK[area],
        }

    # ── Figures ──────────────────────────────────────────────────────────────

    AREAS = [a for a in ['V1', 'V4', 'IT'] if a in results]

    # Figure 1: Cosine similarity
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, area in zip(axes, AREAS):
        r   = results[area]
        t   = r['t_ms']
        cos = r['cos_t']
        color = AREA_COLORS[area]
        mean_c = gaussian_filter1d(np.nanmean(cos, axis=1), 1.5)
        std_c  = np.nanstd(cos, axis=1)
        ax.fill_between(t, mean_c - std_c, mean_c + std_c, alpha=0.2, color=color)
        ax.plot(t, mean_c, color=color, lw=2, label=f'N={r["N_units"]}')
        ax.axvline(t[r['t_ref_idx']], color='k', lw=1, ls='--', alpha=0.6,
                   label=f't_ref={t[r["t_ref_idx"]]:.0f} ms')
        ax.axhline(0, color='k', lw=0.5, ls='--')
        ax.axvline(0, color='gray', lw=0.8, ls='--')
        ax.set_xlabel('Time (ms)'); ax.set_ylabel('Cosine similarity')
        ax.set_title(f'{area}  B{r["block"]+1}  (N={r["N_units"]})', fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_ylim(-0.3, 1.05)
    fig.suptitle('Weight axis stability: cos(w[t], w[t_ref]) per unit, mean ± SD', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'fig_wdyn_cosine.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: Cosine similarity — overlay comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for area in AREAS:
        r   = results[area]
        t   = r['t_ms']
        mean_c = gaussian_filter1d(np.nanmean(r['cos_t'], axis=1), 1.5)
        axes[0].plot(t, mean_c, color=AREA_COLORS[area], lw=2.5,
                     label=f'{area} (t_ref={t[r["t_ref_idx"]]:.0f}ms)')
    axes[0].axhline(0, color='k', lw=0.5, ls='--')
    axes[0].axvline(0, color='gray', lw=0.8, ls='--')
    axes[0].set_xlabel('Time (ms)'); axes[0].set_ylabel('Mean cosine similarity')
    axes[0].set_title('Axis stability comparison'); axes[0].legend()
    axes[0].set_ylim(-0.3, 1.05)

    # Heatmap: cosine similarity sorted by peak R²
    for area in AREAS:
        r  = results[area]
        t  = r['t_ms']
        cos = r['cos_t']
        # sort by mean cosine in pre-ref window
        sort_order = np.argsort(np.nanmean(cos, axis=0))
        im = axes[1].imshow(cos[:, sort_order].T,
                            aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1,
                            extent=[t[0], t[-1], 0, r['N_units']])
        break    # show V1 as example; could make 3 panels
    axes[1].axvline(t[results[AREAS[0]]['t_ref_idx']], color='k', lw=1, ls='--')
    axes[1].set_xlabel('Time (ms)'); axes[1].set_ylabel('Units (sorted)')
    axes[1].set_title(f'{AREAS[0]} cosine similarity heatmap')
    plt.colorbar(im, ax=axes[1], shrink=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'fig_wdyn_cosine_compare.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 3: Canonical angles
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for area in AREAS:
        r = results[area]; t = r['t_ms']
        mean_ang = gaussian_filter1d(r['cang'].mean(axis=1), 1.5)
        axes[0].plot(t, mean_ang, color=AREA_COLORS[area], lw=2.5,
                     label=f'{area}')
        for ki, k in enumerate([0, 2, 4]):
            axes[1].plot(t, gaussian_filter1d(r['cang'][:, k], 1.5),
                         color=AREA_COLORS[area],
                         alpha=[0.9, 0.55, 0.3][ki],
                         lw=1.8,
                         label=f'{area} ang{k+1}' if area == 'IT' else None)
    for ax in axes:
        ax.axvline(0, color='gray', lw=0.8, ls='--')
        ax.set_xlabel('Time (ms)')
    axes[0].set_ylabel('Mean canonical angle (°)')
    axes[0].set_title('Population subspace rotation from t_ref')
    axes[0].legend()
    axes[1].set_ylabel('Canonical angle (°)')
    axes[1].set_title('Leading canonical angles (V1/V4/IT, opacity=rank)')
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'fig_wdyn_canonical_angles.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 4: Trajectory PCA
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for area in AREAS:
        r = results[area]
        axes[0].plot(np.arange(1, 21), r['var_exp'][:20] * 100,
                     'o-', color=AREA_COLORS[area], lw=2, ms=4,
                     label=f'{area}  PR={r["PR"]:.1f}')
    axes[0].axhline(80, color='k', ls='--', lw=0.8)
    axes[0].set_xlabel('PC'); axes[0].set_ylabel('Cumulative variance (%)')
    axes[0].set_title('Weight trajectory dimensionality'); axes[0].legend()

    for area in AREAS:
        r  = results[area]
        t  = r['t_ms']
        coords = r['coords']
        sc = axes[1].scatter(coords[:, 0], coords[:, 1],
                              c=t, cmap='RdYlGn', s=18, alpha=0.7,
                              vmin=t.min(), vmax=t.max())
        axes[1].plot(coords[:, 0], coords[:, 1],
                     color=AREA_COLORS[area], lw=1.2, alpha=0.5,
                     label=area)
    plt.colorbar(sc, ax=axes[1], label='Time (ms)', shrink=0.85)
    axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
    axes[1].set_title('Weight code trajectory (PC1-2)'); axes[1].legend()

    for area in AREAS:
        r   = results[area]
        lds = r['lds']
        axes[2].scatter(lds['eigs'].real, lds['eigs'].imag,
                        s=60, color=AREA_COLORS[area], zorder=3,
                        label=f'{area} R²={lds["r2_openloop"]:.2f}')
    theta = np.linspace(0, 2 * np.pi, 300)
    axes[2].plot(np.cos(theta), np.sin(theta), 'k--', lw=0.8, alpha=0.5)
    axes[2].set_xlim(-1.3, 1.3); axes[2].set_ylim(-1.3, 1.3); axes[2].set_aspect('equal')
    axes[2].set_xlabel('Re(λ)'); axes[2].set_ylabel('Im(λ)')
    axes[2].set_title(f'LDS eigenvalues (K={K_LDS})\nopen-loop R²')
    axes[2].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'fig_wdyn_trajectory_pca.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 5: LDS reconstruction
    fig, axes = plt.subplots(len(AREAS), 3, figsize=(14, 4 * len(AREAS)))
    for row, area in enumerate(AREAS):
        r   = results[area]
        lds = r['lds']
        t   = r['t_ms']
        color = AREA_COLORS[area]
        for ki, label in enumerate(['PC1', 'PC2']):
            ax = axes[row, ki]
            ax.plot(t, lds['Z'][ki], color=color, lw=2, label='Actual')
            ax.plot(t, lds['Z_pred'][ki], 'k--', lw=1.5, label='LDS open-loop')
            ax.set_xlabel('Time (ms)'); ax.set_ylabel(label)
            ax.set_title(f'{area} {label}  (one-step R²={lds["r2_onestep"]:.3f})')
            ax.legend(fontsize=8)
        ax = axes[row, 2]
        eigs = lds['eigs']
        sc = ax.scatter(eigs.real, eigs.imag,
                        c=np.abs(eigs), cmap='RdYlGn_r', s=80, vmin=0.9, vmax=1.05, zorder=3)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=0.8, alpha=0.4)
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_aspect('equal')
        ax.set_xlabel('Re(λ)'); ax.set_ylabel('Im(λ)')
        ax.set_title(f'{area} A eigenvalues\nopen-loop R²={lds["r2_openloop"]:.3f}')
        plt.colorbar(sc, ax=ax, shrink=0.85, label='|λ|')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'fig_wdyn_lds.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nAll figures saved to {FIGDIR}")
    for area in AREAS:
        lds = results[area]['lds']
        print(f"  {area}  N={results[area]['N_units']}  PR={results[area]['PR']:.1f}"
              f"  LDS one-step R²={lds['r2_onestep']:.3f}"
              f"  open-loop R²={lds['r2_openloop']:.3f}"
              f"  |λ| range [{np.abs(lds['eigs']).min():.3f}, {np.abs(lds['eigs']).max():.3f}]")


if __name__ == '__main__':
    main()
