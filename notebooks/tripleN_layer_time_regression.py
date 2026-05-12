"""
Triple-N: DINOv2 Layer × Time Regression
=========================================
For each brain area (V1, V4, IT), regress neural responses at each time bin
against DINOv2 CLS features from each of 12 transformer blocks.
Produces R²(layer, time) heatmaps revealing the cortical hierarchy.

Usage:
    python notebooks/tripleN_layer_time_regression.py

Output:
    figures/tripleN_layer_time/fig_layer_time_main.png
    figures/tripleN_layer_time/fig_layer_time_heatmap_unified.png
    figures/tripleN_layer_time/fig_layer_time_summary.png
"""
import os, sys, glob, pickle
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.tripleN import (
    TRIPLE_N_ROOT, area_metadata, load_session, get_area_mask
)
from NSD_utils.h5_dataset_utils import load_data_from_GoodUnitStrc
import h5py

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'tripleN_layer_time')
FEAT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache', 'dinov2_nsd_features.pkl')
MIN_VAR = 1e-6
ALPHAS   = np.logspace(-2, 6, 20)
N_PCA    = 100
REL_THRESH = 0.2
STRIDE   = 5   # sample every 5th time bin


# ── helpers ──────────────────────────────────────────────────────────────────

def safe_r2(y_true, y_pred):
    ss_res = ((y_true - y_pred)**2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0))**2).sum(axis=0)
    r2 = np.where(ss_tot > MIN_VAR, 1 - ss_res / ss_tot, np.nan)
    return np.clip(r2, -1, 1).astype(np.float32)


def load_goodunit_for_session(ses_idx, root=TRIPLE_N_ROOT):
    """Match GoodUnit file by date and probe number from Processed filename."""
    proc_files = glob.glob(f'{root}/Processed/Processed_ses{ses_idx:02d}_*.mat')
    if not proc_files:
        return None, None
    parts = os.path.basename(proc_files[0]).replace('.mat', '').split('_')
    date, probe_num = parts[2], parts[4]
    gu = glob.glob(f'{root}/GoodUnit/GoodUnit_{date}_*_NSD1000_LOC_g{probe_num}.mat')
    if not gu:
        return None, None
    return h5py.File(gu[0], 'r'), os.path.basename(gu[0])


def layer_time_regression(psth, feat_dict, layer_keys, time_ms,
                           reliability, rel_thresh=REL_THRESH,
                           stride=STRIDE, n_pca=N_PCA):
    """
    psth        : (n_units, n_time, n_images) float32
    feat_dict   : {key: (n_images, dim)}
    Returns
    -------
    r2_lt : (n_layers, n_times, n_units_sel) float32
    t_idx : time indices used
    sel   : boolean mask of selected units
    """
    sel = reliability >= rel_thresh
    psth_sel = psth[sel]
    N = psth_sel.shape[0]
    t_idx = np.where((time_ms >= -49) & (np.arange(len(time_ms)) % stride == 0))[0]
    T, L  = len(t_idx), len(layer_keys)

    n_img     = psth_sel.shape[2]
    rng       = np.random.default_rng(42)
    train_idx = rng.choice(n_img, int(0.8 * n_img), replace=False)
    test_idx  = np.setdiff1d(np.arange(n_img), train_idx)

    r2_lt = np.full((L, T, N), np.nan, dtype=np.float32)

    for li, key in enumerate(layer_keys):
        feat  = feat_dict[key]
        pca   = PCA(n_components=min(n_pca, feat.shape[1]))
        X_tr  = pca.fit_transform(feat[train_idx])
        X_te  = pca.transform(feat[test_idx])
        for ti, tidx in enumerate(t_idx):
            y = psth_sel[:, tidx, :].T        # (n_images, N)
            clf = RidgeCV(alphas=ALPHAS, alpha_per_target=True)
            clf.fit(X_tr, y[train_idx])
            r2_lt[li, ti] = safe_r2(y[test_idx], clf.predict(X_te))
        print(f"  layer {li+1}/{L}", end='\r')

    return r2_lt, t_idx, sel


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIGDIR, exist_ok=True)

    # Load DINOv2 features
    with open(FEAT_CACHE, 'rb') as f:
        feat_cache = pickle.load(f)
    layer_keys = [f'blocks.{i}_cls' for i in range(12)]

    df = area_metadata()
    area_sessions = {
        'V1': list(df[df['AREALABEL'] == 'V1']['SesIdx'].unique()),
        'V4': list(df[df['AREALABEL'] == 'V4']['SesIdx'].unique()),
        'IT': list(df[df['Area']      == 'IT']['SesIdx'].unique()[:5]),
    }

    AREAS = ['V1', 'V4', 'IT']
    area_r2    = {}
    PsthRange  = None

    for area in AREAS:
        sessions = area_sessions[area][:3]
        print(f"\n=== {area} — sessions {sessions} ===")
        r2_list = []

        for ses_idx in sessions:
            fh, fname = load_goodunit_for_session(ses_idx)
            if fh is None:
                continue
            d = load_data_from_GoodUnitStrc(fh)
            psth     = d['response_matrix_img'].astype(np.float32)
            time_ms  = d['PsthRange']
            fh.close()
            if PsthRange is None:
                PsthRange = time_ms

            proc = glob.glob(f'{TRIPLE_N_ROOT}/Processed/Processed_ses{ses_idx:02d}_*.mat')
            pd_  = sio.loadmat(proc[0])
            rel  = np.array(pd_['reliability_best']).ravel().astype(np.float32)
            pos  = np.array(pd_['pos']).ravel().astype(np.float32)

            area_mask  = get_area_mask(ses_idx, pos, area)
            psth_area  = psth[area_mask]
            rel_area   = rel[area_mask]

            n_sel = (rel_area >= REL_THRESH).sum()
            print(f"  ses{ses_idx}: {area_mask.sum()} {area} units, {n_sel} reliable")
            if n_sel == 0:
                continue

            r2_lt, t_idx, sel = layer_time_regression(
                psth_area, feat_cache, layer_keys, time_ms, rel_area
            )
            print(f"  → {r2_lt.shape}")
            r2_list.append(r2_lt)

        if r2_list:
            area_r2[area] = np.concatenate(r2_list, axis=2)
            print(f"{area} total: {area_r2[area].shape[2]} units")

    # Time axis for plotting
    t_plot  = PsthRange[t_idx]
    vmax    = max(np.nanmax(np.nanmean(area_r2[a], axis=2)) for a in AREAS)
    mean_r2 = {a: np.nanmean(area_r2[a], axis=2) for a in AREAS}

    # ── Figure 1: main heatmap with IT-V1 difference ──────────────────────────
    cmaps = {'V1': 'Blues', 'V4': 'Greens', 'IT': 'Reds'}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, area in zip(axes.flat[:3], AREAS):
        mat = mean_r2[area]
        im  = ax.imshow(mat, aspect='auto', origin='lower',
                        cmap=cmaps[area], vmin=0, vmax=vmax,
                        extent=[t_plot[0], t_plot[-1], 0.5, 12.5])
        ax.set_xlabel('Time (ms)', fontsize=11)
        ax.set_title(f'{area}  (N={area_r2[area].shape[2]:,})', fontsize=13, fontweight='bold')
        ax.axvline(0, color='white', lw=1, ls='--', alpha=0.7)
        plt.colorbar(im, ax=ax, shrink=0.88).set_label('mean R²', fontsize=9)
        ax.set_ylabel('DINOv2 block', fontsize=11)
        ax.set_yticks(range(1, 13))
        ax.set_yticklabels([f'B{i}' for i in range(1, 13)], fontsize=8)
        peak = np.unravel_index(np.argmax(mat), mat.shape)
        ax.plot(t_plot[peak[1]], peak[0] + 1, 'w*', ms=12, zorder=10)

    ax = axes[1, 1]
    diff = mean_r2['IT'] - mean_r2['V1']
    lim  = np.abs(diff).max()
    im   = ax.imshow(diff, aspect='auto', origin='lower',
                     cmap='RdBu_r', vmin=-lim, vmax=lim,
                     extent=[t_plot[0], t_plot[-1], 0.5, 12.5])
    ax.set_xlabel('Time (ms)', fontsize=11)
    ax.set_title('IT − V1  (higher layer & later → IT)', fontsize=12, fontweight='bold')
    ax.axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    plt.colorbar(im, ax=ax, shrink=0.88).set_label('ΔR²', fontsize=9)
    ax.set_ylabel('DINOv2 block', fontsize=11)
    ax.set_yticks(range(1, 13))
    ax.set_yticklabels([f'B{i}' for i in range(1, 13)], fontsize=8)

    fig.suptitle('DINOv2 Layer × Time Regression (Triple-N, 3 sessions/area)\n'
                 'White ★ = peak R²',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'fig_layer_time_main.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 2: summary curves ───────────────────────────────────────────────
    palette = {'V1': '#2166ac', 'V4': '#1a9641', 'IT': '#d73027'}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, area in enumerate(AREAS):
        mat   = mean_r2[area]
        color = palette[area]

        ax = axes[0, col]
        best_l = np.argmax(mat.max(axis=1))
        for li in range(12):
            ax.plot(t_plot, mat[li], color=color, alpha=0.15, lw=0.8)
        ax.plot(t_plot, mat[best_l], color=color, lw=2.5, label=f'Best: B{best_l+1}')
        ax.plot(t_plot, mat.max(axis=0), 'k--', lw=1.5, label='Max/layer')
        ax.axvline(0, color='gray', lw=0.8, ls='--')
        ax.set_title(f'{area} R² vs time', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('mean R²')
        ax.legend(fontsize=8)

        ax = axes[1, col]
        peak_t = np.argmax(mat.max(axis=0))
        xs = np.arange(1, 13)
        for ti in range(0, len(t_plot), 5):
            ax.plot(xs, mat[:, ti], color=color, alpha=0.12, lw=0.7)
        ax.plot(xs, mat[:, peak_t], color=color, lw=2.5,
                label=f'Peak t={t_plot[peak_t]:.0f}ms')
        ax.plot(xs, mat.max(axis=1), 'k--', lw=1.5, label='Max/time')
        ax.set_title(f'{area} R² vs layer', fontsize=12, fontweight='bold')
        ax.set_xlabel('DINOv2 block')
        ax.set_ylabel('mean R²')
        ax.set_xticks(range(1, 13))
        ax.legend(fontsize=8)

    plt.suptitle('Layer × Time R² summary: V1 / V4 / IT', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'fig_layer_time_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("\nAll figures saved to", FIGDIR)


if __name__ == '__main__':
    main()
