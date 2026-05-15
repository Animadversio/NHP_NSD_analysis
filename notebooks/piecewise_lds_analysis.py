"""
Piecewise LDS analysis of regression weight trajectory dynamics.

For each NSD_N3 monkey:
  1. Load or compute weight coefficient tensor W (n_t, n_units, n_pca)
     using DINOv2 CLS block 7 features (best block from prior analysis).
  2. Restrict to stimulus-driven window (t >= 0ms) to avoid degenerate
     pre-stimulus segment.
  3. Diagnose global LDS residuals vs time (k=10).
  4. Fit piecewise LDS with K=2 biologically-motivated breakpoints
     at 150ms and 250ms (onset / peak / sustained) using k=5 (properly
     overdetermined: 30 bins > k^2=25 params per segment).
  5. Also run BIC changepoint detection (Dynp algorithm) for comparison.
  6. Save all results to WCOEF_DIR/<monkey>_piecewise_lds.pkl.

Run with:
    /n/home12/binxuwang/.conda/envs/torch2/bin/python notebooks/piecewise_lds_analysis.py
"""

import sys, os, pickle as pkl, numpy as np, h5py, matplotlib.pyplot as plt
import ruptures as rpt
from tqdm import tqdm
sys.path.insert(0, '/n/home12/binxuwang/Github/NHP_NSD_analysis')

from NSD_utils.h5_dataset_utils import load_data_from_GoodUnitStrc
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from core.piecewise_lds import (
    get_latent_trajectory, diagnose_lds_residuals,
    fit_piecewise_lds, find_changepoints_bic, compare_lds_vs_piecewise,
)

# ── paths ─────────────────────────────────────────────────────────────────────
CACHE_DIR  = '/n/home12/binxuwang/Github/NHP_NSD_analysis/notebooks/cache'
DATA_ROOT  = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/NSD_N3'
STORE_DIR  = os.environ.get('STORE_DIR',
             '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang')
WCOEF_DIR  = os.path.join(STORE_DIR, 'weight_coefs')
FIG_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'piecewise_lds')
os.makedirs(WCOEF_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── hyperparams ───────────────────────────────────────────────────────────────
N_PCA       = 200
ALPHAS      = np.logspace(-2, 6, 25)
TIME_STRIDE = 5
FEAT_TYPE   = 'cls'
BEST_BLOCK  = 7
K_GLOBAL    = 10   # LDS dim for global analysis
K_PW        = 5    # LDS dim for piecewise (ensures overdetermined segments)
ALPHA_RIDGE = 0.0  # ridge on A (0 = OLS; segments are overdetermined at K_PW=5)
MAX_BKPS    = 5    # BIC sweep range
# Fixed neural-phase breakpoints (ms) for biologically-grounded K=2 segmentation
BKP_MS      = [150, 250]    # onset/peak/sustained

SESSIONS = {
    'JianJian':     'GoodUnit_240629_JianJian_NSD1000_LOC_g2.mat',
    'FaCai':        'GoodUnit_240711_FaCai_NSD1000_LOC_g4.mat',
    'TuTu':         'GoodUnit_240724_TuTu_NSD1000_LOC_g2.mat',
    'ZhuangZhuang': 'GoodUnit_240817_ZhuangZhuang_NSD1000_LOC_g6.mat',
    'MaoDan':       'GoodUnit_240815_MaoDan_NSD1000_LOC_g5.mat',
}

# ── load DINOv2 features ──────────────────────────────────────────────────────
print('Loading DINOv2 features...')
FEAT_CACHE = os.path.join(CACHE_DIR, 'dinov2_nsd_features.pkl')
with open(FEAT_CACHE, 'rb') as f:
    feat_dict = pkl.load(f)

ln = f'blocks.{BEST_BLOCK}_{FEAT_TYPE}'
feats = feat_dict[ln]
n_images = feats.shape[0]

rng = np.random.RandomState(42)
train_idx = rng.choice(n_images, int(0.8 * n_images), replace=False)
test_idx  = np.setdiff1d(np.arange(n_images), train_idx)

pca = PCA(n_components=N_PCA)
X_train_pca = pca.fit_transform(feats[train_idx])
X_test_pca  = pca.transform(feats[test_idx])
print(f'PCA fitted: {N_PCA} components, {pca.explained_variance_ratio_.sum():.2%} variance')

# ── per-monkey loop ───────────────────────────────────────────────────────────
all_results = {}

for monkey, fname in SESSIONS.items():
    print(f'\n{"="*60}')
    print(f'  {monkey}')
    print(f'{"="*60}')

    # ── load neural data ──────────────────────────────────────────────────
    fpath = os.path.join(DATA_ROOT, fname)
    fh = h5py.File(fpath, 'r')
    d  = load_data_from_GoodUnitStrc(fh)
    R  = d['response_matrix_img']          # (n_units, n_time, n_images)
    t_full = d['PsthRange']
    fh.close()

    t_indices = np.where(
        (t_full >= -49) &
        (np.arange(len(t_full)) % TIME_STRIDE == 0)
    )[0]
    t_ms    = t_full[t_indices]
    n_t     = len(t_indices)
    n_units = R.shape[0]
    print(f'  {n_units} units, {n_t} time bins, t=[{t_ms[0]:.0f}, {t_ms[-1]:.0f}] ms')

    # ── compute or load coefficient tensor W ─────────────────────────────
    coef_path = os.path.join(WCOEF_DIR, f'{monkey}_cls_block{BEST_BLOCK}_coefs.npy')
    if os.path.exists(coef_path):
        print(f'  Loading cached coefs from {coef_path}')
        W = np.load(coef_path)
    else:
        print(f'  Computing ridge regression coefs ({n_t} time bins × {n_units} units)...')
        W = np.zeros((n_t, n_units, N_PCA), dtype=np.float32)
        for ti, tidx in enumerate(tqdm(t_indices, desc=f'  {monkey} coefs')):
            y = R[:, tidx, :].T              # (n_images, n_units)
            clf = RidgeCV(alphas=ALPHAS, alpha_per_target=True)
            clf.fit(X_train_pca, y[train_idx])
            W[ti] = clf.coef_                # (n_units, N_PCA)
        np.save(coef_path, W)
        print(f'  Saved to {coef_path}  ({W.nbytes / 1e6:.1f} MB)')

    # ── trim to stimulus-driven window (t >= 0ms) ─────────────────────────
    stim_start = int(np.searchsorted(t_ms, 0))
    W_stim  = W[stim_start:]
    t_stim  = t_ms[stim_start:]
    n_stim  = len(t_stim)
    print(f'  Stim window: {n_stim} bins, t=[{t_stim[0]:.0f}, {t_stim[-1]:.0f}] ms')

    # ── Step 1: global LDS on stim window ────────────────────────────────
    global_res = diagnose_lds_residuals(W_stim, k=K_GLOBAL)
    print(f'  Global LDS (k={K_GLOBAL}): one-step R²={global_res["r2_onestep"]:.3f}, '
          f'open-loop R²={global_res["r2_openloop"]:.3f}')

    # ── Step 2: BIC changepoint detection (Dynp, simple penalty) ─────────
    Z_bic, _, _ = get_latent_trajectory(W_stim, k=K_PW)
    algo_dynp = rpt.Dynp(model='rbf', min_size=5).fit(Z_bic.T.astype(np.float64))
    bic_costs, bic_vals, bic_all_bkps = [], [], []
    for n in range(0, MAX_BKPS + 1):
        bkps_n = algo_dynp.predict(n_bkps=n)
        cost_n = algo_dynp.cost.sum_of_costs(bkps_n)
        bic_n  = cost_n + n * np.log(n_stim)
        bic_vals.append(bic_n)
        bic_costs.append(cost_n)
        bic_all_bkps.append(bkps_n[:-1])
    best_n_bic  = int(np.argmin(bic_vals))
    bkps_bic    = bic_all_bkps[best_n_bic]
    bkp_ms_bic  = [t_stim[b] for b in bkps_bic]
    print(f'  BIC optimal: K={best_n_bic} breakpoints at bins {bkps_bic} → {[f"{m:.0f}ms" for m in bkp_ms_bic]}')

    # ── Step 3: fixed neural-phase piecewise LDS ─────────────────────────
    bkps_fixed = [int(np.searchsorted(t_stim, ms)) for ms in BKP_MS]
    bkps_fixed = [b for b in bkps_fixed if 0 < b < n_stim]
    bkp_ms_fixed = [t_stim[b] for b in bkps_fixed]
    print(f'  Fixed breakpoints: bins {bkps_fixed} → {[f"{m:.0f}ms" for m in bkp_ms_fixed]}')

    pw_fixed = fit_piecewise_lds(W_stim, breakpoints=bkps_fixed, k=K_PW, alpha=ALPHA_RIDGE)
    print(f'  Piecewise LDS (k={K_PW}, fixed): open-loop R²={pw_fixed["r2_openloop"]:.3f}  '
          f'(Δ vs global: {pw_fixed["r2_openloop"] - global_res["r2_openloop"]:+.3f})')
    for si, (seg, r2, eigs) in enumerate(zip(pw_fixed['segments'],
                                              pw_fixed['r2_onestep'],
                                              pw_fixed['eigs_list'])):
        t0 = t_stim[seg[0]]
        t1 = t_stim[min(seg[1], n_stim - 1)]
        n_pairs = seg[1] - seg[0] - 1
        print(f'    Seg {si} [{t0:.0f}–{t1:.0f}ms, {n_pairs} pairs]: '
              f'1-step R²={r2:.4f}, max|λ|={max(abs(eigs)):.4f}')

    # ── Step 4: BIC-breakpoint piecewise LDS ─────────────────────────────
    if bkps_bic:
        pw_bic = fit_piecewise_lds(W_stim, breakpoints=bkps_bic, k=K_PW, alpha=ALPHA_RIDGE)
        print(f'  Piecewise LDS (k={K_PW}, BIC): open-loop R²={pw_bic["r2_openloop"]:.3f}')
    else:
        pw_bic = pw_fixed  # fall back if BIC prefers no splits

    # ── save ──────────────────────────────────────────────────────────────
    result = {
        'monkey':        monkey,
        't_ms':          t_ms,
        't_stim':        t_stim,
        'stim_start':    stim_start,
        'n_units':       n_units,
        'K_global':      K_GLOBAL,
        'K_pw':          K_PW,
        'global_lds':    global_res,
        'bkps_fixed':    bkps_fixed,
        'bkp_ms_fixed':  bkp_ms_fixed,
        'pw_fixed':      pw_fixed,
        'bkps_bic':      bkps_bic,
        'bkp_ms_bic':    bkp_ms_bic,
        'bic_vals':      bic_vals,
        'bic_costs':     bic_costs,
        'bic_all_bkps':  bic_all_bkps,
        'pw_bic':        pw_bic,
    }
    out_path = os.path.join(WCOEF_DIR, f'{monkey}_piecewise_lds.pkl')
    with open(out_path, 'wb') as f:
        pkl.dump(result, f)
    print(f'  Results saved → {out_path}')

    all_results[monkey] = result

# ── summary table ─────────────────────────────────────────────────────────────
print('\n\n' + '='*75)
print(f'{"Monkey":<14} {"N_units":<9} {"Global R²ol":<14} {"PW R²ol (fixed)":<18} {"BIC K":<8} {"BIC bkps (ms)"}')
print('-'*75)
for monkey, res in all_results.items():
    g  = res['global_lds']['r2_openloop']
    pw = res['pw_fixed']['r2_openloop']
    bk = res['best_n_bic'] if 'best_n_bic' in res else len(res['bkps_bic'])
    bk_str = ', '.join(f'{m:.0f}' for m in res['bkp_ms_bic']) or '—'
    print(f'{monkey:<14} {res["n_units"]:<9} {g:<14.3f} {pw:<18.3f} {bk:<8} {bk_str}')

# ── summary figure ────────────────────────────────────────────────────────────
n_monkeys = len(all_results)
fig, axes = plt.subplots(2, n_monkeys, figsize=(4 * n_monkeys, 8))
if n_monkeys == 1:
    axes = axes[:, np.newaxis]

for col, (monkey, res) in enumerate(all_results.items()):
    t_stim_m = res['t_stim']
    g   = res['global_lds']
    pw  = res['pw_fixed']

    # Top row: per-step residuals with breakpoints marked
    ax = axes[0, col]
    ax.plot(t_stim_m[:-1], g['residuals'], color='steelblue', lw=1.5)
    for bk in pw['breakpoints']:
        ax.axvline(t_stim_m[bk], color='tomato', lw=1.5, ls='--', alpha=0.8)
    ax.set_title(f'{monkey}\n(n={res["n_units"]})', fontsize=10)
    ax.set_xlabel('Time (ms)')
    if col == 0:
        ax.set_ylabel('||z[t+1]−Az[t]||²')

    # Bottom row: latent trajectory PC1 vs PC2 — true vs open-loop predictions
    # Use pw['Z'] (k=K_PW, same SVD as Z_pred) as the reference to keep
    # all three curves in the same latent space.  Fit a k=K_PW global LDS
    # on pw['Z'] for a fair comparison (avoids sign-flip across SVD calls).
    Z_pw   = pw['Z']          # (K_PW, n_t) — true trajectory in k=5 space
    Z_pred = pw['Z_pred']     # (K_PW, n_t) — piecewise open-loop
    # Re-fit global A in the same k=5 space
    from numpy.linalg import lstsq as _lstsq
    A_gl5, _, _, _ = _lstsq(Z_pw[:, :-1].T, Z_pw[:, 1:].T, rcond=None)
    A_gl5 = A_gl5.T
    Z_gl5 = np.zeros_like(Z_pw)
    Z_gl5[:, 0] = Z_pw[:, 0]
    for t in range(1, Z_pw.shape[1]):
        Z_gl5[:, t] = A_gl5 @ Z_gl5[:, t - 1]

    ax2 = axes[1, col]
    ax2.plot(Z_pw[0], Z_pw[1], 'k-', lw=2, alpha=0.8, label='True')
    ax2.plot(Z_gl5[0], Z_gl5[1], color='steelblue', lw=1.5, ls='--', alpha=0.7,
             label=f'Global OL R²={g["r2_openloop"]:.2f}')
    ax2.plot(Z_pred[0], Z_pred[1], color='tomato', lw=1.5, ls='--', alpha=0.7,
             label=f'PW OL R²={pw["r2_openloop"]:.2f}')
    ax2.scatter(Z_pw[0, 0], Z_pw[1, 0], s=60, color='green', zorder=5)
    ax2.set_xlabel('PC1')
    if col == 0:
        ax2.set_ylabel('PC2')
        ax2.legend(fontsize=7)

fig.suptitle(f'Piecewise LDS — DINOv2 CLS block {BEST_BLOCK}  '
             f'(k_pw={K_PW}, breakpoints={BKP_MS} ms)', fontsize=12)
plt.tight_layout()
figpath = os.path.join(FIG_DIR, 'fig_piecewise_lds_summary.png')
plt.savefig(figpath, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSummary figure → {figpath}')

# ── eigenvalue figure (all monkeys, all segments) ─────────────────────────────
fig2, axes2 = plt.subplots(1, n_monkeys, figsize=(4 * n_monkeys, 4))
if n_monkeys == 1:
    axes2 = [axes2]

seg_colors = ['#2196F3', '#FF9800', '#4CAF50']  # onset / peak / sustained
seg_labels = ['Onset (0–150ms)', 'Peak (150–250ms)', 'Sustained (250–400ms)']
theta = np.linspace(0, 2 * np.pi, 200)

for col, (monkey, res) in enumerate(all_results.items()):
    ax = axes2[col]
    ax.plot(np.cos(theta), np.sin(theta), 'k-', lw=0.8, alpha=0.3, label='unit circle')
    pw = res['pw_fixed']
    for si, (eigs, color, label) in enumerate(zip(pw['eigs_list'], seg_colors, seg_labels)):
        ax.scatter(eigs.real, eigs.imag, color=color, s=40, alpha=0.8,
                   label=label if col == 0 else '_', zorder=3)
    ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    ax.axvline(0, color='k', lw=0.5, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_title(monkey, fontsize=10)
    ax.set_xlabel('Re(λ)')
    if col == 0:
        ax.set_ylabel('Im(λ)')

axes2[0].legend(fontsize=7, loc='lower left')
fig2.suptitle(f'Eigenvalues per segment — Piecewise LDS (k={K_PW})', fontsize=12)
plt.tight_layout()
figpath2 = os.path.join(FIG_DIR, 'fig_piecewise_eigenvalues.png')
plt.savefig(figpath2, dpi=150, bbox_inches='tight')
plt.close()
print(f'Eigenvalue figure → {figpath2}')
print('\nDone.')
