"""
Time-resolved neural regression across multiple monkeys/sessions.
Uses cached ResNet50 features. Runs one representative session per monkey.
"""
import sys, os, pickle, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter1d
import h5py

sys.path.insert(0, '/n/home12/binxuwang/Github/NHP_NSD_analysis')
sys.path.insert(0, '/n/home12/binxuwang/Github/Closed-loop-visual-insilico')

from NSD_utils.h5_dataset_utils import load_data_from_GoodUnitStrc
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# ── Config ──────────────────────────────────────────────────────────────────
DATA_ROOT = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/NSD_N3'
CACHE_DIR = '/n/home12/binxuwang/Github/NHP_NSD_analysis/notebooks/cache'
OUT_DIR   = '/n/home12/binxuwang/Github/NHP_NSD_analysis/notebooks'
os.makedirs(CACHE_DIR, exist_ok=True)

# One representative session per monkey
SESSIONS = {
    'JianJian':     'GoodUnit_240629_JianJian_NSD1000_LOC_g2.mat',
    'FaCai':        'GoodUnit_240711_FaCai_NSD1000_LOC_g4.mat',
    'TuTu':         'GoodUnit_240724_TuTu_NSD1000_LOC_g2.mat',
    'ZhuangZhuang': 'GoodUnit_240817_ZhuangZhuang_NSD1000_LOC_g6.mat',
    'MaoDan':       'GoodUnit_240815_MaoDan_NSD1000_LOC_g5.mat',
}

LAYERS = ['relu', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
LAYER_LABELS = ['input\n(relu)', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
LAYER_COLORS = plt.cm.viridis(np.linspace(0.1, 0.9, len(LAYERS)))
TIME_STRIDE = 5   # subsample time axis every N ms
SMOOTH_SIGMA = 10  # ms Gaussian smoothing for neural responses
N_PCA = 200        # PCA dims for features
ALPHAS = np.logspace(-2, 5, 14)

# ── Load cached features ────────────────────────────────────────────────────
feat_cache = os.path.join(CACHE_DIR, 'resnet50_nsd_features.pkl')
print("Loading cached ResNet50 features...")
with open(feat_cache, 'rb') as fh:
    feat_dict = pickle.load(fh)
# feat_dict[layer] = (n_images, flat_features)
print({k: v.shape for k, v in feat_dict.items()})

# ── Per-session PCA-reduced features (shared across time bins) ──────────────
def get_pca_features(feat_dict, n_pca=N_PCA):
    """Fit PCA per layer, return reduced features."""
    Xpca = {}
    pcas = {}
    for layer in LAYERS:
        X = feat_dict[layer].astype(np.float32)
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        pca = PCA(n_components=min(n_pca, X.shape[1], X.shape[0]-1), random_state=0)
        Xpca[layer] = pca.fit_transform(X)
        pcas[layer] = pca
        print(f"  {layer}: {X.shape} -> PCA {Xpca[layer].shape}")
    return Xpca, pcas


def run_timeresolved_regression(monkey, fname):
    """Run time-resolved ridge regression for one session."""
    result_cache = os.path.join(CACHE_DIR, f'time_resolved_{monkey}.pkl')
    if os.path.exists(result_cache):
        print(f"[{monkey}] Loading cached results...")
        with open(result_cache, 'rb') as fh:
            return pickle.load(fh)

    print(f"\n{'='*60}")
    print(f"[{monkey}] Loading neural data: {fname}")
    f = h5py.File(os.path.join(DATA_ROOT, fname), 'r')
    data = load_data_from_GoodUnitStrc(f)
    f.close()

    R = data['response_matrix_img']        # (units, time, images)
    psth_range = data['PsthRange']         # (time,) in ms
    n_units, n_time, n_images = R.shape
    print(f"  {n_units} units, {n_time} time pts, {n_images} images")
    print(f"  Time: {psth_range[0]:.0f} to {psth_range[-1]:.0f} ms")

    # Gaussian smooth along time
    R_smooth = gaussian_filter1d(R.astype(np.float32), sigma=SMOOTH_SIGMA, axis=1)

    # PCA-reduced features
    Xpca, _ = get_pca_features(feat_dict)

    # Time-resolved regression
    time_idx = np.arange(0, n_time, TIME_STRIDE)
    times_ms = psth_range[time_idx]

    # Train/test split on images
    n_img = n_images
    img_idx = np.arange(n_img)
    tr_idx, te_idx = train_test_split(img_idx, test_size=0.2, random_state=42)

    results = {layer: {'r2_mean': [], 'r2_std': []} for layer in LAYERS}

    for ti, t in zip(time_idx, times_ms):
        y = R_smooth[:, ti, :].T  # (n_images, n_units)
        y_tr, y_te = y[tr_idx], y[te_idx]

        for layer in LAYERS:
            X = Xpca[layer]
            X_tr, X_te = X[tr_idx], X[te_idx]

            # RidgeCV per-unit
            clf = RidgeCV(alphas=ALPHAS, fit_intercept=True)
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)

            # R² per unit
            ss_res = ((y_te - y_pred) ** 2).sum(axis=0)
            ss_tot = ((y_te - y_te.mean(axis=0)) ** 2).sum(axis=0)
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            r2 = np.clip(r2, -1, 1)

            results[layer]['r2_mean'].append(r2.mean())
            results[layer]['r2_std'].append(r2.std())

        if ti % 50 == 0:
            best = max(LAYERS, key=lambda l: results[l]['r2_mean'][-1])
            print(f"  t={t:+.0f}ms  best={best}  R²={results[best]['r2_mean'][-1]:.3f}")

    # Convert to arrays
    for layer in LAYERS:
        results[layer]['r2_mean'] = np.array(results[layer]['r2_mean'])
        results[layer]['r2_std'] = np.array(results[layer]['r2_std'])

    out = {'monkey': monkey, 'times_ms': times_ms, 'results': results,
           'n_units': n_units, 'n_images': n_images}

    with open(result_cache, 'wb') as fh:
        pickle.dump(out, fh)
    print(f"  Saved to {result_cache}")
    return out


# ── Run all sessions ─────────────────────────────────────────────────────────
all_results = {}
for monkey, fname in SESSIONS.items():
    all_results[monkey] = run_timeresolved_regression(monkey, fname)

print("\nAll sessions done!")

# ── Plot 1: R² over time per monkey (grid) ──────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)
axes = axes.flatten()

for ax, (monkey, res) in zip(axes, all_results.items()):
    times = res['times_ms']
    for i, (layer, lbl) in enumerate(zip(LAYERS, LAYER_LABELS)):
        r2 = res['results'][layer]['r2_mean']
        ax.plot(times, r2, color=LAYER_COLORS[i], lw=1.8, label=lbl)
    ax.axvline(0, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.set_title(f"{monkey} ({res['n_units']} units)", fontsize=11, fontweight='bold')
    ax.set_xlabel('Time from stimulus onset (ms)')
    ax.set_ylabel('Mean R²')
    ax.set_xlim(times[0], times[-1])
    ax.spines[['top','right']].set_visible(False)

# Legend in last panel
axes[-1].set_visible(False)
handles = [plt.Line2D([0],[0], color=LAYER_COLORS[i], lw=2, label=lbl)
           for i, lbl in enumerate(LAYER_LABELS)]
fig.legend(handles=handles, title='ResNet50 Layer', loc='lower right',
           bbox_to_anchor=(0.98, 0.05), fontsize=10, title_fontsize=10)

fig.suptitle('Time-resolved Neural Regression (R²) — All Monkeys\nResNet50 layers → LOC population response',
             fontsize=13, y=1.01)
plt.tight_layout()
p1 = os.path.join(OUT_DIR, 'fig_multimonkey_R2_time.png')
plt.savefig(p1, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {p1}")

# ── Plot 2: Best layer over time per monkey ──────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
axes = axes.flatten()
layer_to_num = {l: i for i, l in enumerate(LAYERS)}
cmap = matplotlib.colormaps['viridis']

for ax, (monkey, res) in zip(axes, all_results.items()):
    times = res['times_ms']
    # Best layer index at each time point
    r2_stack = np.stack([res['results'][l]['r2_mean'] for l in LAYERS], axis=0)
    best_layer_idx = r2_stack.argmax(axis=0)
    best_r2 = r2_stack.max(axis=0)

    sc = ax.scatter(times, best_layer_idx, c=best_r2, cmap='magma',
                    s=15, vmin=0, vmax=0.4, zorder=3)
    ax.axvline(0, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.set_yticks(range(len(LAYERS)))
    ax.set_yticklabels(LAYER_LABELS, fontsize=8)
    ax.set_title(f"{monkey}", fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (ms)')
    ax.set_ylim(-0.5, len(LAYERS) - 0.5)
    ax.spines[['top','right']].set_visible(False)

axes[-1].set_visible(False)
cbar = fig.colorbar(sc, ax=axes[:5], shrink=0.6, pad=0.02)
cbar.set_label('Best R²', fontsize=10)
fig.suptitle('Optimal ResNet50 Layer Over Time — All Monkeys\n(color = R² of best layer)',
             fontsize=13, y=1.01)
plt.tight_layout()
p2 = os.path.join(OUT_DIR, 'fig_multimonkey_best_layer.png')
plt.savefig(p2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {p2}")

# ── Plot 3: Peak R² comparison across monkeys per layer ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
monkeys = list(all_results.keys())
x = np.arange(len(LAYERS))
bar_width = 0.15
monkey_colors = plt.cm.Set2(np.linspace(0, 1, len(monkeys)))

ax = axes[0]
for mi, (monkey, res) in enumerate(all_results.items()):
    peak_r2 = [res['results'][l]['r2_mean'].max() for l in LAYERS]
    ax.bar(x + mi * bar_width, peak_r2, bar_width,
           label=monkey, color=monkey_colors[mi], alpha=0.85)
ax.set_xticks(x + bar_width * (len(monkeys)-1) / 2)
ax.set_xticklabels(LAYER_LABELS, fontsize=9)
ax.set_ylabel('Peak R²')
ax.set_title('Peak R² per Layer across Monkeys', fontweight='bold')
ax.legend(fontsize=9)
ax.spines[['top','right']].set_visible(False)

# Time of peak R² (overall best layer)
ax = axes[1]
for mi, (monkey, res) in enumerate(all_results.items()):
    times = res['times_ms']
    r2_stack = np.stack([res['results'][l]['r2_mean'] for l in LAYERS], axis=0)
    best_r2_curve = r2_stack.max(axis=0)
    peak_t = times[best_r2_curve.argmax()]
    # Plot best R² over time
    ax.plot(times, best_r2_curve, color=monkey_colors[mi], lw=2, label=monkey)
    ax.axvline(peak_t, color=monkey_colors[mi], ls=':', lw=1.2, alpha=0.7)

ax.axvline(0, color='k', ls='--', lw=0.8, alpha=0.5)
ax.set_xlabel('Time from stimulus onset (ms)')
ax.set_ylabel('Best layer R² (envelope)')
ax.set_title('Peak-layer R² over Time — All Monkeys', fontweight='bold')
ax.legend(fontsize=9)
ax.spines[['top','right']].set_visible(False)

plt.suptitle('Multi-monkey Comparison — Time-resolved ResNet50 Regression',
             fontsize=13, fontweight='bold')
plt.tight_layout()
p3 = os.path.join(OUT_DIR, 'fig_multimonkey_comparison.png')
plt.savefig(p3, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {p3}")

print("\nDone! Generated figures:")
for p in [p1, p2, p3]:
    print(f"  {p}")
