"""
Per-Unit Time-Resolved Neural Encoding Dynamics
Atlas agent — Apr 7, 2026

Multi-output RidgeCV regression: ResNet50 layers → each unit separately.
Stores r2_perunit[n_layers, n_times, n_units] and generates per-unit figures.
"""
import sys, os, glob, pickle
sys.path.insert(0, '/n/home12/binxuwang/Github/NHP_NSD_analysis')
sys.path.insert(0, '/n/home12/binxuwang/Github/Closed-loop-visual-insilico')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py, torch
from os.path import join, exists
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tqdm import tqdm

from NSD_utils.h5_dataset_utils import load_data_from_GoodUnitStrc

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_ROOT    = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/NSD_N3'
SESSION_FILE = join(DATA_ROOT, 'GoodUnit_240629_JianJian_NSD1000_LOC_g2.mat')
CACHE_DIR    = '/n/home12/binxuwang/Github/NHP_NSD_analysis/notebooks/cache'
OUT_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

LAYERS       = ['relu', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
LAYER_LABELS = ['stem/relu', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
LAYER_COLORS = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3', '#a65628']
TIME_STRIDE  = 5    # ms
SMOOTH_SIGMA = 10   # ms
ALPHAS       = [1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5]

# ─── Load features ────────────────────────────────────────────────────────────
feat_cache = join(CACHE_DIR, 'resnet50_nsd_features.pkl')
print("Loading cached ResNet50 features...")
with open(feat_cache, 'rb') as fh:
    feat_dict = pickle.load(fh)
print({k: v.shape for k, v in feat_dict.items()})

# ─── Load neural data ─────────────────────────────────────────────────────────
print(f"\nLoading neural data: {os.path.basename(SESSION_FILE)}")
f = h5py.File(SESSION_FILE, 'r')
d = load_data_from_GoodUnitStrc(f)
f.close()

psth = d['PsthRange']          # (450,) ms
R    = d['response_matrix_img'] # (n_units, n_time, n_images)
n_units, n_time, n_images = R.shape
dt   = float(psth[1] - psth[0])
print(f"  {n_units} units, {n_time} timepoints, {n_images} images, dt={dt:.1f}ms")

# Smooth + convert to firing rate
sigma_bins = SMOOTH_SIGMA / dt
R_smooth   = gaussian_filter1d(R.astype(np.float32), sigma=sigma_bins, axis=1)
R_fr       = R_smooth / dt * 1000  # sp/s

# Train/test split (fixed)
np.random.seed(42)
idx_train, idx_test = train_test_split(np.arange(n_images), test_size=0.2, random_state=42)
print(f"  Train: {len(idx_train)}, Test: {len(idx_test)}")

# ─── Time-resolved per-unit regression ────────────────────────────────────────
t_indices = np.arange(0, n_time, TIME_STRIDE)
t_ms      = psth[t_indices]
n_t       = len(t_indices)

# r2_perunit[layer, time, unit]
perunit_cache = join(CACHE_DIR, 'time_resolved_perunit_JianJian.pkl')

if exists(perunit_cache):
    print(f"\nLoading cached per-unit results from {perunit_cache}")
    with open(perunit_cache, 'rb') as fh:
        res = pickle.load(fh)
    r2_pu      = res['r2_perunit']   # (n_layers, n_t, n_units)
    r2_pu_train = res['r2_perunit_train']
else:
    print(f"\nRunning per-unit regression ({len(LAYERS)} layers × {n_t} time pts × {n_units} units)...")
    r2_pu       = np.zeros((len(LAYERS), n_t, n_units), dtype=np.float32)
    r2_pu_train = np.zeros((len(LAYERS), n_t, n_units), dtype=np.float32)

    for li, layer in enumerate(LAYERS):
        X       = feat_dict[layer]          # (n_images, n_feat)
        X_train = X[idx_train]
        X_test  = X[idx_test]
        print(f"\n  Layer {layer} (dim={X.shape[1]}):")
        for ti, t_idx in enumerate(tqdm(t_indices, desc=f'  {layer}', leave=False)):
            y        = R_fr[:, t_idx, :].T   # (n_images, n_units)
            y_train  = y[idx_train]
            y_test   = y[idx_test]
            clf      = RidgeCV(alphas=ALPHAS, alpha_per_target=True)
            clf.fit(X_train, y_train)
            y_pred_test  = clf.predict(X_test)
            y_pred_train = clf.predict(X_train)
            # per-unit R² — clip negatives at -1 for stability
            r2_pu[li, ti]       = r2_score(y_test,  y_pred_test,  multioutput='raw_values').clip(-1)
            r2_pu_train[li, ti] = r2_score(y_train, y_pred_train, multioutput='raw_values').clip(-1)

    with open(perunit_cache, 'wb') as fh:
        pickle.dump({'r2_perunit': r2_pu, 'r2_perunit_train': r2_pu_train,
                     't_ms': t_ms, 'layers': LAYERS, 'n_units': n_units,
                     'psth': psth, 'spikepos': d['spikepos']}, fh)
    print(f"\n  Saved to {perunit_cache}")

print(f"r2_perunit shape: {r2_pu.shape}  (layers × time × units)")

# ─── Derived quantities ────────────────────────────────────────────────────────
# Mean R² across units (same as before, sanity check)
r2_mean = r2_pu.mean(axis=2)   # (n_layers, n_t)

# Best layer per unit per time: argmax over layers
best_layer_t_unit = r2_pu.argmax(axis=0)   # (n_t, n_units)

# Peak R² per unit (max over layers and time)
peak_r2_per_unit  = r2_pu.max(axis=(0, 1)) # (n_units,)

# Preferred layer per unit (at peak time, using best overall layer)
# Method: which layer gives max R² averaged over evoked window (100–250ms)
evk_mask = (t_ms >= 100) & (t_ms <= 250)
r2_evk   = r2_pu[:, evk_mask, :].mean(axis=1)  # (n_layers, n_units)
pref_layer_unit = r2_evk.argmax(axis=0)          # (n_units,)

# Latency of peak R² per unit (using best layer per unit)
# For each unit: R²(t) = max over layers at each t
r2_max_layer_t = r2_pu.max(axis=0)  # (n_t, n_units) — best layer at each t
peak_time_unit  = t_ms[r2_max_layer_t.argmax(axis=0)]  # (n_units,)

print(f"\nSummary:")
print(f"  peak_r2_per_unit: mean={peak_r2_per_unit.mean():.4f}, "
      f"median={np.median(peak_r2_per_unit):.4f}, "
      f"top10%={np.percentile(peak_r2_per_unit,90):.4f}")
print(f"  preferred layer counts: {dict(zip(LAYER_LABELS, np.bincount(pref_layer_unit, minlength=len(LAYERS))))}")
print(f"  peak latency: mean={peak_time_unit.mean():.1f}ms, median={np.median(peak_time_unit):.1f}ms")

# ─── Fig 1: Violin plots of per-unit R² distribution over time ────────────────
print("\nGenerating figures...")

# Select a subset of time points for violin plots
t_select_ms  = [-40, 0, 50, 80, 100, 125, 150, 175, 200, 250, 300, 350]
t_select_idx = [np.argmin(np.abs(t_ms - t)) for t in t_select_ms]

# Use best-layer R² per unit at each time
r2_bestlayer_t = r2_pu.max(axis=0)   # (n_t, n_units)

fig, axes = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle('Per-Unit R² Distribution Over Time — JianJian 240629', fontsize=13, fontweight='bold')

ax = axes[0]
data_violin = [r2_bestlayer_t[ti] for ti in t_select_idx]
labels_v    = [f'{t_ms[ti]:.0f}ms' for ti in t_select_idx]
vp = ax.violinplot(data_violin, positions=range(len(t_select_idx)), showmedians=True,
                   showextrema=False)
for body in vp['bodies']:
    body.set_alpha(0.6)
    body.set_facecolor('#377eb8')
vp['cmedians'].set_color('red')
ax.axvline(labels_v.index('0ms') if '0ms' in labels_v else 1, color='gray', lw=1, ls='--', alpha=0.5)
ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)
ax.set_xticks(range(len(t_select_idx)))
ax.set_xticklabels(labels_v, rotation=30)
ax.set_ylabel('Per-Unit R² (best layer)')
ax.set_title('Distribution of per-unit R² (best layer at each time point)')

# Overlay mean curve
ax2 = ax.twinx()
mean_sel = [r2_bestlayer_t[ti].mean() for ti in t_select_idx]
ax2.plot(range(len(t_select_idx)), mean_sel, 'r-o', lw=2, ms=5, label='mean')
ax2.set_ylabel('Mean R²', color='red')
ax2.tick_params(axis='y', colors='red')

# Bottom: fraction of units with R² > threshold
ax = axes[1]
thresholds = [0.0, 0.05, 0.1, 0.2]
colors_thr = ['#a8d8ea', '#4a90d9', '#1a5276', '#0a2342']
for thr, col in zip(thresholds, colors_thr):
    frac = (r2_bestlayer_t > thr).mean(axis=1)  # over units
    ax.plot(t_ms, frac, color=col, lw=2, label=f'R² > {thr}')
ax.axvline(0, color='k', lw=1, ls='--', alpha=0.5)
ax.axvspan(100, 250, alpha=0.1, color='orange')
ax.set_xlabel('Time from stimulus onset (ms)')
ax.set_ylabel('Fraction of units')
ax.set_title('Fraction of well-predicted units over time (by R² threshold)')
ax.legend(fontsize=9)
ax.set_xlim(t_ms[0], t_ms[-1])

plt.tight_layout()
fig1_path = join(OUT_DIR, 'fig_perunit_1_distribution.png')
plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved {fig1_path}")

# ─── Fig 2: Preferred layer distribution & layer curves per unit subgroup ─────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Per-Unit Preferred Layer Analysis', fontsize=13, fontweight='bold')

# Panel A: histogram of preferred layer
ax = axes[0]
counts = np.bincount(pref_layer_unit, minlength=len(LAYERS))
bars   = ax.bar(LAYER_LABELS, counts, color=LAYER_COLORS, edgecolor='k', linewidth=0.5)
ax.set_title('Preferred Layer Distribution\n(evoked window 100–250ms)')
ax.set_ylabel('# Units')
ax.tick_params(axis='x', rotation=20)
for bar, c in zip(bars, counts):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, str(c),
            ha='center', fontsize=9, fontweight='bold')

# Panel B: mean R²(t) curves split by preferred layer
ax = axes[1]
for li, (label, color) in enumerate(zip(LAYER_LABELS, LAYER_COLORS)):
    unit_mask = pref_layer_unit == li
    if unit_mask.sum() < 3:
        continue
    # mean of best-layer R² for units preferring this layer
    r2_group = r2_bestlayer_t[:, unit_mask].mean(axis=1)  # (n_t,)
    ax.plot(t_ms, r2_group, color=color, lw=2,
            label=f'{label} (n={unit_mask.sum()})')
ax.axvline(0, color='k', lw=1, ls='--', alpha=0.5)
ax.axvspan(100, 250, alpha=0.1, color='orange')
ax.set_xlabel('Time (ms)'); ax.set_ylabel('Mean R² (best layer)')
ax.set_title('R²(t) by preferred-layer group')
ax.legend(fontsize=8); ax.set_xlim(t_ms[0], t_ms[-1])

# Panel C: peak latency distribution
ax = axes[2]
ax.hist(peak_time_unit[(peak_r2_per_unit > 0.02)], bins=30,
        color='steelblue', edgecolor='k', linewidth=0.5)
ax.axvline(0, color='k', lw=1, ls='--')
ax.set_xlabel('Peak R² latency (ms)')
ax.set_ylabel('# Units')
ax.set_title('Distribution of peak R² latency\n(units with R² > 0.02)')

plt.tight_layout()
fig2_path = join(OUT_DIR, 'fig_perunit_2_preferred_layer.png')
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved {fig2_path}")

# ─── Fig 3: Per-unit R² heatmap sorted by peak latency ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Per-Unit R² Dynamics (units sorted by peak latency)', fontsize=13, fontweight='bold')

# Sort units by peak latency, only show units with peak R² > 0.02
good_units = np.where(peak_r2_per_unit > 0.02)[0]
sort_order = good_units[np.argsort(peak_time_unit[good_units])]
r2_sorted  = r2_bestlayer_t[:, sort_order].T  # (good_units, n_t)
pref_sorted = pref_layer_unit[sort_order]

ax = axes[0]
im = ax.imshow(r2_sorted, aspect='auto', cmap='hot',
               extent=[t_ms[0], t_ms[-1], len(sort_order), 0],
               vmin=0, vmax=np.percentile(r2_sorted, 98))
ax.axvline(0, color='cyan', lw=1, ls='--')
ax.set_xlabel('Time (ms)'); ax.set_ylabel(f'Unit (sorted by peak latency, n={len(sort_order)})')
ax.set_title('Per-unit R²(t) heatmap\n(best layer, sorted by peak latency)')
plt.colorbar(im, ax=ax, label='R² (best layer)', shrink=0.8)

# Right panel: color bar showing preferred layer
ax = axes[1]
pref_colors = np.array([LAYER_COLORS[li] for li in pref_sorted])
for i, (unit_idx, pref) in enumerate(zip(sort_order, pref_sorted)):
    ax.barh(i, peak_r2_per_unit[unit_idx], color=LAYER_COLORS[pref], height=1, linewidth=0)
ax.set_xlabel('Peak R²')
ax.set_ylabel(f'Unit (same order)')
ax.set_title('Peak R² per unit\n(color = preferred layer)')
# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=LAYER_COLORS[li], label=LAYER_LABELS[li]) for li in range(len(LAYERS))]
ax.legend(handles=legend_elements, fontsize=8, loc='lower right')
ax.set_ylim(0, len(sort_order))

plt.tight_layout()
fig3_path = join(OUT_DIR, 'fig_perunit_3_heatmap.png')
plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved {fig3_path}")

# ─── Fig 4: Per-layer R² curves with unit-level uncertainty ──────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Per-Layer R²(t) with Unit Distribution — JianJian', fontsize=13, fontweight='bold')

for li, (layer, label, color, ax) in enumerate(zip(LAYERS, LAYER_LABELS, LAYER_COLORS, axes.flat)):
    r2_layer_t = r2_pu[li]  # (n_t, n_units)
    mean  = r2_layer_t.mean(axis=1)
    med   = np.median(r2_layer_t, axis=1)
    p25   = np.percentile(r2_layer_t, 25, axis=1)
    p75   = np.percentile(r2_layer_t, 75, axis=1)
    p90   = np.percentile(r2_layer_t, 90, axis=1)

    ax.fill_between(t_ms, p25, p75, alpha=0.25, color=color, label='IQR')
    ax.fill_between(t_ms, med, p90, alpha=0.12, color=color, label='50–90th pct')
    ax.plot(t_ms, mean, color=color, lw=2.5, label='mean')
    ax.plot(t_ms, med,  color=color, lw=1.5, ls='--', alpha=0.7, label='median')
    ax.axvline(0, color='k', lw=1, ls='--', alpha=0.5)
    ax.axhline(0, color='k', lw=0.5, ls=':', alpha=0.5)
    ax.axvspan(100, 250, alpha=0.08, color='orange')
    ax.set_title(f'{label}  (peak mean R²={mean.max():.3f})')
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('R²')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlim(t_ms[0], t_ms[-1])

plt.tight_layout()
fig4_path = join(OUT_DIR, 'fig_perunit_4_per_layer_bands.png')
plt.savefig(fig4_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved {fig4_path}")

print(f"\nAll figures saved to {OUT_DIR}")
print("Done!")
