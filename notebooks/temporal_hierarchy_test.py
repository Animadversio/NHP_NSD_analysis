import sys, os
sys.path.insert(0, '/n/home12/binxuwang/Github/NHP_NSD_analysis')
import numpy as np
import pickle as pkl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, pearsonr
from scipy.ndimage import gaussian_filter1d

CACHE = '/n/home12/binxuwang/Github/NHP_NSD_analysis/notebooks/cache/time_resolved_perunit_JianJian.pkl'
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

with open(CACHE, 'rb') as f:
    res = pkl.load(f)

R2 = res['r2_perunit']       # (6, 90, 456)  layers × time × units
t_ms = res['t_ms']            # (90,) ms
layers = res['layers']        # ['relu','layer1','layer2','layer3','layer4','avgpool']
n_layers, n_time, n_units = R2.shape
depth_idx = np.arange(n_layers)  # 0..5

print(f"R2 shape: {R2.shape}")
print(f"Time: {t_ms[0]:.0f} to {t_ms[-1]:.0f} ms, {n_time} points")
print(f"Layers: {layers}")

# Clip negative R² to 0 for weighting
R2_pos = np.clip(R2, 0, None)

# ── 1. Population-level weighted layer depth over time ──────────────────────
# depth_com[t] = weighted center of mass across layers, averaged over units
depth_com_per_unit = np.zeros((n_time, n_units))
for t in range(n_time):
    total = R2_pos[:, t, :].sum(axis=0) + 1e-10   # (n_units,)
    depth_com_per_unit[t] = (depth_idx[:, None] * R2_pos[:, t, :]).sum(axis=0) / total

mean_depth = depth_com_per_unit.mean(axis=1)
sem_depth  = depth_com_per_unit.std(axis=1) / np.sqrt(n_units)

# ── 2. Per-unit early vs late preferred layer ────────────────────────────────
early_mask = (t_ms >= 80) & (t_ms <= 150)
late_mask  = (t_ms >= 180) & (t_ms <= 350)

# Best layer = argmax R² within window (use mean R² over window bins)
R2_early = R2[:, early_mask, :].mean(axis=1)   # (6, 456)
R2_late  = R2[:, late_mask,  :].mean(axis=1)   # (6, 456)

pref_layer_early = R2_early.argmax(axis=0)   # (456,)
pref_layer_late  = R2_late.argmax(axis=0)    # (456,)

# Peak R² per unit (to filter responsive units)
peak_r2_unit = R2.max(axis=(0,1))             # (456,)

# Use multiple thresholds
thresholds = [0.0, 0.02, 0.05, 0.10, 0.15]
print("\nFraction of units with late_depth > early_depth (all layers as predicted):")
for thresh in thresholds:
    mask = peak_r2_unit >= thresh
    n = mask.sum()
    if n == 0:
        continue
    diff = pref_layer_late[mask] - pref_layer_early[mask]
    frac_deeper = (diff > 0).mean()
    frac_same   = (diff == 0).mean()
    frac_shallower = (diff < 0).mean()
    # Wilcoxon test on the depth difference
    if (diff != 0).sum() >= 10:
        stat, pval = wilcoxon(pref_layer_early[mask], pref_layer_late[mask], alternative='less')
    else:
        pval = np.nan
    mean_diff = diff.mean()
    print(f"  thresh>={thresh:.2f}: n={n:3d}  deeper={frac_deeper:.2f}  same={frac_same:.2f}  "
          f"shallower={frac_shallower:.2f}  mean_Δdepth={mean_diff:+.2f}  p={pval:.4f}")

# ── 3. Per-layer R² time course ──────────────────────────────────────────────
# Mean R² over units for each layer × time
mean_r2_layer_time = R2.mean(axis=2)    # (6, 90)  
# Smooth for display
smooth = lambda x: gaussian_filter1d(x, sigma=2)

# ── PLOTTING ─────────────────────────────────────────────────────────────────
LAYER_COLORS = ['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f','#b07aa1']
LAYER_LABELS = ['stem/relu', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Temporal Hierarchy Test — JianJian (456 units)", fontsize=13, fontweight='bold')

# ── Panel 1: R² per layer over time (mean across units) ─────────────────────
ax = axes[0, 0]
for li in range(n_layers):
    ax.plot(t_ms, smooth(mean_r2_layer_time[li]), color=LAYER_COLORS[li], 
            label=LAYER_LABELS[li], lw=2)
ax.axvline(0, color='k', ls='--', lw=0.8, alpha=0.5, label='stim onset')
ax.axvspan(80, 150,  alpha=0.12, color='steelblue', label='early win')
ax.axvspan(180, 350, alpha=0.12, color='tomato',    label='late win')
ax.set_xlabel('Time (ms)'); ax.set_ylabel('Mean R²')
ax.set_title('Mean R² per layer over time')
ax.legend(fontsize=7, ncol=2); ax.set_xlim(t_ms[0], t_ms[-1])

# ── Panel 2: Weighted layer depth (population CoM) ───────────────────────────
ax = axes[0, 1]
ax.plot(t_ms, smooth(mean_depth), color='k', lw=2)
ax.fill_between(t_ms, smooth(mean_depth - sem_depth), smooth(mean_depth + sem_depth),
                alpha=0.2, color='k')
ax.axvline(0, color='k', ls='--', lw=0.8, alpha=0.5)
ax.axvspan(80, 150,  alpha=0.12, color='steelblue')
ax.axvspan(180, 350, alpha=0.12, color='tomato')
ax.set_yticks(range(n_layers)); ax.set_yticklabels(LAYER_LABELS, fontsize=8)
ax.set_xlabel('Time (ms)'); ax.set_title('Weighted layer depth (pop. CoM)')
ax.set_xlim(t_ms[0], t_ms[-1])

# ── Panel 3: Scatter — early vs late preferred layer (responsive units) ──────
ax = axes[0, 2]
thresh = 0.05
mask = peak_r2_unit >= thresh
jitter = lambda x: x + np.random.randn(len(x)) * 0.15
sc = ax.scatter(jitter(pref_layer_early[mask]), jitter(pref_layer_late[mask]),
                alpha=0.3, s=15, c=peak_r2_unit[mask], cmap='viridis')
ax.plot([-0.5, n_layers-0.5], [-0.5, n_layers-0.5], 'r--', lw=1.5, label='no change')
plt.colorbar(sc, ax=ax, label='Peak R²')
ax.set_xticks(range(n_layers)); ax.set_xticklabels(LAYER_LABELS, rotation=35, fontsize=7)
ax.set_yticks(range(n_layers)); ax.set_yticklabels(LAYER_LABELS, fontsize=7)
ax.set_xlabel('Early preferred layer (80-150ms)')
ax.set_ylabel('Late preferred layer (180-350ms)')
ax.set_title(f'Per-unit layer shift (R²>{thresh}, n={mask.sum()})')

# ── Panel 4: Histogram of (late_depth - early_depth) ─────────────────────────
ax = axes[1, 0]
diff = pref_layer_late[mask] - pref_layer_early[mask]
bins = np.arange(-n_layers+0.5, n_layers+0.5, 1)
ax.hist(diff, bins=bins, color='steelblue', edgecolor='white', alpha=0.85)
ax.axvline(0, color='r', ls='--', lw=1.5, label='no shift')
ax.axvline(diff.mean(), color='k', ls='-', lw=1.5, label=f'mean={diff.mean():+.2f}')
frac_pos = (diff > 0).mean()
ax.set_xlabel('Late depth − Early depth'); ax.set_ylabel('# units')
ax.set_title(f'Layer depth shift (deeper-late={frac_pos:.0%})')
ax.legend(fontsize=9)

# ── Panel 5: Fraction of "deeper-late" units vs R² threshold ─────────────────
ax = axes[1, 1]
thresh_range = np.linspace(0, 0.3, 40)
frac_deeper_all, n_units_thresh = [], []
for th in thresh_range:
    m = peak_r2_unit >= th
    n_units_thresh.append(m.sum())
    if m.sum() >= 5:
        d = pref_layer_late[m] - pref_layer_early[m]
        frac_deeper_all.append((d > 0).mean())
    else:
        frac_deeper_all.append(np.nan)
ax2 = ax.twinx()
ax.plot(thresh_range, frac_deeper_all, color='steelblue', lw=2, label='frac deeper-late')
ax.axhline(0.33, color='gray', ls='--', lw=1, label='chance')
ax2.plot(thresh_range, n_units_thresh, color='tomato', lw=1.5, ls='--', alpha=0.7)
ax.set_xlabel('R² threshold'); ax.set_ylabel('Fraction with late_depth > early_depth', color='steelblue')
ax2.set_ylabel('# units above threshold', color='tomato')
ax.set_title('Fraction "deeper-late" vs threshold')
ax.set_ylim(0, 1); ax.legend(fontsize=8)

# ── Panel 6: Layer R² ratio (late/early) per layer ───────────────────────────
ax = axes[1, 2]
# Mean R² in early vs late window per layer (across responsive units)
mask_all = peak_r2_unit >= 0.05
r2_e = R2[:, early_mask, :][:, :, mask_all].mean(axis=(1, 2))  # (6,)
r2_l = R2[:, late_mask,  :][:, :, mask_all].mean(axis=(1, 2))  # (6,)
x = np.arange(n_layers)
w = 0.35
ax.bar(x - w/2, r2_e, w, label='Early (80-150ms)', color='steelblue', alpha=0.85)
ax.bar(x + w/2, r2_l, w, label='Late (180-350ms)',  color='tomato',    alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(LAYER_LABELS, rotation=30, fontsize=9)
ax.set_ylabel('Mean R²'); ax.set_title('R² by layer: early vs late window')
ax.legend(fontsize=9)
# Add ratio labels
for i in range(n_layers):
    ratio = r2_l[i] / (r2_e[i] + 1e-8)
    ax.text(i, max(r2_e[i], r2_l[i]) + 0.001, f'{ratio:.2f}x', ha='center', fontsize=7)

plt.tight_layout()
outpath = f'{FIGDIR}/fig_temporal_hierarchy.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {outpath}")

# ── Print clean summary ──────────────────────────────────────────────────────
print("\n=== TEMPORAL HIERARCHY SUMMARY ===")
print(f"Session: JianJian, {n_units} units, {n_layers} layers, {n_time} time points")
print(f"Time range: {t_ms[0]:.0f} to {t_ms[-1]:.0f} ms")
print(f"\nPopulation weighted depth:")
early_com = mean_depth[early_mask].mean()
late_com  = mean_depth[late_mask].mean()
print(f"  Early window (80-150ms) mean depth: {early_com:.3f} ({layers[int(round(early_com))]})")
print(f"  Late window (180-350ms) mean depth:  {late_com:.3f} ({layers[int(round(late_com))]})")
print(f"  Shift: {late_com - early_com:+.3f} layer units")

print(f"\nPer-unit analysis (peak R² >= 0.05, n={mask.sum()}):")
diff_05 = pref_layer_late[mask] - pref_layer_early[mask]
print(f"  Deeper-late: {(diff_05>0).mean():.1%}")
print(f"  Same:        {(diff_05==0).mean():.1%}")
print(f"  Shallower-late: {(diff_05<0).mean():.1%}")
print(f"  Mean depth shift: {diff_05.mean():+.3f}")
if (diff_05 != 0).sum() >= 10:
    stat, pval = wilcoxon(pref_layer_early[mask], pref_layer_late[mask], alternative='less')
    print(f"  Wilcoxon (early < late): p={pval:.4f}")

print(f"\nLayer R² ratio (late/early) for responsive units:")
for i in range(n_layers):
    ratio = r2_l[i] / (r2_e[i] + 1e-8)
    print(f"  {layers[i]:10s}: early={r2_e[i]:.4f}  late={r2_l[i]:.4f}  ratio={ratio:.2f}x")

print("\nDone!")
