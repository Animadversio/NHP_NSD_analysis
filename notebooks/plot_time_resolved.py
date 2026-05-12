"""Generate final figures from cached time-resolved regression results."""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CACHE_DIR = '/n/home12/binxuwang/Github/NHP_NSD_analysis/notebooks/cache'
OUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')

with open(f'{CACHE_DIR}/time_resolved_regression_results.pkl', 'rb') as f:
    res = pickle.load(f)

t_ms         = res['t_ms']
r2_test      = res['r2_test']   # (6, 90)
LAYER_LABELS = res['layer_labels']
LAYER_COLORS = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3', '#a65628']

optimal_layer_idx = np.argmax(r2_test, axis=0)
best_r2 = r2_test[optimal_layer_idx, np.arange(len(t_ms))]

def get_runs(mask, times):
    runs = []
    in_run = False
    for i, m in enumerate(mask):
        if m and not in_run:
            start = times[i] - 2.5; in_run = True
        elif not m and in_run:
            runs.append((start, times[i-1]+2.5)); in_run = False
    if in_run: runs.append((start, times[-1]+2.5))
    return runs

# ─── Fig A: R² curves ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Time-Resolved ResNet50 Layer Regression — JianJian 240629\n'
             '(RidgeCV, sp_avg features, 80/20 split)', fontsize=12, fontweight='bold')

ax = axes[0]
for li, (label, color) in enumerate(zip(LAYER_LABELS, LAYER_COLORS)):
    ax.plot(t_ms, r2_test[li], color=color, lw=2.5, label=label)
ax.axvline(0, color='k', lw=1, ls='--', alpha=0.5, label='onset')
ax.axvspan(100, 250, alpha=0.1, color='orange', label='evoked window')
ax.set_xlabel('Time from stimulus onset (ms)', fontsize=11)
ax.set_ylabel('Test R²', fontsize=11)
ax.set_title('Prediction R² per layer over time')
ax.legend(fontsize=8); ax.set_xlim(t_ms[0], t_ms[-1]); ax.set_ylim(bottom=0)

ax = axes[1]
# Best layer R² envelope
ax.fill_between(t_ms, 0, best_r2, alpha=0.15, color='gray')
ax.plot(t_ms, best_r2, 'k-', lw=2, label='best-layer R²')
# Color background by dominant layer
for li, color in enumerate(LAYER_COLORS):
    for start, end in get_runs(optimal_layer_idx == li, t_ms):
        ax.axvspan(start, end, alpha=0.25, color=color, lw=0)
ax.axvline(0, color='k', lw=1, ls='--', alpha=0.5)
ax.axvspan(100, 250, alpha=0.1, color='orange')
ax.set_xlabel('Time (ms)', fontsize=11); ax.set_ylabel('Test R²', fontsize=11)
ax.set_title('Best-layer R² (background = optimal layer)')
ax.legend(fontsize=8); ax.set_xlim(t_ms[0], t_ms[-1]); ax.set_ylim(bottom=0)
# Add legend patches for layers
from matplotlib.patches import Patch
patches = [Patch(fc=c, alpha=0.5, label=l) for c, l in zip(LAYER_COLORS, LAYER_LABELS)]
ax.legend(handles=[ax.get_legend_handles_labels()[0][0]]+patches, fontsize=7, ncol=2)

plt.tight_layout()
figA = f'{OUT_DIR}/figA_time_resolved_R2_curves.png'
plt.savefig(figA, dpi=150, bbox_inches='tight'); plt.close()
print(f"Saved {figA}")

# ─── Fig B: Heatmap ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 9))
fig.suptitle('Time-Resolved Regression — Layer × Time R² Heatmap', fontsize=13, fontweight='bold')

ax = axes[0]
im = ax.imshow(np.clip(r2_test, 0, None), aspect='auto', cmap='hot',
               extent=[t_ms[0], t_ms[-1], len(LAYER_LABELS)-0.5, -0.5], vmin=0)
ax.set_yticks(range(len(LAYER_LABELS))); ax.set_yticklabels(LAYER_LABELS, fontsize=10)
ax.set_xlabel('Time (ms)', fontsize=11); ax.set_title('Test R² (layer × time)')
ax.axvline(0, color='cyan', lw=1.5, ls='--')
ax.axvline(100, color='white', lw=1, ls=':', alpha=0.7)
ax.axvline(250, color='white', lw=1, ls=':', alpha=0.7)
plt.colorbar(im, ax=ax, label='Test R²')
# White dots at optimal layer
ax.scatter(t_ms, optimal_layer_idx, c='white', s=12, zorder=5, alpha=0.8, label='optimal layer')
ax.legend(fontsize=8)

ax = axes[1]
for li, (label, color) in enumerate(zip(LAYER_LABELS, LAYER_COLORS)):
    ax.plot(t_ms, r2_test[li], color=color, lw=2, label=label)
    ax.fill_between(t_ms, 0, np.clip(r2_test[li],0,None), alpha=0.07, color=color)
for li, color in enumerate(LAYER_COLORS):
    for start, end in get_runs(optimal_layer_idx == li, t_ms):
        ax.axvspan(start, end, alpha=0.12, color=color, lw=0)
ax.axvline(0, color='k', lw=1, ls='--', alpha=0.5)
ax.axvspan(100, 250, alpha=0.08, color='orange')
ax.set_xlabel('Time (ms)', fontsize=11); ax.set_ylabel('Test R²', fontsize=11)
ax.set_title('R² curves (background shading = dominant layer)')
ax.legend(fontsize=8, ncol=3); ax.set_xlim(t_ms[0], t_ms[-1]); ax.set_ylim(bottom=0)

plt.tight_layout()
figB = f'{OUT_DIR}/figB_time_resolved_heatmap.png'
plt.savefig(figB, dpi=150, bbox_inches='tight'); plt.close()
print(f"Saved {figB}")

# ─── Fig C: Summary bars ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Layer Regression Summary — JianJian 240629', fontsize=13, fontweight='bold')

peak_r2 = r2_test.max(axis=1)
ax = axes[0]
bars = ax.bar(LAYER_LABELS, peak_r2, color=LAYER_COLORS)
ax.set_title('Peak Test R² per Layer'); ax.set_ylabel('Peak R²')
ax.tick_params(axis='x', rotation=20)
for bar, v in zip(bars, peak_r2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001, f'{v:.3f}', ha='center', fontsize=8)

peak_latency = t_ms[np.argmax(r2_test, axis=1)]
ax = axes[1]
ax.bar(LAYER_LABELS, peak_latency, color=LAYER_COLORS)
ax.set_title('Latency of Peak R² per Layer'); ax.set_ylabel('Time (ms)')
ax.tick_params(axis='x', rotation=20); ax.axhline(0, color='k', lw=1, ls='--', alpha=0.5)

layer_dom = np.array([(optimal_layer_idx == li).mean() for li in range(len(LAYER_LABELS))])
ax = axes[2]
ax.bar(LAYER_LABELS, layer_dom, color=LAYER_COLORS)
ax.set_title('Fraction of Time as Optimal Layer'); ax.set_ylabel('Fraction')
ax.tick_params(axis='x', rotation=20)

plt.tight_layout()
figC = f'{OUT_DIR}/figC_layer_summary.png'
plt.savefig(figC, dpi=150, bbox_inches='tight'); plt.close()
print(f"Saved {figC}")

# ─── Fig D: Layer preference timeline ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))
fig.suptitle('Layer Preference Over Time — Which ResNet50 Layer Best Explains Neural Activity',
             fontsize=12, fontweight='bold')

for li, (label, color) in enumerate(zip(LAYER_LABELS, LAYER_COLORS)):
    mask = optimal_layer_idx == li
    if mask.any():
        ax.fill_between(t_ms, li-0.45, li+0.45, where=mask, color=color, alpha=0.75, step='mid')

ax.plot(t_ms, optimal_layer_idx, 'k-', lw=2, alpha=0.7)
ax.axvline(0, color='k', lw=2, ls='--', label='stimulus onset')
ax.axvspan(100, 250, alpha=0.1, color='orange', label='evoked window (100-250ms)')
ax.set_yticks(range(len(LAYER_LABELS))); ax.set_yticklabels(LAYER_LABELS, fontsize=11)
ax.set_xlabel('Time from stimulus onset (ms)', fontsize=12)
ax.set_ylabel('Optimal ResNet50 Layer', fontsize=12)
ax.set_xlim(t_ms[0], t_ms[-1]); ax.set_ylim(-0.6, len(LAYER_LABELS)-0.4)
ax.legend(fontsize=9); ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
figD = f'{OUT_DIR}/figD_layer_preference_timeline.png'
plt.savefig(figD, dpi=150, bbox_inches='tight'); plt.close()
print(f"Saved {figD}")

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RESULTS SUMMARY — JianJian 240629, ResNet50 sp_avg features")
print("="*60)
for li, label in enumerate(LAYER_LABELS):
    peak = r2_test[li].max()
    lat  = t_ms[np.argmax(r2_test[li])]
    dom  = (optimal_layer_idx == li).mean()
    evk  = r2_test[li][(t_ms>=100)&(t_ms<=250)].mean()
    print(f"  {label:12s}: peak R²={peak:.4f} @ {lat:4.0f}ms | evoked R²={evk:.4f} | dominant {dom*100:.0f}% of time")

best_li, best_ti = np.unravel_index(r2_test.argmax(), r2_test.shape)
print(f"\n  Global peak: {LAYER_LABELS[best_li]} @ {t_ms[best_ti]:.0f}ms, R²={r2_test.max():.4f}")
print("\nAll figures saved.")
