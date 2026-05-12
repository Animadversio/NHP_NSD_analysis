"""
Single-neuron & population weighted layer depth over time.
Handles degenerate cases: mask where sum(R²+) < min_signal.
"""
import sys, os
sys.path.insert(0, '/n/home12/binxuwang/Github/NHP_NSD_analysis')
import numpy as np
import pickle as pkl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter1d

CACHE  = '/n/home12/binxuwang/Github/NHP_NSD_analysis/notebooks/cache/time_resolved_perunit_JianJian.pkl'
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

with open(CACHE, 'rb') as f:
    res = pkl.load(f)

R2      = res['r2_perunit'].astype(np.float64)   # (6, 90, 456)  layers×time×units
t_ms    = res['t_ms']                             # (90,)
layers  = res['layers']                           # 6 layer names
n_layers, n_time, n_units = R2.shape
depth_idx = np.arange(n_layers, dtype=float)      # 0..5

LAYER_LABELS = ['relu', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
LAYER_COLORS = ['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f','#b07aa1']

# ── Compute per-unit weighted depth curve ────────────────────────────────────
# R²⁺ = max(R², 0)  — clip negatives to 0 before weighting
# depth_com[t, u] = Σ_l (l × R²⁺[l,t,u]) / Σ_l R²⁺[l,t,u]
# Set to NaN when Σ_l R²⁺ < min_signal (degenerate)
MIN_SIGNAL = 0.02   # minimum total positive R² to trust depth estimate

R2pos = np.clip(R2, 0, None)                       # (6, 90, 456)
total_r2 = R2pos.sum(axis=0)                       # (90, 456)
weighted  = (depth_idx[:, None, None] * R2pos).sum(axis=0)  # (90, 456)

depth_com = np.where(total_r2 >= MIN_SIGNAL,
                     weighted / (total_r2 + 1e-12),
                     np.nan)                       # (90, 456)

# Peak R² per unit (for sorting / filtering)
peak_r2   = R2.max(axis=(0, 1))                   # (456,)
valid_frac = np.isfinite(depth_com).mean(axis=0)  # fraction of time points that are valid

print(f"R2 range: {R2.min():.3f} to {R2.max():.3f}")
print(f"MIN_SIGNAL={MIN_SIGNAL}: fraction of (t,u) cells that are valid: "
      f"{np.isfinite(depth_com).mean():.2%}")
print(f"Units with >50% valid time points: {(valid_frac > 0.5).sum()}")
print(f"Units with >20% valid time points: {(valid_frac > 0.2).sum()}")

# ── Population weighted depth: two versions ──────────────────────────────────
# V1: nanmean of per-unit depth curves (equal unit weight when valid)
pop_depth_mean = np.nanmean(depth_com, axis=1)      # (90,)
pop_depth_sem  = np.nanstd(depth_com, axis=1) / np.sqrt(np.isfinite(depth_com).sum(axis=1).clip(1))

# V2: signal-weighted mean (high-R² units contribute more)
# weight each unit by its total R²+ at each time point
weighted_sum   = np.nansum(total_r2 * np.where(np.isfinite(depth_com), depth_com, 0), axis=1)
weight_total   = np.nansum(total_r2 * np.isfinite(depth_com), axis=1)
pop_depth_w    = np.where(weight_total > 0, weighted_sum / weight_total, np.nan)

smooth = lambda x: gaussian_filter1d(np.nan_to_num(x, nan=np.nanmean(x)), sigma=2)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Population depth + example single neurons
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Weighted Layer Depth Curves — JianJian (456 units)", fontsize=13, fontweight='bold')

# Panel 1: Population mean (both versions)
ax = axes[0, 0]
ax.plot(t_ms, smooth(pop_depth_mean), color='k', lw=2.5, label='unweighted mean')
ax.fill_between(t_ms,
                smooth(pop_depth_mean - pop_depth_sem),
                smooth(pop_depth_mean + pop_depth_sem),
                alpha=0.25, color='k')
ax.plot(t_ms, smooth(pop_depth_w), color='steelblue', lw=2, ls='--', label='R²-weighted mean')
ax.axvline(0, color='gray', ls='--', lw=0.8)
ax.axvspan(80, 150,  alpha=0.1, color='steelblue')
ax.axvspan(180, 350, alpha=0.1, color='tomato')
ax.set_yticks(range(n_layers)); ax.set_yticklabels(LAYER_LABELS, fontsize=8)
ax.set_xlabel('Time (ms)'); ax.set_title('Population weighted depth')
ax.legend(fontsize=8); ax.set_xlim(t_ms[0], t_ms[-1])
n_valid = np.isfinite(depth_com).sum(axis=1)
ax2 = ax.twinx()
ax2.fill_between(t_ms, n_valid, alpha=0.15, color='green', label='# valid units')
ax2.set_ylabel('# valid units', color='green', fontsize=7)
ax2.tick_params(axis='y', colors='green', labelsize=7)

# Panel 2: Best 25 units — individual depth curves (colored by peak R²)
ax = axes[0, 1]
top_units = np.argsort(peak_r2)[::-1][:25]
cmap = cm.get_cmap('plasma')
for rank, u in enumerate(top_units):
    dc = depth_com[:, u]
    valid = np.isfinite(dc)
    color = cmap(rank / 25)
    ax.plot(t_ms[valid], dc[valid], color=color, lw=1.2, alpha=0.75)
ax.axvline(0, color='gray', ls='--', lw=0.8)
ax.set_yticks(range(n_layers)); ax.set_yticklabels(LAYER_LABELS, fontsize=8)
ax.set_xlabel('Time (ms)'); ax.set_title('Top 25 units (colored by rank)')
ax.set_xlim(t_ms[0], t_ms[-1])
sm = cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(0, 25))
plt.colorbar(sm, ax=ax, label='R² rank')

# Panel 3: Example "late-deepening" units — shift from layer3 to layer4/avgpool
# Find units where late mean depth > early mean depth by ≥1
early_mask = (t_ms >= 80) & (t_ms <= 150)
late_mask  = (t_ms >= 180) & (t_ms <= 350)
early_depth_mean = np.nanmean(depth_com[early_mask, :], axis=0)
late_depth_mean  = np.nanmean(depth_com[late_mask,  :], axis=0)
shift = late_depth_mean - early_depth_mean

# Find top "deeper-late" units with high R²
cand_mask = (peak_r2 >= 0.10) & np.isfinite(shift) & (shift > 0.3)
cand_units = np.where(cand_mask)[0]
cand_units = cand_units[np.argsort(shift[cand_units])[::-1]][:8]

ax = axes[0, 2]
cmap2 = cm.get_cmap('tab10')
for i, u in enumerate(cand_units):
    dc = depth_com[:, u]
    valid = np.isfinite(dc)
    ax.plot(t_ms[valid], gaussian_filter1d(dc[valid], sigma=2),
            color=cmap2(i), lw=2, alpha=0.85,
            label=f'u{u} R²={peak_r2[u]:.2f} Δ={shift[u]:+.1f}')
ax.axvline(0, color='gray', ls='--', lw=0.8)
ax.axvspan(80, 150,  alpha=0.1, color='steelblue')
ax.axvspan(180, 350, alpha=0.1, color='tomato')
ax.set_yticks(range(n_layers)); ax.set_yticklabels(LAYER_LABELS, fontsize=8)
ax.set_xlabel('Time (ms)'); ax.set_title(f'"Deeper-late" example units (n={len(cand_units)})')
ax.legend(fontsize=6); ax.set_xlim(t_ms[0], t_ms[-1])

# Panel 4: Heatmap of depth_com across all responsive units × time
ax = axes[1, 0]
resp_units = np.where(peak_r2 >= 0.10)[0]
sort_by = np.nanargmax(depth_com[:, resp_units], axis=0)  # time of max depth
order = resp_units[np.argsort(sort_by)]
hmap = depth_com[:, order].T  # (n_resp_units, n_time)
im = ax.imshow(hmap, aspect='auto', origin='lower', cmap='RdBu_r',
               extent=[t_ms[0], t_ms[-1], 0, len(order)],
               vmin=0, vmax=n_layers-1)
ax.axvline(0, color='white', lw=0.8, ls='--')
plt.colorbar(im, ax=ax, ticks=range(n_layers),
             label='Weighted depth').set_ticklabels(LAYER_LABELS)
ax.set_xlabel('Time (ms)'); ax.set_ylabel('Unit (sorted by peak depth time)')
ax.set_title(f'Depth heatmap (R²≥0.10, n={len(order)})')

# Panel 5: Distribution of early vs late depth (violin)
ax = axes[1, 1]
early_d = early_depth_mean[peak_r2 >= 0.10]
late_d  = late_depth_mean[peak_r2 >= 0.10]
early_d = early_d[np.isfinite(early_d)]
late_d  = late_d[np.isfinite(late_d)]
vp = ax.violinplot([early_d, late_d], positions=[0, 1], showmedians=True, showextrema=True)
for pc, col in zip(vp['bodies'], ['steelblue', 'tomato']):
    pc.set_facecolor(col); pc.set_alpha(0.7)
ax.set_xticks([0, 1]); ax.set_xticklabels(['Early\n80-150ms', 'Late\n180-350ms'])
ax.set_yticks(range(n_layers)); ax.set_yticklabels(LAYER_LABELS, fontsize=8)
ax.set_ylabel('Weighted preferred depth')
ax.set_title(f'Depth distribution: early vs late (n={len(early_d)})')
from scipy.stats import wilcoxon as wlcx
stat, pval = wlcx(early_d[:len(late_d)], late_d[:len(early_d)])
ax.text(0.5, 0.05, f'Wilcoxon p={pval:.4f}', transform=ax.transAxes,
        ha='center', fontsize=9, color='darkred' if pval < 0.05 else 'gray')

# Panel 6: Scatter of early vs late depth (per unit, colored by peak R²)
ax = axes[1, 2]
resp = peak_r2 >= 0.05
ed = early_depth_mean[resp]; ld = late_depth_mean[resp]; pr = peak_r2[resp]
sc = ax.scatter(ed, ld, c=pr, cmap='viridis', alpha=0.4, s=15,
                vmin=0, vmax=pr.max())
ax.plot([0, n_layers-1], [0, n_layers-1], 'r--', lw=1.5)
plt.colorbar(sc, ax=ax, label='Peak R²')
ax.set_xticks(range(n_layers)); ax.set_xticklabels(LAYER_LABELS, rotation=30, fontsize=7)
ax.set_yticks(range(n_layers)); ax.set_yticklabels(LAYER_LABELS, fontsize=7)
ax.set_xlabel('Early weighted depth'); ax.set_ylabel('Late weighted depth')
ax.set_title('Scatter: early vs late depth per unit')

plt.tight_layout()
outpath = f'{FIGDIR}/fig_weighted_depth_curves.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {outpath}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Focused single-neuron examples (4 panels, nice curves)
# ─────────────────────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 4, figsize=(16, 7))
fig2.suptitle("Single-neuron weighted depth + R² layer profiles", fontsize=12, fontweight='bold')

# Pick 4 representative units: 2 "deeper-late", 2 stable/control
deeper_late = cand_units[:2] if len(cand_units) >= 2 else []
stable_cand = np.where((peak_r2 >= 0.10) & np.isfinite(shift) & (np.abs(shift) < 0.2))[0]
stable_cand = stable_cand[np.argsort(peak_r2[stable_cand])[::-1]][:2]
example_units = list(deeper_late) + list(stable_cand)
titles = ['Deep-late unit 1', 'Deep-late unit 2', 'Stable unit 1', 'Stable unit 2']

for col, (u, title) in enumerate(zip(example_units, titles)):
    # Top row: R² per layer over time
    ax = axes2[0, col]
    for li in range(n_layers):
        ax.plot(t_ms, gaussian_filter1d(R2[li, :, u], sigma=2),
                color=LAYER_COLORS[li], label=LAYER_LABELS[li], lw=1.5)
    ax.axvline(0, color='gray', ls='--', lw=0.8)
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('R²')
    ax.set_title(f'{title}\n(unit {u}, peak R²={peak_r2[u]:.2f})', fontsize=8)
    ax.legend(fontsize=5.5, ncol=2)
    ax.set_xlim(t_ms[0], t_ms[-1])
    
    # Bottom row: weighted depth curve
    ax = axes2[1, col]
    dc = depth_com[:, u]
    valid = np.isfinite(dc)
    ax.plot(t_ms[valid], gaussian_filter1d(dc[valid], sigma=2), 'k-', lw=2.5)
    ax.fill_between(t_ms[valid],
                    gaussian_filter1d(dc[valid], sigma=2) - 0.2,
                    gaussian_filter1d(dc[valid], sigma=2) + 0.2,
                    alpha=0.2, color='k')
    ax.axvline(0, color='gray', ls='--', lw=0.8)
    ax.axvspan(80, 150,  alpha=0.1, color='steelblue')
    ax.axvspan(180, 350, alpha=0.1, color='tomato')
    ax.set_yticks(range(n_layers)); ax.set_yticklabels(LAYER_LABELS, fontsize=7)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Weighted depth')
    ed_u = np.nanmean(dc[early_mask]) if np.isfinite(dc[early_mask]).any() else np.nan
    ld_u = np.nanmean(dc[late_mask])  if np.isfinite(dc[late_mask]).any()  else np.nan
    ax.set_title(f'Δdepth={ld_u-ed_u:+.2f}' if np.isfinite(ld_u-ed_u) else 'Δdepth=NaN', fontsize=8)
    ax.set_xlim(t_ms[0], t_ms[-1])

plt.tight_layout()
outpath2 = f'{FIGDIR}/fig_single_neuron_depth.png'
plt.savefig(outpath2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {outpath2}")

# ── Summary stats ────────────────────────────────────────────────────────────
print(f"\n=== WEIGHTED DEPTH SUMMARY ===")
resp_mask = peak_r2 >= 0.10
ed_all = early_depth_mean[resp_mask & np.isfinite(early_depth_mean)]
ld_all = late_depth_mean[resp_mask & np.isfinite(late_depth_mean)]
print(f"Responsive units (R²≥0.10): {resp_mask.sum()}")
print(f"  Early mean depth: {ed_all.mean():.3f} ± {ed_all.std():.3f}")
print(f"  Late  mean depth: {ld_all.mean():.3f} ± {ld_all.std():.3f}")
print(f"  Δ = {ld_all.mean() - ed_all.mean():+.3f}")
n = min(len(ed_all), len(ld_all))
stat, pval = wlcx(ed_all[:n], ld_all[:n])
print(f"  Wilcoxon p = {pval:.4f}")
print(f"\nDeeper-late units (Δ>0.3, R²≥0.10): {cand_mask.sum()} / {resp_mask.sum()} = {cand_mask.sum()/resp_mask.sum():.1%}")
print("Done!")
