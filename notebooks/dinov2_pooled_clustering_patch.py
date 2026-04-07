"""
Pool all monkey sessions and cluster neurons by DINOv2 time-depth profile.
Uses CLS token (best performing) weighted depth over 12 blocks.
"""
import sys, os
sys.path.insert(0, '/n/home12/binxuwang/Github/NHP_NSD_analysis')
import numpy as np, pickle as pkl
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

CACHE_DIR = '/n/home12/binxuwang/Github/NHP_NSD_analysis/notebooks/cache'
FIGDIR    = '/n/home12/binxuwang/Github/NHP_NSD_analysis/notebooks/figures'
os.makedirs(FIGDIR, exist_ok=True)

MONKEYS = ['JianJian','FaCai','TuTu','ZhuangZhuang','MaoDan']
N_BLOCKS = 12
MIN_SIG  = 0.02
depth_idx = np.arange(N_BLOCKS, dtype=float)
MONKEY_COLORS = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']
SMOOTH = lambda x: gaussian_filter1d(np.nan_to_num(x, nan=0.0), sigma=2)

def weighted_depth(R2, min_sig=MIN_SIG):
    R2p = np.clip(R2, 0, None)
    tot = R2p.sum(axis=0)
    return np.where(tot >= min_sig,
                    (depth_idx[:,None,None]*R2p).sum(axis=0)/(tot+1e-12), np.nan)

# ── Load and pool ─────────────────────────────────────────────────────────────
all_R2_cls = []; all_R2_patch = []; all_depth_patch = []; all_depth_patch = []
all_peak_patch = []; all_peak_patch = []; all_monkey_id = []
t_ms = None

for mi, monkey in enumerate(MONKEYS):
    with open(f'{CACHE_DIR}/time_resolved_perunit_dinov2_{monkey}.pkl','rb') as f:
        res = pkl.load(f)
    R2_cls   = res['r2_all']['patch'].astype(np.float64)    # (12, 90, n_units)
    R2_patch = res['r2_all']['patch'].astype(np.float64)
    if t_ms is None: t_ms = res['t_ms']
    n_units = R2_cls.shape[2]
    all_R2_cls.append(R2_cls);   all_R2_patch.append(R2_patch)
    all_depth_patch.append(weighted_depth(R2_cls))
    all_depth_patch.append(weighted_depth(R2_patch))
    pk_cls   = np.nanmax(R2_cls,   axis=(0,1))
    pk_patch = np.nanmax(R2_patch, axis=(0,1))
    all_peak_patch.append(pk_cls);  all_peak_patch.append(pk_patch)
    all_monkey_id.append(np.full(n_units, mi))
    print(f"  {monkey}: {n_units} units | CLS peak R² mean={pk_cls.mean():.3f} "
          f"| patch={pk_patch.mean():.3f}")

R2_cls_pool   = np.concatenate(all_R2_cls,   axis=2)   # (12,90,N)
R2_patch_pool = np.concatenate(all_R2_patch, axis=2)
depth_patch     = np.concatenate(all_depth_patch,  axis=1)  # (90,N)
depth_patch   = np.concatenate(all_depth_patch,axis=1)
peak_patch      = np.concatenate(all_peak_patch)             # (N,)
peak_patch    = np.concatenate(all_peak_patch)
monkey_id     = np.concatenate(all_monkey_id)
N = R2_cls_pool.shape[2]
print(f"\nPooled: {N} units from 5 monkeys")

early_t = (t_ms>=80)&(t_ms<=150)
late_t  = (t_ms>=180)&(t_ms<=350)
win_t   = (t_ms>=50)&(t_ms<=380)

# ── Feature construction (CLS token depth curve + R² envelope) ───────────────
def fill_depth(v):
    v = v.copy(); ok = np.isfinite(v)
    if ok.sum() < 3: return None
    xi = np.arange(len(v))
    v[~ok] = np.interp(xi[~ok], xi[ok], v[ok])
    return v

feat_rows, good_units = [], []
for u in range(N):
    if peak_patch[u] < 0.05: continue
    # CLS depth curve
    dc = fill_depth(depth_patch[win_t, u])
    if dc is None: continue
    # Patch depth curve
    dp = fill_depth(depth_patch[win_t, u])
    if dp is None: dp = np.full(win_t.sum(), np.nan)
    # R² envelope (max over blocks) for CLS
    r2_env = np.clip(R2_cls_pool[:, win_t, u], 0, None).max(axis=0)
    ed = np.nanmean(depth_patch[early_t, u])
    ld = np.nanmean(depth_patch[late_t,  u])
    pk_t_val = t_ms[np.clip(R2_cls_pool[:,:,u].max(axis=0).argmax(), 0, len(t_ms)-1)]
    dc_norm  = (dc - dc.min()) / (np.ptp(dc) + 1e-6)
    env_norm = r2_env / (r2_env.max() + 1e-8)
    feat_rows.append(np.concatenate([
        dc_norm, env_norm,
        [ed/N_BLOCKS, ld/N_BLOCKS, (ld-ed)/N_BLOCKS,
         peak_patch[u], pk_t_val/400.0]
    ]))
    good_units.append(u)

X = np.array(feat_rows); good_units = np.array(good_units)
n_good = len(good_units)
monkey_good = monkey_id[good_units]
print(f"Feature matrix: {X.shape}")
print("Units per monkey:", {MONKEYS[m]: int((monkey_good==m).sum()) for m in range(5)})

# PCA + UMAP
X_sc = StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(X))
pca  = PCA(n_components=min(25, n_good-1), whiten=True)
X_pca = pca.fit_transform(X_sc)
var80 = np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.80) + 1
X_red = X_pca[:, :max(var80, 5)]
print(f"PCA: {var80} PCs → 80% var")

try:
    import umap
    emb = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, random_state=42).fit_transform(X_red)
    emb_name = 'UMAP'
except:
    from sklearn.manifold import TSNE
    emb = TSNE(n_components=2, perplexity=50, random_state=42).fit_transform(X_red)
    emb_name = 't-SNE'
print(f"{emb_name} done")

# Silhouette sweep
ks = range(2, 9)
sil_scores = []
for k in ks:
    labs = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(X_red)
    sil_scores.append(silhouette_score(X_red, labs))
    print(f"  k={k}: sil={sil_scores[-1]:.3f}")
best_k = list(ks)[np.argmax(sil_scores)]
print(f"Best k={best_k}")

BLOCK_CMAP  = plt.cm.plasma
BLOCK_COLORS = [BLOCK_CMAP(i/(N_BLOCKS-1)) for i in range(N_BLOCKS)]

for k_use in sorted(set([best_k, 4, 5])):
    labels_good = KMeans(n_clusters=k_use, random_state=42, n_init=30).fit_predict(X_red)
    COLS = plt.cm.tab10.colors[:k_use]

    cluster_stats = []
    for c in range(k_use):
        cidx     = np.where(labels_good==c)[0]
        unit_idx = good_units[cidx]
        r2_c     = R2_cls_pool[:,:,unit_idx]
        dc_c     = depth_patch[:, unit_idx]
        pk_c     = peak_patch[unit_idx]
        ed = np.nanmean(dc_c[early_t]); ld = np.nanmean(dc_c[late_t])
        mk_comp  = {MONKEYS[m]: (monkey_id[unit_idx]==m).mean() for m in range(5)}
        cluster_stats.append(dict(
            label=c, n=len(cidx), cidx=cidx, unit_idx=unit_idx,
            mean_r2=np.nanmean(r2_c, axis=2), mean_depth=np.nanmean(dc_c,axis=1),
            mean_pk=np.nanmean(pk_c), ed=ed, ld=ld, shift=ld-ed,
            mk_comp=mk_comp,
        ))
    cluster_stats.sort(key=lambda x: x['shift'])
    for ci, cs in enumerate(cluster_stats):
        print(f"  C{ci}: n={cs['n']:3d} pk_R²={cs['mean_pk']:.3f} "
              f"ed={cs['ed']:.2f} ld={cs['ld']:.2f} Δd={cs['shift']:+.2f} | "
              + " ".join(f"{m[:4]}:{cs['mk_comp'][m]:.0%}" for m in MONKEYS))

    # ── FIGURE ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(5*(k_use+1), 14))
    fig.suptitle(f'DINOv2 Patch Pooled Clusters (k={k_use}, N={n_good}, 5 monkeys)',
                 fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(4, k_use+1, figure=fig, hspace=0.42, wspace=0.28)

    # Row 0: embedding (by cluster + by monkey) + silhouette
    ax = fig.add_subplot(gs[0, :2])
    for ci, cs in enumerate(cluster_stats):
        m = labels_good == cs['label']
        ax.scatter(emb[m,0], emb[m,1], c=[COLS[ci]], s=15, alpha=0.5,
                   label=f'C{ci}(Δd={cs["shift"]:+.1f},n={cs["n"]})')
    ax.set_title(f'{emb_name} — cluster'); ax.legend(fontsize=7)

    ax2 = fig.add_subplot(gs[0, 2])
    for mi, mk in enumerate(MONKEYS):
        mm = monkey_good==mi
        ax2.scatter(emb[mm,0], emb[mm,1], c=[MONKEY_COLORS[mi]], s=12, alpha=0.5, label=mk)
    ax2.set_title(f'{emb_name} — monkey'); ax2.legend(fontsize=7)

    ax3 = fig.add_subplot(gs[0, 3])
    ax3.plot(list(ks), sil_scores, 'ko-', lw=2)
    ax3.axvline(best_k, color='r', ls='--', lw=1.5, label=f'best k={best_k}')
    ax3.set_xlabel('k'); ax3.set_ylabel('Silhouette'); ax3.legend(fontsize=8)
    ax3.set_title('Cluster quality')

    # Row 1: monkey composition per cluster
    ax4 = fig.add_subplot(gs[1, :2])
    bottoms = np.zeros(k_use)
    for mi, mk in enumerate(MONKEYS):
        fracs = [cs['mk_comp'][mk] for cs in cluster_stats]
        ax4.bar(range(k_use), fracs, bottom=bottoms, color=MONKEY_COLORS[mi], label=mk, alpha=0.85)
        bottoms += np.array(fracs)
    ax4.set_xticks(range(k_use))
    ax4.set_xticklabels([f'C{i}\nΔd={cs["shift"]:+.2f}' for i,cs in enumerate(cluster_stats)], fontsize=8)
    ax4.set_ylabel('Fraction'); ax4.set_title('Monkey composition per cluster')
    ax4.legend(fontsize=8, loc='upper right')

    # Depth shift bar
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.bar(range(k_use), [cs['shift'] for cs in cluster_stats], color=COLS)
    ax5.axhline(0, color='k', lw=0.8)
    ax5.set_xticks(range(k_use)); ax5.set_xticklabels([f'C{i}' for i in range(k_use)])
    ax5.set_ylabel('Late−Early depth'); ax5.set_title('Depth shift per cluster')

    # Row 2: mean R² per block over time
    for ci, cs in enumerate(cluster_stats):
        ax = fig.add_subplot(gs[2, ci])
        for bi in range(N_BLOCKS):
            ax.plot(t_ms, SMOOTH(cs['mean_r2'][bi]), color=BLOCK_COLORS[bi],
                    label=f'B{bi}', lw=1.3, alpha=0.85)
        ax.axvline(0, color='gray', ls='--', lw=0.7); ax.axhline(0, color='gray', lw=0.4)
        ax.set_title(f'C{ci} n={cs["n"]}\nΔd={cs["shift"]:+.2f} R²={cs["mean_pk"]:.2f}',
                     fontsize=9, color=COLS[ci])
        ax.set_xlabel('ms', fontsize=7); ax.set_xlim(t_ms[0], t_ms[-1])
        if ci==0: ax.set_ylabel('Mean R²'); ax.legend(fontsize=4, ncol=2)

    # Row 3: individual + mean weighted depth curves
    for ci, cs in enumerate(cluster_stats):
        ax = fig.add_subplot(gs[3, ci])
        for u in cs['unit_idx'][:60]:
            dc = depth_patch[:, u]; valid = np.isfinite(dc)
            if valid.sum() > 5:
                ax.plot(t_ms[valid], dc[valid], color=COLS[ci], lw=0.4, alpha=0.1)
        md = cs['mean_depth']; vm = np.isfinite(md)
        ax.plot(t_ms[vm], SMOOTH(md[vm]), color='k', lw=2.5)
        ax.axvline(0, color='gray', ls='--', lw=0.7)
        ax.axvspan(80,150,alpha=0.1,color='steelblue')
        ax.axvspan(180,350,alpha=0.1,color='tomato')
        ax.set_yticks(range(0,N_BLOCKS,2))
        ax.set_yticklabels([f'B{i}' for i in range(0,N_BLOCKS,2)], fontsize=6)
        ax.set_xlabel('ms', fontsize=7); ax.set_xlim(t_ms[0], t_ms[-1])
        if ci==0: ax.set_ylabel('Weighted depth')

    outpath = f'{FIGDIR}/fig_dinov2_patch_clusters_k{k_use}.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")

# ── Also generate a per-monkey comparison of mean R²×time heatmaps ────────────
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('DINOv2 CLS — Mean R² (block × time) per monkey', fontsize=12, fontweight='bold')
for mi, monkey in enumerate(MONKEYS):
    with open(f'{CACHE_DIR}/time_resolved_perunit_dinov2_{monkey}.pkl','rb') as f:
        res = pkl.load(f)
    R2m = res['r2_all']['patch'].astype(np.float64)  # (12,90,n_units)
    mr2 = np.nanmean(R2m, axis=2)                  # (12,90)
    vmax = np.nanpercentile(mr2, 98)

    ax = axes[0, mi]
    im = ax.imshow(mr2, aspect='auto', origin='lower', cmap='hot',
                   extent=[t_ms[0], t_ms[-1], -0.5, N_BLOCKS-0.5], vmin=0, vmax=vmax)
    ax.axvline(0, color='cyan', lw=0.8, ls='--')
    plt.colorbar(im, ax=ax, shrink=0.7)
    ax.set_yticks(range(0,N_BLOCKS,2))
    ax.set_yticklabels([f'B{i}' for i in range(0,N_BLOCKS,2)], fontsize=7)
    ax.set_title(f'{monkey}\n({R2m.shape[2]} units, max R²={vmax:.3f})', fontsize=9)
    ax.set_xlabel('ms'); ax.set_ylabel('Block') if mi==0 else None

    # Population depth curve
    dc_m = weighted_depth(R2m)
    pk_m = np.nanmax(R2m, axis=(0,1))
    resp = pk_m >= 0.05
    pop_d = np.nanmean(dc_m[:, resp], axis=1) if resp.sum()>0 else np.full(len(t_ms), np.nan)
    ax2 = axes[1, mi]
    ax2.plot(t_ms, SMOOTH(pop_d), 'k-', lw=2.5)
    ax2.axvline(0, color='gray', ls='--', lw=0.8)
    ax2.axvspan(80,150,alpha=0.12,color='steelblue')
    ax2.axvspan(180,350,alpha=0.12,color='tomato')
    ed = np.nanmean(pop_d[early_t]); ld = np.nanmean(pop_d[late_t])
    ax2.set_yticks(range(0,N_BLOCKS,2))
    ax2.set_yticklabels([f'B{i}' for i in range(0,N_BLOCKS,2)], fontsize=7)
    ax2.set_title(f'Depth: early={ed:.1f} late={ld:.1f} Δ={ld-ed:+.2f}', fontsize=8)
    ax2.set_xlabel('ms'); ax2.set_ylabel('Weighted depth') if mi==0 else None
    ax2.set_xlim(t_ms[0], t_ms[-1])

plt.tight_layout()
outpath_mk = f'{FIGDIR}/fig_dinov2_patch_per_monkey.png'
plt.savefig(outpath_mk, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {outpath_mk}")

print("\nAll done!")
