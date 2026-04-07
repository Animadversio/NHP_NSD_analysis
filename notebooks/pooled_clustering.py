"""Pool neurons across all monkey sessions and run clustering."""
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
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

CACHE_DIR = '/n/home12/binxuwang/Github/NHP_NSD_analysis/notebooks/cache'
FIGDIR    = '/n/home12/binxuwang/Github/NHP_NSD_analysis/notebooks/figures'
os.makedirs(FIGDIR, exist_ok=True)

MONKEYS = ['JianJian', 'FaCai', 'TuTu', 'ZhuangZhuang', 'MaoDan']
LAYER_LABELS = ['relu','layer1','layer2','layer3','layer4','avgpool']
LAYER_COLORS = ['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f','#b07aa1']
SMOOTH = lambda x: gaussian_filter1d(x, sigma=2)
MIN_SIG = 0.02
depth_idx = np.arange(6, dtype=float)

def compute_depth(R2):
    """R2: (6, 90, n_units) → depth_com (90, n_units), NaN where signal < MIN_SIG"""
    R2pos = np.clip(R2, 0, None)
    tot = R2pos.sum(axis=0)
    return np.where(tot >= MIN_SIG,
                    (depth_idx[:,None,None]*R2pos).sum(axis=0)/(tot+1e-12),
                    np.nan)

# ── Load & pool all sessions ─────────────────────────────────────────────────
all_R2, all_depth, all_peak_r2, all_monkey_id = [], [], [], []
t_ms = None

for mi, monkey in enumerate(MONKEYS):
    path = f'{CACHE_DIR}/time_resolved_perunit_{monkey}.pkl'
    with open(path, 'rb') as f: res = pkl.load(f)
    R2 = res['r2_perunit'].astype(np.float64)   # (6, 90, n_units)
    if t_ms is None: t_ms = res['t_ms']
    n_units = R2.shape[2]
    dc = compute_depth(R2)                       # (90, n_units)
    pk = np.nanmax(R2, axis=(0,1))                      # (n_units,)
    all_R2.append(R2)
    all_depth.append(dc)
    all_peak_r2.append(pk)
    all_monkey_id.append(np.full(n_units, mi))
    print(f"  {monkey}: {n_units} units, peak R² mean={pk.mean():.3f}")

# Concatenate along unit axis
R2_pool    = np.concatenate(all_R2,      axis=2)  # (6, 90, N_total)
depth_pool = np.concatenate(all_depth,   axis=1)  # (90, N_total)
# Use nanmax so NaN time bins are skipped
peak_r2    = np.concatenate(all_peak_r2, axis=0)  # (N_total,)
monkey_id  = np.concatenate(all_monkey_id)         # (N_total,)
N = R2_pool.shape[2]
print(f"\nPooled: {N} units from {len(MONKEYS)} monkeys")

early_t = (t_ms>=80)&(t_ms<=150)
late_t  = (t_ms>=180)&(t_ms<=350)
win_t   = (t_ms>=50)&(t_ms<=380)
n_win   = win_t.sum()

# ── Build features ────────────────────────────────────────────────────────────
def fill_depth(dc_vec):
    v = dc_vec.copy(); ok = np.isfinite(v)
    if ok.sum() < 3: return None
    xi = np.arange(len(v))
    v[~ok] = np.interp(xi[~ok], xi[ok], v[ok])
    return v

feat_rows, good_units = [], []
for u in range(N):
    if peak_r2[u] < 0.05: continue
    dc_win = depth_pool[win_t, u]
    dc_f = fill_depth(dc_win)
    if dc_f is None: continue
    r2_env = np.clip(R2_pool[:, win_t, u], 0, None).max(axis=0)
    ed = np.nanmean(depth_pool[early_t, u])
    ld = np.nanmean(depth_pool[late_t,  u])
    pk_t_val = t_ms[R2_pool.max(axis=0)[:, u].argmax()]
    dc_norm  = (dc_f - dc_f.min()) / (np.ptp(dc_f) + 1e-6)
    env_norm = r2_env / (r2_env.max() + 1e-8)
    feat_rows.append(np.concatenate([
        dc_norm, env_norm,
        [ed/6, ld/6, (ld-ed)/6, peak_r2[u], pk_t_val/400.0]
    ]))
    good_units.append(u)

X = np.array(feat_rows); good_units = np.array(good_units)
n_good = len(good_units)
monkey_good = monkey_id[good_units]
print(f"Feature matrix: {X.shape}")
print("Units per monkey:", {MONKEYS[m]: (monkey_good==m).sum() for m in range(len(MONKEYS))})

# Scale + impute
X_sc = StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(X))
pca = PCA(n_components=min(25, n_good-1), whiten=True)
X_pca = pca.fit_transform(X_sc)
var80 = np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.80) + 1
X_red = X_pca[:, :max(var80, 5)]
print(f"PCA: {var80} PCs → 80% variance")

# t-SNE
try:
    import umap
    emb = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, random_state=42).fit_transform(X_red)
    emb_name = 'UMAP'
except:
    emb = TSNE(n_components=2, perplexity=50, random_state=42, n_iter=1000).fit_transform(X_red)
    emb_name = 't-SNE'
print(f"{emb_name} done")

# k-means sweep
ks = range(2, 9)
sil_scores = []
for k in ks:
    labs = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(X_red)
    sil_scores.append(silhouette_score(X_red, labs))
    print(f"  k={k}: sil={sil_scores[-1]:.3f}")
best_k = list(ks)[np.argmax(sil_scores)]
print(f"Best k={best_k}")

MONKEY_COLORS = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']
MONKEY_MARKERS = ['o','s','^','D','v']

for k_use in sorted(set([best_k, 4, 5])):
    labels_good = KMeans(n_clusters=k_use, random_state=42, n_init=30).fit_predict(X_red)
    COLS = plt.cm.tab10.colors[:k_use]

    cluster_stats = []
    for c in range(k_use):
        cidx = np.where(labels_good==c)[0]
        unit_idx = good_units[cidx]
        r2_c = R2_pool[:,:,unit_idx]
        dc_c = depth_pool[:,unit_idx]
        pk_c = peak_r2[unit_idx]
        ed = np.nanmean(dc_c[early_t])
        ld = np.nanmean(dc_c[late_t])
        # Monkey composition
        mk_comp = {MONKEYS[m]: (monkey_id[unit_idx]==m).mean() for m in range(len(MONKEYS))}
        cluster_stats.append(dict(
            label=c, n=len(cidx), cidx=cidx, unit_idx=unit_idx,
            mean_r2=r2_c.mean(axis=2), mean_depth=np.nanmean(dc_c,axis=1),
            mean_pk_r2=np.nanmean(pk_c), early_d=ed, late_d=ld, shift=ld-ed,
            mk_comp=mk_comp,
        ))
    cluster_stats.sort(key=lambda x: x['shift'])
    for ci, cs in enumerate(cluster_stats):
        print(f"  C{ci}: n={cs['n']:3d} pk_R²={cs['mean_pk_r2']:.3f} "
              f"ed={cs['early_d']:.2f} ld={cs['late_d']:.2f} Δd={cs['shift']:+.2f} "
              f"| " + " ".join(f"{m}:{cs['mk_comp'][m]:.0%}" for m in MONKEYS))

    # ── FIGURE ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(5*(k_use+1), 13))
    fig.suptitle(f'Pooled Neuron Clusters (k={k_use}, N={n_good} units, 5 monkeys)',
                 fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(4, k_use+1, figure=fig, hspace=0.45, wspace=0.30)

    # Row 0a: embedding colored by cluster
    ax = fig.add_subplot(gs[0, :2])
    for ci, cs in enumerate(cluster_stats):
        m = labels_good==cs['label']
        ax.scatter(emb[m,0], emb[m,1], c=[COLS[ci]], s=15, alpha=0.5,
                   label=f'C{ci}(Δd={cs["shift"]:+.1f},n={cs["n"]})')
    ax.set_title(f'{emb_name} — cluster'); ax.legend(fontsize=7)

    # Row 0b: embedding colored by monkey
    ax2 = fig.add_subplot(gs[0, 2])
    for mi, mk in enumerate(MONKEYS):
        m = monkey_good==mi
        ax2.scatter(emb[m,0], emb[m,1], c=[MONKEY_COLORS[mi]], marker=MONKEY_MARKERS[mi],
                    s=12, alpha=0.5, label=mk)
    ax2.set_title(f'{emb_name} — monkey'); ax2.legend(fontsize=7)

    # Row 0c: silhouette + depth shift bar
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.plot(list(ks), sil_scores, 'ko-', lw=2)
    ax3.axvline(best_k, color='r', ls='--', lw=1.5)
    ax3.set_xlabel('k'); ax3.set_ylabel('Silhouette'); ax3.set_title('Cluster quality')

    # Row 1: monkey composition per cluster (stacked bar)
    ax4 = fig.add_subplot(gs[1, :2])
    bottoms = np.zeros(k_use)
    for mi, mk in enumerate(MONKEYS):
        fracs = [cs['mk_comp'][mk] for cs in cluster_stats]
        ax4.bar(range(k_use), fracs, bottom=bottoms, color=MONKEY_COLORS[mi], label=mk, alpha=0.85)
        bottoms += np.array(fracs)
    ax4.set_xticks(range(k_use))
    ax4.set_xticklabels([f'C{i}\n(Δd={cs["shift"]:+.1f})' for i,cs in enumerate(cluster_stats)])
    ax4.set_ylabel('Fraction of units'); ax4.set_title('Monkey composition per cluster')
    ax4.legend(fontsize=8, loc='upper right')

    # Row 2: mean R² per layer over time per cluster
    for ci, cs in enumerate(cluster_stats):
        ax = fig.add_subplot(gs[2, ci])
        for li in range(6):
            ax.plot(t_ms, SMOOTH(cs['mean_r2'][li]), color=LAYER_COLORS[li],
                    label=LAYER_LABELS[li], lw=1.5)
        ax.axvline(0, color='gray', ls='--', lw=0.7); ax.axhline(0, color='gray', lw=0.4)
        ax.set_title(f'C{ci} n={cs["n"]}\nΔd={cs["shift"]:+.2f}', fontsize=9, color=COLS[ci])
        ax.set_xlabel('ms', fontsize=7); ax.set_xlim(t_ms[0], t_ms[-1])
        if ci==0: ax.set_ylabel('Mean R²'); ax.legend(fontsize=5)

    # Row 3: individual + mean weighted depth curves per cluster
    for ci, cs in enumerate(cluster_stats):
        ax = fig.add_subplot(gs[3, ci])
        for u in cs['unit_idx'][:60]:
            dc = depth_pool[:, u]; valid = np.isfinite(dc)
            if valid.sum() > 5:
                ax.plot(t_ms[valid], dc[valid], color=COLS[ci], lw=0.4, alpha=0.1)
        md = cs['mean_depth']; vm = np.isfinite(md)
        ax.plot(t_ms[vm], SMOOTH(md[vm]), color='k', lw=2.5)
        ax.axvline(0, color='gray', ls='--', lw=0.7)
        ax.axvspan(80,150,alpha=0.1,color='steelblue')
        ax.axvspan(180,350,alpha=0.1,color='tomato')
        ax.set_yticks(range(6)); ax.set_yticklabels(LAYER_LABELS, fontsize=6)
        ax.set_xlabel('ms', fontsize=7); ax.set_xlim(t_ms[0], t_ms[-1])
        if ci==0: ax.set_ylabel('Weighted depth')

    outpath = f'{FIGDIR}/fig_pooled_clusters_k{k_use}.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")

# ── Cross-monkey consistency: per-monkey cluster assignment ──────────────────
# Use best_k, check if same clusters emerge in each monkey separately
print(f"\n=== Cross-monkey cluster depth shifts (pooled k={best_k}) ===")
labels_best = KMeans(n_clusters=best_k, random_state=42, n_init=30).fit_predict(X_red)
cluster_stats_best = []
for c in range(best_k):
    cidx = np.where(labels_best==c)[0]
    unit_idx = good_units[cidx]
    dc_c = depth_pool[:, unit_idx]
    ed = np.nanmean(dc_c[early_t]); ld = np.nanmean(dc_c[late_t])
    cluster_stats_best.append(dict(c=c, n=len(cidx), shift=ld-ed))
cluster_stats_best.sort(key=lambda x: x['shift'])

for ci, cs in enumerate(cluster_stats_best):
    orig = cs['c']
    print(f"  C{ci} (orig {orig}): n={cs['n']}, Δd={cs['shift']:+.3f}")
    for mi, mk in enumerate(MONKEYS):
        mk_mask = monkey_good == mi
        mk_in_c = mk_mask & (labels_best == orig)
        n_mk_total = mk_mask.sum()
        n_mk_c = mk_in_c.sum()
        dc_mk = depth_pool[:, good_units[mk_in_c]]
        ed_mk = np.nanmean(dc_mk[early_t]); ld_mk = np.nanmean(dc_mk[late_t])
        print(f"    {mk:15s}: {n_mk_c:3d}/{n_mk_total:3d} ({n_mk_c/n_mk_total:.0%}), "
              f"Δd={ld_mk-ed_mk:+.2f}")

print("\nAll done!")
