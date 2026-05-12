"""
Cluster neurons by their time × layer R² profile.
Two feature sets: (A) summary stats, (B) full depth curve + R² envelope.
Dim reduction: PCA → UMAP (or t-SNE fallback).
Clustering: k-means with silhouette selection.
"""
import sys, os
sys.path.insert(0, '/n/home12/binxuwang/Github/NHP_NSD_analysis')
import numpy as np, pickle as pkl
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

CACHE  = '/n/home12/binxuwang/Github/NHP_NSD_analysis/notebooks/cache/time_resolved_perunit_JianJian.pkl'
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

with open(CACHE, 'rb') as f: res = pkl.load(f)
R2      = res['r2_perunit'].astype(np.float64)   # (6, 90, 456)
t_ms    = res['t_ms']                             # (90,)
layers  = res['layers']
n_layers, n_time, n_units = R2.shape
depth_idx = np.arange(n_layers, dtype=float)
LAYER_LABELS = ['relu','layer1','layer2','layer3','layer4','avgpool']
LAYER_COLORS = ['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f','#b07aa1']
SMOOTH = lambda x: gaussian_filter1d(x, sigma=2)

# ── Weighted depth curve per unit ────────────────────────────────────────────
MIN_SIG = 0.02
R2pos    = np.clip(R2, 0, None)
total_r2 = R2pos.sum(axis=0)                      # (90, 456)
depth_com = np.where(total_r2 >= MIN_SIG,
                     (depth_idx[:,None,None]*R2pos).sum(axis=0)/(total_r2+1e-12),
                     np.nan)                       # (90, 456)
peak_r2   = R2.max(axis=(0,1))                     # (456,)

early_t = (t_ms>=80)&(t_ms<=150)
late_t  = (t_ms>=180)&(t_ms<=350)

# ── FEATURE SET: combined depth curve + magnitude ───────────────────────────
# Sampled at 10ms resolution in 50-380ms window, plus peak R² and latency
win = (t_ms>=50)&(t_ms<=380)
t_win  = t_ms[win]

# Per unit: fill NaN in depth_com with interpolation only inside the signal window
def fill_depth(dc_vec):
    """Linearly interpolate NaN in dc_vec; still-NaN → median."""
    v = dc_vec.copy()
    ok = np.isfinite(v)
    if ok.sum() < 3: return None
    xi = np.arange(len(v))
    v[~ok] = np.interp(xi[~ok], xi[ok], v[ok])
    return v

# Build feature matrix
feat_rows, good_units = [], []
for u in range(n_units):
    if peak_r2[u] < 0.05: continue
    dc_win = depth_com[win, u]
    dc_filled = fill_depth(dc_win)
    if dc_filled is None: continue
    # Max R² over layers at each time point in window (magnitude envelope)
    r2_env = R2pos[:, win, u].max(axis=0)         # (n_win,)
    # Summary: early/late depth, depth shift, peak R², peak latency
    ed = np.nanmean(depth_com[early_t, u])
    ld = np.nanmean(depth_com[late_t,  u])
    pk_t = t_ms[R2.max(axis=0)[:, u].argmax()]    # time of peak total R²
    # Normalize depth curve to [0,1] range for shape focus
    dc_norm = (dc_filled - dc_filled.min()) / (np.ptp(dc_filled) + 1e-6)
    # Normalize envelope by peak
    env_norm = r2_env / (r2_env.max() + 1e-8)
    feat_vec = np.concatenate([
        dc_norm,                         # depth curve shape (~66 dims)
        env_norm,                        # R² temporal envelope (~66 dims)
        [ed/n_layers, ld/n_layers,       # normalized early/late depth
         (ld-ed)/n_layers,               # depth shift
         peak_r2[u],                     # peak R² (magnitude)
         pk_t/400.0,                     # peak latency (normalized)
    ]])
    feat_rows.append(feat_vec)
    good_units.append(u)

X = np.array(feat_rows)
good_units = np.array(good_units)
n_good = len(good_units)
print(f"Feature matrix: {X.shape}  ({n_good} units × {X.shape[1]} features)")

# Scale
scaler = StandardScaler()
# Impute NaN with column median before scaling
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
X_imp = imputer.fit_transform(X)
X_sc = scaler.fit_transform(X_imp)

# PCA
pca = PCA(n_components=min(20, n_good-1), whiten=True)
X_pca = pca.fit_transform(X_sc)
var80 = np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.80) + 1
print(f"PCA: {var80} PCs explain 80% variance")
X_red = X_pca[:, :max(var80, 5)]

# Dim reduction for viz
try:
    import umap
    emb = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.15,
                    metric='euclidean', random_state=42).fit_transform(X_red)
    emb_name = 'UMAP'
    print("UMAP ok")
except:
    emb = TSNE(n_components=2, perplexity=40, random_state=42,
               n_iter=1000).fit_transform(X_red)
    emb_name = 't-SNE'
    print("t-SNE fallback")

# ── k-means sweep ─────────────────────────────────────────────────────────────
ks = range(2, 9)
sil_scores = []
for k in ks:
    labs = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(X_red)
    s = silhouette_score(X_red, labs)
    sil_scores.append(s)
    print(f"  k={k}: sil={s:.3f}")
best_k = list(ks)[np.argmax(sil_scores)]
print(f"Best k={best_k}")

# Produce figures for k=best_k AND k=5 (usually interpretable)
for k_use in sorted(set([best_k, 5])):
    labels_good = KMeans(n_clusters=k_use, random_state=42, n_init=30).fit_predict(X_red)
    
    # Characterize each cluster
    cluster_stats = []
    for c in range(k_use):
        cidx = np.where(labels_good == c)[0]
        unit_idx = good_units[cidx]
        r2_c  = R2[:, :, unit_idx]
        dc_c  = depth_com[:, unit_idx]
        pr2_c = peak_r2[unit_idx]
        ed = np.nanmean(dc_c[early_t])
        ld = np.nanmean(dc_c[late_t])
        cluster_stats.append(dict(
            label=c, n=len(cidx), cidx=cidx, unit_idx=unit_idx,
            mean_r2=r2_c.mean(axis=2),           # (6,90)
            mean_depth=np.nanmean(dc_c, axis=1), # (90,)
            mean_pk_r2=pr2_c.mean(),
            early_d=ed, late_d=ld, shift=ld-ed,
        ))
        print(f"  C{c}: n={len(cidx):3d} pk_R²={pr2_c.mean():.3f} "
              f"ed={ed:.2f} ld={ld:.2f} Δd={ld-ed:+.2f}")
    # Sort by depth shift for interpretable ordering
    cluster_stats.sort(key=lambda x: x['shift'])
    COLS = plt.cm.tab10.colors[:k_use]

    # ── FIGURE ────────────────────────────────────────────────────────────────
    nr, nc_plot = 3, k_use+1
    fig = plt.figure(figsize=(4*(k_use+1), 11))
    fig.suptitle(f'Neuron Clusters (k={k_use}, n={n_good}, JianJian) — sorted by depth shift',
                 fontsize=12, fontweight='bold')
    gs = gridspec.GridSpec(nr, nc_plot, figure=fig, hspace=0.45, wspace=0.30)

    # Row 0: embedding + silhouette
    ax_emb = fig.add_subplot(gs[0, :2])
    for ci, cs in enumerate(cluster_stats):
        m = labels_good == cs['label']
        ax_emb.scatter(emb[m,0], emb[m,1], c=[COLS[ci]], s=18, alpha=0.6,
                       label=f'C{ci}(Δd={cs["shift"]:+.1f})')
    ax_emb.set_title(f'{emb_name} embedding'); ax_emb.legend(fontsize=7)
    ax_emb.set_xlabel('Dim 1'); ax_emb.set_ylabel('Dim 2')

    ax_sil = fig.add_subplot(gs[0, 2])
    ax_sil.plot(list(ks), sil_scores, 'ko-', lw=2)
    ax_sil.axvline(best_k, color='r', ls='--', lw=1.5, label=f'best k={best_k}')
    ax_sil.set_xlabel('k'); ax_sil.set_ylabel('Silhouette')
    ax_sil.set_title('Cluster quality'); ax_sil.legend(fontsize=8)

    ax_bar = fig.add_subplot(gs[0, 3])
    shifts = [cs['shift'] for cs in cluster_stats]
    bars = ax_bar.bar(range(k_use), shifts, color=COLS)
    ax_bar.axhline(0, color='k', lw=0.8)
    ax_bar.set_xticks(range(k_use))
    ax_bar.set_xticklabels([f'C{i}\nn={cs["n"]}' for i,cs in enumerate(cluster_stats)], fontsize=8)
    ax_bar.set_ylabel('Late−Early depth'); ax_bar.set_title('Depth shift per cluster')

    # Row 1: mean R² per layer over time
    for ci, cs in enumerate(cluster_stats):
        ax = fig.add_subplot(gs[1, ci])
        for li in range(n_layers):
            ax.plot(t_ms, SMOOTH(cs['mean_r2'][li]), color=LAYER_COLORS[li],
                    label=LAYER_LABELS[li], lw=1.5)
        ax.axvline(0, color='gray', ls='--', lw=0.7)
        ax.axhline(0, color='gray', lw=0.4)
        ax.set_title(f'C{ci} (n={cs["n"]}, Δd={cs["shift"]:+.2f})',
                     fontsize=9, color=COLS[ci])
        ax.set_xlabel('ms', fontsize=7); ax.set_xlim(t_ms[0], t_ms[-1])
        if ci == 0:
            ax.set_ylabel('Mean R²')
            ax.legend(fontsize=5)

    # Row 2: individual + mean weighted depth curves
    for ci, cs in enumerate(cluster_stats):
        ax = fig.add_subplot(gs[2, ci])
        for u in cs['unit_idx'][:50]:
            dc = depth_com[:, u]; valid = np.isfinite(dc)
            if valid.sum() > 5:
                ax.plot(t_ms[valid], dc[valid], color=COLS[ci], lw=0.5, alpha=0.12)
        md = cs['mean_depth']; vm = np.isfinite(md)
        ax.plot(t_ms[vm], SMOOTH(md[vm]), color='k', lw=2.5)
        ax.axvline(0, color='gray', ls='--', lw=0.7)
        ax.axvspan(80,150,alpha=0.1,color='steelblue')
        ax.axvspan(180,350,alpha=0.1,color='tomato')
        ax.set_yticks(range(n_layers)); ax.set_yticklabels(LAYER_LABELS, fontsize=6)
        ax.set_xlabel('ms', fontsize=7); ax.set_xlim(t_ms[0], t_ms[-1])
        if ci == 0: ax.set_ylabel('Weighted depth')

    outpath = f'{FIGDIR}/fig_clusters_k{k_use}.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")

print("\nAll done!")
