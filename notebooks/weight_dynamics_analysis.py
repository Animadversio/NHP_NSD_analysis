"""
Weight dynamics analysis: cosine similarity, subspace angles, trajectory PCA, LDS fit.
JianJian CLS block 7 — run after dinov2_all_sessions.py has cached R² results.
"""
import sys, os, pickle as pkl, numpy as np, h5py, matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import subspace_angles, svd
sys.path.insert(0, '/n/home12/binxuwang/Github/NHP_NSD_analysis')
from NSD_utils.h5_dataset_utils import load_data_from_GoodUnitStrc
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from tqdm import tqdm

CACHE_DIR  = '/n/home12/binxuwang/Github/NHP_NSD_analysis/notebooks/cache'
DATA_ROOT  = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/NSD_N3'
STORE_DIR  = os.environ.get('STORE_DIR', '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang')
WCOEF_DIR  = os.path.join(STORE_DIR, 'weight_coefs')
FIG_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(WCOEF_DIR, exist_ok=True)

N_BLOCKS = 12; N_PCA = 200; ALPHAS = np.logspace(-2, 6, 25)
TIME_STRIDE = 5; MIN_VAR = 1e-6; FEAT_TYPE = 'cls'; BEST_BLOCK = 7; K_LDS = 10

# ── features & PCA ───────────────────────────────────────────────────────────
FEAT_CACHE = os.path.join(CACHE_DIR, 'dinov2_nsd_features.pkl')
with open(FEAT_CACHE, 'rb') as f: feat_dict = pkl.load(f)
n_images = feat_dict['blocks.0_cls'].shape[0]
rng = np.random.RandomState(42)
train_idx = rng.choice(n_images, int(0.8*n_images), replace=False)
test_idx  = np.setdiff1d(np.arange(n_images), train_idx)

ln = f'blocks.{BEST_BLOCK}_{FEAT_TYPE}'
pca = PCA(n_components=N_PCA); Xtrain = pca.fit_transform(feat_dict[ln][train_idx])

# ── neural data ───────────────────────────────────────────────────────────────
fname = 'GoodUnit_240629_JianJian_NSD1000_LOC_g2.mat'
fh = h5py.File(os.path.join(DATA_ROOT, fname), 'r')
d  = load_data_from_GoodUnitStrc(fh)
R  = d['response_matrix_img']; t_full = d['PsthRange']; fh.close()
n_units = R.shape[0]
t_indices = np.where((t_full >= -49) & (np.arange(len(t_full)) % TIME_STRIDE == 0))[0]
t_ms = t_full[t_indices]; n_t = len(t_indices)

# ── weight extraction ─────────────────────────────────────────────────────────
OUT_PATH = os.path.join(WCOEF_DIR, 'JianJian_cls_block7_coefs.npy')
if os.path.exists(OUT_PATH):
    W = np.load(OUT_PATH)
else:
    W = np.zeros((n_t, n_units, N_PCA), dtype=np.float32)
    for ti, tidx in enumerate(tqdm(t_indices, desc='Block7 coefs')):
        y = R[:, tidx, :].T
        clf = RidgeCV(alphas=ALPHAS, alpha_per_target=True)
        clf.fit(Xtrain, y[train_idx])
        W[ti] = clf.coef_
    np.save(OUT_PATH, W)

# ── load prev R² for reference time ──────────────────────────────────────────
with open(os.path.join(CACHE_DIR, 'time_resolved_perunit_dinov2_JianJian.pkl'), 'rb') as f:
    prev = pkl.load(f)
r2_block7 = prev['r2_all']['cls'][7]
t_ref_idx = int(np.nanargmax(np.nanmean(r2_block7, axis=1)))
t_ref_ms  = t_ms[t_ref_idx]
best_unit  = int(np.nanargmax(np.nanmax(r2_block7, axis=0)))

# ── A: cosine similarity ──────────────────────────────────────────────────────
def cos_sim_matrix(A, B):
    num = np.einsum('ij,ij->i', A, B)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12
    return num / den

W_ref  = W[t_ref_idx]
cos_t  = np.array([cos_sim_matrix(W[ti], W_ref) for ti in range(n_t)])

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
ax = axes[0]
ax.plot(t_ms, r2_block7[:, best_unit], 'steelblue', lw=2, label='R² (block 7)')
ax2 = ax.twinx()
ax2.plot(t_ms, cos_t[:, best_unit], 'tomato', lw=2, label='cos(w[t], w[t_ref])')
ax2.axhline(0, color='tomato', lw=0.5, ls='--')
ax.axvline(t_ref_ms, color='k', lw=1, ls='--', alpha=0.5)
ax.set_xlabel('Time (ms)'); ax.set_ylabel('R²', color='steelblue')
ax2.set_ylabel('Cosine similarity', color='tomato'); ax.set_title(f'Single unit {best_unit} (best R²)')
ax.legend(loc='upper left'); ax2.legend(loc='upper right')
mean_cos = gaussian_filter1d(np.nanmean(cos_t, axis=1), sigma=1)
std_cos  = np.nanstd(cos_t, axis=1)
ax = axes[1]
ax.fill_between(t_ms, mean_cos - std_cos, mean_cos + std_cos, alpha=0.2, color='tomato')
ax.plot(t_ms, mean_cos, 'tomato', lw=2)
ax.axvline(t_ref_ms, color='k', lw=1, ls='--', alpha=0.5, label=f't_ref={t_ref_ms:.0f}ms')
ax.axhline(0, color='k', lw=0.5, ls='--'); ax.set_xlabel('Time (ms)'); ax.set_ylabel('Mean cosine similarity')
ax.set_title('Population: cos(w[t], w[t_ref])'); ax.legend()
sort_order = np.argsort(np.nanmax(r2_block7, axis=0))[::-1]
im = axes[2].imshow(cos_t[:, sort_order].T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1,
               extent=[t_ms[0], t_ms[-1], 0, n_units])
axes[2].axvline(t_ref_ms, color='k', lw=1, ls='--'); axes[2].set_xlabel('Time (ms)')
axes[2].set_ylabel('Units (sorted by peak R²)'); axes[2].set_title('Cosine similarity heatmap')
plt.colorbar(im, ax=axes[2], label='cos sim')
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/fig_weight_cosine_JianJian.png', dpi=130, bbox_inches='tight')

# ── B: canonical angles ───────────────────────────────────────────────────────
from numpy.linalg import svd as npsvd
def canonical_angles_vs_ref(W, ref_idx, top_k=20):
    U_ref, _, _ = npsvd(W[ref_idx], full_matrices=False); Q_ref = U_ref[:, :top_k]
    angles_all = []
    for ti in range(W.shape[0]):
        U_t, _, _ = npsvd(W[ti], full_matrices=False)
        angles_all.append(np.degrees(subspace_angles(Q_ref, U_t[:, :top_k])))
    return np.array(angles_all)

cang = canonical_angles_vs_ref(W, t_ref_idx, top_k=20)
mean_ang = np.mean(cang, axis=1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
colors = plt.cm.plasma(np.linspace(0, 0.85, 5))
for k_i, k in enumerate([0, 1, 2, 4, 9]):
    axes[0].plot(t_ms, gaussian_filter1d(cang[:, k], 1.5), color=colors[k_i], lw=2, label=f'angle {k+1}')
axes[0].axvline(t_ref_ms, color='k', lw=1, ls='--', alpha=0.5); axes[0].set_xlabel('Time (ms)')
axes[0].set_ylabel('Canonical angle (°)'); axes[0].set_title('Subspace canonical angles vs reference')
axes[0].legend(fontsize=8)
axes[1].plot(t_ms, gaussian_filter1d(mean_ang, 1.5), 'purple', lw=2)
axes[1].axvline(t_ref_ms, color='k', lw=1, ls='--', alpha=0.5); axes[1].set_xlabel('Time (ms)')
axes[1].set_ylabel('Mean canonical angle (°)'); axes[1].set_title('Mean subspace rotation from t_ref')
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/fig_subspace_angles_JianJian.png', dpi=130, bbox_inches='tight')

# ── C: trajectory dimensionality ──────────────────────────────────────────────
W_flat = W.reshape(n_t, -1).astype(np.float64); W_flat -= W_flat.mean(axis=0, keepdims=True)
U, sv, Vt = npsvd(W_flat, full_matrices=False)
var_explained = sv**2 / (sv**2).sum(); cumvar = np.cumsum(var_explained)
PR = (sv**2).sum()**2 / (sv**4).sum()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].bar(np.arange(1,21), var_explained[:20]*100, color='steelblue', alpha=0.8)
axes[0].plot(np.arange(1,21), cumvar[:20]*100, 'ro-', ms=4)
axes[0].set_xlabel('PC'); axes[0].set_ylabel('Variance explained (%)')
axes[0].set_title(f'W trajectory dimensionality (PR={PR:.1f})')
coords = U[:, :2] * sv[:2]
sc = axes[1].scatter(coords[:,0], coords[:,1], c=t_ms, cmap='RdYlGn', s=30, zorder=3)
axes[1].plot(coords[:,0], coords[:,1], 'k-', lw=0.8, alpha=0.5)
for label, t_mark in [('0ms',0),('100ms',100),('200ms',200),('350ms',350)]:
    idx = np.argmin(np.abs(t_ms - t_mark)); axes[1].annotate(label, coords[idx], fontsize=8, xytext=(5,5), textcoords='offset points')
    axes[1].scatter(*coords[idx], c='k', s=60, zorder=4)
plt.colorbar(sc, ax=axes[1], label='Time (ms)'); axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2'); axes[1].set_title('Weight code trajectory')
from mpl_toolkits.mplot3d import Axes3D
ax3d = fig.add_subplot(1,3,3,projection='3d'); coords3 = U[:,:3]*sv[:3]
ax3d.scatter(coords3[:,0],coords3[:,1],coords3[:,2],c=t_ms,cmap='RdYlGn',s=20)
ax3d.plot(coords3[:,0],coords3[:,1],coords3[:,2],'k-',lw=0.5,alpha=0.4)
ax3d.set_xlabel('PC1'); ax3d.set_ylabel('PC2'); ax3d.set_zlabel('PC3'); ax3d.set_title('3D trajectory')
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/fig_weight_trajectory_JianJian.png', dpi=130, bbox_inches='tight')

# ── D: LDS ────────────────────────────────────────────────────────────────────
Z = (U[:, :K_LDS] * sv[:K_LDS]).T
Z1 = Z[:, :-1]; Z2 = Z[:, 1:]
A_hat, _, _, _ = np.linalg.lstsq(Z1.T, Z2.T, rcond=None); A_hat = A_hat.T
eigs = np.linalg.eigvals(A_hat)
Z_pred = np.zeros_like(Z); Z_pred[:,0] = Z[:,0]
for t in range(1, n_t): Z_pred[:,t] = A_hat @ Z_pred[:,t-1]
Z_onestep = A_hat @ Z[:,:-1]
r2_lds = 1 - np.sum((Z-Z_pred)**2)/np.sum((Z-Z.mean(axis=1,keepdims=True))**2)
r2_onestep = 1 - np.sum((Z[:,1:]-Z_onestep)**2)/np.sum((Z[:,1:]-Z[:,1:].mean(axis=1,keepdims=True))**2)
print(f"LDS open-loop R²={r2_lds:.4f}, one-step R²={r2_onestep:.4f}")

fig, axes = plt.subplots(2,3,figsize=(16,8))
for ki,(ax,label) in enumerate(zip(axes[0,:2],['PC1','PC2'])):
    ax.plot(t_ms,Z[ki],'steelblue',lw=2,label='Actual'); ax.plot(t_ms,Z_pred[ki],'tomato',lw=1.5,ls='--',label='LDS open-loop')
    ax.set_xlabel('Time (ms)'); ax.set_ylabel(f'Z {label}'); ax.set_title(f'LDS reconstruction {label}'); ax.legend(fontsize=8)
theta = np.linspace(0,2*np.pi,300)
axes[0,2].plot(np.cos(theta),np.sin(theta),'k--',lw=0.8,alpha=0.4)
axes[0,2].scatter(eigs.real,eigs.imag,c=np.abs(eigs),cmap='RdYlGn',s=80,zorder=3,vmin=0,vmax=1.2)
axes[0,2].set_xlim(-1.5,1.5); axes[0,2].set_ylim(-1.5,1.5); axes[0,2].set_aspect('equal')
axes[0,2].set_xlabel('Re'); axes[0,2].set_ylabel('Im'); axes[0,2].set_title(f'A eigenvalues (R²={r2_lds:.3f})')
resid = Z[:,1:]-Z_onestep
axes[1,0].plot(t_ms[1:],np.sqrt((resid**2).sum(axis=0)),'purple',lw=1.5)
axes[1,0].set_xlabel('Time (ms)'); axes[1,0].set_ylabel('Prediction error (L2)'); axes[1,0].set_title(f'One-step error (R²={r2_onestep:.3f})')
cos_lds = np.array([np.dot(Z[:,t],Z_pred[:,t])/(np.linalg.norm(Z[:,t])*np.linalg.norm(Z_pred[:,t])+1e-12) for t in range(n_t)])
axes[1,1].plot(t_ms,cos_lds,'darkgreen',lw=2); axes[1,1].set_xlabel('Time (ms)'); axes[1,1].set_ylabel('Cosine similarity')
axes[1,1].set_title('LDS vs actual: direction alignment')
axes[1,2].plot(np.arange(1,21),cumvar[:20]*100,'o-',color='steelblue',ms=5)
axes[1,2].axhline(80,color='k',ls='--',lw=0.8); axes[1,2].axvline(K_LDS,color='tomato',lw=1,ls='--',label=f'K={K_LDS}')
axes[1,2].set_xlabel('K'); axes[1,2].set_ylabel('Cumulative variance (%)'); axes[1,2].set_title(f'Trajectory dimensionality (PR={PR:.1f})'); axes[1,2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/fig_LDS_JianJian.png', dpi=130, bbox_inches='tight')
print("All figures saved.")
