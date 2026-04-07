"""
Time-resolved regression using DINOv2 ViT-B/14-reg (12 transformer blocks).
Features: CLS token, avg patch token, and PCA of all tokens per block.
Single session: JianJian 240629.
"""
import sys, os
sys.path.insert(0, '/n/home12/binxuwang/Github/NHP_NSD_analysis')
sys.path.insert(0, '/n/home12/binxuwang/Github/Closed-loop-visual-insilico')
import numpy as np, pickle as pkl, h5py, torch
from os.path import join
from tqdm import tqdm
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from NSD_utils.h5_dataset_utils import load_data_from_GoodUnitStrc

CACHE_DIR  = '/n/home12/binxuwang/Github/NHP_NSD_analysis/notebooks/cache'
FIGDIR     = '/n/home12/binxuwang/Github/NHP_NSD_analysis/notebooks/figures'
DATA_ROOT  = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/NSD_N3'
SESSION    = 'GoodUnit_240629_JianJian_NSD1000_LOC_g2.mat'
IMG_PKL    = join(CACHE_DIR, 'nsd_image_paths.pkl')
FEAT_CACHE = join(CACHE_DIR, 'dinov2_nsd_features.pkl')
REGR_CACHE = join(CACHE_DIR, 'time_resolved_perunit_dinov2_JianJian.pkl')
os.makedirs(FIGDIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

N_BLOCKS  = 12
N_PCA     = 200
ALPHAS    = np.logspace(-2, 6, 25)
TIME_STRIDE = 5
MIN_VAR   = 1e-6

# ── STEP 1: Load DINOv2 and extract features ─────────────────────────────────
if os.path.exists(FEAT_CACHE):
    print("Loading cached DINOv2 features...")
    with open(FEAT_CACHE, 'rb') as f: feat_dict = pkl.load(f)
    print({k: v.shape for k, v in feat_dict.items()})
else:
    print("Extracting DINOv2 features...")
    from core.model_load_utils import load_model_transform
    model, tfm = load_model_transform('dinov2_vitb14_reg', device=DEVICE)
    model.eval()

    # Get image paths from the NSD stimuli used in this dataset
    # Use the ResNet cache to find n_images, then load images from NSD path
    with open(join(CACHE_DIR, 'resnet50_nsd_features.pkl'), 'rb') as f:
        resnet_feats = pkl.load(f)
    n_images = resnet_feats['relu'].shape[0]
    print(f"n_images: {n_images}")

    # Load NSD image paths
    nsd_stim_dir = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/NSD_N3/NSD1000_LOC'
    if not os.path.exists(nsd_stim_dir):
        # Try alternate path
        nsd_stim_dir = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/NSD'
    print(f"Looking for stimuli in: {nsd_stim_dir}")
    img_files = sorted([f for f in os.listdir(nsd_stim_dir) if f.endswith('.bmp')])[:n_images]
    img_paths = [join(nsd_stim_dir, f) for f in img_files]
    print(f"Found {len(img_paths)} images")

    # Hook all 12 blocks
    hooks = {f'blocks.{i}': [] for i in range(N_BLOCKS)}
    handles = []
    def make_hook(name):
        def hook(mod, inp, out):
            # out: (B, n_tokens, 768) — save to cpu immediately
            hooks[name].append(out.detach().cpu())
        return hook
    for i in range(N_BLOCKS):
        handles.append(model.blocks[i].register_forward_hook(make_hook(f'blocks.{i}')))

    BATCH = 32
    for start in tqdm(range(0, len(img_paths), BATCH), desc='Extracting DINOv2 features'):
        batch_imgs = []
        for p in img_paths[start:start+BATCH]:
            img = Image.open(p).convert('RGB')
            batch_imgs.append(tfm(img))
        batch_tensor = torch.stack(batch_imgs).to(DEVICE)
        with torch.no_grad():
            model(batch_tensor)

    for h in handles: h.remove()

    # Build feat_dict: 3 feature types × 12 blocks
    num_register = getattr(model, 'num_register_tokens', 0)
    print(f"Register tokens: {num_register}")
    feat_dict = {}
    for i in range(N_BLOCKS):
        block_feats = torch.cat(hooks[f'blocks.{i}'], dim=0).numpy()  # (N, n_tok, 768)
        cls = block_feats[:, 0, :]                                      # (N, 768)
        # Skip CLS + register tokens for patch avg
        patch = block_feats[:, 1 + num_register:, :].mean(axis=1)     # (N, 768)
        feat_dict[f'blocks.{i}_cls']   = cls
        feat_dict[f'blocks.{i}_patch'] = patch

    with open(FEAT_CACHE, 'wb') as f: pkl.dump(feat_dict, f)
    print(f"Saved DINOv2 features: {list(feat_dict.keys())[:4]}...")
    print({k: v.shape for k, v in list(feat_dict.items())[:4]})

# ── STEP 2: Load neural data ─────────────────────────────────────────────────
print(f"\nLoading neural data: {SESSION}")
fh = h5py.File(join(DATA_ROOT, SESSION), 'r')
d  = load_data_from_GoodUnitStrc(fh)
R  = d['response_matrix_img']         # (n_units, n_time, n_images)
t_full = d['PsthRange']
n_units, n_time_full, n_images = R.shape
fh.close()
print(f"  {n_units} units, {n_images} images, {n_time_full} time pts")
print(f"  Time: {t_full[0]:.0f} to {t_full[-1]:.0f} ms")

# Train/test split
rng = np.random.RandomState(42)
train_idx = rng.choice(n_images, int(0.8*n_images), replace=False)
test_idx  = np.setdiff1d(np.arange(n_images), train_idx)
print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

# ── STEP 3: PCA per feature type ─────────────────────────────────────────────
# Run both CLS and patch tokens; we'll compare them
feature_types = ['cls', 'patch']
layer_names = {ft: [f'blocks.{i}_{ft}' for i in range(N_BLOCKS)] for ft in feature_types}
Xdict = {}
for ft in feature_types:
    for ln in layer_names[ft]:
        feat = feat_dict[ln]   # (n_images, 768)
        pca  = PCA(n_components=min(N_PCA, feat.shape[1]))
        Xtrain = pca.fit_transform(feat[train_idx])
        Xtest  = pca.transform(feat[test_idx])
        Xdict[ln] = (Xtrain, Xtest)
print(f"PCA done for {len(Xdict)} layer×feature combos")

# ── STEP 4: Time-resolved regression ─────────────────────────────────────────
if os.path.exists(REGR_CACHE):
    print("Loading cached regression results...")
    with open(REGR_CACHE, 'rb') as f: regr_res = pkl.load(f)
else:
    t_indices = np.where(
        (t_full >= -49) & (np.arange(len(t_full)) % TIME_STRIDE == 0)
    )[0]
    t_ms = t_full[t_indices]
    n_t  = len(t_indices)

    # Store R² for each feature_type × block × time × unit
    r2_all = {ft: np.full((N_BLOCKS, n_t, n_units), np.nan, dtype=np.float32)
              for ft in feature_types}

    for ft in feature_types:
        for bi in range(N_BLOCKS):
            ln = f'blocks.{bi}_{ft}'
            Xtrain, Xtest = Xdict[ln]
            for ti, tidx in enumerate(tqdm(t_indices, desc=f'{ft} block{bi}', leave=False)):
                y = R[:, tidx, :].T          # (n_images, n_units)
                ytrain = y[train_idx]; ytest = y[test_idx]
                clf = RidgeCV(alphas=ALPHAS, alpha_per_target=True)
                clf.fit(Xtrain, ytrain)
                yhat = clf.predict(Xtest)
                ss_res = ((ytest - yhat)**2).sum(axis=0)
                ss_tot = ((ytest - ytest.mean(axis=0))**2).sum(axis=0)
                r2 = np.where(ss_tot > MIN_VAR, 1 - ss_res/ss_tot, np.nan)
                r2_all[ft][bi, ti] = np.clip(r2, -1, 1).astype(np.float32)
            peak = np.nanmax(r2_all[ft][bi], axis=(0,1)).mean()
            print(f"  {ft} block{bi:2d}: mean peak R²={peak:.3f}")

    regr_res = dict(r2_all=r2_all, t_ms=t_ms, n_units=n_units, feature_types=feature_types)
    with open(REGR_CACHE, 'wb') as f: pkl.dump(regr_res, f)
    print(f"Saved to {REGR_CACHE}")

r2_all  = regr_res['r2_all']
t_ms    = regr_res['t_ms']
# ── STEP 5: Compute weighted depth ───────────────────────────────────────────
MIN_SIG = 0.02
depth_idx = np.arange(N_BLOCKS, dtype=float)

def weighted_depth(R2, min_sig=MIN_SIG):
    """R2: (n_blocks, n_time, n_units) → (n_time, n_units) NaN-masked depth"""
    R2p  = np.clip(R2, 0, None)
    tot  = R2p.sum(axis=0)
    return np.where(tot >= min_sig,
                    (depth_idx[:,None,None]*R2p).sum(axis=0)/(tot+1e-12), np.nan)

peak_r2 = {ft: np.nanmax(r2_all[ft], axis=(0,1)) for ft in feature_types}

# ── STEP 6: Figures ───────────────────────────────────────────────────────────
BLOCK_CMAP = plt.cm.plasma
BLOCK_COLORS = [BLOCK_CMAP(i/(N_BLOCKS-1)) for i in range(N_BLOCKS)]
SMOOTH = lambda x: gaussian_filter1d(np.nan_to_num(x, nan=0.0), sigma=2)
early_t = (t_ms>=80)&(t_ms<=150)
late_t  = (t_ms>=180)&(t_ms<=350)

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('DINOv2 ViT-B/14-reg (12 blocks) — JianJian Time-Resolved Regression', fontsize=13, fontweight='bold')

for col, ft in enumerate(['cls', 'patch']):
    R2  = r2_all[ft]         # (12, 90, n_units)
    dc  = weighted_depth(R2) # (90, n_units)
    pk  = peak_r2[ft]
    
    # Panel: mean R² per block over time
    ax = axes[0, col]
    mean_r2 = np.nanmean(R2, axis=2)  # (12, 90)
    for bi in range(N_BLOCKS):
        ax.plot(t_ms, SMOOTH(mean_r2[bi]), color=BLOCK_COLORS[bi], 
                label=f'B{bi}', lw=1.5, alpha=0.85)
    ax.axvline(0, color='gray', ls='--', lw=0.8)
    ax.axhline(0, color='gray', lw=0.4)
    ax.set_title(f'{ft.upper()} token — mean R² per block', fontsize=10)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Mean R²')
    ax.set_xlim(t_ms[0], t_ms[-1])
    if col == 0: ax.legend(fontsize=5, ncol=3)

    # Panel: population weighted depth
    ax = axes[1, col]
    resp = pk >= 0.05
    pop_depth = np.nanmean(dc[:, resp], axis=1)
    pop_sem   = np.nanstd(dc[:, resp], axis=1) / np.sqrt(np.isfinite(dc[:, resp]).sum(axis=1).clip(1))
    ax.plot(t_ms, SMOOTH(pop_depth), 'k-', lw=2.5)
    ax.fill_between(t_ms, SMOOTH(pop_depth-pop_sem), SMOOTH(pop_depth+pop_sem), alpha=0.2, color='k')
    ax.axvline(0, color='gray', ls='--', lw=0.8)
    ax.axvspan(80,150,alpha=0.1,color='steelblue')
    ax.axvspan(180,350,alpha=0.1,color='tomato')
    ax.set_yticks(range(0,N_BLOCKS,2)); ax.set_yticklabels([f'B{i}' for i in range(0,N_BLOCKS,2)], fontsize=7)
    ed = np.nanmean(pop_depth[early_t]); ld = np.nanmean(pop_depth[late_t])
    ax.set_title(f'{ft.upper()} weighted depth | early={ed:.2f} late={ld:.2f} Δ={ld-ed:+.2f}', fontsize=9)
    ax.set_xlabel('Time (ms)'); ax.set_xlim(t_ms[0], t_ms[-1])
    if col == 0: ax.set_ylabel('Weighted block depth')

    # Panel: heatmap R² (blocks × time), averaged over units
    ax = axes[2, col]
    im = ax.imshow(mean_r2, aspect='auto', origin='lower', cmap='hot',
                   extent=[t_ms[0], t_ms[-1], -0.5, N_BLOCKS-0.5],
                   vmin=0, vmax=np.nanpercentile(mean_r2, 98))
    ax.axvline(0, color='cyan', lw=0.8, ls='--')
    plt.colorbar(im, ax=ax, label='Mean R²')
    ax.set_yticks(range(N_BLOCKS)); ax.set_yticklabels([f'B{i}' for i in range(N_BLOCKS)], fontsize=7)
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Block')
    ax.set_title(f'{ft.upper()} — R² heatmap (blocks × time)', fontsize=9)

# Panel: CLS vs patch comparison of peak R² by block
ax = axes[0, 2]
pk_cls   = np.nanmean(np.nanmax(r2_all['cls'],   axis=1), axis=1)  # (12,) mean over units
pk_patch = np.nanmean(np.nanmax(r2_all['patch'], axis=1), axis=1)
ax.plot(range(N_BLOCKS), pk_cls,   'o-', color='steelblue', lw=2, label='CLS token')
ax.plot(range(N_BLOCKS), pk_patch, 's-', color='tomato',    lw=2, label='Avg patch')
ax.set_xticks(range(N_BLOCKS)); ax.set_xticklabels([f'B{i}' for i in range(N_BLOCKS)], fontsize=7)
ax.set_xlabel('Block'); ax.set_ylabel('Mean peak R² (across units)')
ax.set_title('CLS vs Avg-patch: peak R² by block', fontsize=10)
ax.legend(fontsize=9)

# Panel: depth shift CLS vs patch
ax = axes[1, 2]
from scipy.stats import wilcoxon as wlcx
for ci, (ft, col_c) in enumerate([('cls','steelblue'), ('patch','tomato')]):
    dc_ft = weighted_depth(r2_all[ft])
    resp  = peak_r2[ft] >= 0.10
    ed    = np.nanmean(dc_ft[early_t][:,resp], axis=0)
    ld    = np.nanmean(dc_ft[late_t][:,resp],  axis=0)
    diff  = ld - ed
    diff  = diff[np.isfinite(diff)]
    n = len(diff)
    mn = diff.mean()
    if n >= 10:
        _, pv = wlcx(np.zeros(n), diff, alternative='greater')
    else: pv = np.nan
    ax.hist(diff, bins=30, alpha=0.6, color=col_c,
            label=f'{ft.upper()} mean={mn:+.2f} p={pv:.4f}')
ax.axvline(0, color='k', ls='--', lw=1.5)
ax.set_xlabel('Late − Early depth'); ax.set_ylabel('# units')
ax.set_title('Depth shift (R²≥0.10): CLS vs patch', fontsize=9)
ax.legend(fontsize=8)

# Panel: per-unit best block distribution at peak response window
ax = axes[2, 2]
for ci, (ft, col_c) in enumerate([('cls','steelblue'), ('patch','tomato')]):
    resp = peak_r2[ft] >= 0.05
    R2_win = r2_all[ft][:, early_t, :][:,:,resp].mean(axis=1)  # (12, n_resp)
    best_blocks = R2_win.argmax(axis=0)
    ax.hist(best_blocks, bins=np.arange(-0.5, N_BLOCKS+0.5), alpha=0.6, color=col_c,
            label=f'{ft.upper()} (n={resp.sum()})', density=True)
ax.set_xticks(range(N_BLOCKS)); ax.set_xticklabels([f'B{i}' for i in range(N_BLOCKS)], fontsize=7)
ax.set_xlabel('Best block (80-150ms window)'); ax.set_ylabel('Fraction of units')
ax.set_title('Preferred block at peak response', fontsize=9)
ax.legend(fontsize=8)

plt.tight_layout()
outpath = f'{FIGDIR}/fig_dinov2_time_resolved.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {outpath}")

# Print summary
print("\n=== DINOv2 SUMMARY ===")
for ft in feature_types:
    R2 = r2_all[ft]; pk = peak_r2[ft]
    dc = weighted_depth(R2)
    resp = pk >= 0.10
    ed = np.nanmean(dc[early_t][:,resp]); ld = np.nanmean(dc[late_t][:,resp])
    best_block_early = np.nanmean(r2_all[ft][:,early_t,:][:,:,resp].mean(axis=1).argmax(axis=0))
    print(f"\n{ft.upper()} token:")
    print(f"  Best block at peak: {best_block_early:.1f} / {N_BLOCKS-1}")
    print(f"  Peak R² (mean): {np.nanmean(pk):.3f}, top10%: {np.nanpercentile(pk,90):.3f}")
    print(f"  Early depth: {ed:.2f}, Late depth: {ld:.2f}, Δ={ld-ed:+.3f}")
    diff = np.nanmean(dc[late_t][:,resp],axis=0) - np.nanmean(dc[early_t][:,resp],axis=0)
    diff = diff[np.isfinite(diff)]
    _, pv = wlcx(np.zeros(len(diff)), diff, alternative='greater')
    print(f"  Deeper-late: {(diff>0).mean():.1%}, Wilcoxon p={pv:.4f}")

print("\nDone!")
