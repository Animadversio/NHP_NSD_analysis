"""
Time-Resolved Neural Encoding Dynamics
Atlas agent — Apr 7, 2026

Tracks how ResNet50 layer → neural response regression accuracy changes
over the 450ms PSTH window.
"""
import sys, os, glob, pickle
sys.path.insert(0, '/n/home12/binxuwang/Github/NHP_NSD_analysis')
sys.path.insert(0, '/n/home12/binxuwang/Github/Closed-loop-visual-insilico')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import h5py, torch
from os.path import join, exists
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from NSD_utils.h5_dataset_utils import load_data_from_GoodUnitStrc
from neural_regress.regress_lib import sweep_regressors

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_ROOT = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/NSD_N3'
IMG_DIR   = join(DATA_ROOT, 'NSD1000_LOC')
SESSION_FILE = join(DATA_ROOT, 'GoodUnit_240629_JianJian_NSD1000_LOC_g2.mat')
OUT_DIR   = '/n/home12/binxuwang/Github/NHP_NSD_analysis/notebooks/figures'
CACHE_DIR = '/n/home12/binxuwang/Github/NHP_NSD_analysis/notebooks/cache'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

LAYERS = ['relu', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
LAYER_LABELS = ['stem/relu', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
LAYER_COLORS = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3', '#a65628']

TIME_STRIDE = 5   # ms — sample every 5ms
SMOOTH_SIGMA = 10  # ms Gaussian smoothing

# ─── STEP 1: Extract ResNet50 features ────────────────────────────────────────
feat_cache = join(CACHE_DIR, 'resnet50_nsd_features.pkl')
if exists(feat_cache):
    print("\n[Step 1] Loading cached ResNet50 features...")
    with open(feat_cache, 'rb') as f:
        feat_dict_sp_avg = pickle.load(f)
    print("  Loaded.")
else:
    print("\n[Step 1] Extracting ResNet50 features for 1072 NSD images...")
    import torchvision.models as models
    import torchvision.transforms as T
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(DEVICE).eval()
    transform = T.Compose([
        T.Resize(256), T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    img_files = sorted(glob.glob(join(IMG_DIR, '*.bmp')))
    assert len(img_files) == 1072, f"Expected 1072 images, got {len(img_files)}"

    # Register plain PyTorch hooks on named modules
    activations = {l: [] for l in LAYERS}
    hooks = []
    named_mods = dict(model.named_modules())
    for layer in LAYERS:
        assert layer in named_mods, f"Layer {layer} not found in model. Available: {list(named_mods.keys())[:20]}"
        def make_hook(name):
            def hook(module, input, output):
                activations[name].append(output.detach().cpu())
            return hook
        hooks.append(named_mods[layer].register_forward_hook(make_hook(layer)))

    # Run in batches
    batch_size = 64
    with torch.no_grad():
        for i in tqdm(range(0, len(img_files), batch_size), desc='Extracting features'):
            batch_paths = img_files[i:i+batch_size]
            imgs = torch.stack([transform(Image.open(p).convert('RGB')) for p in batch_paths]).to(DEVICE)
            model(imgs)

    for h in hooks:
        h.remove()

    # Concatenate
    feat_dict_raw = {l: torch.cat(activations[l], dim=0) for l in LAYERS}
    print("  Raw shapes:", {l: v.shape for l, v in feat_dict_raw.items()})

    # Apply sp_avg (spatial average pooling for conv layers, flatten for linear)
    feat_dict_sp_avg = {}
    for layer in LAYERS:
        t = feat_dict_raw[layer]
        if t.dim() == 4:  # (N, C, H, W)
            feat_dict_sp_avg[layer] = t.mean(dim=(2,3)).numpy()  # (N, C)
        else:  # (N, C) already
            feat_dict_sp_avg[layer] = t.numpy()
    print("  sp_avg shapes:", {l: v.shape for l, v in feat_dict_sp_avg.items()})

    with open(feat_cache, 'wb') as f:
        pickle.dump(feat_dict_sp_avg, f)
    print(f"  Saved to {feat_cache}")

# ─── STEP 2: Load neural data ─────────────────────────────────────────────────
print("\n[Step 2] Loading neural data...")
f = h5py.File(SESSION_FILE, 'r')
d = load_data_from_GoodUnitStrc(f)
f.close()

psth = d['PsthRange']           # (450,) ms
R = d['response_matrix_img']    # (456 units, 450 time, 1072 images)
n_units, n_time, n_images = R.shape
dt = float(psth[1] - psth[0])  # ms per bin
print(f"  {n_units} units, {n_time} timepoints ({psth[0]:.0f}–{psth[1]:.0f} to {psth[-1]:.0f}ms), {n_images} images")
print(f"  dt = {dt:.1f}ms, smooth sigma = {SMOOTH_SIGMA}ms ({SMOOTH_SIGMA/dt:.1f} bins)")

# Gaussian smooth along time axis
sigma_bins = SMOOTH_SIGMA / dt
R_smooth = gaussian_filter1d(R.astype(np.float32), sigma=sigma_bins, axis=1)
print(f"  Smoothed response matrix: {R_smooth.shape}")

# Convert to firing rate (sp/s)
R_smooth_fr = R_smooth / dt * 1000  # sp/s

# ─── STEP 3: Train/test split (fixed across time) ────────────────────────────
np.random.seed(42)
idx_train, idx_test = train_test_split(np.arange(n_images), test_size=0.2, random_state=42)
print(f"\n[Step 3] Train/test split: {len(idx_train)} train, {len(idx_test)} test images")

# ─── STEP 4: Time-resolved regression ─────────────────────────────────────────
print("\n[Step 4] Running time-resolved regression...")

# Select time points
t_indices = np.arange(0, n_time, TIME_STRIDE)
t_ms = psth[t_indices]
print(f"  {len(t_indices)} time points at stride {TIME_STRIDE}ms")

# Prepare regression — RidgeCV with fixed alpha list
ALPHAS = [1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5]

# Result storage: (layer, time) → r2_test
r2_test_mat  = np.zeros((len(LAYERS), len(t_indices)))
r2_train_mat = np.zeros((len(LAYERS), len(t_indices)))

# Prepare X matrices (fixed for all time steps)
X_dict = {layer: feat_dict_sp_avg[layer] for layer in LAYERS}  # (1072, n_features)

for li, layer in enumerate(LAYERS):
    X = X_dict[layer]  # (1072, n_feat)
    X_train = X[idx_train]
    X_test  = X[idx_test]
    print(f"\n  Layer {layer} (feat dim={X.shape[1]}):")
    for ti, t_idx in enumerate(tqdm(t_indices, desc=f'  {layer}', leave=False)):
        # Neural response at this time point: (1072, n_units)
        y = R_smooth_fr[:, t_idx, :].T  # (n_images, n_units)
        y_train = y[idx_train]
        y_test  = y[idx_test]
        # Fit RidgeCV
        clf = RidgeCV(alphas=ALPHAS, alpha_per_target=True)
        clf.fit(X_train, y_train)
        r2_test_mat[li, ti]  = clf.score(X_test, y_test)
        r2_train_mat[li, ti] = clf.score(X_train, y_train)

print("\n  Regression complete.")
print(f"  r2_test_mat shape: {r2_test_mat.shape}")

# Save results
results = {
    't_ms': t_ms, 'psth': psth,
    'r2_test': r2_test_mat, 'r2_train': r2_train_mat,
    'layers': LAYERS, 'layer_labels': LAYER_LABELS,
}
with open(join(CACHE_DIR, 'time_resolved_regression_results.pkl'), 'wb') as f:
    pickle.dump(results, f)
print(f"  Results saved to cache.")

# ─── STEP 5: Visualization ────────────────────────────────────────────────────
print("\n[Step 5] Generating figures...")

# --- Fig A: R²(t) per layer ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Time-Resolved ResNet50 Layer Regression — JianJian 240629', fontsize=13, fontweight='bold')

ax = axes[0]
for li, (layer, label, color) in enumerate(zip(LAYERS, LAYER_LABELS, LAYER_COLORS)):
    ax.plot(t_ms, r2_test_mat[li], color=color, lw=2, label=label)
ax.axvline(0, color='k', lw=1, ls='--', alpha=0.5)
ax.axvspan(100, 250, alpha=0.1, color='orange', label='evoked window')
ax.set_xlabel('Time from stimulus onset (ms)')
ax.set_ylabel('Test R²')
ax.set_title('Prediction accuracy (R²) per layer over time')
ax.legend(fontsize=8, loc='upper left')
ax.set_xlim(t_ms[0], t_ms[-1])
ax.set_ylim(bottom=0)

# --- Fig B: Optimal layer index over time ---
ax = axes[1]
optimal_layer_idx = np.argmax(r2_test_mat, axis=0)  # (n_times,)
optimal_r2 = r2_test_mat[optimal_layer_idx, np.arange(len(t_indices))]

# Plot as colored scatter (color = which layer)
for li, (label, color) in enumerate(zip(LAYER_LABELS, LAYER_COLORS)):
    mask = optimal_layer_idx == li
    if mask.any():
        ax.scatter(t_ms[mask], np.ones(mask.sum()) * li, color=color,
                   s=optimal_r2[mask]*500, alpha=0.8, label=label)
ax.plot(t_ms, optimal_layer_idx, 'k-', alpha=0.4, lw=1)
ax.set_yticks(range(len(LAYERS)))
ax.set_yticklabels(LAYER_LABELS)
ax.set_xlabel('Time from stimulus onset (ms)')
ax.set_title('Optimal Layer Index over Time\n(dot size = R²)')
ax.axvline(0, color='k', lw=1, ls='--', alpha=0.5)
ax.set_xlim(t_ms[0], t_ms[-1])
ax.legend(fontsize=7, loc='upper left')

plt.tight_layout()
figA_path = join(OUT_DIR, 'figA_time_resolved_R2_curves.png')
plt.savefig(figA_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved {figA_path}")

# --- Fig B2: Heatmap of R²(layer × time) ---
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle('Time-Resolved Regression — Layer × Time R²', fontsize=13, fontweight='bold')

ax = axes[0]
im = ax.imshow(r2_test_mat, aspect='auto', cmap='hot',
               extent=[t_ms[0], t_ms[-1], len(LAYERS)-0.5, -0.5],
               vmin=0, vmax=r2_test_mat.max())
ax.set_yticks(range(len(LAYERS))); ax.set_yticklabels(LAYER_LABELS)
ax.set_xlabel('Time (ms)'); ax.set_title('Test R² heatmap (layer × time)')
ax.axvline(0, color='cyan', lw=1, ls='--')
plt.colorbar(im, ax=ax, label='Test R²')

# R² curves with fill
ax = axes[1]
for li, (label, color) in enumerate(zip(LAYER_LABELS, LAYER_COLORS)):
    ax.plot(t_ms, r2_test_mat[li], color=color, lw=2, label=label)
    ax.fill_between(t_ms, 0, r2_test_mat[li], alpha=0.1, color=color)
ax.axvline(0, color='k', lw=1, ls='--', alpha=0.5)
ax.axvspan(100, 250, alpha=0.1, color='orange')
ax.set_xlabel('Time (ms)'); ax.set_ylabel('Test R²')
ax.set_title('Layer R² trajectories')
ax.legend(fontsize=8); ax.set_xlim(t_ms[0], t_ms[-1]); ax.set_ylim(bottom=0)

plt.tight_layout()
figB_path = join(OUT_DIR, 'figB_time_resolved_heatmap.png')
plt.savefig(figB_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved {figB_path}")

# --- Fig C: Summary stats per layer ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Layer Regression Summary', fontsize=13, fontweight='bold')

# Peak R² per layer
peak_r2 = r2_test_mat.max(axis=1)
ax = axes[0]
bars = ax.bar(LAYER_LABELS, peak_r2, color=LAYER_COLORS)
ax.set_title('Peak Test R² per Layer')
ax.set_ylabel('Peak R²')
ax.tick_params(axis='x', rotation=20)
for bar, v in zip(bars, peak_r2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002, f'{v:.3f}', ha='center', fontsize=8)

# Latency of peak R² per layer
peak_latency = t_ms[np.argmax(r2_test_mat, axis=1)]
ax = axes[1]
ax.bar(LAYER_LABELS, peak_latency, color=LAYER_COLORS)
ax.set_title('Latency of Peak R² (ms)')
ax.set_ylabel('Time (ms)')
ax.tick_params(axis='x', rotation=20)
ax.axhline(0, color='k', lw=1, ls='--', alpha=0.5)

# Fraction of time each layer is optimal
layer_dominance = np.zeros(len(LAYERS))
for li in range(len(LAYERS)):
    layer_dominance[li] = (optimal_layer_idx == li).mean()
ax = axes[2]
ax.bar(LAYER_LABELS, layer_dominance, color=LAYER_COLORS)
ax.set_title('Fraction of Time as Optimal Layer')
ax.set_ylabel('Fraction')
ax.tick_params(axis='x', rotation=20)

plt.tight_layout()
figC_path = join(OUT_DIR, 'figC_layer_summary.png')
plt.savefig(figC_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved {figC_path}")

# ─── Print summary stats ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
for li, label in enumerate(LAYER_LABELS):
    peak = r2_test_mat[li].max()
    lat  = t_ms[np.argmax(r2_test_mat[li])]
    evk_r2 = r2_test_mat[li][(t_ms>=100)&(t_ms<=250)].mean()
    print(f"  {label:12s}: peak R²={peak:.4f} at {lat:.0f}ms | evoked-window R²={evk_r2:.4f}")

print(f"\n  Overall peak: layer={LAYER_LABELS[np.unravel_index(r2_test_mat.argmax(), r2_test_mat.shape)[0]]}, "
      f"t={t_ms[np.unravel_index(r2_test_mat.argmax(), r2_test_mat.shape)[1]]:.0f}ms, "
      f"R²={r2_test_mat.max():.4f}")

print(f"\nAll figures saved to {OUT_DIR}")
print("Done!")
