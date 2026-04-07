"""DINOv2 per-unit time-resolved regression for all 5 monkey sessions."""
import sys, os
sys.path.insert(0, '/n/home12/binxuwang/Github/NHP_NSD_analysis')
import numpy as np, pickle as pkl, h5py, torch
from os.path import join
from tqdm import tqdm
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from NSD_utils.h5_dataset_utils import load_data_from_GoodUnitStrc

CACHE_DIR = '/n/home12/binxuwang/Github/NHP_NSD_analysis/notebooks/cache'
DATA_ROOT = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/NSD_N3'
FEAT_CACHE = join(CACHE_DIR, 'dinov2_nsd_features.pkl')

SESSIONS = {
    'JianJian':    'GoodUnit_240629_JianJian_NSD1000_LOC_g2.mat',
    'FaCai':       'GoodUnit_240711_FaCai_NSD1000_LOC_g4.mat',
    'TuTu':        'GoodUnit_240724_TuTu_NSD1000_LOC_g2.mat',
    'ZhuangZhuang':'GoodUnit_240817_ZhuangZhuang_NSD1000_LOC_g6.mat',
    'MaoDan':      'GoodUnit_240815_MaoDan_NSD1000_LOC_g5.mat',
}
N_BLOCKS = 12; N_PCA = 200; ALPHAS = np.logspace(-2, 6, 25)
TIME_STRIDE = 5; MIN_VAR = 1e-6

print("Loading DINOv2 features...")
with open(FEAT_CACHE, 'rb') as f: feat_dict = pkl.load(f)
feature_types = ['cls', 'patch']
layer_names = {ft: [f'blocks.{i}_{ft}' for i in range(N_BLOCKS)] for ft in feature_types}
n_images = feat_dict['blocks.0_cls'].shape[0]
print(f"  {n_images} images, {N_BLOCKS} blocks × {len(feature_types)} feature types")

# Fixed train/test split (same as ResNet50 analysis)
rng = np.random.RandomState(42)
train_idx = rng.choice(n_images, int(0.8*n_images), replace=False)
test_idx  = np.setdiff1d(np.arange(n_images), train_idx)

# PCA per layer (fit once on train, apply to all)
print("Fitting PCA per layer...")
Xdict = {}
for ft in feature_types:
    for ln in layer_names[ft]:
        feat = feat_dict[ln]
        pca = PCA(n_components=min(N_PCA, feat.shape[1]))
        Xtr = pca.fit_transform(feat[train_idx])
        Xte = pca.transform(feat[test_idx])
        Xdict[ln] = (Xtr, Xte)
print(f"  PCA done for {len(Xdict)} combos")

def run_session(monkey, fname):
    out_path = join(CACHE_DIR, f'time_resolved_perunit_dinov2_{monkey}.pkl')
    if os.path.exists(out_path):
        print(f"[{monkey}] Already cached.")
        return

    print(f"\n{'='*60}\n[{monkey}] Loading {fname}")
    fh = h5py.File(join(DATA_ROOT, fname), 'r')
    d  = load_data_from_GoodUnitStrc(fh)
    R  = d['response_matrix_img']   # (n_units, n_time, n_images)
    t_full = d['PsthRange']
    n_units = R.shape[0]; fh.close()
    print(f"  {n_units} units")

    t_indices = np.where(
        (t_full >= -49) & (np.arange(len(t_full)) % TIME_STRIDE == 0)
    )[0]
    t_ms = t_full[t_indices]; n_t = len(t_indices)

    r2_all = {ft: np.full((N_BLOCKS, n_t, n_units), np.nan, dtype=np.float32)
              for ft in feature_types}

    for ft in feature_types:
        for bi in range(N_BLOCKS):
            ln = f'blocks.{bi}_{ft}'
            Xtrain, Xtest = Xdict[ln]
            for ti, tidx in enumerate(tqdm(t_indices, desc=f'{ft} B{bi}', leave=False)):
                y = R[:, tidx, :].T
                ytrain = y[train_idx]; ytest = y[test_idx]
                clf = RidgeCV(alphas=ALPHAS, alpha_per_target=True)
                clf.fit(Xtrain, ytrain)
                yhat = clf.predict(Xtest)
                ss_res = ((ytest - yhat)**2).sum(axis=0)
                ss_tot = ((ytest - ytest.mean(axis=0))**2).sum(axis=0)
                r2 = np.where(ss_tot > MIN_VAR, 1 - ss_res/ss_tot, np.nan)
                r2_all[ft][bi, ti] = np.clip(r2, -1, 1).astype(np.float32)
            pk = np.nanmean(np.nanmax(r2_all[ft][bi], axis=0))
            print(f"  {ft} B{bi:2d}: mean peak R²={pk:.3f}")

    result = dict(r2_all=r2_all, t_ms=t_ms, n_units=n_units, monkey=monkey,
                  feature_types=feature_types)
    with open(out_path, 'wb') as f: pkl.dump(result, f)
    print(f"  Saved → {out_path}")

for monkey, fname in SESSIONS.items():
    run_session(monkey, fname)
print("\nAll sessions done!")
