"""Run per-unit time-resolved regression for all monkey sessions."""
import sys, os
sys.path.insert(0, '/n/home12/binxuwang/Github/NHP_NSD_analysis')
import numpy as np, pickle as pkl, h5py, glob
from os.path import join, basename
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from tqdm import tqdm
from NSD_utils.h5_dataset_utils import load_data_from_GoodUnitStrc

DATA_ROOT  = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/NSD_N3'
CACHE_DIR  = '/n/home12/binxuwang/Github/NHP_NSD_analysis/notebooks/cache'
FEAT_CACHE = join(CACHE_DIR, 'resnet50_nsd_features.pkl')
TIME_STRIDE = 5; N_PCA = 200; ALPHAS = np.logspace(-2, 6, 25)

# One representative session per monkey (first by date)
SESSIONS = {
    'FaCai':       'GoodUnit_240711_FaCai_NSD1000_LOC_g4.mat',
    'TuTu':        'GoodUnit_240724_TuTu_NSD1000_LOC_g2.mat',
    'ZhuangZhuang':'GoodUnit_240817_ZhuangZhuang_NSD1000_LOC_g6.mat',
    'MaoDan':      'GoodUnit_240815_MaoDan_NSD1000_LOC_g5.mat',
}

# Load cached features
with open(FEAT_CACHE, 'rb') as f: feat_dict = pkl.load(f)
layer_names = list(feat_dict.keys())
print("Layers:", layer_names)

def run_session(monkey, fname):
    out_path = join(CACHE_DIR, f'time_resolved_perunit_{monkey}.pkl')
    if os.path.exists(out_path):
        print(f"[{monkey}] Already cached, skipping.")
        return

    print(f"\n{'='*60}\n[{monkey}] Loading {fname}")
    f = h5py.File(join(DATA_ROOT, fname), 'r')
    d = load_data_from_GoodUnitStrc(f)
    R = d['response_matrix_img']        # (n_units, 450, n_images) — shape varies
    t_ms_full = d['PsthRange']
    n_units, n_time_full, n_images = R.shape
    print(f"  {n_units} units, {n_images} images, time {t_ms_full[0]:.0f}-{t_ms_full[-1]:.0f}ms")
    f.close()

    # Transpose to (n_images, n_units) per time — use only images that have features
    n_feat_images = feat_dict[layer_names[0]].shape[0]
    assert n_images == n_feat_images, f"Image count mismatch: {n_images} vs {n_feat_images}"
    # R is (n_units, n_time, n_images); transpose to (n_images, n_units) per time
    # Use same train/test split
    rng = np.random.RandomState(42)
    idx = np.arange(n_images)
    train_n = int(0.8 * n_images)
    train_idx = rng.choice(idx, train_n, replace=False)
    test_idx  = np.setdiff1d(idx, train_idx)

    # PCA per layer
    Xdict = {}
    for ln in layer_names:
        feat = feat_dict[ln]  # (n_images, d)
        pca = PCA(n_components=min(N_PCA, feat.shape[1]))
        Xtrain = pca.fit_transform(feat[train_idx])
        Xtest  = pca.transform(feat[test_idx])
        Xdict[ln] = (Xtrain, Xtest)
        print(f"  {ln}: {feat.shape[1]} -> {Xtrain.shape[1]} PCA dims")

    # Time points
    t_indices = np.where(
        (t_ms_full >= -49) & (np.arange(len(t_ms_full)) % TIME_STRIDE == 0)
    )[0]
    t_ms = t_ms_full[t_indices]

    r2_perunit       = np.zeros((len(layer_names), len(t_indices), n_units), dtype=np.float32)
    r2_perunit_train = np.zeros_like(r2_perunit)

    for li, ln in enumerate(layer_names):
        Xtrain, Xtest = Xdict[ln]
        print(f"  Layer {ln}:", end='', flush=True)
        for ti, tidx in enumerate(tqdm(t_indices, desc=f'  {ln}', leave=False)):
            y = R[:, tidx, :].T  # (n_images, n_units)
            ytrain = y[train_idx]; ytest = y[test_idx]
            clf = RidgeCV(alphas=ALPHAS, alpha_per_target=True)
            clf.fit(Xtrain, ytrain)
            yhat_test  = clf.predict(Xtest)
            yhat_train = clf.predict(Xtrain)
            # Safe R²: set to NaN when variance is near-zero (unit has no signal)
            MIN_VAR = 1e-6
            ss_res   = ((ytest  - yhat_test )**2).sum(axis=0)
            ss_tot   = ((ytest  - ytest.mean(axis=0))**2).sum(axis=0)
            ss_res_tr= ((ytrain - yhat_train)**2).sum(axis=0)
            ss_tot_tr= ((ytrain - ytrain.mean(axis=0))**2).sum(axis=0)
            r2  = np.where(ss_tot   > MIN_VAR, 1 - ss_res   / ss_tot,   np.nan)
            r2t = np.where(ss_tot_tr > MIN_VAR, 1 - ss_res_tr / ss_tot_tr, np.nan)
            r2_perunit[li, ti]       = np.clip(r2,  -1, 1).astype(np.float32)
            r2_perunit_train[li, ti] = np.clip(r2t, -1, 1).astype(np.float32)
        print()

    result = dict(r2_perunit=r2_perunit, r2_perunit_train=r2_perunit_train,
                  t_ms=t_ms, layers=layer_names, n_units=n_units, monkey=monkey,
                  psth=R[:, :, :].mean(axis=(0,2)), spikepos=d.get('spikepos', None))
    with open(out_path, 'wb') as fh: pkl.dump(result, fh)
    print(f"  Saved to {out_path}")

    # Print quick summary
    peak_r2 = r2_perunit.max(axis=(0,1))
    print(f"  Peak R²: mean={peak_r2.mean():.3f}, top10%={np.percentile(peak_r2,90):.3f}")

for monkey, fname in SESSIONS.items():
    run_session(monkey, fname)

print("\nAll sessions done!")
