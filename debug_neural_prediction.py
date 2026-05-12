"""
Debug neural prediction step by step
"""
import sys
import numpy as np
from os.path import join
import h5py

# Import our utilities
from NSD_utils.h5_dataset_utils import load_data_from_GoodUnitStrc
from neural_prediction_utils import (
    transform_features2Xdict,
    sweep_regressors,
    RidgeCV, MultiTaskLassoCV
)

def test_small_nsd_data():
    """Test with a small subset of NSD data"""
    print("=== Testing with Small NSD Data ===")
    
    # Load data
    NSD_root = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/NSD_N3"
    data_path = join(NSD_root, "GoodUnit_240709_JianJian_NSD1000_LOC_g2.mat")
    
    print(f"Loading data from {data_path}")
    h5_file = h5py.File(data_path, "r")
    neural_data = load_data_from_GoodUnitStrc(h5_file)
    
    # Get a small subset
    Raster = neural_data['Raster']  # (675, 450, 4308)
    trial_valid_idx = neural_data['trial_valid_idx'][:, 0]
    dataset_valid_idx = neural_data['dataset_valid_idx'][:, 0]
    
    # Take only first 100 images and 50 units for testing
    n_units_test = 50
    n_images_test = 100
    
    # Compute responses for subset
    evk_slice = slice(100, 250)
    bsl_slice = slice(0, 90)
    
    evk_resp = Raster[:n_units_test, evk_slice, :].mean(axis=1) * 1000
    bsl_resp = Raster[:n_units_test, bsl_slice, :].mean(axis=1) * 1000
    
    # Get valid trials
    n_trials_raster = Raster.shape[2]
    valid_trials = dataset_valid_idx[:n_trials_raster].astype(bool)
    
    evk_resp_valid = evk_resp[:, valid_trials]
    bsl_resp_valid = bsl_resp[:, valid_trials]
    resp_corrected = evk_resp_valid - bsl_resp_valid
    
    valid_stim_idx = trial_valid_idx[dataset_valid_idx.astype(bool)][:valid_trials.sum()]
    
    # Average per image for small subset
    unique_images = np.unique(valid_stim_idx)[:n_images_test]
    resp_matrix = np.zeros((len(unique_images), n_units_test))
    
    for i, img_idx in enumerate(unique_images):
        trial_mask = valid_stim_idx == img_idx
        if trial_mask.sum() > 0:
            resp_matrix[i, :] = resp_corrected[:, trial_mask].mean(axis=1)
    
    h5_file.close()
    
    print(f"Created response matrix: {resp_matrix.shape}")
    print(f"Response range: {resp_matrix.min():.2f} to {resp_matrix.max():.2f}")
    
    # Create dummy features
    np.random.seed(42)
    features = np.random.randn(len(unique_images), 512)  # Smaller features
    
    print(f"Created features: {features.shape}")
    
    # Test feature transformation
    print("\nTesting feature transformation...")
    feat_dict = {'layer1': features}
    
    Xdict, tfm_dict = transform_features2Xdict(
        feat_dict,
        dimred_list=['pca100', 'sp_avg'],
        train_split_idx=None
    )
    
    print("Feature transformation successful!")
    for key, val in Xdict.items():
        print(f"  {key}: {val.shape}")
    
    # Test regression
    print("\nTesting regression...")
    alpha_list = [1e-3, 1e-2, 1e-1, 1, 10, 100]
    regressors = [RidgeCV(alphas=alpha_list, alpha_per_target=True)]
    regressor_names = ['RidgeCV']
    
    result_df, models = sweep_regressors(
        Xdict, resp_matrix, regressors, regressor_names,
        verbose=True, train_split_idx=None
    )
    
    print("\nRegression Results:")
    print(result_df)
    
    print("\n=== Test Successful! ===")
    return result_df, models

if __name__ == "__main__":
    try:
        result_df, models = test_small_nsd_data()
        print("All tests passed!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()