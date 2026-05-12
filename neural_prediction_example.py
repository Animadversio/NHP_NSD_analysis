"""
Neural Prediction Example

This script demonstrates how to use the neural prediction utilities 
with NSD neural data to predict neural responses from deep network features.

Usage:
    python neural_prediction_example.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import h5py

# Add the current directory to path to import our utilities
sys.path.append("/n/home12/binxuwang/Github/NHP_NSD_analysis")
from NSD_utils.h5_dataset_utils import load_data_from_GoodUnitStrc
from neural_prediction_utils import (
    neural_prediction_pipeline, 
    plot_prediction_results,
    load_model_transform
)

def load_nsd_neural_data(data_path):
    """
    Load and prepare NSD neural data for prediction.
    
    Args:
        data_path: Path to NSD GoodUnit .mat file
        
    Returns:
        resp_matrix: Neural response matrix (n_images, n_units)
        image_indices: Image indices for stimulus mapping
        metadata: Additional metadata
    """
    print(f"Loading neural data from {data_path}")
    
    # Load neural data using NSD utilities
    h5_file = h5py.File(data_path, "r")
    neural_data = load_data_from_GoodUnitStrc(h5_file)
    
    # Extract key variables
    Raster = neural_data['Raster']  # (n_units, n_timepoints, n_trials)
    trial_valid_idx = neural_data['trial_valid_idx'][:, 0]  # Valid stimulus indices
    dataset_valid_idx = neural_data['dataset_valid_idx'][:, 0]  # Valid trials
    
    # Define time windows for response calculation
    evk_slice = slice(100, 250)  # Evoked response window (100-250ms)
    bsl_slice = slice(0, 90)     # Baseline window (0-90ms)
    
    # Compute baseline-corrected responses
    evk_resp = Raster[:, evk_slice, :].mean(axis=1) * 1000  # Convert to spikes/sec
    bsl_resp = Raster[:, bsl_slice, :].mean(axis=1) * 1000
    
    # Get valid trials and corresponding stimulus indices
    valid_trials = dataset_valid_idx.astype(bool)
    
    # Debug info
    print(f"Number of trials in Raster: {Raster.shape[2]}")
    print(f"Number of valid trials: {valid_trials.sum()}")
    print(f"dataset_valid_idx shape: {dataset_valid_idx.shape}")
    
    # Only select valid trials that exist in the Raster data
    n_trials_raster = Raster.shape[2]
    valid_trials_truncated = valid_trials[:n_trials_raster]
    
    evk_resp_valid = evk_resp[:, valid_trials_truncated]  # (n_units, n_valid_trials)
    bsl_resp_valid = bsl_resp[:, valid_trials_truncated]
    
    # Get corresponding stimulus indices
    valid_stim_idx = trial_valid_idx[valid_trials][:valid_trials_truncated.sum()]
    
    # Baseline-correct responses
    resp_corrected = evk_resp_valid - bsl_resp_valid  # (n_units, n_valid_trials)
    
    # Average responses per image (handling multiple repetitions)
    unique_images = np.unique(valid_stim_idx)
    n_images = len(unique_images)
    n_units = resp_corrected.shape[0]
    
    resp_matrix = np.zeros((n_images, n_units))
    
    for i, img_idx in enumerate(unique_images):
        trial_mask = valid_stim_idx == img_idx
        resp_matrix[i, :] = resp_corrected[:, trial_mask].mean(axis=1)
    
    # Transpose to get (n_images, n_units) format expected by prediction pipeline
    h5_file.close()
    
    print(f"Loaded neural data: {n_images} images, {n_units} units")
    
    metadata = {
        'n_images': n_images,
        'n_units': n_units,
        'unique_images': unique_images,
        'evk_window': evk_slice,
        'bsl_window': bsl_slice
    }
    
    return resp_matrix, unique_images, metadata


def create_dummy_features(n_images, feature_dim=2048):
    """
    Create dummy deep network features for demonstration.
    In practice, these would be extracted from a CNN using actual images.
    
    Args:
        n_images: Number of images
        feature_dim: Feature dimensionality
        
    Returns:
        Dummy feature matrix
    """
    print(f"Creating dummy features: {n_images} images, {feature_dim} features")
    
    # Create structured dummy features with some correlation structure
    np.random.seed(42)
    
    # Create some basis patterns
    n_patterns = 50
    patterns = np.random.randn(n_patterns, feature_dim)
    
    # Combine patterns with random weights
    weights = np.random.randn(n_images, n_patterns)
    features = weights @ patterns
    
    # Add some noise
    features += 0.1 * np.random.randn(n_images, feature_dim)
    
    return features


def run_neural_prediction_example():
    """
    Run complete neural prediction example using NSD data.
    """
    print("=== Neural Prediction Example ===\n")
    
    # Example data path - modify this to point to your actual data
    NSD_root = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/NSD_N3"
    data_path = join(NSD_root, "GoodUnit_240709_JianJian_NSD1000_LOC_g2.mat")
    
    try:
        # Step 1: Load neural data
        resp_matrix, image_indices, metadata = load_nsd_neural_data(data_path)
        
        # Step 2: Create or load deep network features
        # In practice, you would extract these from actual images using a CNN
        features = create_dummy_features(metadata['n_images'], feature_dim=2048)
        
        # Step 3: Run neural prediction pipeline
        print("\nRunning neural prediction pipeline...")
        
        # Adaptive PCA components based on data size
        n_train = int(0.8 * metadata['n_images'])  # 80% for training
        max_pca_components = min(500, n_train - 10, features.shape[1] // 4)
        pca_methods = [f'pca{max_pca_components//2}', f'pca{max_pca_components}', 'sp_avg']
        
        print(f"Using PCA components: {pca_methods}")
        
        results = neural_prediction_pipeline(
            features=features,
            responses=resp_matrix,
            dimred_methods=pca_methods,
            regression_methods=['RidgeCV', 'LassoCV'],
            test_size=0.2,
            random_state=42
        )
        
        # Step 4: Display results
        print("\n=== Prediction Results ===")
        print("\nRegression Results Summary:")
        print(results['result_df'].round(3))
        
        print("\nEvaluation Results Summary:")
        print(results['eval_df'].round(3))
        
        # Step 5: Plot best performing method
        best_method = results['result_df']['test_score'].idxmax()
        print(f"\nBest performing method: {best_method}")
        
        if best_method in results['predictions']:
            y_pred = results['predictions'][best_method]
            y_test = resp_matrix[results['splits']['idx_test']]
            
            # Plot results for first few units
            for unit_idx in range(min(3, resp_matrix.shape[1])):
                plot_prediction_results(
                    y_test, y_pred, 
                    title=f"Neural Prediction Results - {best_method}",
                    unit_idx=unit_idx
                )
        
        print("\n=== Analysis Complete ===")
        print(f"Processed {metadata['n_images']} images and {metadata['n_units']} neural units")
        print(f"Best test R² score: {results['result_df'].loc[best_method, 'test_score']:.3f}")
        
        return results, metadata
        
    except FileNotFoundError:
        print(f"Data file not found: {data_path}")
        print("Please update the data_path variable to point to your NSD data file.")
        return None, None
    
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def demonstrate_with_synthetic_data():
    """
    Demonstrate the pipeline with fully synthetic data if real data is not available.
    """
    print("=== Synthetic Data Demo ===\n")
    
    # Create synthetic neural data
    n_images = 500
    n_units = 100
    feature_dim = 2048
    
    print(f"Creating synthetic data: {n_images} images, {n_units} units")
    
    # Create features
    np.random.seed(42)
    features = np.random.randn(n_images, feature_dim)
    
    # Create neural responses with some structure
    # Simulate a simple linear relationship between features and responses
    true_weights = np.random.randn(feature_dim, n_units) * 0.1
    responses = features @ true_weights + 0.5 * np.random.randn(n_images, n_units)
    
    # Run prediction pipeline
    print("Running prediction pipeline on synthetic data...")
    
    # Use appropriate PCA components for synthetic data
    n_train = int(0.8 * n_images)
    max_pca = min(300, n_train - 10)
    
    results = neural_prediction_pipeline(
        features=features,
        responses=responses,
        dimred_methods=[f'pca{max_pca//2}', f'pca{max_pca}', 'sp_avg'],
        regression_methods=['RidgeCV', 'LassoCV'],
        test_size=0.2,
        random_state=42
    )
    
    # Display results
    print("\n=== Synthetic Data Results ===")
    print(results['result_df'].round(3))
    
    # Plot best method
    best_method = results['result_df']['test_score'].idxmax()
    print(f"\nBest method: {best_method} (R² = {results['result_df'].loc[best_method, 'test_score']:.3f})")
    
    if best_method in results['predictions']:
        y_pred = results['predictions'][best_method]
        y_test = responses[results['splits']['idx_test']]
        
        plot_prediction_results(
            y_test, y_pred, 
            title=f"Synthetic Data Results - {best_method}",
            unit_idx=0
        )
    
    return results


if __name__ == "__main__":
    print("Neural Prediction Example Script")
    print("================================\n")
    
    # Try to run with real NSD data first
    results, metadata = run_neural_prediction_example()
    
    # If real data is not available, run synthetic demo
    if results is None:
        print("\nFalling back to synthetic data demonstration...")
        results = demonstrate_with_synthetic_data()
    
    print("\nExample complete! Check the neural_prediction_utils.py file for more advanced usage.")