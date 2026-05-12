"""
Optimized Neural Prediction Example

This script demonstrates neural prediction with NSD data using a more memory-efficient approach.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import h5py

# Import utilities
from NSD_utils.h5_dataset_utils import load_data_from_GoodUnitStrc
from neural_prediction_utils import (
    transform_features2Xdict,
    sweep_regressors,
    evaluate_prediction,
    plot_prediction_results,
    RidgeCV, MultiTaskLassoCV
)

def load_nsd_subset(data_path, max_images=500, max_units=200):
    """
    Load a manageable subset of NSD data for neural prediction.
    
    Args:
        data_path: Path to NSD GoodUnit .mat file
        max_images: Maximum number of images to process
        max_units: Maximum number of units to use
        
    Returns:
        resp_matrix: Neural response matrix (n_images, n_units)
        metadata: Metadata dictionary
    """
    print(f"Loading NSD data subset from {data_path}")
    
    h5_file = h5py.File(data_path, "r")
    neural_data = load_data_from_GoodUnitStrc(h5_file)
    
    # Extract variables
    Raster = neural_data['Raster']  # (n_units, n_timepoints, n_trials)
    trial_valid_idx = neural_data['trial_valid_idx'][:, 0]
    dataset_valid_idx = neural_data['dataset_valid_idx'][:, 0]
    
    print(f"Original data: {Raster.shape[0]} units, {Raster.shape[2]} trials")
    
    # Time windows
    evk_slice = slice(100, 250)  # 100-250ms
    bsl_slice = slice(0, 90)     # 0-90ms
    
    # Use subset of units for efficiency
    n_units_use = min(max_units, Raster.shape[0])
    
    # Compute responses
    evk_resp = Raster[:n_units_use, evk_slice, :].mean(axis=1) * 1000
    bsl_resp = Raster[:n_units_use, bsl_slice, :].mean(axis=1) * 1000
    
    # Get valid trials
    n_trials_raster = Raster.shape[2]
    valid_trials = dataset_valid_idx[:n_trials_raster].astype(bool)
    
    evk_resp_valid = evk_resp[:, valid_trials]
    bsl_resp_valid = bsl_resp[:, valid_trials]
    resp_corrected = evk_resp_valid - bsl_resp_valid
    
    valid_stim_idx = trial_valid_idx[dataset_valid_idx.astype(bool)][:valid_trials.sum()]
    
    # Average responses per image
    unique_images = np.unique(valid_stim_idx)
    n_images_use = min(max_images, len(unique_images))
    unique_images = unique_images[:n_images_use]
    
    resp_matrix = np.zeros((n_images_use, n_units_use))
    
    for i, img_idx in enumerate(unique_images):
        trial_mask = valid_stim_idx == img_idx
        if trial_mask.sum() > 0:
            resp_matrix[i, :] = resp_corrected[:, trial_mask].mean(axis=1)
    
    h5_file.close()
    
    print(f"Loaded subset: {n_images_use} images, {n_units_use} units")
    
    metadata = {
        'n_images': n_images_use,
        'n_units': n_units_use,
        'unique_images': unique_images,
        'evk_window': evk_slice,
        'bsl_window': bsl_slice
    }
    
    return resp_matrix, metadata

def create_features(n_images, feature_dim=1024):
    """Create structured dummy features."""
    print(f"Creating features for {n_images} images")
    
    np.random.seed(42)
    
    # Create features with some structure
    n_patterns = min(100, feature_dim // 10)
    patterns = np.random.randn(n_patterns, feature_dim)
    weights = np.random.randn(n_images, n_patterns)
    features = weights @ patterns
    features += 0.1 * np.random.randn(n_images, feature_dim)
    
    return features

def run_neural_prediction():
    """Run optimized neural prediction pipeline."""
    print("=== Optimized Neural Prediction ===\n")
    
    # Data path
    NSD_root = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/NSD_N3"
    data_path = join(NSD_root, "GoodUnit_240709_JianJian_NSD1000_LOC_g2.mat")
    
    try:
        # Step 1: Load subset of neural data
        resp_matrix, metadata = load_nsd_subset(data_path, max_images=400, max_units=150)
        
        # Step 2: Create features
        features = create_features(metadata['n_images'], feature_dim=1024)
        
        # Step 3: Transform features
        print("\nTransforming features...")
        feat_dict = {'cnn_features': features}
        
        # Adaptive PCA components
        n_train = int(0.8 * metadata['n_images'])
        max_pca = min(200, n_train - 20, features.shape[1] // 8)
        dimred_methods = [f'pca{max_pca//2}', f'pca{max_pca}', 'sp_avg']
        
        Xdict, tfm_dict = transform_features2Xdict(
            feat_dict,
            dimred_list=dimred_methods
        )
        
        # Step 4: Run regression sweep
        print(f"\nRunning regression with methods: {dimred_methods}")
        
        alpha_list = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
        regressors = [RidgeCV(alphas=alpha_list, alpha_per_target=True)]
        regressor_names = ['RidgeCV']
        
        result_df, models = sweep_regressors(
            Xdict, resp_matrix, regressors, regressor_names, verbose=True
        )
        
        # Step 5: Evaluate models
        print("\nEvaluating models...")
        eval_df, eval_dict, y_pred_dict = evaluate_prediction(
            models, Xdict, resp_matrix, label="nsd_prediction"
        )
        
        # Step 6: Save and display results
        print("\n=== Results Summary ===")
        
        # Save results to CSV
        result_df.to_csv('neural_prediction_regression_results.csv')
        eval_df.to_csv('neural_prediction_evaluation_results.csv')
        print("✓ Saved regression_results.csv and evaluation_results.csv")
        
        print("\nRegression Performance:")
        print(result_df[['train_score', 'test_score', 'n_feat']].round(3))
        
        print("\nEvaluation Metrics:")
        print(eval_df[['rho_p', 'rho_s', 'D2']].round(3))
        
        # Find best method
        best_method = result_df['test_score'].idxmax()
        print(f"\nBest method: {best_method}")
        print(f"Test R²: {result_df.loc[best_method, 'test_score']:.3f}")
        
        # Step 7: Plot results for best method
        if best_method in y_pred_dict:
            # Get test split (same split used in sweep_regressors)
            from sklearn.model_selection import train_test_split
            indices = np.arange(len(resp_matrix))
            idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)
            
            # Get predictions for test set only
            y_pred_all = y_pred_dict[best_method]
            y_pred_test = y_pred_all[idx_test]  # Select test predictions
            y_test = resp_matrix[idx_test]
            
            print(f"\nPlotting results for {len(idx_test)} test samples...")
            print(f"y_test shape: {y_test.shape}, y_pred_test shape: {y_pred_test.shape}")
            
            # Plot a few example units and save figures
            for unit_idx in range(min(3, metadata['n_units'])):
                save_path = f"neural_prediction_unit_{unit_idx}_results.png"
                plot_prediction_results(
                    y_test, y_pred_test,
                    title=f"Neural Prediction - {best_method}",
                    unit_idx=unit_idx,
                    save_path=save_path
                )
                print(f"✓ Saved {save_path}")
            
            # Create summary plot comparing all methods
            plt.figure(figsize=(12, 8))
            
            # Plot 1: R² comparison
            plt.subplot(2, 2, 1)
            methods = [str(idx) for idx in result_df.index]
            train_scores = result_df['train_score'].values
            test_scores = result_df['test_score'].values
            
            x = np.arange(len(methods))
            width = 0.35
            plt.bar(x - width/2, train_scores, width, label='Train R²', alpha=0.7)
            plt.bar(x + width/2, test_scores, width, label='Test R²', alpha=0.7)
            plt.xlabel('Method')
            plt.ylabel('R²')
            plt.title('Train vs Test R² by Method')
            plt.xticks(x, [m.split('_')[1] for m in methods], rotation=45)
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            # Plot 2: Correlation comparison
            plt.subplot(2, 2, 2)
            pearson_corr = eval_df['rho_p'].values
            spearman_corr = eval_df['rho_s'].values
            
            plt.bar(x - width/2, pearson_corr, width, label='Pearson r', alpha=0.7)
            plt.bar(x + width/2, spearman_corr, width, label='Spearman ρ', alpha=0.7)
            plt.xlabel('Method')
            plt.ylabel('Correlation')
            plt.title('Prediction Correlations by Method')
            plt.xticks(x, [m.split('_')[1] for m in methods], rotation=45)
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            # Plot 3: Feature dimensions
            plt.subplot(2, 2, 3)
            n_features = result_df['n_feat'].values
            plt.bar(x, n_features, alpha=0.7, color='green')
            plt.xlabel('Method')
            plt.ylabel('Number of Features')
            plt.title('Feature Dimensions by Method')
            plt.xticks(x, [m.split('_')[1] for m in methods], rotation=45)
            plt.grid(axis='y', alpha=0.3)
            
            # Plot 4: Best method scatter plot
            plt.subplot(2, 2, 4)
            # Average across units for visualization
            y_true_mean = y_test.mean(axis=1)
            y_pred_mean = y_pred_test.mean(axis=1)
            plt.scatter(y_true_mean, y_pred_mean, alpha=0.6, s=30)
            plt.xlabel('True Response (mean across units)')
            plt.ylabel('Predicted Response (mean across units)')
            plt.title(f'Best Method: {best_method[0].split("_")[1]}')
            
            # Add diagonal line
            min_val = min(plt.xlim()[0], plt.ylim()[0])
            max_val = max(plt.xlim()[1], plt.ylim()[1])
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            plt.tight_layout()
            plt.savefig('neural_prediction_summary.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("✓ Saved neural_prediction_summary.png")
        
        print(f"\n=== Success! ===")
        print(f"Processed {metadata['n_images']} images and {metadata['n_units']} units")
        print(f"Best performance: R² = {result_df.loc[best_method, 'test_score']:.3f}")
        
        return result_df, eval_df, models, metadata
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

if __name__ == "__main__":
    print("Optimized Neural Prediction Script")
    print("=================================\n")
    
    result_df, eval_df, models, metadata = run_neural_prediction()
    
    if result_df is not None:
        print("\nScript completed successfully!")
    else:
        print("\nScript encountered errors.")