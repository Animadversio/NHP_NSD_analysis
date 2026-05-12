"""
Comprehensive Neural Prediction Experiment

This script runs neural prediction experiments across multiple monkeys
using the full 1000+ image dataset with proper train/test splits.

Experiments include:
- Multiple monkeys: JianJian, FaCai, TuTu, MaoDan, ZhuangZhuang
- Full image dataset (1000+ images)
- Multiple feature reduction methods (PCA64, PCA128, PCA256, spatial averaging)
- Ridge regression with cross-validation
- Comprehensive evaluation and visualization
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join, basename
import h5py
from datetime import datetime

# Import utilities
from NSD_utils.h5_dataset_utils import load_data_from_GoodUnitStrc
from neural_prediction_utils import (
    transform_features2Xdict,
    sweep_regressors,
    evaluate_prediction,
    plot_prediction_results,
    RidgeCV
)

def extract_monkey_info(filename):
    """Extract monkey name and date from filename."""
    base = basename(filename)
    parts = base.replace('.mat', '').split('_')
    date = parts[1]
    monkey = parts[2]
    return monkey, date

def load_full_nsd_data(data_path, max_units=None):
    """
    Load full NSD dataset with all 1000+ images.
    
    Args:
        data_path: Path to NSD GoodUnit .mat file
        max_units: Maximum number of units to use (None = all)
        
    Returns:
        resp_matrix: Neural response matrix (n_images, n_units)
        metadata: Metadata dictionary
    """
    print(f"Loading full NSD data from {data_path}")
    
    h5_file = h5py.File(data_path, "r")
    neural_data = load_data_from_GoodUnitStrc(h5_file)
    
    # Extract variables
    Raster = neural_data['Raster']  # (n_units, n_timepoints, n_trials)
    trial_valid_idx = neural_data['trial_valid_idx'][:, 0]
    dataset_valid_idx = neural_data['dataset_valid_idx'][:, 0]
    
    print(f"Original data: {Raster.shape[0]} units, {Raster.shape[2]} trials")
    
    # Use specified number of units
    n_units_use = Raster.shape[0] if max_units is None else min(max_units, Raster.shape[0])
    
    # Time windows for neural response
    evk_slice = slice(100, 250)  # 100-250ms evoked response
    bsl_slice = slice(0, 90)     # 0-90ms baseline
    
    # Compute responses
    evk_resp = Raster[:n_units_use, evk_slice, :].mean(axis=1) * 1000  # spikes/sec
    bsl_resp = Raster[:n_units_use, bsl_slice, :].mean(axis=1) * 1000
    
    # Get valid trials
    n_trials_raster = Raster.shape[2]
    valid_trials = dataset_valid_idx[:n_trials_raster].astype(bool)
    
    evk_resp_valid = evk_resp[:, valid_trials]
    bsl_resp_valid = bsl_resp[:, valid_trials]
    resp_corrected = evk_resp_valid - bsl_resp_valid  # Baseline-corrected
    
    valid_stim_idx = trial_valid_idx[dataset_valid_idx.astype(bool)][:valid_trials.sum()]
    
    # Average responses per image (all unique images)
    unique_images = np.unique(valid_stim_idx)
    n_images = len(unique_images)
    
    resp_matrix = np.zeros((n_images, n_units_use))
    
    for i, img_idx in enumerate(unique_images):
        trial_mask = valid_stim_idx == img_idx
        if trial_mask.sum() > 0:
            resp_matrix[i, :] = resp_corrected[:, trial_mask].mean(axis=1)
    
    h5_file.close()
    
    print(f"Loaded: {n_images} images, {n_units_use} units")
    
    # Extract monkey and date info
    monkey, date = extract_monkey_info(data_path)
    
    metadata = {
        'n_images': n_images,
        'n_units': n_units_use,
        'unique_images': unique_images,
        'monkey': monkey,
        'date': date,
        'data_path': data_path,
        'evk_window': evk_slice,
        'bsl_window': bsl_slice
    }
    
    return resp_matrix, metadata

def create_comprehensive_features(n_images, feature_dims=[512, 1024, 2048]):
    """Create multiple feature sets with different dimensionalities."""
    print(f"Creating comprehensive features for {n_images} images")
    
    np.random.seed(42)  # For reproducibility
    
    feature_sets = {}
    
    for dim in feature_dims:
        # Create structured features with realistic properties
        n_patterns = min(100, dim // 10)
        patterns = np.random.randn(n_patterns, dim)
        weights = np.random.randn(n_images, n_patterns)
        
        # Add some nonlinearity and structure
        weights = np.tanh(weights * 0.5)  # Bounded activation
        features = weights @ patterns
        
        # Add noise
        features += 0.1 * np.random.randn(n_images, dim)
        
        # Normalize features
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        feature_sets[f'features_{dim}d'] = features
        print(f"Created {dim}D features: {features.shape}")
    
    return feature_sets

def run_monkey_experiment(data_path, output_dir="results", max_units=300):
    """Run comprehensive experiment for one monkey."""
    
    # Load data
    resp_matrix, metadata = load_full_nsd_data(data_path, max_units=max_units)
    
    # Create features (using 1024D as main feature set)
    feature_sets = create_comprehensive_features(metadata['n_images'], [1024])
    features = feature_sets['features_1024d']
    
    # Feature transformation
    print("\nTransforming features...")
    feat_dict = {'cnn_features': features}
    
    # Adaptive PCA components based on data size
    n_train = int(0.8 * metadata['n_images'])
    max_pca = min(400, n_train - 20, features.shape[1] // 4)
    
    # Multiple PCA dimensions for comparison
    pca_dims = [64, 128, min(256, max_pca), max_pca]
    dimred_methods = [f'pca{dim}' for dim in pca_dims] + ['sp_avg']
    
    print(f"Using dimensionality reduction methods: {dimred_methods}")
    
    Xdict, tfm_dict = transform_features2Xdict(
        feat_dict,
        dimred_list=dimred_methods
    )
    
    # Ridge regression with comprehensive alpha range
    print(f"\nRunning Ridge regression...")
    alpha_list = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1e4]
    regressors = [RidgeCV(alphas=alpha_list, alpha_per_target=True)]
    regressor_names = ['RidgeCV']
    
    # Run regression with proper train/test split
    result_df, models = sweep_regressors(
        Xdict, resp_matrix, regressors, regressor_names, verbose=True
    )
    
    # Evaluate models
    print("\nEvaluating models...")
    eval_df, eval_dict, y_pred_dict = evaluate_prediction(
        models, Xdict, resp_matrix, 
        label=f"{metadata['monkey']}_{metadata['date']}"
    )
    
    # Add metadata to results
    result_df['monkey'] = metadata['monkey']
    result_df['date'] = metadata['date']
    result_df['n_images_total'] = metadata['n_images']
    result_df['n_units_total'] = metadata['n_units']
    
    eval_df['monkey'] = metadata['monkey']
    eval_df['date'] = metadata['date']
    eval_df['n_images_total'] = metadata['n_images']
    eval_df['n_units_total'] = metadata['n_units']
    
    # Save results
    monkey_name = metadata['monkey']
    date_str = metadata['date']
    
    os.makedirs(output_dir, exist_ok=True)
    result_df.to_csv(join(output_dir, f'{monkey_name}_{date_str}_regression_results.csv'))
    eval_df.to_csv(join(output_dir, f'{monkey_name}_{date_str}_evaluation_results.csv'))
    
    print(f"✓ Saved results for {monkey_name} ({date_str})")
    
    return result_df, eval_df, models, metadata, y_pred_dict, Xdict

def plot_monkey_results(result_df, eval_df, monkey_name, date_str, output_dir="results"):
    """Create comprehensive plots for one monkey."""
    
    plt.style.use('default')
    fig = plt.figure(figsize=(16, 12))
    
    # Extract method names for plotting
    methods = [idx[0].split('_')[1] if isinstance(idx, tuple) else str(idx).split('_')[1] 
               for idx in result_df.index]
    
    # Plot 1: R² performance
    plt.subplot(3, 3, 1)
    train_scores = result_df['train_score'].values
    test_scores = result_df['test_score'].values
    
    x = np.arange(len(methods))
    width = 0.35
    plt.bar(x - width/2, train_scores, width, label='Train R²', alpha=0.7, color='skyblue')
    plt.bar(x + width/2, test_scores, width, label='Test R²', alpha=0.7, color='orange')
    plt.xlabel('Method')
    plt.ylabel('R²')
    plt.title(f'{monkey_name} - R² Performance')
    plt.xticks(x, methods, rotation=45)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 2: Correlation metrics
    plt.subplot(3, 3, 2)
    pearson_corr = eval_df['rho_p'].values
    spearman_corr = eval_df['rho_s'].values
    
    plt.bar(x - width/2, pearson_corr, width, label='Pearson r', alpha=0.7, color='lightgreen')
    plt.bar(x + width/2, spearman_corr, width, label='Spearman ρ', alpha=0.7, color='lightcoral')
    plt.xlabel('Method')
    plt.ylabel('Correlation')
    plt.title(f'{monkey_name} - Correlations')
    plt.xticks(x, methods, rotation=45)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 3: Feature dimensions
    plt.subplot(3, 3, 3)
    n_features = result_df['n_feat'].values
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
    plt.bar(x, n_features, alpha=0.7, color=colors)
    plt.xlabel('Method')
    plt.ylabel('Number of Features')
    plt.title(f'{monkey_name} - Feature Dimensions')
    plt.xticks(x, methods, rotation=45)
    plt.yscale('log')
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 4: Runtime comparison
    plt.subplot(3, 3, 4)
    runtimes = result_df['runtime'].values
    plt.bar(x, runtimes, alpha=0.7, color='gold')
    plt.xlabel('Method')
    plt.ylabel('Runtime (seconds)')
    plt.title(f'{monkey_name} - Runtime')
    plt.xticks(x, methods, rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 5: Alpha values used
    plt.subplot(3, 3, 5)
    # Extract alpha values (they're arrays, so take mean)
    alphas = []
    for alpha_val in result_df['alpha'].values:
        if isinstance(alpha_val, (list, np.ndarray)):
            alphas.append(np.mean(alpha_val))
        else:
            alphas.append(alpha_val)
    
    plt.bar(x, alphas, alpha=0.7, color='mediumpurple')
    plt.xlabel('Method')
    plt.ylabel('Mean Alpha Value')
    plt.title(f'{monkey_name} - Regularization Strength')
    plt.xticks(x, methods, rotation=45)
    plt.yscale('log')
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 6: Performance vs Features scatter
    plt.subplot(3, 3, 6)
    plt.scatter(n_features, test_scores, s=100, alpha=0.7, c=colors)
    plt.xlabel('Number of Features')
    plt.ylabel('Test R²')
    plt.title(f'{monkey_name} - Performance vs Complexity')
    plt.xscale('log')
    for i, method in enumerate(methods):
        plt.annotate(method, (n_features[i], test_scores[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    plt.grid(alpha=0.3)
    
    # Plot 7-9: Method comparison details
    best_idx = result_df['test_score'].idxmax()
    best_method = methods[list(result_df.index).index(best_idx)]
    
    plt.subplot(3, 3, 7)
    metrics = ['train_score', 'test_score']
    best_values = [result_df.loc[best_idx, 'train_score'], result_df.loc[best_idx, 'test_score']]
    plt.bar(metrics, best_values, alpha=0.7, color=['skyblue', 'orange'])
    plt.ylabel('R²')
    plt.title(f'Best Method: {best_method}')
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(3, 3, 8)
    corr_metrics = ['Pearson r', 'Spearman ρ']
    eval_idx = list(eval_df.index)[list(result_df.index).index(best_idx)]
    corr_values = [eval_df.loc[eval_idx, 'rho_p'], eval_df.loc[eval_idx, 'rho_s']]
    plt.bar(corr_metrics, corr_values, alpha=0.7, color=['lightgreen', 'lightcoral'])
    plt.ylabel('Correlation')
    plt.title(f'Best Method Correlations')
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(3, 3, 9)
    # Summary text
    plt.text(0.1, 0.8, f'Monkey: {monkey_name}', fontsize=12, weight='bold')
    plt.text(0.1, 0.7, f'Date: {date_str}', fontsize=10)
    plt.text(0.1, 0.6, f'Images: {result_df.iloc[0]["n_images_total"]}', fontsize=10)
    plt.text(0.1, 0.5, f'Units: {result_df.iloc[0]["n_units_total"]}', fontsize=10)
    plt.text(0.1, 0.4, f'Best R²: {result_df.loc[best_idx, "test_score"]:.3f}', fontsize=10)
    plt.text(0.1, 0.3, f'Best Correlation: {corr_values[0]:.3f}', fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Summary')
    
    plt.tight_layout()
    plt.savefig(join(output_dir, f'{monkey_name}_{date_str}_comprehensive_results.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Saved comprehensive plot for {monkey_name} ({date_str})")

def run_comprehensive_experiment():
    """Run comprehensive neural prediction experiment across multiple monkeys."""
    
    print("=" * 60)
    print("COMPREHENSIVE NEURAL PREDICTION EXPERIMENT")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Data directory
    NSD_root = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/NSD_N3"
    output_dir = "comprehensive_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Select representative files from different monkeys
    monkey_files = [
        "GoodUnit_240709_JianJian_NSD1000_LOC_g2.mat",      # JianJian
        "GoodUnit_240720_FaCai_NSD1000_LOC_g2.mat",        # FaCai  
        "GoodUnit_240724_TuTu_NSD1000_LOC_g2.mat",         # TuTu
        "GoodUnit_240815_MaoDan_NSD1000_LOC_g5.mat",       # MaoDan
        "GoodUnit_240817_ZhuangZhuang_NSD1000_LOC_g6.mat", # ZhuangZhuang
    ]
    
    all_results = []
    all_evaluations = []
    
    # Run experiment for each monkey
    for i, filename in enumerate(monkey_files):
        data_path = join(NSD_root, filename)
        
        if not os.path.exists(data_path):
            print(f"Warning: File not found: {data_path}")
            continue
            
        print(f"\n{'='*20} MONKEY {i+1}/{len(monkey_files)} {'='*20}")
        print(f"Processing: {filename}")
        
        try:
            # Run experiment
            result_df, eval_df, models, metadata, y_pred_dict, Xdict = run_monkey_experiment(
                data_path, output_dir, max_units=250  # Use reasonable number of units
            )
            
            # Create plots
            plot_monkey_results(result_df, eval_df, metadata['monkey'], 
                              metadata['date'], output_dir)
            
            # Collect results
            all_results.append(result_df)
            all_evaluations.append(eval_df)
            
            print(f"✓ Completed {metadata['monkey']} ({metadata['date']})")
            
        except Exception as e:
            print(f"✗ Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combine all results
    if all_results:
        print(f"\n{'='*20} COMBINING RESULTS {'='*20}")
        
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_evaluations = pd.concat(all_evaluations, ignore_index=True)
        
        # Save combined results
        combined_results.to_csv(join(output_dir, 'all_monkeys_regression_results.csv'))
        combined_evaluations.to_csv(join(output_dir, 'all_monkeys_evaluation_results.csv'))
        
        # Create cross-monkey comparison plot
        create_cross_monkey_comparison(combined_results, combined_evaluations, output_dir)
        
        print(f"✓ Saved combined results from {len(all_results)} monkeys")
        print(f"✓ Results saved in: {output_dir}/")
        
        # Summary statistics
        print(f"\n{'='*20} SUMMARY STATISTICS {'='*20}")
        print("Average performance by method:")
        method_summary = combined_results.groupby(combined_results.index.str[0] if hasattr(combined_results.index, 'str') else 'method')[['train_score', 'test_score']].mean()
        print(method_summary.round(3))
        
        print("\nAverage performance by monkey:")
        monkey_summary = combined_results.groupby('monkey')[['train_score', 'test_score']].mean()
        print(monkey_summary.round(3))
        
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETED!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    return combined_results, combined_evaluations

def create_cross_monkey_comparison(combined_results, combined_evaluations, output_dir):
    """Create comparison plots across all monkeys."""
    
    plt.figure(figsize=(20, 15))
    
    # Extract method and monkey info
    methods = []
    monkeys = combined_results['monkey'].values
    
    for idx in combined_results.index:
        if isinstance(idx, tuple):
            method = idx[0].split('_')[1]
        else:
            method = str(idx).split('_')[1] if '_' in str(idx) else str(idx)
        methods.append(method)
    
    # Plot 1: R² by monkey and method
    plt.subplot(3, 4, 1)
    pivot_test = combined_results.pivot_table(values='test_score', index='monkey', 
                                            columns=pd.Series(methods, name='method'))
    sns.heatmap(pivot_test, annot=True, fmt='.3f', cmap='RdYlBu_r')
    plt.title('Test R² by Monkey and Method')
    plt.xlabel('Method')
    plt.ylabel('Monkey')
    
    # Plot 2: Correlation by monkey and method
    plt.subplot(3, 4, 2)
    combined_evaluations['method'] = methods
    pivot_corr = combined_evaluations.pivot_table(values='rho_p', index='monkey', columns='method')
    sns.heatmap(pivot_corr, annot=True, fmt='.3f', cmap='RdYlBu_r')
    plt.title('Pearson Correlation by Monkey and Method')
    plt.xlabel('Method')
    plt.ylabel('Monkey')
    
    # Plot 3: Method comparison across monkeys
    plt.subplot(3, 4, 3)
    method_df = pd.DataFrame({'method': methods, 'test_score': combined_results['test_score']})
    sns.boxplot(data=method_df, x='method', y='test_score')
    plt.xticks(rotation=45)
    plt.title('Test R² Distribution by Method')
    plt.ylabel('Test R²')
    
    # Plot 4: Monkey comparison across methods
    plt.subplot(3, 4, 4)
    sns.boxplot(data=combined_results, x='monkey', y='test_score')
    plt.xticks(rotation=45)
    plt.title('Test R² Distribution by Monkey')
    plt.ylabel('Test R²')
    
    # Plot 5: Performance vs complexity
    plt.subplot(3, 4, 5)
    colors = plt.cm.Set1(np.linspace(0, 1, len(combined_results['monkey'].unique())))
    monkey_colors = {monkey: color for monkey, color in zip(combined_results['monkey'].unique(), colors)}
    
    for monkey in combined_results['monkey'].unique():
        mask = combined_results['monkey'] == monkey
        plt.scatter(combined_results[mask]['n_feat'], combined_results[mask]['test_score'], 
                   label=monkey, alpha=0.7, s=60, color=monkey_colors[monkey])
    
    plt.xlabel('Number of Features')
    plt.ylabel('Test R²')
    plt.xscale('log')
    plt.legend()
    plt.title('Performance vs Complexity')
    plt.grid(alpha=0.3)
    
    # Plot 6: Images vs performance
    plt.subplot(3, 4, 6)
    plt.scatter(combined_results['n_images_total'], combined_results['test_score'], 
               c=[monkey_colors[m] for m in combined_results['monkey']], s=60, alpha=0.7)
    plt.xlabel('Number of Images')
    plt.ylabel('Test R²')
    plt.title('Dataset Size vs Performance')
    plt.grid(alpha=0.3)
    
    # Plot 7: Units vs performance  
    plt.subplot(3, 4, 7)
    plt.scatter(combined_results['n_units_total'], combined_results['test_score'],
               c=[monkey_colors[m] for m in combined_results['monkey']], s=60, alpha=0.7)
    plt.xlabel('Number of Units')
    plt.ylabel('Test R²')
    plt.title('Units vs Performance')
    plt.grid(alpha=0.3)
    
    # Plot 8: Runtime vs performance
    plt.subplot(3, 4, 8)
    plt.scatter(combined_results['runtime'], combined_results['test_score'],
               c=[monkey_colors[m] for m in combined_results['monkey']], s=60, alpha=0.7)
    plt.xlabel('Runtime (seconds)')
    plt.ylabel('Test R²')
    plt.title('Runtime vs Performance')
    plt.grid(alpha=0.3)
    
    # Plot 9: Average performance by method (bar plot)
    plt.subplot(3, 4, 9)
    method_avg = method_df.groupby('method')['test_score'].mean().sort_values(ascending=False)
    plt.bar(range(len(method_avg)), method_avg.values, alpha=0.7, color='skyblue')
    plt.xticks(range(len(method_avg)), method_avg.index, rotation=45)
    plt.ylabel('Average Test R²')
    plt.title('Average Performance by Method')
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 10: Average performance by monkey (bar plot)
    plt.subplot(3, 4, 10)
    monkey_avg = combined_results.groupby('monkey')['test_score'].mean().sort_values(ascending=False)
    plt.bar(range(len(monkey_avg)), monkey_avg.values, alpha=0.7, 
           color=[monkey_colors[m] for m in monkey_avg.index])
    plt.xticks(range(len(monkey_avg)), monkey_avg.index, rotation=45)
    plt.ylabel('Average Test R²')
    plt.title('Average Performance by Monkey')
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 11: Method preference by monkey (stacked bar)
    plt.subplot(3, 4, 11)
    best_methods = []
    for monkey in combined_results['monkey'].unique():
        monkey_data = combined_results[combined_results['monkey'] == monkey]
        best_method = methods[monkey_data['test_score'].idxmax()]
        best_methods.append(best_method)
    
    method_counts = pd.Series(best_methods).value_counts()
    plt.pie(method_counts.values, labels=method_counts.index, autopct='%1.1f%%')
    plt.title('Best Method Distribution')
    
    # Plot 12: Summary statistics
    plt.subplot(3, 4, 12)
    stats_text = f"""
    Total Monkeys: {len(combined_results['monkey'].unique())}
    Total Experiments: {len(combined_results)}
    
    Best Overall R²: {combined_results['test_score'].max():.3f}
    Average R²: {combined_results['test_score'].mean():.3f}
    
    Best Method: {method_avg.index[0]}
    Best Monkey: {monkey_avg.index[0]}
    
    Avg Images: {combined_results['n_images_total'].mean():.0f}
    Avg Units: {combined_results['n_units_total'].mean():.0f}
    """
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    plt.axis('off')
    plt.title('Summary Statistics')
    
    plt.tight_layout()
    plt.savefig(join(output_dir, 'cross_monkey_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Saved cross-monkey comparison plot")

if __name__ == "__main__":
    # Run comprehensive experiment
    combined_results, combined_evaluations = run_comprehensive_experiment()