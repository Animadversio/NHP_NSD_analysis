"""
Analyze and summarize comprehensive neural prediction results
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join

def load_and_clean_results():
    """Load and clean the results data."""
    
    # Load combined results
    results_df = pd.read_csv('comprehensive_results/all_monkeys_regression_results.csv', index_col=0)
    eval_df = pd.read_csv('comprehensive_results/all_monkeys_evaluation_results.csv', index_col=0)
    
    # Extract method names from index
    results_df['method'] = results_df.index.str.extract(r'cnn_features_(\w+)')[0]
    eval_df['method'] = eval_df.index.str.extract(r'cnn_features_(\w+)')[0]
    
    return results_df, eval_df

def print_summary_statistics(results_df, eval_df):
    """Print comprehensive summary statistics."""
    
    print("=" * 70)
    print("COMPREHENSIVE NEURAL PREDICTION RESULTS SUMMARY")
    print("=" * 70)
    
    # Basic dataset info
    n_monkeys = results_df['monkey'].nunique()
    total_experiments = len(results_df)
    avg_images = results_df['n_images_total'].iloc[0]  # Same for all
    avg_units = results_df['n_units_total'].iloc[0]   # Same for all
    
    print(f"\nDataset Overview:")
    print(f"  • Monkeys tested: {n_monkeys}")
    print(f"  • Total experiments: {total_experiments}")
    print(f"  • Images per experiment: {avg_images}")
    print(f"  • Units per experiment: {avg_units}")
    print(f"  • Train/Test split: 80%/20%")
    
    # Performance summary
    print(f"\nOverall Performance:")
    print(f"  • Best R²: {results_df['test_score'].max():.3f}")
    print(f"  • Average R²: {results_df['test_score'].mean():.3f}")
    print(f"  • Std R²: {results_df['test_score'].std():.3f}")
    print(f"  • Best Correlation: {eval_df['rho_p'].max():.3f}")
    print(f"  • Average Correlation: {eval_df['rho_p'].mean():.3f}")
    
    # By monkey performance
    print(f"\nPerformance by Monkey:")
    monkey_perf = results_df.groupby('monkey').agg({
        'test_score': ['mean', 'max'],
        'train_score': 'mean'
    }).round(3)
    
    for monkey in results_df['monkey'].unique():
        monkey_data = results_df[results_df['monkey'] == monkey]
        avg_test = monkey_data['test_score'].mean()
        max_test = monkey_data['test_score'].max()
        avg_corr = eval_df[eval_df['monkey'] == monkey]['rho_p'].mean()
        print(f"  • {monkey:12}: R² = {avg_test:.3f} (max: {max_test:.3f}), Corr = {avg_corr:.3f}")
    
    # By method performance  
    print(f"\nPerformance by Method:")
    for method in results_df['method'].unique():
        method_data = results_df[results_df['method'] == method]
        avg_test = method_data['test_score'].mean()
        avg_train = method_data['train_score'].mean()
        avg_feats = method_data['n_feat'].iloc[0]
        avg_time = method_data['runtime'].mean()
        print(f"  • {method:8}: R² = {avg_test:.3f} (train: {avg_train:.3f}), Features: {avg_feats:4.0f}, Time: {avg_time:.3f}s")
    
    # Best combinations
    print(f"\nBest Monkey-Method Combinations:")
    results_df['combo'] = results_df['monkey'] + '_' + results_df['method']
    top_combos = results_df.nlargest(5, 'test_score')[['combo', 'test_score', 'monkey', 'method']]
    for idx, row in top_combos.iterrows():
        print(f"  • {row['combo']:20}: R² = {row['test_score']:.3f}")

def create_summary_plots(results_df, eval_df):
    """Create summary visualization plots."""
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Performance by monkey
    ax1 = axes[0, 0]
    monkey_stats = results_df.groupby('monkey')['test_score'].agg(['mean', 'std']).reset_index()
    bars = ax1.bar(monkey_stats['monkey'], monkey_stats['mean'], 
                   yerr=monkey_stats['std'], capsize=5, alpha=0.7, color='skyblue')
    ax1.set_title('Average Test R² by Monkey')
    ax1.set_ylabel('Test R²')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, monkey_stats['mean']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Performance by method
    ax2 = axes[0, 1]
    method_stats = results_df.groupby('method')['test_score'].agg(['mean', 'std']).reset_index()
    bars = ax2.bar(method_stats['method'], method_stats['mean'],
                   yerr=method_stats['std'], capsize=5, alpha=0.7, color='lightcoral')
    ax2.set_title('Average Test R² by Method')
    ax2.set_ylabel('Test R²')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, mean_val in zip(bars, method_stats['mean']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Correlation by monkey
    ax3 = axes[0, 2]
    corr_stats = eval_df.groupby('monkey')['rho_p'].agg(['mean', 'std']).reset_index()
    bars = ax3.bar(corr_stats['monkey'], corr_stats['mean'],
                   yerr=corr_stats['std'], capsize=5, alpha=0.7, color='lightgreen')
    ax3.set_title('Average Correlation by Monkey')
    ax3.set_ylabel('Pearson Correlation')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Feature complexity vs performance
    ax4 = axes[1, 0]
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, monkey in enumerate(results_df['monkey'].unique()):
        monkey_data = results_df[results_df['monkey'] == monkey]
        ax4.scatter(monkey_data['n_feat'], monkey_data['test_score'], 
                   label=monkey, alpha=0.7, s=60, color=colors[i])
    
    ax4.set_xlabel('Number of Features')
    ax4.set_ylabel('Test R²')
    ax4.set_title('Performance vs Feature Complexity')
    ax4.set_xscale('log')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # Plot 5: Runtime vs performance
    ax5 = axes[1, 1]
    for i, monkey in enumerate(results_df['monkey'].unique()):
        monkey_data = results_df[results_df['monkey'] == monkey]
        ax5.scatter(monkey_data['runtime'], monkey_data['test_score'],
                   label=monkey, alpha=0.7, s=60, color=colors[i])
    
    ax5.set_xlabel('Runtime (seconds)')
    ax5.set_ylabel('Test R²')
    ax5.set_title('Performance vs Runtime')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # Plot 6: Method distribution pie chart
    ax6 = axes[1, 2]
    best_methods = []
    for monkey in results_df['monkey'].unique():
        monkey_data = results_df[results_df['monkey'] == monkey]
        best_method = monkey_data.loc[monkey_data['test_score'].idxmax(), 'method']
        best_methods.append(best_method)
    
    method_counts = pd.Series(best_methods).value_counts()
    ax6.pie(method_counts.values, labels=method_counts.index, autopct='%1.1f%%',
           startangle=90, colors=['lightblue', 'lightcoral', 'lightgreen', 'gold'])
    ax6.set_title('Best Method Distribution\nAcross Monkeys')
    
    plt.tight_layout()
    plt.savefig('comprehensive_results/results_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Saved summary plots to comprehensive_results/results_summary.png")

def create_detailed_comparison(results_df, eval_df):
    """Create detailed heatmap comparison."""
    
    plt.figure(figsize=(15, 8))
    
    # Create pivot table for R² scores
    pivot_r2 = results_df.pivot_table(values='test_score', index='monkey', columns='method')
    
    # Create pivot table for correlations
    pivot_corr = eval_df.pivot_table(values='rho_p', index='monkey', columns='method')
    
    # Plot R² heatmap
    plt.subplot(1, 2, 1)
    sns.heatmap(pivot_r2, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                cbar_kws={'label': 'Test R²'})
    plt.title('Test R² by Monkey and Method')
    plt.xlabel('Method')
    plt.ylabel('Monkey')
    
    # Plot correlation heatmap
    plt.subplot(1, 2, 2)
    sns.heatmap(pivot_corr, annot=True, fmt='.3f', cmap='RdYlBu_r',
                cbar_kws={'label': 'Pearson Correlation'})
    plt.title('Correlation by Monkey and Method')
    plt.xlabel('Method')
    plt.ylabel('Monkey')
    
    plt.tight_layout()
    plt.savefig('comprehensive_results/detailed_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Saved detailed heatmaps to comprehensive_results/detailed_heatmaps.png")

def main():
    """Main analysis function."""
    
    # Load data
    results_df, eval_df = load_and_clean_results()
    
    # Print summary statistics
    print_summary_statistics(results_df, eval_df)
    
    # Create plots
    create_summary_plots(results_df, eval_df)
    create_detailed_comparison(results_df, eval_df)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("Files generated:")
    print("  • comprehensive_results/results_summary.png")
    print("  • comprehensive_results/detailed_heatmaps.png")
    print("  • Individual monkey plots (already saved)")
    print("=" * 70)

if __name__ == "__main__":
    main()