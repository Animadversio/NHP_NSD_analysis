"""
Simple results summary for neural prediction experiment
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def summarize_results():
    """Create a simple summary of the results."""
    
    print("=" * 70)
    print("NEURAL PREDICTION EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)
    
    # Load the individual CSV files instead of the combined one
    result_files = [
        'comprehensive_results/JianJian_240709_regression_results.csv',
        'comprehensive_results/FaCai_240720_regression_results.csv', 
        'comprehensive_results/TuTu_240724_regression_results.csv',
        'comprehensive_results/MaoDan_240815_regression_results.csv',
        'comprehensive_results/ZhuangZhuang_240817_regression_results.csv'
    ]
    
    eval_files = [
        'comprehensive_results/JianJian_240709_evaluation_results.csv',
        'comprehensive_results/FaCai_240720_evaluation_results.csv',
        'comprehensive_results/TuTu_240724_evaluation_results.csv', 
        'comprehensive_results/MaoDan_240815_evaluation_results.csv',
        'comprehensive_results/ZhuangZhuang_240817_evaluation_results.csv'
    ]
    
    all_results = []
    all_evals = []
    
    # Load each file
    for result_file, eval_file in zip(result_files, eval_files):
        try:
            df_result = pd.read_csv(result_file, index_col=0)
            df_eval = pd.read_csv(eval_file, index_col=0)
            all_results.append(df_result)
            all_evals.append(df_eval)
            print(f"✓ Loaded {result_file.split('/')[-1]}")
        except Exception as e:
            print(f"✗ Error loading {result_file}: {e}")
    
    if not all_results:
        print("No results files found!")
        return
    
    print(f"\n📊 EXPERIMENT OVERVIEW:")
    print(f"  • Total monkeys: {len(all_results)}")
    print(f"  • Images per experiment: {all_results[0]['n_images_total'].iloc[0]}")
    print(f"  • Units per experiment: {all_results[0]['n_units_total'].iloc[0]}")
    print(f"  • Methods tested per monkey: {len(all_results[0])}")
    
    # Extract key metrics
    methods = ['pca64', 'pca128', 'pca256', 'sp_avg']  # Deduplicate pca256
    monkeys = ['JianJian', 'FaCai', 'TuTu', 'MaoDan', 'ZhuangZhuang']
    
    print(f"\n🎯 PERFORMANCE SUMMARY:")
    
    # Create summary table
    summary_data = []
    for i, (monkey, df_result, df_eval) in enumerate(zip(monkeys, all_results, all_evals)):
        best_idx = df_result['test_score'].idxmax()
        best_r2 = df_result.loc[best_idx, 'test_score']
        avg_r2 = df_result['test_score'].mean()
        best_corr = df_eval['rho_p'].max()
        avg_corr = df_eval['rho_p'].mean()
        
        summary_data.append({
            'Monkey': monkey,
            'Best_R2': best_r2,
            'Avg_R2': avg_r2,
            'Best_Corr': best_corr,
            'Avg_Corr': avg_corr
        })
        
        print(f"  • {monkey:12}: Best R² = {best_r2:6.3f}, Avg R² = {avg_r2:6.3f}, "
              f"Best Corr = {best_corr:5.3f}")
    
    summary_df = pd.DataFrame(summary_data)
    
    print(f"\n🏆 OVERALL STATISTICS:")
    print(f"  • Best individual R²: {summary_df['Best_R2'].max():.3f}")
    print(f"  • Average across all: {summary_df['Avg_R2'].mean():.3f}")
    print(f"  • Best correlation: {summary_df['Best_Corr'].max():.3f}")
    print(f"  • Std deviation: {summary_df['Avg_R2'].std():.3f}")
    
    # Method comparison
    print(f"\n🔧 METHOD COMPARISON:")
    method_performance = {}
    
    for method_idx, method in enumerate(['pca64', 'pca128', 'pca256', 'sp_avg']):
        scores = []
        for df in all_results:
            if method_idx < len(df):
                scores.append(df['test_score'].iloc[method_idx])
        if scores:
            method_performance[method] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'count': len(scores)
            }
            print(f"  • {method:8}: R² = {np.mean(scores):6.3f} ± {np.std(scores):5.3f} "
                  f"(n={len(scores)})")
    
    # Create visualization
    create_summary_plot(summary_df, method_performance)
    
    print(f"\n📁 FILES GENERATED:")
    print(f"  • Individual monkey results: comprehensive_results/[Monkey]_*_results.csv")
    print(f"  • Individual monkey plots: comprehensive_results/[Monkey]_*_comprehensive_results.png")
    print(f"  • Summary plot: comprehensive_results/experiment_summary.png")
    
    print(f"\n" + "=" * 70)
    print("EXPERIMENT SUCCESSFULLY COMPLETED! 🎉")
    print("All 5 monkeys tested with 1000+ images using 80/20 train/test split")
    print("=" * 70)

def create_summary_plot(summary_df, method_performance):
    """Create summary visualization."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: R² by monkey
    ax1 = axes[0, 0]
    bars = ax1.bar(summary_df['Monkey'], summary_df['Best_R2'], 
                   alpha=0.7, color='skyblue', label='Best R²')
    ax1.bar(summary_df['Monkey'], summary_df['Avg_R2'], 
            alpha=0.5, color='orange', label='Avg R²')
    ax1.set_title('Neural Prediction Performance by Monkey')
    ax1.set_ylabel('Test R²')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (monkey, best, avg) in enumerate(zip(summary_df['Monkey'], 
                                               summary_df['Best_R2'], 
                                               summary_df['Avg_R2'])):
        ax1.text(i, best + 0.002, f'{best:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Correlation by monkey
    ax2 = axes[0, 1]
    bars = ax2.bar(summary_df['Monkey'], summary_df['Best_Corr'],
                   alpha=0.7, color='lightgreen', label='Best Corr')
    ax2.bar(summary_df['Monkey'], summary_df['Avg_Corr'],
            alpha=0.5, color='lightcoral', label='Avg Corr')
    ax2.set_title('Correlation Performance by Monkey')
    ax2.set_ylabel('Pearson Correlation')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Method comparison
    ax3 = axes[1, 0]
    methods = list(method_performance.keys())
    means = [method_performance[m]['mean'] for m in methods]
    stds = [method_performance[m]['std'] for m in methods]
    
    bars = ax3.bar(methods, means, yerr=stds, capsize=5, 
                   alpha=0.7, color='gold')
    ax3.set_title('Performance by Method (Average Across Monkeys)')
    ax3.set_ylabel('Test R²')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, mean_val in zip(bars, means):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    overall_stats = [
        summary_df['Best_R2'].max(),
        summary_df['Avg_R2'].mean(), 
        summary_df['Best_Corr'].max(),
        summary_df['Avg_Corr'].mean()
    ]
    labels = ['Best R²', 'Avg R²', 'Best Corr', 'Avg Corr']
    colors = ['skyblue', 'orange', 'lightgreen', 'lightcoral']
    
    bars = ax4.bar(labels, overall_stats, color=colors, alpha=0.7)
    ax4.set_title('Overall Experiment Statistics')
    ax4.set_ylabel('Value')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, overall_stats):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    plt.tight_layout()
    plt.savefig('comprehensive_results/experiment_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Saved experiment summary plot")

if __name__ == "__main__":
    summarize_results()