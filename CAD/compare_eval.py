"""
Comparison script for evaluating AD vs CAD performance.

This script loads evaluation results from both models and creates visualizations
comparing their per-episode rewards averaged across test environments.

Usage:
    python compare_eval.py
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_eval_results(result_path):
    """Load evaluation results from .npy file."""
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Evaluation result not found: {result_path}")
    return np.load(result_path)


def plot_comparison(ad_results, cad_results, save_path=None, figsize=(14, 10)):
    """
    Plot comparison of AD vs CAD evaluation results.
    
    Args:
        ad_results: numpy array of shape (n_envs, n_episodes)
        cad_results: numpy array of shape (n_envs, n_episodes)
        save_path: path to save the figure (optional)
        figsize: figure size tuple
    """
    n_envs = ad_results.shape[0]
    n_episodes = ad_results.shape[1]
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    # --- Plot 1: Average reward across all environments per episode ---
    ax1 = axes[0]
    episodes = np.arange(1, n_episodes + 1)
    
    ad_mean = ad_results.mean(axis=0)
    cad_mean = cad_results.mean(axis=0)
    
    # Apply smoothing for better visualization
    window_size = 10
    ad_smooth = np.convolve(ad_mean, np.ones(window_size)/window_size, mode='valid')
    cad_smooth = np.convolve(cad_mean, np.ones(window_size)/window_size, mode='valid')
    smooth_episodes = episodes[window_size-1:]
    
    ax1.plot(smooth_episodes, ad_smooth, label=f'AD (n_transit=240)', color='blue', linewidth=2)
    ax1.plot(smooth_episodes, cad_smooth, label=f'CAD (n_transit=60)', color='red', linewidth=2)
    
    # Add raw data with transparency
    ax1.plot(episodes, ad_mean, alpha=0.2, color='blue', linewidth=0.5)
    ax1.plot(episodes, cad_mean, alpha=0.2, color='red', linewidth=0.5)
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Average Reward', fontsize=12)
    ax1.set_title('Average Reward per Episode (Averaged Across Test Environments)', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([1, n_episodes])
    
    # --- Plot 2: Cumulative average reward ---
    ax2 = axes[1]
    ad_cumavg = np.cumsum(ad_mean) / np.arange(1, n_episodes + 1)
    cad_cumavg = np.cumsum(cad_mean) / np.arange(1, n_episodes + 1)
    
    ax2.plot(episodes, ad_cumavg, label='AD', color='blue', linewidth=2)
    ax2.plot(episodes, cad_cumavg, label='CAD', color='red', linewidth=2)
    
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Cumulative Average Reward', fontsize=12)
    ax2.set_title('Cumulative Average Reward Over Episodes', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([1, n_episodes])
    
    # --- Plot 3: Per-environment comparison (bar chart) ---
    ax3 = axes[2]
    env_indices = np.arange(n_envs)
    bar_width = 0.35
    
    ad_env_means = ad_results.mean(axis=1)
    cad_env_means = cad_results.mean(axis=1)
    
    bars1 = ax3.bar(env_indices - bar_width/2, ad_env_means, bar_width, 
                    label='AD', color='blue', alpha=0.7)
    bars2 = ax3.bar(env_indices + bar_width/2, cad_env_means, bar_width, 
                    label='CAD', color='red', alpha=0.7)
    
    ax3.set_xlabel('Environment', fontsize=12)
    ax3.set_ylabel('Mean Reward', fontsize=12)
    ax3.set_title('Mean Reward per Environment', fontsize=14)
    ax3.set_xticks(env_indices)
    ax3.set_xticklabels([f'Env {i}' for i in range(n_envs)])
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax3.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax3.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return fig


def plot_learning_curves(ad_results, cad_results, save_path=None, figsize=(16, 8)):
    """
    Plot learning curves for each environment separately.
    
    Args:
        ad_results: numpy array of shape (n_envs, n_episodes)
        cad_results: numpy array of shape (n_envs, n_episodes)
        save_path: path to save the figure (optional)
        figsize: figure size tuple
    """
    n_envs = ad_results.shape[0]
    n_episodes = ad_results.shape[1]
    episodes = np.arange(1, n_episodes + 1)
    
    # Create subplots grid
    n_cols = 4
    n_rows = (n_envs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    window_size = 20
    
    for env_idx in range(n_envs):
        ax = axes[env_idx]
        
        ad_env = ad_results[env_idx]
        cad_env = cad_results[env_idx]
        
        # Smoothing
        ad_smooth = np.convolve(ad_env, np.ones(window_size)/window_size, mode='valid')
        cad_smooth = np.convolve(cad_env, np.ones(window_size)/window_size, mode='valid')
        smooth_episodes = episodes[window_size-1:]
        
        ax.plot(smooth_episodes, ad_smooth, label='AD', color='blue', linewidth=1.5)
        ax.plot(smooth_episodes, cad_smooth, label='CAD', color='red', linewidth=1.5)
        
        # Add raw data with transparency
        ax.plot(episodes, ad_env, alpha=0.15, color='blue', linewidth=0.3)
        ax.plot(episodes, cad_env, alpha=0.15, color='red', linewidth=0.3)
        
        ax.set_title(f'Env {env_idx}', fontsize=10)
        ax.set_xlabel('Episode', fontsize=8)
        ax.set_ylabel('Reward', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=7)
        
        if env_idx == 0:
            ax.legend(fontsize=8, loc='lower right')
    
    # Hide unused subplots
    for idx in range(n_envs, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Learning Curves per Environment: AD vs CAD', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    
    return fig


def print_statistics(ad_results, cad_results):
    """Print detailed statistics comparing AD and CAD."""
    n_envs = ad_results.shape[0]
    n_episodes = ad_results.shape[1]
    
    print("=" * 70)
    print("EVALUATION COMPARISON: AD vs CAD")
    print("=" * 70)
    
    print(f"\nDataset: {n_envs} environments, {n_episodes} episodes each")
    print("-" * 70)
    
    print("\n{:<15} {:>15} {:>15} {:>15}".format(
        "Metric", "AD", "CAD", "Difference"))
    print("-" * 70)
    
    # Overall statistics
    ad_overall_mean = ad_results.mean()
    cad_overall_mean = cad_results.mean()
    diff_mean = cad_overall_mean - ad_overall_mean
    print("{:<15} {:>15.3f} {:>15.3f} {:>+15.3f}".format(
        "Overall Mean", ad_overall_mean, cad_overall_mean, diff_mean))
    
    ad_overall_std = ad_results.std()
    cad_overall_std = cad_results.std()
    diff_std = cad_overall_std - ad_overall_std
    print("{:<15} {:>15.3f} {:>15.3f} {:>+15.3f}".format(
        "Overall Std", ad_overall_std, cad_overall_std, diff_std))
    
    # Per-environment comparison
    print("\n" + "-" * 70)
    print("\nPer-Environment Mean Rewards:")
    print("-" * 70)
    print("{:<10} {:>12} {:>12} {:>12} {:>12}".format(
        "Env", "AD Mean", "CAD Mean", "Diff", "Ratio"))
    print("-" * 70)
    
    for env_idx in range(n_envs):
        ad_mean = ad_results[env_idx].mean()
        cad_mean = cad_results[env_idx].mean()
        diff = cad_mean - ad_mean
        ratio = cad_mean / ad_mean if ad_mean != 0 else float('inf')
        print("{:<10} {:>12.3f} {:>12.3f} {:>+12.3f} {:>12.2%}".format(
            f"Env {env_idx}", ad_mean, cad_mean, diff, ratio))
    
    # Learning speed analysis
    print("\n" + "-" * 70)
    print("\nLearning Speed Analysis (Episodes to reach threshold):")
    print("-" * 70)
    
    # Find episodes to reach certain performance thresholds
    ad_env_means = ad_results.mean(axis=0)
    cad_env_means = cad_results.mean(axis=0)
    
    final_ad_perf = ad_env_means[-50:].mean()  # Last 50 episodes
    final_cad_perf = cad_env_means[-50:].mean()
    
    print(f"Final performance (last 50 episodes): AD={final_ad_perf:.3f}, CAD={final_cad_perf:.3f}")
    
    for threshold_pct in [0.5, 0.75, 0.9]:
        threshold = final_ad_perf * threshold_pct
        
        ad_reached = np.where(np.convolve(ad_env_means, np.ones(20)/20, mode='valid') >= threshold)[0]
        cad_reached = np.where(np.convolve(cad_env_means, np.ones(20)/20, mode='valid') >= threshold)[0]
        
        ad_episode = ad_reached[0] + 20 if len(ad_reached) > 0 else "Never"
        cad_episode = cad_reached[0] + 20 if len(cad_reached) > 0 else "Never"
        
        print(f"  {threshold_pct*100:.0f}% of AD final ({threshold:.1f}): AD={ad_episode}, CAD={cad_episode}")


def main():
    parser = argparse.ArgumentParser(description='Compare AD vs CAD evaluation results')
    parser.add_argument('--ad_dir', type=str, default='./runs/AD-darkroom-seed0',
                       help='Directory containing AD evaluation results')
    parser.add_argument('--cad_dir', type=str, default='./runs/CAD-darkroom-seed0',
                       help='Directory containing CAD evaluation results')
    parser.add_argument('--output_dir', type=str, default='./runs/comparison',
                       help='Directory to save comparison figures')
    parser.add_argument('--no_show', action='store_true',
                       help='Do not display plots (only save)')
    args = parser.parse_args()
    
    # Load results
    ad_result_path = os.path.join(args.ad_dir, 'eval_result.npy')
    cad_result_path = os.path.join(args.cad_dir, 'eval_result.npy')
    
    print(f"Loading AD results from: {ad_result_path}")
    print(f"Loading CAD results from: {cad_result_path}")
    
    ad_results = load_eval_results(ad_result_path)
    cad_results = load_eval_results(cad_result_path)
    
    print(f"AD results shape: {ad_results.shape}")
    print(f"CAD results shape: {cad_results.shape}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print statistics
    print_statistics(ad_results, cad_results)
    
    # Plot comparisons
    if not args.no_show:
        print("\nGenerating comparison plots...")
        
        # Main comparison plot
        plot_comparison(
            ad_results, cad_results,
            save_path=os.path.join(args.output_dir, 'comparison_main.png')
        )
        
        # Per-environment learning curves
        plot_learning_curves(
            ad_results, cad_results,
            save_path=os.path.join(args.output_dir, 'comparison_per_env.png')
        )
    else:
        # Save without showing
        import matplotlib
        matplotlib.use('Agg')
        
        plot_comparison(
            ad_results, cad_results,
            save_path=os.path.join(args.output_dir, 'comparison_main.png')
        )
        plot_learning_curves(
            ad_results, cad_results,
            save_path=os.path.join(args.output_dir, 'comparison_per_env.png')
        )
    
    print(f"\nComparison complete! Figures saved to {args.output_dir}")


if __name__ == '__main__':
    main()
