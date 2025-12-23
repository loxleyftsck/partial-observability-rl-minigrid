"""
Visualization Utilities for RL Results
Creates plots and analysis visualizations
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json


def plot_learning_curve(metrics_file: str, save_path: str = "results/figures/learning_curve.png"):
    """Plot training learning curve from metrics"""
    # This would parse TensorBoard logs or metrics CSV
    # Placeholder for now
    pass


def plot_success_comparison(results: dict, save_path: str = "results/figures/comparison.png"):
    """
    Compare success rates across different approaches.
    
    Args:
        results: Dict with keys like "baseline", "with_shaping", etc.
                 Each value is dict with "success_rate", "mean_reward"
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    methods = list(results.keys())
    success_rates = [results[m]["success_rate"] for m in methods]
    mean_rewards = [results[m]["mean_reward"] for m in methods]
    
    # Success rate bar plot
    ax1.bar(methods, success_rates, color=sns.color_palette("husl", len(methods)))
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("Success Rate Comparison")
    ax1.set_ylim(0, 100)
    
    # Mean reward bar plot
    ax2.bar(methods, mean_rewards, color=sns.color_palette("husl", len(methods)))
    ax2.set_ylabel("Mean Reward")
    ax2.set_title("Mean Reward Comparison")
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f" Saved comparison plot: {save_path}")
    plt.close()


def create_metrics_table(results: dict, save_path: str = "results/metrics_table.md"):
    """Create markdown table of metrics"""
    
    table = "# Experiment Results\n\n"
    table += "| Method | Success Rate | Mean Reward | Mean Steps | Failures |\n"
    table += "|--------|--------------|-------------|------------|----------|\n"
    
    for method, metrics in results.items():
        table += f"| {method} | "
        table += f"{metrics.get('success_rate', 0):.1f}% | "
        table += f"{metrics.get('mean_reward', 0):.3f} | "
        table += f"{metrics.get('mean_length', 0):.1f} | "
        table += f"{100 - metrics.get('success_rate', 0):.1f}% |\n"
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        f.write(table)
    
    print(f" Saved metrics table: {save_path}")
    return table
