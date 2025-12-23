"""
Analysis Script - Compare Different Approaches
"""
import json
from pathlib import Path
from src.utils.visualizations import plot_success_comparison, create_metrics_table


def analyze_results(results_dir: str = "results"):
    """
    Analyze and compare all experiment results.
    
    Expected files:
    - results/baseline_metrics.json
    - results/with_shaping_metrics.json
    - results/memory_lstm_metrics.json (Week 2)
    """
    results_path = Path(results_dir)
    
    # Load all available results
    experiments = {}
    
    for metrics_file in results_path.glob("*_metrics.json"):
        exp_name = metrics_file.stem.replace("_metrics", "")
        with open(metrics_file) as f:
            experiments[exp_name] = json.load(f)
    
    print(f"\n Found {len(experiments)} experiments:\n")
    
    # Create comparison table
    table = create_metrics_table(experiments)
    print(table)
    
    # Create plots
    if len(experiments) > 1:
        plot_success_comparison(experiments)
        print("\n Comparison plots saved!")
    
    return experiments


if __name__ == "__main__":
    results = analyze_results()
    
    print("\n Analysis complete!")
    print(f"Check: results/metrics_table.md")
    print(f"Check: results/figures/comparison.png")
