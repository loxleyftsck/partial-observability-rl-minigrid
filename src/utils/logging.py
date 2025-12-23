"""
Logging utilities for RL experiments
"""
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns


class ExperimentLogger:
    """Logger for tracking RL experiment metrics."""
    
    def __init__(self, experiment_name: str, log_dir: str = "results/logs"):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.log_dir / "metrics.csv"
        self.config_file = self.log_dir / "config.json"
        
        self.metrics = []
        
    def log_config(self, config: Dict[str, Any]):
        """Save experiment configuration."""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f" Config saved to {self.config_file}")
        
    def log_metrics(self, step: int, metrics: Dict[str, float]):
        """Log metrics for a training step."""
        entry = {'step': step, 'timestamp': datetime.now().isoformat()}
        entry.update(metrics)
        self.metrics.append(entry)
        
        # Append to CSV
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=entry.keys())
            if f.tell() == 0:  # File is empty
                writer.writeheader()
            writer.writerow(entry)
    
    def plot_metrics(self, metric_names: List[str], save_path: str = None):
        """Plot training metrics."""
        if not self.metrics:
            print(" No metrics to plot")
            return
            
        fig, axes = plt.subplots(len(metric_names), 1, figsize=(10, 4*len(metric_names)))
        if len(metric_names) == 1:
            axes = [axes]
        
        for ax, metric_name in zip(axes, metric_names):
            steps = [m['step'] for m in self.metrics]
            values = [m.get(metric_name, 0) for m in self.metrics]
            
            ax.plot(steps, values, linewidth=2)
            ax.set_xlabel('Training Steps')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} over Training')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Plot saved to {save_path}")
        else:
            plt.show()
            
    def summary(self) -> Dict:
        """Get experiment summary."""
        if not self.metrics:
            return {}
            
        last_metrics = self.metrics[-1]
        return {
            'experiment': self.experiment_name,
            'total_steps': last_metrics['step'],
            'final_metrics': {k: v for k, v in last_metrics.items() 
                            if k not in ['step', 'timestamp']}
        }


def create_comparison_table(results: List[Dict], save_path: str = None):
    """Create comparison table for multiple experiments."""
    import pandas as pd
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"\n Comparison saved to {save_path}")
    
    return df
