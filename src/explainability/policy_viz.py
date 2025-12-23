"""
Policy Visualization Tools
Visualize learned policies and decision boundaries
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns


def visualize_value_function(model, env, save_path=None):
    """
    Visualize learned value function across grid states.
    
    For MiniGrid environments, shows V(s) heatmap.
    """
    # This is environment-specific
    # Placeholder for now - would need to implement state space iteration
    pass


def plot_action_distribution(model, observation, save_path=None):
    """
    Plot action probabilities for given observation.
    
    Shows what actions agent prefers in this state.
    """
    import torch
    
    obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
    
    with torch.no_grad():
        actions, values, log_probs = model.policy(obs_tensor)
        action_probs = torch.softmax(actions, dim=-1).squeeze().numpy()
    
    # Action names (MiniGrid specific)
    action_names = ["Left", "Right", "Forward", "Pickup", "Drop", "Toggle", "Done"]
    
    plt.figure(figsize=(10, 6))
    plt.bar(action_names, action_probs)
    plt.ylabel("Probability")
    plt.title("Action Distribution for Current State")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value
    value = values.item()
    plt.text(0.02, 0.98, f"V(s) = {value:.3f}", 
             transform=plt.gca().transAxes, 
             va="top", fontsize=12, 
             bbox=dict(boxstyle="round", facecolor="wheat"))
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f" Action distribution saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_policy_comparison_plot(results_dict, save_path="results/figures/policy_comparison.png"):
    """
    Compare multiple policies visually.
    
    Args:
        results_dict: {policy_name: {metrics}}
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    policies = list(results_dict.keys())
    
    # Success rates
    success_rates = [results_dict[p]["success_rate"] for p in policies]
    axes[0, 0].bar(policies, success_rates, color=sns.color_palette("Set2"))
    axes[0, 0].set_ylabel("Success Rate (%)")
    axes[0, 0].set_title("Success Rate Comparison")
    axes[0, 0].set_ylim(0, 100)
    
    # Mean rewards
    mean_rewards = [results_dict[p]["mean_reward"] for p in policies]
    axes[0, 1].bar(policies, mean_rewards, color=sns.color_palette("Set2"))
    axes[0, 1].set_ylabel("Mean Reward")
    axes[0, 1].set_title("Mean Reward Comparison")
    
    # Episode lengths
    mean_lengths = [results_dict[p]["mean_length"] for p in policies]
    axes[1, 0].bar(policies, mean_lengths, color=sns.color_palette("Set2"))
    axes[1, 0].set_ylabel("Mean Episode Length")
    axes[1, 0].set_title("Episode Length Comparison")
    
    # Sample efficiency (if available)
    if "sample_efficiency" in results_dict[policies[0]]:
        efficiencies = [results_dict[p]["sample_efficiency"] for p in policies]
        axes[1, 1].bar(policies, efficiencies, color=sns.color_palette("Set2"))
        axes[1, 1].set_ylabel("Sample Efficiency")
        axes[1, 1].set_title("Sample Efficiency Comparison")
    else:
        axes[1, 1].axis("off")
    
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f" Policy comparison saved: {save_path}")
    plt.close()
