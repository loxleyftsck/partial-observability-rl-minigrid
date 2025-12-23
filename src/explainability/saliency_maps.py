"""
Saliency Maps & Attention Visualization for RL
Understand what agent focuses on
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class SaliencyAnalyzer:
    """
    Generate saliency maps for RL policies.
    Shows which parts of observation influence action decisions.
    """
    
    def __init__(self, model):
        self.model = model
        self.model.policy.eval()
        
    def compute_saliency(self, observation):
        """
        Compute saliency map using gradient-based method.
        
        Args:
            observation: Input observation [obs_dim]
            
        Returns:
            saliency: Gradient magnitudes [obs_dim]
        """
        # Convert to tensor
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        obs_tensor.requires_grad = True
        
        # Forward pass
        with torch.enable_grad():
            actions, values, log_probs = self.model.policy(obs_tensor)
            
            # Get action probabilities
            action_probs = F.softmax(actions, dim=-1)
            max_prob = action_probs.max()
            
            # Backward
            max_prob.backward()
        
        # Get gradients
        saliency = obs_tensor.grad.abs().squeeze().numpy()
        
        return saliency
    
    def visualize_saliency(self, observation, saliency, save_path=None):
        """Visualize saliency map as heatmap"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original observation (if image)
        if len(observation.shape) == 3:  # H x W x C
            ax1.imshow(observation)
            ax1.set_title("Original Observation")
            ax1.axis("off")
            
            # Reshape saliency to match
            saliency_img = saliency.reshape(observation.shape)
            saliency_magnitude = np.mean(saliency_img, axis=-1)
            
            im = ax2.imshow(saliency_magnitude, cmap="hot")
            ax2.set_title("Saliency Map (What Agent Focuses On)")
            ax2.axis("off")
            plt.colorbar(im, ax=ax2)
        else:
            # For flat observations
            ax1.bar(range(len(observation)), observation)
            ax1.set_title("Observation Values")
            ax1.set_xlabel("Feature Index")
            
            ax2.bar(range(len(saliency)), saliency, color="red")
            ax2.set_title("Saliency (Feature Importance)")
            ax2.set_xlabel("Feature Index")
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f" Saliency map saved: {save_path}")
        else:
            plt.show()
        
        plt.close()


class AttentionVisualizer:
    """Visualize attention patterns in LSTM policies"""
    
    def __init__(self, lstm_model):
        self.model = lstm_model
        
    def visualize_hidden_states(self, episode_observations, save_path=None):
        """
        Visualize LSTM hidden state evolution over episode.
        
        Args:
            episode_observations: List of observations from one episode
            save_path: Where to save plot
        """
        hidden_states = []
        
        # Collect hidden states
        self.model.policy.features_extractor.reset_hidden_state()
        
        for obs in episode_observations:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            features = self.model.policy.features_extractor(obs_tensor)
            
            # Get hidden state
            h_state = self.model.policy.features_extractor.hidden_state[0].detach().numpy()
            hidden_states.append(h_state.squeeze())
        
        # Plot
        hidden_states = np.array(hidden_states)  # [timesteps, hidden_dim]
        
        plt.figure(figsize=(12, 6))
        plt.imshow(hidden_states.T, aspect="auto", cmap="viridis")
        plt.colorbar(label="Activation")
        plt.xlabel("Timestep")
        plt.ylabel("Hidden Unit")
        plt.title("LSTM Hidden State Evolution")
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f" Attention plot saved: {save_path}")
        else:
            plt.show()
        
        plt.close()


class FailureTaxonomy:
    """Categorize and analyze agent failures"""
    
    def __init__(self):
        self.categories = {
            "stuck": [],      # Agent stuck in loop
            "timeout": [],    # Reached max steps
            "wrong_action": [],  # Correct state, wrong action
            "exploration": [],   # Poor exploration
            "other": []
        }
    
    def categorize_failure(self, trajectory, category):
        """Add failure trajectory to category"""
        if category in self.categories:
            self.categories[category].append(trajectory)
        else:
            self.categories["other"].append(trajectory)
    
    def generate_report(self, save_path="results/failure_taxonomy.md"):
        """Generate failure analysis report"""
        report = "# Failure Taxonomy\n\n"
        
        total = sum(len(v) for v in self.categories.values())
        
        report += f"**Total Failures**: {total}\n\n"
        report += "## Breakdown\n\n"
        
        for category, trajectories in self.categories.items():
            count = len(trajectories)
            pct = (count / total * 100) if total > 0 else 0
            report += f"- **{category.title()}**: {count} ({pct:.1f}%)\n"
        
        report += "\n## Analysis\n\n"
        
        # Add detailed analysis per category
        for category, trajectories in self.categories.items():
            if trajectories:
                report += f"### {category.title()}\n\n"
                report += f"Examples: {min(3, len(trajectories))}\n\n"
                # Could add trajectory details here
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(report)
        
        print(f" Failure taxonomy saved: {save_path}")
        return report
