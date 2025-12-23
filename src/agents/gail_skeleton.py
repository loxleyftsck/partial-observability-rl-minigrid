"""
GAIL (Generative Adversarial Imitation Learning) Skeleton
Week 4: Inverse RL - Learn from Demonstrations
"""
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


class Discriminator(nn.Module):
    """
    Discriminator network for GAIL.
    Distinguishes expert demonstrations from agent trajectories.
    """
    
    def __init__(self, obs_dim, act_dim, hidden_size=128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  # Output probability [0,1]
        )
        
    def forward(self, obs, act):
        """
        Forward pass.
        
        Args:
            obs: Observations [batch, obs_dim]
            act: Actions [batch, act_dim]
            
        Returns:
            prob: Probability of being expert [batch, 1]
        """
        x = torch.cat([obs, act], dim=1)
        return self.network(x)


class GAILTrainer:
    """
    GAIL training framework.
    
    Algorithm:
    1. Collect expert demonstrations
    2. Train discriminator to distinguish expert vs agent
    3. Use discriminator as reward for PPO
    4. Iterate
    """
    
    def __init__(self, env, expert_demos, obs_dim, act_dim):
        self.env = env
        self.expert_demos = expert_demos  # List of (obs, act) tuples
        self.discriminator = Discriminator(obs_dim, act_dim)
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=3e-4)
        
        # PPO agent
        self.agent = PPO("MlpPolicy", env, verbose=1)
        
    def train_discriminator(self, agent_samples, n_epochs=5):
        """Train discriminator on expert vs agent samples"""
        for epoch in range(n_epochs):
            # Sample from expert and agent
            expert_batch = self._sample_expert(batch_size=64)
            agent_batch = agent_samples  # From recent agent rollouts
            
            # Labels: 1 for expert, 0 for agent
            expert_labels = torch.ones(len(expert_batch), 1)
            agent_labels = torch.zeros(len(agent_batch), 1)
            
            # Discriminator loss (binary cross-entropy)
            expert_pred = self.discriminator(expert_batch[0], expert_batch[1])
            agent_pred = self.discriminator(agent_batch[0], agent_batch[1])
            
            loss = -torch.mean(torch.log(expert_pred + 1e-8) + torch.log(1 - agent_pred + 1e-8))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return loss.item()
    
    def _sample_expert(self, batch_size):
        """Sample from expert demonstrations"""
        indices = np.random.choice(len(self.expert_demos), batch_size)
        samples = [self.expert_demos[i] for i in indices]
        obs = torch.tensor([s[0] for s in samples])
        act = torch.tensor([s[1] for s in samples])
        return obs, act
    
    def get_gail_reward(self, obs, act):
        """
        Get GAIL reward from discriminator.
        Reward = -log(D(s,a)) where D is discriminator
        """
        with torch.no_grad():
            disc_output = self.discriminator(obs, act)
            reward = -torch.log(disc_output + 1e-8)
        return reward.numpy()
    
    def train(self, total_timesteps=100000):
        """
        Main GAIL training loop.
        
        TODO: Integrate discriminator rewards into PPO
        This is a skeleton - full implementation needs custom PPO callback
        """
        print(" GAIL training skeleton - needs full integration")
        print(f"Expert demos: {len(self.expert_demos)}")
        print(f"Training timesteps: {total_timesteps}")
        
        # Placeholder - would need custom reward wrapper
        self.agent.learn(total_timesteps=total_timesteps)
        
        return self.agent


def collect_expert_demos(env, model_path, n_episodes=10):
    """
    Collect expert demonstrations using trained agent.
    
    Args:
        env: Environment
        model_path: Path to expert model
        n_episodes: Number of episodes to collect
        
    Returns:
        demonstrations: List of (obs, action) tuples
    """
    model = PPO.load(model_path)
    demonstrations = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            demonstrations.append((obs.copy(), action))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
    
    print(f" Collected {len(demonstrations)} demonstrations from {n_episodes} episodes")
    return demonstrations


# NOTE: Full GAIL implementation requires:
# 1. Custom reward wrapper that queries discriminator
# 2. Alternating discriminator/policy training
# 3. Proper batching and sampling
# This skeleton provides the foundation!
