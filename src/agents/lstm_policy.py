"""
LSTM Policy Wrapper for Recurrent RL
Handles partial observability with memory
"""
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces
import numpy as np


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    LSTM-based feature extractor for sequential observations.
    
    Processes sequences of observations to maintain memory.
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, lstm_hidden_size: int = 128):
        super().__init__(observation_space, features_dim)
        
        # Input size from observation space
        self.input_size = int(np.prod(observation_space.shape))
        self.lstm_hidden_size = lstm_hidden_size
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # Output projection
        self.linear = nn.Linear(lstm_hidden_size, features_dim)
        
        # Hidden state (will be maintained across steps)
        self.hidden_state = None
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM.
        
        Args:
            observations: Batch of observations [batch_size, obs_dim]
            
        Returns:
            features: Extracted features [batch_size, features_dim]
        """
        batch_size = observations.shape[0]
        
        # Flatten observations if needed
        if len(observations.shape) > 2:
            observations = observations.reshape(batch_size, -1)
        
        # Add sequence dimension [batch_size, 1, obs_dim]
        observations = observations.unsqueeze(1)
        
        # Initialize hidden state if needed
        if self.hidden_state is None or self.hidden_state[0].shape[1] != batch_size:
            self.hidden_state = (
                torch.zeros(1, batch_size, self.lstm_hidden_size).to(observations.device),
                torch.zeros(1, batch_size, self.lstm_hidden_size).to(observations.device)
            )
        
        # LSTM forward
        lstm_out, self.hidden_state = self.lstm(observations, self.hidden_state)
        
        # Get last output and project
        lstm_out = lstm_out[:, -1, :]  # [batch_size, lstm_hidden_size]
        features = self.linear(lstm_out)
        
        return features
    
    def reset_hidden_state(self):
        """Reset LSTM hidden state (call at episode reset)"""
        self.hidden_state = None


def create_lstm_policy(env, **policy_kwargs):
    """
    Create PPO policy with LSTM feature extractor.
    
    Usage:
        from stable_baselines3 import PPO
        policy_kwargs = dict(
            features_extractor_class=LSTMFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256, lstm_hidden_size=128),
        )
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs)
    """
    return {
        "features_extractor_class": LSTMFeatureExtractor,
        "features_extractor_kwargs": {
            "features_dim": 256,
            "lstm_hidden_size": 128,
        }
    }
