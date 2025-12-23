"""
Reward Shaping Wrapper for Sparse Reward Environments
Adds intermediate rewards to guide learning
"""
import gymnasium as gym


class RewardShapingWrapper(gym.Wrapper):
    """
    Add intermediate rewards to MiniGrid environments.
    
    Modifications:
    - Bonus for picking up key
    - Small step penalty (encourage efficiency)
    - Progress rewards (optional)
    """
    
    def __init__(self, env, key_bonus=0.5, step_penalty=0.001):
        super().__init__(env)
        self.key_bonus = key_bonus
        self.step_penalty = step_penalty
        self.has_key = False
        
    def reset(self, **kwargs):
        self.has_key = False
        return super().reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        shaped_reward = reward
        
        # Bonus for picking up key (one-time reward)
        if hasattr(self.env.unwrapped, 'carrying') and self.env.unwrapped.carrying:
            if not self.has_key:
                shaped_reward += self.key_bonus
                self.has_key = True
        
        # Small penalty per step (encourage efficiency)
        if not terminated and not truncated:
            shaped_reward -= self.step_penalty
        
        return obs, shaped_reward, terminated, truncated, info
