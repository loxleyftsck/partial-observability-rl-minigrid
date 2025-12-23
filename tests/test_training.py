"""
Integration tests for training pipeline
"""
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from src.envs.reward_shaping import RewardShapingWrapper
from stable_baselines3 import PPO
import tempfile
import shutil


def test_training_pipeline():
    """Test full training pipeline with minimal steps"""
    print("Testing training pipeline...")
    
    # Create environment
    env = gym.make("MiniGrid-Empty-5x5-v0")
    env = FlatObsWrapper(env)
    env = RewardShapingWrapper(env)
    
    # Train for minimal steps
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=100)  # Just 100 steps for testing
    
    # Test saving/loading
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = f"{tmpdir}/test_model"
        model.save(save_path)
        
        loaded_model = PPO.load(save_path, env=env)
        assert loaded_model is not None
    
    env.close()
    print(" Training pipeline test passed")


def test_evaluation():
    """Test evaluation on random policy"""
    print("Testing evaluation...")
    
    env = gym.make("MiniGrid-Empty-5x5-v0")
    env = FlatObsWrapper(env)
    
    # Random policy
    obs, info = env.reset()
    done = False
    steps = 0
    
    while not done and steps < 50:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    
    assert steps > 0
    env.close()
    print(" Evaluation test passed")


if __name__ == "__main__":
    test_training_pipeline()
    test_evaluation()
    print("\n All integration tests passed!")
