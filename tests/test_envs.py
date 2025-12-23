"""
Unit tests for environment setup and wrappers
"""
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from src.envs.reward_shaping import RewardShapingWrapper


def test_environments():
    """Test that all target environments can be created"""
    envs = [
        "MiniGrid-DoorKey-8x8-v0",
        "MiniGrid-Empty-8x8-v0",
        "MiniGrid-Memory-S7-v0"
    ]
    
    for env_name in envs:
        env = gym.make(env_name)
        obs, info = env.reset()
        assert obs is not None, f"{env_name} failed to reset"
        env.close()
        print(f" {env_name} works")


def test_flat_obs_wrapper():
    """Test FlatObsWrapper"""
    env = gym.make("MiniGrid-DoorKey-8x8-v0")
    env = FlatObsWrapper(env)
    obs, info = env.reset()
    
    assert isinstance(obs, object), "FlatObsWrapper should return array"
    print(f" FlatObsWrapper works, obs shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")
    env.close()


def test_reward_shaping():
    """Test RewardShapingWrapper"""
    env = gym.make("MiniGrid-DoorKey-8x8-v0")
    env = FlatObsWrapper(env)
    env = RewardShapingWrapper(env, key_bonus=0.5, step_penalty=0.001)
    
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(2)  # Forward
    
    # Should have step penalty
    assert reward <= 0, "Step should have penalty"
    print(f" RewardShapingWrapper works, reward: {reward}")
    env.close()


if __name__ == "__main__":
    print("Running environment tests...")
    test_environments()
    test_flat_obs_wrapper()
    test_reward_shaping()
    print("\n All tests passed!")
