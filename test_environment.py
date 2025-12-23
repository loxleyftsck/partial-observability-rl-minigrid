"""
Test MiniGrid Environment Installation
Run this to verify everything is installed correctly
"""
import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper


def test_environment():
    """Test that MiniGrid environments work."""
    print("="*60)
    print("Testing MiniGrid Installation")
    print("="*60)
    
    # Test environments
    test_envs = [
        "MiniGrid-DoorKey-8x8-v0",
        "MiniGrid-Empty-5x5-v0"
    ]
    
    for env_name in test_envs:
        try:
            print(f"\n Testing {env_name}...")
            env = gym.make(env_name, render_mode="rgb_array")
            env = ImgObsWrapper(env)
            obs, info = env.reset(seed=42)
            
            # Test a few steps
            for _ in range(5):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    obs, info = env.reset()
            
            env.close()
            print(f"   {env_name} works!")
            print(f"  Observation shape: {obs.shape}")
            
        except Exception as e:
            print(f"   Error: {e}")
            return False
    
    print("\n" + "="*60)
    print(" All tests passed! MiniGrid is working correctly.")
    print("="*60)
    print("\nNext steps:")
    print("1. Create a virtual environment: python -m venv venv")
    print("2. Activate it: venv\\Scripts\\activate")
    print("3. Install dependencies: pip install -r requirements.txt")
    print("4. Run training: python src/training/train_baseline.py")
    return True


if __name__ == "__main__":
    test_environment()
