"""Quick diagnostic to check agent behavior"""
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
import numpy as np

# Load model
model = PPO.load("results/models/baseline/ppo_baseline_final")

# Create env
env = gym.make("MiniGrid-DoorKey-8x8-v0", render_mode="human")
env = ImgObsWrapper(env)

print("="*60)
print("DIAGNOSTIC: Watching agent for 3 episodes")
print("="*60)

for ep in range(3):
    obs, info = env.reset()
    print(f"\nEpisode {ep+1}:")
    print(f"  Obs shape: {obs.shape}")
    print(f"  Obs sample: {obs.flatten()[:10]}")  # First 10 values
    
    done = False
    steps = 0
    total_reward = 0
    
    while not done and steps < 50:  # Watch first 50 steps
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        steps += 1
        
        if steps % 10 == 0:
            print(f"  Step {steps}: action={action}, reward={reward}, total={total_reward}")
    
    print(f"  Final: {steps} steps, reward={total_reward}, done={done}")

env.close()
print("\nDiagnostic complete. Did you see the agent moving intelligently?")
