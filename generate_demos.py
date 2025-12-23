"""
Demo Generation Script
Create GIFs and videos for portfolio showcase
"""
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from src.envs.reward_shaping import RewardShapingWrapper
from stable_baselines3 import PPO
import imageio
from pathlib import Path


def generate_demo_video(model_path, env_name, output_path, n_episodes=3):
    """
    Generate demo video showing trained agent.
    
    Args:
        model_path: Path to trained model
        env_name: Environment name
        output_path: Where to save video
        n_episodes: Number of episodes to record
    """
    model = PPO.load(model_path)
    env = gym.make(env_name, render_mode="rgb_array")
    env = FlatObsWrapper(env)
    
    all_frames = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        frames = []
        
        while not done:
            frame = env.render()
            frames.append(frame)
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        # Add frames to collection
        all_frames.extend(frames)
        
        # Add separator frame (black screen)
        if ep < n_episodes - 1:
            black_frame = frames[0] * 0  # Same size, all black
            all_frames.extend([black_frame] * 10)  # 1 second separator
    
    # Save as MP4
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, all_frames, fps=10)
    print(f" Demo video saved: {output_path}")
    
    env.close()


def create_comparison_gif(model_paths, env_name, output_path):
    """
    Create side-by-side comparison GIF.
    
    Shows multiple policies on same task.
    """
    # Placeholder - would need PIL or similar for side-by-side
    pass


if __name__ == "__main__":
    # Example usage
    print("Generate demo videos for portfolio")
    print("Edit model paths and run")
    
    # generate_demo_video(
    #     "results/models/baseline/ppo_baseline_final",
    #     "MiniGrid-DoorKey-8x8-v0",
    #     "results/demos/baseline_demo.mp4",
    #     n_episodes=3
    # )
