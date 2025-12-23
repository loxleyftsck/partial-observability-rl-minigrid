"""
Evaluation script for trained RL agents
"""
import gymnasium as gym
import numpy as np
from pathlib import Path
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO
import imageio
from typing import Dict, List, Tuple
import json


def evaluate_agent(
    model_path: str,
    env_name: str = "MiniGrid-DoorKey-8x8-v0",
    n_episodes: int = 100,
    render_video: bool = True,
    video_episodes: int = 5,
    seed: int = 42
) -> Dict:
    """
    Evaluate trained agent and collect comprehensive metrics.
    
    Args:
        model_path: Path to trained model
        env_name: Gymnasium environment name
        n_episodes: Number of evaluation episodes
        render_video: Whether to record videos
        video_episodes: Number of episodes to record
        seed: Random seed
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f" Evaluating agent on {env_name}...")
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}\n")
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    env = gym.make(env_name, render_mode="rgb_array")
    env = FlatObsWrapper(env)
    
    # Metrics storage
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    failure_trajectories = []
    
    # Video frames
    video_frames = {i: [] for i in range(video_episodes)} if render_video else {}
    
    # Run evaluation
    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0
        episode_length = 0
        trajectory = []
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            trajectory.append({
                'step': episode_length,
                'action': int(action),
                'reward': float(reward)
            })
            
            # Capture frame for video
            if render_video and episode < video_episodes:
                video_frames[episode].append(env.render())
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Check success (positive reward = reached goal)
        if episode_reward > 0:
            success_count += 1
        else:
            # Save failure trajectory (up to 10)
            if len(failure_trajectories) < 10:
                failure_trajectories.append({
                    'episode': episode,
                    'length': episode_length,
                    'trajectory': trajectory
                })
        
        # Progress update
        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode+1}/{n_episodes}: "
                  f"Reward={episode_reward:.2f}, Length={episode_length}")
    
    # Calculate statistics
    results = {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'std_length': float(np.std(episode_lengths)),
        'min_length': int(np.min(episode_lengths)),
        'max_length': int(np.max(episode_lengths)),
        'success_rate': (success_count / n_episodes) * 100,
        'n_episodes': n_episodes,
        'env_name': env_name,
        'model_path': model_path,
        'all_rewards': episode_rewards,
        'all_lengths': episode_lengths,
        'num_failures': len(failure_trajectories)
    }
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Environment:     {env_name}")
    print(f"Episodes:        {n_episodes}")
    print(f"Success Rate:    {results['success_rate']:.1f}%")
    print(f"Mean Reward:     {results['mean_reward']:.3f}  {results['std_reward']:.3f}")
    print(f"Mean Length:     {results['mean_length']:.1f}  {results['std_length']:.1f} steps")
    print(f"Best Episode:    {results['max_reward']:.2f} reward in {results['min_length']} steps")
    print(f"Worst Episode:   {results['min_reward']:.2f} reward in {results['max_length']} steps")
    print(f"Failures Saved:  {results['num_failures']}")
    print("="*70)
    
    # Save videos
    if render_video:
        video_dir = Path("results/videos")
        video_dir.mkdir(parents=True, exist_ok=True)
        
        for ep_idx, frames in video_frames.items():
            if frames:
                video_path = video_dir / f"eval_episode_{ep_idx}.mp4"
                imageio.mimsave(str(video_path), frames, fps=10)
        
        print(f"\n Saved {len(video_frames)} videos to {video_dir}/")
    
    # Save results to JSON
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics (without large arrays)
    metrics_file = results_dir / "evaluation_metrics.json"
    metrics_only = {k: v for k, v in results.items() 
                   if k not in ['all_rewards', 'all_lengths']}
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_only, f, indent=2)
    print(f" Saved metrics to {metrics_file}")
    
    # Save failure trajectories
    if failure_trajectories:
        failures_file = results_dir / "failure_trajectories.json"
        with open(failures_file, 'w') as f:
            json.dump(failure_trajectories, f, indent=2)
        print(f" Saved failure analysis to {failures_file}")
    
    env.close()
    return results


if __name__ == "__main__":
    # Default evaluation
    MODEL_PATH = "results/models/baseline/ppo_baseline_final"
    
    print("\n Starting Evaluation...\n")
    
    try:
        results = evaluate_agent(
            model_path=MODEL_PATH,
            env_name="MiniGrid-DoorKey-8x8-v0",
            n_episodes=100,
            render_video=True,
            video_episodes=5
        )
        
        print("\n Evaluation complete!")
        
    except FileNotFoundError:
        print(f"\n Model not found: {MODEL_PATH}")
        print("Train a model first with: python src/training/train_baseline.py")
    except Exception as e:
        print(f"\n Error during evaluation: {e}")

