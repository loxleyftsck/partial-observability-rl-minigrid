"""
Video Recording Utilities for RL Agents
Records agent episodes as MP4 videos
"""
import gymnasium as gym
import imageio
from pathlib import Path
from typing import Optional, List
import numpy as np


class VideoRecorder:
    """Record agent interactions as video"""
    
    def __init__(self, save_dir: str = "results/videos", fps: int = 10):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.frames: List[np.ndarray] = []
        
    def capture(self, frame: np.ndarray):
        """Capture a single frame"""
        self.frames.append(frame)
    
    def save(self, filename: str):
        """Save captured frames as video"""
        if not self.frames:
            print(" No frames to save")
            return
            
        video_path = self.save_dir / f"{filename}.mp4"
        imageio.mimsave(str(video_path), self.frames, fps=self.fps)
        print(f" Saved video: {video_path} ({len(self.frames)} frames)")
        
    def reset(self):
        """Clear frames for new recording"""
        self.frames = []


def record_episode(env, model, filename: str, recorder: Optional[VideoRecorder] = None):
    """
    Record a single episode with agent.
    
    Args:
        env: Gym environment
        model: Trained agent model
        filename: Output filename
        recorder: VideoRecorder instance (creates new if None)
    """
    if recorder is None:
        recorder = VideoRecorder()
    
    recorder.reset()
    obs, info = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done:
        # Capture frame
        recorder.capture(env.render())
        
        # Get action and step
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        steps += 1
        
        if steps > 1000:  # Safety limit
            break
    
    # Save video
    recorder.save(filename)
    
    return {
        "reward": total_reward,
        "steps": steps,
        "success": total_reward > 0
    }
