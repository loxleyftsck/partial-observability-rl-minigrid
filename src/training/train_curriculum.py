"""
Curriculum Learning for RL
Progressively increase task difficulty
"""
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from src.envs.reward_shaping import RewardShapingWrapper
from stable_baselines3 import PPO
from pathlib import Path


class CurriculumTrainer:
    """
    Train agent with curriculum learning.
    Start with easy tasks, gradually increase difficulty.
    """
    
    def __init__(self, curriculum_stages, base_config):
        self.stages = curriculum_stages
        self.config = base_config
        self.current_stage = 0
        
    def train_stage(self, stage_name, env_name, timesteps):
        """Train one curriculum stage"""
        print(f"\n Stage {self.current_stage + 1}: {stage_name}")
        print(f"   Environment: {env_name}")
        print(f"   Timesteps: {timesteps:,}")
        
        # Create environment
        env = gym.make(env_name, render_mode="rgb_array")
        env = FlatObsWrapper(env)
        env = RewardShapingWrapper(env)
        
        # Load previous model or create new
        if self.current_stage == 0:
            model = PPO("MlpPolicy", env, **self.config)
        else:
            # Load from previous stage
            prev_model_path = f"results/models/curriculum/stage_{self.current_stage}"
            model = PPO.load(prev_model_path, env=env)
        
        # Train
        model.learn(total_timesteps=timesteps)
        
        # Save
        save_path = Path(f"results/models/curriculum/stage_{self.current_stage + 1}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
        
        print(f" Stage {self.current_stage + 1} complete!")
        self.current_stage += 1
        
        env.close()
        return model


def run_curriculum():
    """
    Example curriculum for MiniGrid.
    
    Stage 1: Empty-5x5 (just reach goal)
    Stage 2: Empty-8x8 (larger grid)
    Stage 3: DoorKey-5x5 (add key/door)
    Stage 4: DoorKey-8x8 (full task)
    """
    curriculum = [
        ("Easy Navigation", "MiniGrid-Empty-5x5-v0", 50000),
        ("Larger Grid", "MiniGrid-Empty-8x8-v0", 100000),
        ("Key & Door (Small)", "MiniGrid-DoorKey-5x5-v0", 200000),
        ("Full Task", "MiniGrid-DoorKey-8x8-v0", 500000),
    ]
    
    config = {
        "learning_rate": 3e-4,
        "n_steps": 128,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "verbose": 1,
    }
    
    trainer = CurriculumTrainer(curriculum, config)
    
    for stage_name, env_name, timesteps in curriculum:
        trainer.train_stage(stage_name, env_name, timesteps)
    
    print("\n Curriculum training complete!")
    print("Final model: results/models/curriculum/stage_4")


if __name__ == "__main__":
    run_curriculum()
