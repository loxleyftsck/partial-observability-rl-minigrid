"""
Baseline PPO Training Script - Fixed for MiniGrid
"""
import gymnasium as gym
import yaml
from pathlib import Path
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed


def make_env(env_name: str, seed: int = 0):
    """Create environment with wrappers."""
    def _init():
        env = gym.make(env_name, render_mode="rgb_array")
        env = ImgObsWrapper(env)  # Converts dict obs to image array
        env.reset(seed=seed)
        return env
    set_random_seed(seed)
    return _init


def train_baseline(config_path: str = "config/baseline_ppo.yaml"):
    """Train baseline PPO agent."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Create directories
    Path(config["logging"]["tensorboard_log"]).mkdir(parents=True, exist_ok=True)
    Path(config["logging"]["save_path"]).mkdir(parents=True, exist_ok=True)
    
    # Create vectorized environments (DummyVecEnv is safer than SubprocVecEnv)
    n_envs = config["training"]["n_envs"]
    env = DummyVecEnv([
        make_env(config["environment"]["name"], seed=config["training"]["seed"] + i) 
        for i in range(n_envs)
    ])
    
    # Create eval environment
    eval_env = DummyVecEnv([
        make_env(config["environment"]["name"], seed=config["training"]["seed"] + 1000)
    ])
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(config["logging"]["checkpoint_freq"] // n_envs, 1),
        save_path=config["logging"]["save_path"],
        name_prefix="ppo_baseline"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config["logging"]["save_path"],
        log_path=config["logging"]["tensorboard_log"],
        eval_freq=max(config["logging"]["eval_freq"] // n_envs,1),
        n_eval_episodes=config["logging"]["n_eval_episodes"],
        deterministic=True
    )
    
    # Create model - Use MlpPolicy instead of CnnPolicy because ImgObsWrapper flattens
    # Actually for 7x7x3 images, we should use CnnPolicy
    model = PPO(
        "MlpPolicy",  # Switch to MLP since obs is small 7x7x3
        env,
        learning_rate=config["model"]["learning_rate"],
        n_steps=config["model"]["n_steps"],
        batch_size=config["model"]["batch_size"],
        n_epochs=config["model"]["n_epochs"],
        gamma=config["model"]["gamma"],
        gae_lambda=config["model"]["gae_lambda"],
        clip_range=config["model"]["clip_range"],
        ent_coef=config["model"]["ent_coef"],
        vf_coef=config["model"]["vf_coef"],
        max_grad_norm=config["model"]["max_grad_norm"],
        verbose=1,
        tensorboard_log=config["logging"]["tensorboard_log"],
        seed=config["training"]["seed"]
    )
    
    print(f" Starting training on {config['environment']['name']}")
    print(f"Total timesteps: {config['training']['total_timesteps']:,}")
    print(f"Using {n_envs} parallel environments")
    print(f"Policy: MlpPolicy (for small 7x7x3 obs)")
    print(f"\nMonitor training: tensorboard --logdir {config['logging']['tensorboard_log']}\n")
    
    # Train
    model.learn(
        total_timesteps=config["training"]["total_timesteps"],
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # Save final model
    final_path = Path(config["logging"]["save_path"]) / "ppo_baseline_final"
    model.save(final_path)
    print(f"\n Training complete! Model saved to {final_path}")
    
    env.close()
    eval_env.close()


if __name__ == "__main__":
    train_baseline()
