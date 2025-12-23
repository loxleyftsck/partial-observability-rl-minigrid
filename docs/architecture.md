# Project Architecture

## System Overview

This project implements an end-to-end Reinforcement Learning system with focus on interpretability and inverse RL.

## Components

### 1. Training Pipeline (`src/training/`)
- **train_baseline.py**: PPO baseline training with multi-env parallelization
- Callbacks: Checkpointing, evaluation, TensorBoard logging
- Configuration: YAML-based hyperparameter management

### 2. Evaluation (`src/evaluation/`)
- Model evaluation scripts
- Metrics calculation (success rate, episode length)
- Failure case collection

### 3. Agents (`src/agents/`)
- PPO baseline (via Stable-Baselines3)
- Future: LSTM-PPO, GAIL (IRL)

### 4. Explainability (`src/explainability/`)
- Policy visualization
- Saliency maps
- Failure analysis

### 5. Utilities (`src/utils/`)
- Logging helpers
- Video recording
- Data processing

## Data Flow

```
Config (YAML) 
     Environment Creation
     Vectorized Envs (4 parallel)
     PPO Training
     Checkpoints + Logs
     Evaluation
     Results Analysis
```

## Technologies

- **RL Framework**: Stable-Baselines3
- **Environment**: Gymnasium + Minigrid
- **Deep Learning**: PyTorch
- **Logging**: TensorBoard
- **Config**: YAML

## Future Extensions

- Week 2: LSTM for memory
- Week 3: Reward shaping
- Week 4: GAIL (Inverse RL)
- Week 5: Explainability tools
