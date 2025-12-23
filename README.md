# Interpretable RL & Inverse RL Portfolio Project

##  Project Overview

End-to-end Reinforcement Learning system focused on:
- **Interpretability**: Understanding agent decisions
- **Inverse RL**: Learning from demonstrations  
- **POMDP**: Handling partial observability with memory

**Target Roles**: AI/ML Engineer, Research Engineer  
**Duration**: 6 weeks (15-20 hours/week)  
**Current Status**: Week 1-2 infrastructure complete

---

##  Quick Start

### 1. Setup Environment
```bash
cd c:\Users\LENOVO\Documents\Minigrid-IRL\minigrid-irl-portfolio
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run Tests
```bash
python tests\test_envs.py
```

### 3. Train Baseline (Week 1)
```bash
python src\training\train_baseline.py
```

### 4. Evaluate
```bash
python src\evaluation\evaluate.py
python analyze_results.py
```

---

##  Project Structure

```
minigrid-irl-portfolio/
 config/                  # YAML configurations
    baseline_ppo.yaml   # Week 1: DoorKey + reward shaping
    memory_lstm.yaml    # Week 2: LSTM for memory
 src/
    agents/             # RL policies
       lstm_policy.py  # LSTM implementation
    envs/               # Environment wrappers
       reward_shaping.py
    training/           # Training scripts
       train_baseline.py
    evaluation/         # Evaluation tools
       evaluate.py
    utils/              # Utilities
        logging.py
        video_recorder.py
        visualizations.py
 tests/                  # Unit tests
    test_envs.py
 results/                # Training outputs
    models/            # Checkpoints
    logs/              # TensorBoard
    videos/            # Agent videos
    figures/           # Plots
 analyze_results.py     # Results comparison

---

##  Progress

**Week 0**:  100% - Infrastructure (9/10 industry standard)  
**Week 1**:  95% - Baseline PPO with reward shaping (ready to execute)  
**Week 2**:  60% - LSTM infrastructure ready  
**Week 3-6**: Planned

---

##  Week 1: Baseline RL

### Challenge
MiniGrid-DoorKey-8x8: Agent must find key, pickup, unlock door, reach goal.

### Solution
- **Algorithm**: PPO with reward shaping
- **Reward Design**: 
  - +0.5 bonus for key pickup
  - -0.001 step penalty (efficiency)
  - +1.0 at goal
- **Expected**: 40-60% success rate

### Run Training
```bash
python src\training\train_baseline.py  # 15-20 mins
python src\evaluation\evaluate.py       # 2 mins
```

---

##  Week 2: Memory & POMDP

### Challenge  
MiniGrid-Memory-S7: Requires remembering information across timesteps.

### Solution
- **Architecture**: LSTM policy  
- **Comparison**: Feedforward vs LSTM
- **Expected**: +20-40% with memory

### Implementation Ready
```bash
# Config already created: config\memory_lstm.yaml
# LSTM policy: src\agents\lstm_policy.py
```

---

##  Debugging Journey

This project includes systematic debugging documentation:
- 2 failed training attempts (0% success)
- Root cause analysis (sparse reward)
- Solution: Reward shaping implementation

See `docs/` folder for complete debugging story (portfolio gold!).

---

##  Metrics to Track

| Metric | Baseline | + Shaping | + Memory | + IRL |
|--------|----------|-----------|----------|-------|
| Success Rate | TBD | TBD | TBD | TBD |
| Mean Steps | TBD | TBD | TBD | TBD |
| Sample Efficiency | TBD | TBD | TBD | TBD |

---

##  Technical Stack

- **Environment**: Gymnasium + MiniGrid
- **RL**: Stable-Baselines3 (PPO)
- **DL**: PyTorch
- **Logging**: TensorBoard
- **Visualization**: Matplotlib, Seaborn

---

##  Documentation

See `.gemini/antigravity/brain/.../` for 21 comprehensive artifacts:
- Planning & roadmaps
- Debugging analysis
- Implementation guides
- Progress reports

---

##  Testing

```bash
python tests\test_envs.py
```

Tests cover:
- Environment creation
- Wrapper functionality  
- Reward shaping behavior

---

##  Learning Outcomes

**Technical**:
- RL implementation (PPO, LSTM)
- Reward design & shaping
- Debugging sparse rewards
- Environment wrappers

**Engineering**:
- Systematic debugging
- Production code structure
- Comprehensive testing
- Documentation best practices

---

##  Current Status

 Infrastructure complete (Week 1-2)  
 Awaiting training execution  
 Ready for results analysis  

**Next**: Execute training, then proceed to Week 2 LSTM experiments!

---

*Built with continuous learning mindset. Failures documented as valuable lessons.* 
