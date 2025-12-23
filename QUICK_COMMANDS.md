#  Quick Commands Cheat Sheet

## Training
```powershell
# Week 1: Baseline + Reward Shaping
python src\training\train_baseline.py

# Week 2: LSTM (edit config first)
python src\training\train_baseline.py

# Week 3: Curriculum
python src\training\train_curriculum.py
```

## Evaluation
```powershell
python src\evaluation\evaluate.py
python analyze_results.py
```

## Testing
```powershell
python tests\test_envs.py
python tests\test_training.py
```

## Demos
```powershell
python generate_demos.py
```

## TensorBoard
```powershell
tensorboard --logdir results\logs
```
