# Quick Commands Reference

## Training
`ash
# Activate environment
venv\Scripts\activate

# Train baseline PPO
python src\training\train_baseline.py

# Monitor with TensorBoard
tensorboard --logdir results/logs
`

## Evaluation
`ash
# Evaluate trained model
python src\evaluation\evaluate.py

# Results saved to:
# - results/evaluation_metrics.json
# - results/videos/*.mp4
`

## Project Structure
- src/training/ - Training scripts
- src/evaluation/ - Evaluation scripts
- src/utils/ - Logging & utilities
- config/ - YAML configurations
- esults/ - Models, logs, videos

## Week 1 Status
- [x] Setup (100%)
- [x] Training script
- [x] Evaluation script
- [ ] Run full training (in progress)
- [ ] Generate results

Current training: ~800/500k steps
