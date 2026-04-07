# Quick Start Guide: Anomaly-Transformer for Ballistic Trajectories

## Setup

1. Ensure you have Python 3.6+ with PyTorch >= 1.4.0 installed
2. Install required packages:
   ```bash
   pip install scipy numpy pandas scikit-learn tqdm matplotlib
   ```

3. Place your ballistic trajectory `.mat` files in `./dataset/ballistic/`

## Training

### Basic Training (Measurements Mode)

```bash
python train_ballistic.py --data_folder ./dataset/ballistic --feature_mode measurements
```

### All Feature Modes

```bash
# 2D measurements only
python train_ballistic.py --feature_mode measurements

# 4D filter outputs only  
python train_ballistic.py --feature_mode filter_outputs

# 6D joint (measurements + filter outputs)
python train_ballistic.py --feature_mode joint

# 6D input, 4D output (contextual filter)
python train_ballistic.py --feature_mode contextual_filter
```

### Custom Parameters

```bash
python train_ballistic.py \
    --data_folder ./dataset/ballistic \
    --feature_mode measurements \
    --win_size 100 \
    --num_epochs 10 \
    --batch_size 256 \
    --lr 1e-4 \
    --k 3
```

## Inference

### Basic Inference

```bash
python infer_ballistic.py --data_folder ./dataset/ballistic --feature_mode measurements
```

### With Threshold Metrics

```bash
python infer_ballistic.py \
    --feature_mode measurements \
    --compute_threshold \
    --anormly_ratio 4.0
```

## Output Files

Results are saved to `results/anomaly_transformer_scores_{feature_mode}/`:
- `{trajectory_filename}.npz` - Per-trajectory scores and metadata

Each `.npz` file contains:
- `scores`: Per-timestep anomaly scores
- `traj_len`: Valid trajectory length
- `anomaly_flags`: Ground truth labels
- `X_measurements`: Original measurements
- `Y_estimated_state`: Filter outputs

## Loading Results

```python
import numpy as np

# Load trajectory results
data = np.load('results/anomaly_transformer_scores_measurements/traj1.mat.npz', allow_pickle=True)

scores = data['scores']
labels = data['anomaly_flags']
traj_len = int(data['traj_len'])

# Use only valid timesteps (exclude padding)
valid_scores = scores[:traj_len]
valid_labels = labels[:traj_len]
```

## Checkpoints

Checkpoints are saved to `checkpoints/AnomalyTransformer_ballistic_{feature_mode}/checkpoint.pth`

To resume training, just run the same command (checkpoint is loaded automatically).

To force retrain:
```bash
python train_ballistic.py --feature_mode measurements --retrain
```

## Plotting Results

Plot model outputs vs ground truth and anomaly scores for a trajectory:

```bash
python plot_ballistic_results.py \
    --results_dir results/anomaly_transformer_scores_measurements \
    --trajectory traj1.mat \
    --data_folder ./dataset/ballistic \
    --save_path plots/traj1.png
```

Use `--model_output measurements` or `--model_output estimated_states` to switch which series is plotted.

