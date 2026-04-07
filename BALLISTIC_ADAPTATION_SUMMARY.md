# Ballistic Trajectory Dataset Adaptation for Anomaly-Transformer

This document summarizes the adaptations made to the Anomaly-Transformer project (ICLR 2022) to support ballistic trajectory anomaly detection.

## Overview

The original Anomaly-Transformer was designed for benchmark datasets (SMD, MSL, SMAP, PSM). We adapted it to work with a custom ballistic trajectory dataset where:
- Each trajectory is an **independent time-series** (no cross-trajectory windowing)
- Data includes **measurements** (2D) and **filter outputs** (4D)
- Timesteps need **individual anomaly scores** for downstream analysis
- Output: Raw per-timestep scores (threshold computation optional)

---

## Files Created/Modified

### New Files

1. **`LoadTrajectoryData.py`** - MATLAB `.mat` file parser
   - Extracts trajectory data from MATLAB structs
   - Handles measurements, filter outputs, anomaly labels

2. **`BallisticDataset.py`** - PyTorch Dataset for ballistic trajectories
   - 4 feature modes supported
   - `to_windows()` method with trajectory tracking
   - No cross-trajectory windowing

3. **`train_ballistic.py`** - Training script
   - Uses Anomaly-Transformer's minimax training strategy
   - Checkpoint save/load with config metadata
   - CUDA support with device detection
   - Contextual filter mode loss handling

4. **`infer_ballistic.py`** - Inference script
   - Computes Association Discrepancy scores
   - Reconstructs per-timestep scores from windows
   - Saves `.npz` files per trajectory
   - Optional threshold-based metrics

5. **`src/ballistic_parser.py`** - CLI argument parser (optional, standalone scripts have their own)

### Modified Files

1. **`data_factory/data_loader.py`**
   - Added `BallisticSegLoader` class
   - Updated `get_loader_segment()` for `dataset='ballistic'`
   - Trajectory-based train/test split

2. **`model/attn.py`**
   - Fixed CUDA hardcoding in `AnomalyAttention`
   - Uses `register_buffer` for device-aware distance matrix

---

## Feature Modes

| Mode | Input Dims | Output Dims | Description |
|------|-----------|-------------|-------------|
| `measurements` | 2 | 2 | Only radar measurements (x, y) |
| `filter_outputs` | 4 | 4 | Only filter outputs (x, y, vx, vy) |
| `joint` | 6 | 6 | Both measurements and filter outputs |
| `contextual_filter` | 6 | 4 | Uses measurements as context, scores only filter outputs |

### Contextual Filter Mode

Special mode where:
- **Input**: Full 6D (measurements + filter outputs)
- **Loss/Score**: Computed only on 4D filter outputs (indices 2:6)
- **Use case**: Measurements provide temporal context to improve filter output reconstruction

---

## Anomaly-Transformer Key Concepts

### Association Discrepancy (from ICLR 2022 paper)

The core idea is that anomalies should have different association patterns than normal points:
- **Prior association**: Based on temporal distance (Gaussian prior)
- **Series association**: Learned from self-attention

The **Association Discrepancy** is the KL divergence between prior and series associations.

### Minimax Training Strategy

Two losses are computed:
```
loss1 = rec_loss - k * series_loss  (minimize)
loss2 = rec_loss + k * prior_loss   (maximize discrepancy)
```

Both are backpropagated to train the model.

### Inference Score

```python
metric = softmax(-series_loss - prior_loss)  # Association discrepancy
score = metric * reconstruction_loss         # Final anomaly score
```

---

## CLI Commands

### Training

```bash
# Basic training with measurements
python train_ballistic.py \
    --data_folder ./dataset/ballistic \
    --feature_mode measurements \
    --win_size 100 \
    --num_epochs 10 \
    --batch_size 256 \
    --lr 1e-4 \
    --k 3

# Contextual filter mode
python train_ballistic.py \
    --feature_mode contextual_filter \
    --num_epochs 10

# Force retrain
python train_ballistic.py \
    --feature_mode measurements \
    --retrain
```

### Inference

```bash
# Basic inference
python infer_ballistic.py \
    --data_folder ./dataset/ballistic \
    --feature_mode measurements

# With threshold metrics
python infer_ballistic.py \
    --feature_mode measurements \
    --compute_threshold \
    --anormly_ratio 4.0

# Different aggregation
python infer_ballistic.py \
    --feature_mode measurements \
    --aggregation mean
```

### Common Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_folder` | `./dataset/ballistic` | Path to .mat files |
| `--feature_mode` | `measurements` | Feature extraction mode |
| `--win_size` | 100 | Window size (as per paper) |
| `--num_epochs` | 10 | Training epochs |
| `--batch_size` | 256 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--k` | 3 | Association discrepancy coefficient |
| `--train_split_ratio` | 0.8 | Train/test split ratio |
| `--seed` | 42 | Random seed |
| `--checkpoint_dir` | `checkpoints` | Checkpoint directory |
| `--results_dir` | `results` | Results directory |

---

## Output Format

### Checkpoint Format (.pth)

```python
{
    'epoch': int,
    'model_state_dict': dict,
    'optimizer_state_dict': dict,
    'accuracy_list': list,
    'config': {
        'feature_mode': str,
        'input_c': int,
        'output_c': int,
        'target_slice': tuple or None,
        'win_size': int,
        'd_model': int,
        'n_heads': int,
        'e_layers': int,
        'd_ff': int,
        'dropout': float,
        'lr': float,
        'k': int,
        'batch_size': int,
        'train_split_ratio': float,
        'seed': int,
    }
}
```

### Trajectory Results Format (.npz)

```python
{
    'scores': np.array,           # 1D per-timestep scores [T]
    'traj_len': int,              # Valid length (excl. padding)
    'source_filename': str,       # Original .mat filename
    'Y_estimated_state': np.array,# Filter outputs [4, T]
    'X_measurements': np.array,   # Measurements [2, T]
    'anomaly_flags': np.array,    # Ground truth labels [T]
    'cli_args': str,              # JSON of CLI arguments
    'config': str,                # JSON of model config
    'model_type': str,            # 'anomaly_transformer'
    'feature_mode': str,
    'score_aggregation': str,
}
```

---

## Directory Structure

```
Anomaly-Transformer/
├── checkpoints/
│   ├── AnomalyTransformer_ballistic_measurements/
│   │   └── checkpoint.pth
│   ├── AnomalyTransformer_ballistic_filter_outputs/
│   │   └── checkpoint.pth
│   ├── AnomalyTransformer_ballistic_joint/
│   │   └── checkpoint.pth
│   └── AnomalyTransformer_ballistic_contextual_filter/
│       └── checkpoint.pth
├── results/
│   ├── anomaly_transformer_scores_measurements/
│   │   ├── traj1.mat.npz
│   │   └── ...
│   └── anomaly_transformer_scores_contextual_filter/
│       └── ...
├── dataset/
│   └── ballistic/
│       ├── traj1.mat
│       └── ...
├── train_ballistic.py
├── infer_ballistic.py
├── LoadTrajectoryData.py
├── BallisticDataset.py
└── BALLISTIC_ADAPTATION_SUMMARY.md
```

---

## Differences from Original Anomaly-Transformer

1. **Data Loading**: Custom `BallisticSegLoader` instead of file-based loaders
2. **Windowing**: Windows created WITHIN trajectories only (no cross-trajectory)
3. **Train/Test Split**: By trajectory, not by timestep
4. **Output**: Per-trajectory `.npz` files with raw scores (not just metrics)
5. **Contextual Filter**: New mode for using measurements as context
6. **Device Support**: Fixed CUDA hardcoding for CPU compatibility

---

## Comparison with TranAD Adaptation

| Aspect | Anomaly-Transformer | TranAD |
|--------|---------------------|--------|
| Window Size | 100 (paper default) | 10 |
| Score Method | Association Discrepancy | MSE reconstruction |
| Training | Minimax strategy | Two-phase reconstruction |
| Architecture | Transformer encoder | Transformer encoder-decoder |
| Output Format | Same `.npz` structure | Same `.npz` structure |

---

## Known Issues / Notes

1. **Window Size**: Default 100 as per paper. Shorter trajectories are handled with padding.

2. **Memory**: Large window size (100) with batch size 256 may require significant GPU memory. Reduce batch size if OOM.

3. **Early Stopping**: Uses patience=3 with dual loss monitoring.

4. **Score Reconstruction**: Each window's last position score is assigned to the corresponding timestep.

---

## Citation

Original Anomaly-Transformer paper:
```bibtex
@inproceedings{xu2022anomaly,
    title={Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy},
    author={Jiehui Xu and Haixu Wu and Jianmin Wang and Mingsheng Long},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=LzQQ89U1qm_}
}
```

---

*Last updated: January 28, 2026*
