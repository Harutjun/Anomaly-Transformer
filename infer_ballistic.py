"""
Inference script for Anomaly-Transformer on ballistic trajectory dataset.
Computes per-timestep anomaly scores using Association Discrepancy metric.
Outputs raw scores for downstream analysis (threshold computation optional).
"""
import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment, BallisticSegLoader
from BallisticDataset import BallisticDataset, FEATURE_DIMS, FEATURE_MODE_CONTEXTUAL_FILTER


class color:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def my_kl_loss(p, q):
    """KL divergence loss for association discrepancy."""
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"{color.GREEN}[CUDA] Using GPU: {torch.cuda.get_device_name(0)}{color.ENDC}")
    else:
        device = torch.device('cpu')
        print(f"{color.WARNING}[CPU] Using CPU{color.ENDC}")
    return device


def get_checkpoint_path(checkpoint_dir, feature_mode):
    """Get checkpoint path based on feature mode."""
    folder = os.path.join(checkpoint_dir, f'AnomalyTransformer_ballistic_{feature_mode}')
    return os.path.join(folder, 'checkpoint.pth')


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"{color.GREEN}Loading model from {checkpoint_path}{color.ENDC}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint.get('config', {})

    # Extract model parameters from config
    input_c = config.get('input_c', 2)
    output_c = config.get('output_c', input_c)
    win_size = config.get('win_size', 100)
    d_model = config.get('d_model', 512)
    n_heads = config.get('n_heads', 8)
    e_layers = config.get('e_layers', 3)
    d_ff = config.get('d_ff', 512)
    dropout = config.get('dropout', 0.0)

    # Build model
    model = AnomalyTransformer(
        win_size=win_size,
        enc_in=input_c,
        c_out=output_c,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        d_ff=d_ff,
        dropout=dropout,
        activation='gelu',
        output_attention=True
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config


def compute_anomaly_scores(model, data_loader, device, win_size, temperature=50, target_slice=None, use_targets=False):
    """
    Compute anomaly scores using Association Discrepancy metric.

    The score combines reconstruction error with the association discrepancy:
    metric = softmax(-series_loss - prior_loss)
    score = metric * reconstruction_loss

    Args:
        model: Trained AnomalyTransformer model
        data_loader: DataLoader for test data
        device: torch device
        win_size: Window size
        temperature: Temperature for softmax
        target_slice: Tuple (start, end) for contextual_filter mode
        use_targets: If True, use loader-provided targets for reconstruction loss

    Returns:
        scores: numpy array of shape [n_windows, win_size]
        labels: numpy array of shape [n_windows]
        reconstructions: numpy array of shape [n_windows, win_size, n_features]
    """
    model.eval()
    criterion = nn.MSELoss(reduction='none')

    all_scores = []
    all_labels = []
    all_reconstructions = []

    with torch.no_grad():
        for input_data, labels in tqdm(data_loader, desc="Computing scores"):
            input_tensor = input_data.float().to(device)
            output, series, prior, _ = model(input_tensor)

            # For contextual_filter mode, compute loss only on filter outputs
            if use_targets:
                target_tensor = labels.float().to(device)
                output_target = output
                input_target = target_tensor
            elif target_slice is not None:
                output_target = output[:, :, target_slice[0]:target_slice[1]]
                input_target = input_tensor[:, :, target_slice[0]:target_slice[1]]
            else:
                output_target = output
                input_target = input_tensor

            # Reconstruction loss per timestep (mean over features)
            loss = torch.mean(criterion(input_target, output_target), dim=-1)  # [B, L]

            # Calculate Association discrepancy
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                win_size)),
                        series[u].detach()) * temperature

            # Association discrepancy metric
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)  # [B, L]

            # Final score: metric * reconstruction_loss
            scores = metric * loss  # [B, L]

            all_scores.append(scores.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_reconstructions.append(output_target.cpu().numpy())

    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0).flatten()
    all_reconstructions = np.concatenate(all_reconstructions, axis=0)

    return all_scores, all_labels, all_reconstructions


def reconstruct_trajectory_scores(window_scores, traj_ids, timestep_indices, dataset, aggregation='last', window_position='last'):
    """
    Reconstruct per-timestep scores from window scores.

    For windows of size W, each window's score at a position corresponds to
    the timestep indicated by timestep_indices. window_position controls which
    within-window position is used for that timestep.

    Args:
        window_scores: [n_windows, win_size] array of scores
        traj_ids: [n_windows] trajectory index for each window
        timestep_indices: [n_windows] timestep index within trajectory
        dataset: BallisticDataset for metadata
        aggregation: How to aggregate overlapping windows ('last', 'mean', 'max')
        window_position: Which position within each window to use ('last', 'mean', 'max')

    Returns:
        Dictionary mapping trajectory index to scores array
    """
    trajectory_scores = {}
    trajectory_counts = {}

    unique_trajs = np.unique(traj_ids)

    for traj_idx in unique_trajs:
        traj_len = dataset.get_trajectory_length(traj_idx)
        trajectory_scores[traj_idx] = np.zeros(traj_len)
        trajectory_counts[traj_idx] = np.zeros(traj_len)

    # Each window's last position corresponds to timestep_indices
    # But the actual anomaly score is aggregated across all window positions
    for i, (traj_idx, t_idx) in enumerate(zip(traj_ids, timestep_indices)):
        if window_position == 'last':
            score = window_scores[i][-1]
        elif window_position == 'max':
            score = window_scores[i].max()
        else:
            score = window_scores[i].mean()

        if aggregation == 'last':
            trajectory_scores[traj_idx][t_idx] = score
            trajectory_counts[traj_idx][t_idx] = 1
        elif aggregation == 'mean':
            trajectory_scores[traj_idx][t_idx] += score
            trajectory_counts[traj_idx][t_idx] += 1
        elif aggregation == 'max':
            trajectory_scores[traj_idx][t_idx] = max(trajectory_scores[traj_idx][t_idx], score)
            trajectory_counts[traj_idx][t_idx] = 1

    # Average if using mean aggregation
    if aggregation == 'mean':
        for traj_idx in trajectory_scores:
            mask = trajectory_counts[traj_idx] > 0
            trajectory_scores[traj_idx][mask] /= trajectory_counts[traj_idx][mask]

    return trajectory_scores


def reconstruct_trajectory_series(window_outputs, traj_ids, timestep_indices, dataset, aggregation='last', window_position='last'):
    """
    Reconstruct per-timestep output series from windowed model outputs.

    Args:
        window_outputs: [n_windows, win_size, n_features] array
        traj_ids: [n_windows] trajectory index for each window
        timestep_indices: [n_windows] timestep index within trajectory
        dataset: BallisticDataset for metadata
        aggregation: How to aggregate overlapping windows ('last', 'mean', 'max')
        window_position: Which position within each window to use ('last', 'mean', 'max')

    Returns:
        Dictionary mapping trajectory index to array [traj_len, n_features]
    """
    trajectory_outputs = {}
    trajectory_counts = {}

    unique_trajs = np.unique(traj_ids)
    n_features = window_outputs.shape[-1]

    for traj_idx in unique_trajs:
        traj_len = dataset.get_trajectory_length(traj_idx)
        trajectory_outputs[traj_idx] = np.zeros((traj_len, n_features), dtype=window_outputs.dtype)
        trajectory_counts[traj_idx] = np.zeros(traj_len, dtype=np.float32)

    for i, (traj_idx, t_idx) in enumerate(zip(traj_ids, timestep_indices)):
        if window_position == 'last':
            vector = window_outputs[i][-1]
        elif window_position == 'max':
            vector = window_outputs[i].max(axis=0)
        else:
            vector = window_outputs[i].mean(axis=0)

        if aggregation == 'last':
            trajectory_outputs[traj_idx][t_idx] = vector
            trajectory_counts[traj_idx][t_idx] = 1
        elif aggregation == 'mean':
            trajectory_outputs[traj_idx][t_idx] += vector
            trajectory_counts[traj_idx][t_idx] += 1
        elif aggregation == 'max':
            trajectory_outputs[traj_idx][t_idx] = np.maximum(trajectory_outputs[traj_idx][t_idx], vector)
            trajectory_counts[traj_idx][t_idx] = 1

    if aggregation == 'mean':
        for traj_idx in trajectory_outputs:
            mask = trajectory_counts[traj_idx] > 0
            trajectory_outputs[traj_idx][mask] /= trajectory_counts[traj_idx][mask][:, None]

    return trajectory_outputs


def save_trajectory_results(trajectory_scores, test_indices, dataset, output_dir,
                           config, args, reconstructions_dict=None):
    """
    Save per-trajectory results as .npz files.

    Args:
        trajectory_scores: Dict mapping traj_idx to scores array
        test_indices: List of test trajectory indices
        dataset: BallisticDataset
        output_dir: Output directory
        config: Model config dict
        args: CLI arguments
        reconstructions_dict: Optional dict of reconstructions per trajectory
    """
    os.makedirs(output_dir, exist_ok=True)

    for traj_idx in test_indices:
        if traj_idx not in trajectory_scores:
            continue

        filename = dataset.get_source_filename(traj_idx)
        traj_len = dataset.get_trajectory_length(traj_idx)
        labels = dataset.get_labels(traj_idx)[:traj_len].numpy()

        # Get measurements and filter outputs
        measurements = dataset.get_measurements(traj_idx)[:, :traj_len].numpy()
        estimated_states = dataset.get_estimated_states(traj_idx)[:, :traj_len].numpy()

        scores = trajectory_scores[traj_idx]

        # Prepare save dict
        save_dict = {
            'scores': scores,
            'traj_len': traj_len,
            'source_filename': filename,
            'Y_estimated_state': estimated_states,
            'X_measurements': measurements,
            'anomaly_flags': labels,
            'cli_args': json.dumps(vars(args)),
            'config': json.dumps(config),
            'model_type': 'anomaly_transformer',
            'feature_mode': args.feature_mode,
            'score_aggregation': args.score_aggregation,
        }

        if reconstructions_dict and traj_idx in reconstructions_dict:
            save_dict['reconstructions'] = reconstructions_dict[traj_idx]

        # Save as .npz
        output_path = os.path.join(output_dir, f"{filename}.npz")
        np.savez(output_path, **save_dict)

    print(f"{color.GREEN}Saved {len(test_indices)} trajectory results to {output_dir}{color.ENDC}")


def infer(args):
    """Main inference function."""
    print("=" * 60)
    print("ANOMALY-TRANSFORMER INFERENCE FOR BALLISTIC DATASET")
    print("=" * 60)

    device = get_device()

    # Load model
    checkpoint_path = get_checkpoint_path(args.checkpoint_dir, args.feature_mode)
    model, config = load_model(checkpoint_path, device)

    # Get parameters from config
    win_size = config.get('win_size', 100)
    target_slice = config.get('target_slice', None)
    if target_slice:
        target_slice = tuple(target_slice)
    use_targets = config.get('target_source') == 'loader'

    print(f"\nModel configuration:")
    print(f"  Feature mode: {config.get('feature_mode', args.feature_mode)}")
    print(f"  Window size: {win_size}")
    print(f"  Target slice: {target_slice}")
    print(f"  Target from loader: {use_targets}")

    # Load test data
    print(f"\nLoading test data from: {args.data_folder}")

    # Create loader to get dataset reference
    test_loader = get_loader_segment(
        args.data_folder,
        batch_size=args.batch_size,
        win_size=win_size,
        step=1,  # Step=1 for inference
        mode='test',
        dataset='ballistic',
        feature_mode=args.feature_mode,
        train_split_ratio=args.train_split_ratio,
        seed=args.seed
    )

    # Get trajectory info from the loader's dataset
    loader_dataset = test_loader.dataset
    traj_info = loader_dataset.get_trajectory_info()

    test_traj_ids = traj_info['test_traj_ids'].numpy()
    test_timestep_indices = traj_info['test_timestep_indices'].numpy()
    test_indices = traj_info['test_indices']
    dataset = traj_info['dataset']

    print(f"Test trajectories: {len(test_indices)}")
    print(f"Test windows: {len(test_loader.dataset)}")

    # Compute scores
    print(f"\nComputing anomaly scores (temperature={args.temperature})...")
    window_scores, labels, reconstructions = compute_anomaly_scores(
        model, test_loader, device, win_size, args.temperature, target_slice, use_targets
    )

    print(f"Window scores shape: {window_scores.shape}")
    print(f"Window scores stats - Min: {window_scores.min():.6f}, Max: {window_scores.max():.6f}, Mean: {window_scores.mean():.6f}")

    # Reconstruct trajectory-level scores
    print(f"\nReconstructing per-timestep scores (aggregation='{args.aggregation}')...")
    trajectory_scores = reconstruct_trajectory_scores(
        window_scores, test_traj_ids, test_timestep_indices, dataset, args.aggregation, args.window_position
    )

    # Aggregate scores across feature dimensions if needed
    if args.score_aggregation != 'none':
        print(f"Score aggregation: {args.score_aggregation}")
        # Scores are already aggregated (mean over features in compute_anomaly_scores)

    # Reconstruct trajectory-level outputs if requested
    reconstructions_dict = None
    if args.save_reconstructions:
        reconstructions_dict = reconstruct_trajectory_series(
            reconstructions, test_traj_ids, test_timestep_indices, dataset, args.aggregation, args.window_position
        )

    # Save results
    if args.feature_mode == 'contextual_filter':
        output_dir = os.path.join(args.results_dir, 'anomaly_transformer_scores_contextual_filter_targets')
    else:
        output_dir = os.path.join(args.results_dir, f'anomaly_transformer_scores_{args.feature_mode}')
    save_trajectory_results(
        trajectory_scores, test_indices, dataset, output_dir, config, args, reconstructions_dict
    )

    # Compute summary statistics
    all_scores = np.concatenate([trajectory_scores[i] for i in test_indices])
    all_labels = np.concatenate([dataset.get_labels(i)[:dataset.get_trajectory_length(i)].numpy() for i in test_indices])

    print(f"\n{'=' * 60}")
    print("INFERENCE SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total test timesteps: {len(all_scores)}")
    print(f"Score statistics:")
    print(f"  Min: {all_scores.min():.6f}")
    print(f"  Max: {all_scores.max():.6f}")
    print(f"  Mean: {all_scores.mean():.6f}")
    print(f"  Std: {all_scores.std():.6f}")
    print(f"Anomaly ratio in test set: {all_labels.mean() * 100:.2f}%")

    # Optional: Compute threshold-based metrics
    if args.compute_threshold:
        thresh = np.percentile(all_scores, 100 - args.anormly_ratio)
        print(f"\nThreshold (at {args.anormly_ratio}% anomaly ratio): {thresh:.6f}")

        pred = (all_scores > thresh).astype(int)
        gt = all_labels.astype(int)

        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, _ = precision_recall_fscore_support(gt, pred, average='binary', zero_division=0)

        print(f"\nMetrics (without point-adjustment):")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-score:  {f_score:.4f}")

    print(f"\nResults saved to: {output_dir}")
    print(f"{'=' * 60}")

    return trajectory_scores


def main():
    parser = argparse.ArgumentParser(description='Inference with Anomaly-Transformer on Ballistic Dataset')

    # Data arguments
    parser.add_argument('--data_folder', type=str, default='./dataset/ballistic',
                        help='Path to folder containing .mat trajectory files')
    parser.add_argument('--feature_mode', type=str, default='measurements',
                        choices=['measurements', 'filter_outputs', 'joint', 'contextual_filter'],
                        help='Feature extraction mode')

    # Inference arguments
    parser.add_argument('--temperature', type=float, default=50,
                        help='Temperature for association discrepancy metric')
    parser.add_argument('--aggregation', type=str, default='last',
                        choices=['last', 'mean', 'max'],
                        help='How to aggregate overlapping windows')
    parser.add_argument('--window_position', type=str, default='last',
                        choices=['last', 'mean', 'max'],
                        help='Which position within each window to use for a timestep score')
    parser.add_argument('--save_reconstructions', action='store_true',
                        help='Save per-timestep model reconstructions for plotting')
    parser.add_argument('--score_aggregation', type=str, default='sum',
                        choices=['sum', 'mean', 'max', 'none'],
                        help='How to aggregate scores across features')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for inference')

    # Threshold arguments (optional)
    parser.add_argument('--compute_threshold', action='store_true',
                        help='Compute threshold-based metrics')
    parser.add_argument('--anormly_ratio', type=float, default=4.0,
                        help='Anomaly ratio for threshold computation')

    # Data split arguments (must match training)
    parser.add_argument('--train_split_ratio', type=float, default=0.8,
                        help='Train split ratio (must match training)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (must match training)')

    # Directory arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory containing model checkpoints')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory for inference results')

    args = parser.parse_args()

    # Print configuration
    print('\n------------ Options -------------')
    for k, v in sorted(vars(args).items()):
        print(f'{k}: {v}')
    print('-------------- End ----------------\n')

    infer(args)


if __name__ == '__main__':
    main()
