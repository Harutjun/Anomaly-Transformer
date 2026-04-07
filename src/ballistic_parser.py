"""
Shared CLI argument parser for Anomaly-Transformer ballistic adaptation.
Used by train_ballistic.py and infer_ballistic.py.
"""
import argparse
import os


def get_parser():
    """Create and return the argument parser for ballistic scripts."""
    parser = argparse.ArgumentParser(description='Anomaly-Transformer for Ballistic Trajectory Anomaly Detection')

    # Data arguments
    parser.add_argument('--data_folder', type=str, default='./dataset/ballistic',
                        help='Path to folder containing .mat trajectory files')
    parser.add_argument('--feature_mode', type=str, default='measurements',
                        choices=['measurements', 'filter_outputs', 'joint', 'contextual_filter'],
                        help='Feature extraction mode')

    # Model arguments
    parser.add_argument('--win_size', type=int, default=100,
                        help='Window size for Anomaly-Transformer (default: 100 as per paper)')
    parser.add_argument('--input_c', type=int, default=None,
                        help='Input channels (auto-detected from feature_mode if not specified)')
    parser.add_argument('--output_c', type=int, default=None,
                        help='Output channels (auto-detected from feature_mode if not specified)')
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=3,
                        help='Number of encoder layers')
    parser.add_argument('--d_ff', type=int, default=512,
                        help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate')

    # Training arguments
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--k', type=int, default=3,
                        help='Association discrepancy coefficient')
    parser.add_argument('--train_split_ratio', type=float, default=0.8,
                        help='Ratio of trajectories for training')

    # Inference arguments
    parser.add_argument('--anormly_ratio', type=float, default=4.0,
                        help='Anomaly ratio for threshold computation (percentage)')
    parser.add_argument('--temperature', type=float, default=50,
                        help='Temperature for association discrepancy metric')
    parser.add_argument('--score_aggregation', type=str, default='sum',
                        choices=['sum', 'mean', 'max'],
                        help='How to aggregate scores across feature dimensions')

    # Checkpoint and output arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory for model checkpoints')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory for inference results')
    parser.add_argument('--model_save_path', type=str, default='checkpoints',
                        help='Path to save model checkpoints')

    # Control arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--retrain', action='store_true',
                        help='Force retrain even if checkpoint exists')
    parser.add_argument('--test', action='store_true',
                        help='Run inference only (skip training)')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'],
                        help='Mode: train or test')

    return parser


def parse_args():
    """Parse command line arguments."""
    parser = get_parser()
    args = parser.parse_args()

    # Auto-detect input/output channels based on feature mode
    feature_dims = {
        'measurements': 2,
        'filter_outputs': 4,
        'joint': 6,
        'contextual_filter': {'input': 2, 'output': 4},
    }

    if args.input_c is None:
        if args.feature_mode == 'contextual_filter':
            args.input_c = feature_dims['contextual_filter']['input']
        else:
            args.input_c = feature_dims[args.feature_mode]

    if args.output_c is None:
        if args.feature_mode == 'contextual_filter':
            args.output_c = feature_dims['contextual_filter']['output']
        else:
            args.output_c = feature_dims[args.feature_mode]

    # Set target slice for contextual_filter mode
    if args.feature_mode == 'contextual_filter':
        args.target_slice = None
    else:
        args.target_slice = None

    return args


# Parse args when module is imported
args = parse_args()


if __name__ == '__main__':
    # Print parsed arguments
    args = parse_args()
    print('------------ Options -------------')
    for k, v in sorted(vars(args).items()):
        print(f'{k}: {v}')
    print('-------------- End ----------------')
