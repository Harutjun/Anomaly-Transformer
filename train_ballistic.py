"""
Training script for Anomaly-Transformer on ballistic trajectory dataset.
Uses the Association Discrepancy with minimax training strategy from ICLR 2022 paper.
"""
import os
import sys
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment, BallisticSegLoader
from BallisticDataset import FEATURE_DIMS, FEATURE_MODE_CONTEXTUAL_FILTER


class color:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def my_kl_loss(p, q):
    """KL divergence loss for association discrepancy."""
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    """Adjust learning rate with decay schedule."""
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Updating learning rate to {lr}')


def get_device():
    """Get the best available device (CUDA if available, otherwise CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"{color.GREEN}[CUDA] Using GPU: {torch.cuda.get_device_name(0)}{color.ENDC}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print(f"{color.WARNING}[CPU] CUDA not available, using CPU (training will be slow){color.ENDC}")
    return device


def get_checkpoint_path(checkpoint_dir, feature_mode):
    """Get checkpoint path based on feature mode."""
    folder = os.path.join(checkpoint_dir, f'AnomalyTransformer_ballistic_{feature_mode}')
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, 'checkpoint.pth')


def save_checkpoint(model, optimizer, epoch, accuracy_list, checkpoint_path, config):
    """Save model checkpoint with config metadata."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy_list': accuracy_list,
        'config': config,
    }, checkpoint_path)
    print(f"{color.GREEN}Saved checkpoint to {checkpoint_path}{color.ENDC}")


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load model checkpoint if exists."""
    if os.path.exists(checkpoint_path):
        print(f"{color.GREEN}Loading checkpoint from {checkpoint_path}{color.ENDC}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        accuracy_list = checkpoint['accuracy_list']
        config = checkpoint.get('config', {})
        return epoch, accuracy_list, config
    return -1, [], {}


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=3, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.val_loss2_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, val_loss2, model, optimizer, epoch, accuracy_list, checkpoint_path, config):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            save_checkpoint(model, optimizer, epoch, accuracy_list, checkpoint_path, config)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            save_checkpoint(model, optimizer, epoch, accuracy_list, checkpoint_path, config)
            self.counter = 0
        self.val_loss_min = min(self.val_loss_min, val_loss)
        self.val_loss2_min = min(self.val_loss2_min, val_loss2)


def validate(model, vali_loader, criterion, k, win_size, device, target_slice=None, target_from_loader=False):
    """Validate model on validation set."""
    model.eval()
    loss_1 = []
    loss_2 = []

    with torch.no_grad():
        for input_data, target_data in vali_loader:
            input_tensor = input_data.float().to(device)
            output, series, prior, _ = model(input_tensor)

            # For contextual_filter mode, compute loss against target windows from loader
            if target_from_loader:
                target_tensor = target_data.float().to(device)
                output_target = output
                input_target = target_tensor
            elif target_slice is not None:
                output_target = output[:, :, target_slice[0]:target_slice[1]]
                input_target = input_tensor[:, :, target_slice[0]:target_slice[1]]
            else:
                output_target = output
                input_target = input_tensor

            # Calculate Association discrepancy
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               win_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       win_size)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = criterion(output_target, input_target)
            loss_1.append((rec_loss - k * series_loss).item())
            loss_2.append((rec_loss + k * prior_loss).item())

    return np.average(loss_1), np.average(loss_2)


def train(args):
    """Main training function."""
    print("=" * 60)
    print("ANOMALY-TRANSFORMER TRAINING FOR BALLISTIC DATASET")
    print("=" * 60)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    # Get device
    device = get_device()

    # Determine feature dimensions
    if args.feature_mode == 'contextual_filter':
        input_c = FEATURE_DIMS[args.feature_mode]['input']
        output_c = FEATURE_DIMS[args.feature_mode]['output']
        target_slice = None
        target_from_loader = True
    else:
        input_c = FEATURE_DIMS[args.feature_mode]
        output_c = input_c
        target_slice = None
        target_from_loader = False

    print(f"\nConfiguration:")
    print(f"  Feature mode: {args.feature_mode}")
    print(f"  Input channels: {input_c}")
    print(f"  Output channels: {output_c}")
    print(f"  Window size: {args.win_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  K (association coef): {args.k}")
    print(f"  Target slice: {target_slice}")
    print(f"  Target from loader: {target_from_loader}")

    # Load data
    print(f"\nLoading data from: {args.data_folder}")
    train_loader = get_loader_segment(
        args.data_folder,
        batch_size=args.batch_size,
        win_size=args.win_size,
        step=args.win_size,  # Non-overlapping for training efficiency
        mode='train',
        dataset='ballistic',
        feature_mode=args.feature_mode,
        train_split_ratio=args.train_split_ratio,
        seed=args.seed
    )

    vali_loader = get_loader_segment(
        args.data_folder,
        batch_size=args.batch_size,
        win_size=args.win_size,
        step=1,
        mode='val',
        dataset='ballistic',
        feature_mode=args.feature_mode,
        train_split_ratio=args.train_split_ratio,
        seed=args.seed
    )

    # Build model
    print(f"\nBuilding Anomaly-Transformer model...")
    model = AnomalyTransformer(
        win_size=args.win_size,
        enc_in=input_c,
        c_out=output_c,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        activation='gelu',
        output_attention=True
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Checkpoint path
    checkpoint_path = get_checkpoint_path(args.checkpoint_dir, args.feature_mode)

    # Load or initialize
    start_epoch = -1
    accuracy_list = []

    if not args.retrain:
        start_epoch, accuracy_list, saved_config = load_checkpoint(model, optimizer, checkpoint_path, device)
        if start_epoch >= 0:
            print(f"Resuming from epoch {start_epoch + 1}")
            if saved_config:
                # Verify config matches
                if saved_config.get('feature_mode') != args.feature_mode:
                    print(f"{color.WARNING}Warning: Feature mode mismatch. Saved: {saved_config.get('feature_mode')}, Current: {args.feature_mode}{color.ENDC}")

    # Config for saving
    config = {
        'feature_mode': args.feature_mode,
        'input_c': input_c,
        'output_c': output_c,
        'target_slice': target_slice,
        'target_source': 'loader' if target_from_loader else 'input',
        'win_size': args.win_size,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'e_layers': args.e_layers,
        'd_ff': args.d_ff,
        'dropout': args.dropout,
        'lr': args.lr,
        'k': args.k,
        'batch_size': args.batch_size,
        'train_split_ratio': args.train_split_ratio,
        'seed': args.seed,
    }

    # Early stopping
    early_stopping = EarlyStopping(patience=3, verbose=True)

    # Training loop
    print(f"\n{'=' * 60}")
    print("Starting training...")
    print(f"{'=' * 60}")

    train_steps = len(train_loader)

    for epoch in range(start_epoch + 1, args.num_epochs):
        epoch_time = time.time()
        model.train()
        loss1_list = []

        pbar = tqdm(enumerate(train_loader), total=train_steps, desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for i, (input_data, labels) in pbar:
            optimizer.zero_grad()
            input_tensor = input_data.float().to(device)

            output, series, prior, _ = model(input_tensor)

            # For contextual_filter mode, compute loss against target windows from loader
            if target_from_loader:
                target_tensor = labels.float().to(device)
                output_target = output
                input_target = target_tensor
            elif target_slice is not None:
                output_target = output[:, :, target_slice[0]:target_slice[1]]
                input_target = input_tensor[:, :, target_slice[0]:target_slice[1]]
            else:
                output_target = output
                input_target = input_tensor

            # Calculate Association discrepancy
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               args.win_size)).detach())) + torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       args.win_size)).detach(),
                               series[u])))
                prior_loss += (torch.mean(my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            args.win_size)),
                    series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(), (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   args.win_size)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = criterion(output_target, input_target)

            # Minimax strategy
            loss1 = rec_loss - args.k * series_loss
            loss2 = rec_loss + args.k * prior_loss

            loss1_list.append(loss1.item())

            loss1.backward(retain_graph=True)
            loss2.backward()
            optimizer.step()

            pbar.set_postfix({'loss1': f'{loss1.item():.4f}', 'loss2': f'{loss2.item():.4f}'})

        train_loss = np.average(loss1_list)

        # Validation
        vali_loss1, vali_loss2 = validate(model, vali_loader, criterion, args.k, args.win_size, device, target_slice, target_from_loader)

        print(f"\nEpoch {epoch+1} | Time: {time.time() - epoch_time:.2f}s | "
              f"Train Loss: {train_loss:.6f} | Val Loss1: {vali_loss1:.6f} | Val Loss2: {vali_loss2:.6f}")

        accuracy_list.append([train_loss, optimizer.param_groups[0]['lr']])

        # Early stopping check
        early_stopping(vali_loss1, vali_loss2, model, optimizer, epoch, accuracy_list, checkpoint_path, config)

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        adjust_learning_rate(optimizer, epoch + 1, args.lr)

    print(f"\n{'=' * 60}")
    print("Training completed!")
    print(f"Best model saved to: {checkpoint_path}")
    print(f"{'=' * 60}")

    return model, checkpoint_path


def main():
    parser = argparse.ArgumentParser(description='Train Anomaly-Transformer on Ballistic Dataset')

    # Data arguments
    parser.add_argument('--data_folder', type=str, default='./dataset/ballistic',
                        help='Path to folder containing .mat trajectory files')
    parser.add_argument('--feature_mode', type=str, default='measurements',
                        choices=['measurements', 'filter_outputs', 'joint', 'contextual_filter'],
                        help='Feature extraction mode')

    # Model arguments
    parser.add_argument('--win_size', type=int, default=100,
                        help='Window size (default: 100 as per paper)')
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

    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory for model checkpoints')

    # Control arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--retrain', action='store_true',
                        help='Force retrain from scratch')

    args = parser.parse_args()

    # Print configuration
    print('\n------------ Options -------------')
    for k, v in sorted(vars(args).items()):
        print(f'{k}: {v}')
    print('-------------- End ----------------\n')

    train(args)


if __name__ == '__main__':
    main()
