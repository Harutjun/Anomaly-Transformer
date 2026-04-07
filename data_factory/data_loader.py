import torch
import os
import sys
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Add parent directory to path for BallisticDataset import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from BallisticDataset import (
    BallisticDataset,
    FEATURE_MODE_MEASUREMENTS,
    FEATURE_MODE_FILTER_OUTPUTS,
    FEATURE_MODE_JOINT,
    FEATURE_MODE_CONTEXTUAL_FILTER,
    FEATURE_DIMS,
)


class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class BallisticSegLoader(object):
    """
    Data loader for ballistic trajectory dataset.

    Loads trajectories, splits by trajectory (not timestep), and creates windows
    WITHIN each trajectory (no cross-trajectory windows).

    Args:
        data_path: Path to folder containing .mat trajectory files
        win_size: Window size for Anomaly-Transformer
        step: Step size for sliding window
        mode: 'train', 'val', 'test', or 'thre'
        feature_mode: Feature extraction mode
        train_split_ratio: Ratio of trajectories for training
        seed: Random seed for reproducible train/test split
    """

    # Class-level cache to avoid reloading for each mode
    _cache = {}

    def __init__(self, data_path, win_size, step, mode="train",
                 feature_mode=FEATURE_MODE_MEASUREMENTS, train_split_ratio=0.8, seed=42):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.feature_mode = feature_mode
        self.train_split_ratio = train_split_ratio
        self.seed = seed

        # Create cache key
        cache_key = (data_path, win_size, feature_mode, train_split_ratio, seed)

        # Check if data is cached
        if cache_key in BallisticSegLoader._cache:
            cached = BallisticSegLoader._cache[cache_key]
            self.dataset = cached['dataset']
            self.train_indices = cached['train_indices']
            self.test_indices = cached['test_indices']
            self.n_features = cached['n_features']
            self.n_output_features = cached['n_output_features']
            self.target_slice = cached['target_slice']
            self.train_windows = cached['train_windows']
            self.train_labels = cached['train_labels']
            self.train_traj_ids = cached['train_traj_ids']
            self.train_timestep_indices = cached['train_timestep_indices']
            self.test_windows = cached['test_windows']
            self.test_labels = cached['test_labels']
            self.test_traj_ids = cached['test_traj_ids']
            self.test_timestep_indices = cached['test_timestep_indices']
            self.train_targets = cached.get('train_targets')
            self.test_targets = cached.get('test_targets')
        else:
            # Load and preprocess dataset
            print(f"Loading ballistic dataset from: {data_path}")
            self.dataset = BallisticDataset(foldername=data_path)
            self.dataset.pre_process()

            n_trajectories = len(self.dataset)
            print(f"Loaded {n_trajectories} trajectories")

            # Random split of trajectory indices
            np.random.seed(seed)
            all_indices = np.random.permutation(n_trajectories)
            n_train = int(n_trajectories * train_split_ratio)

            self.train_indices = all_indices[:n_train].tolist()
            self.test_indices = all_indices[n_train:].tolist()

            print(f"Train trajectories: {len(self.train_indices)}, Test trajectories: {len(self.test_indices)}")

            # Determine feature dimensions
            if feature_mode == FEATURE_MODE_CONTEXTUAL_FILTER:
                self.n_features = FEATURE_DIMS[feature_mode]['input']
                self.n_output_features = FEATURE_DIMS[feature_mode]['output']
                self.target_slice = None
            else:
                self.n_features = FEATURE_DIMS[feature_mode]
                self.n_output_features = self.n_features
                self.target_slice = None

            print(f"Feature mode: {feature_mode} (input: {self.n_features}D, output: {self.n_output_features}D)")

            # Create windows for train and test sets
            print("Creating training windows...")
            if feature_mode == FEATURE_MODE_CONTEXTUAL_FILTER:
                self.train_windows, self.train_targets, self.train_labels, self.train_traj_ids, self.train_timestep_indices = \
                    self.dataset.to_windows_contextual(win_size, self.train_indices, step=step)
            else:
                self.train_windows, self.train_labels, self.train_traj_ids, self.train_timestep_indices = \
                    self.dataset.to_windows(win_size, feature_mode, self.train_indices, step=step)

            print("Creating test windows...")
            # For test/val/thre, always use step=1 for proper reconstruction
            if feature_mode == FEATURE_MODE_CONTEXTUAL_FILTER:
                self.test_windows, self.test_targets, self.test_labels, self.test_traj_ids, self.test_timestep_indices = \
                    self.dataset.to_windows_contextual(win_size, self.test_indices, step=1)
            else:
                self.test_windows, self.test_labels, self.test_traj_ids, self.test_timestep_indices = \
                    self.dataset.to_windows(win_size, feature_mode, self.test_indices, step=1)
                self.train_targets = None
                self.test_targets = None

            print(f"Train windows: {self.train_windows.shape}, Test windows: {self.test_windows.shape}")

            # Cache the data
            BallisticSegLoader._cache[cache_key] = {
                'dataset': self.dataset,
                'train_indices': self.train_indices,
                'test_indices': self.test_indices,
                'n_features': self.n_features,
                'n_output_features': self.n_output_features,
                'target_slice': self.target_slice,
                'train_windows': self.train_windows,
                'train_labels': self.train_labels,
                'train_traj_ids': self.train_traj_ids,
                'train_timestep_indices': self.train_timestep_indices,
                'test_windows': self.test_windows,
                'test_labels': self.test_labels,
                'test_traj_ids': self.test_traj_ids,
                'test_timestep_indices': self.test_timestep_indices,
                'train_targets': self.train_targets,
                'test_targets': self.test_targets,
            }

        # Convert to numpy for compatibility with original loaders
        self.train = self.train_windows.numpy()
        self.test = self.test_windows.numpy()
        self.val = self.test  # Use test set for validation
        self.test_labels_np = self.test_labels.numpy()
        self.train_labels_np = self.train_labels.numpy()
        if self.feature_mode == FEATURE_MODE_CONTEXTUAL_FILTER:
            self.train_targets_np = self.train_targets.numpy()
            self.test_targets_np = self.test_targets.numpy()


    def __len__(self):
        if self.mode == "train":
            return len(self.train)
        elif self.mode == 'val':
            return len(self.val)
        elif self.mode == 'test':
            return len(self.test)
        else:  # 'thre' mode
            return len(self.test)

    def __getitem__(self, index):
        if self.mode == "train":
            if self.feature_mode == FEATURE_MODE_CONTEXTUAL_FILTER:
                return np.float32(self.train[index]), np.float32(self.train_targets_np[index])
            return np.float32(self.train[index]), np.float32(self.train_labels_np[index:index+1])
        elif self.mode == 'val':
            if self.feature_mode == FEATURE_MODE_CONTEXTUAL_FILTER:
                return np.float32(self.val[index]), np.float32(self.test_targets_np[index])
            return np.float32(self.val[index]), np.float32(self.test_labels_np[index:index+1])
        elif self.mode == 'test':
            if self.feature_mode == FEATURE_MODE_CONTEXTUAL_FILTER:
                return np.float32(self.test[index]), np.float32(self.test_targets_np[index])
            return np.float32(self.test[index]), np.float32(self.test_labels_np[index:index+1])
        else:  # 'thre' mode - same as test
            if self.feature_mode == FEATURE_MODE_CONTEXTUAL_FILTER:
                return np.float32(self.test[index]), np.float32(self.test_targets_np[index])
            return np.float32(self.test[index]), np.float32(self.test_labels_np[index:index+1])

    def get_trajectory_info(self):
        """Return trajectory tracking information for score reconstruction."""
        return {
            'train_traj_ids': self.train_traj_ids,
            'train_timestep_indices': self.train_timestep_indices,
            'test_traj_ids': self.test_traj_ids,
            'test_timestep_indices': self.test_timestep_indices,
            'train_indices': self.train_indices,
            'test_indices': self.test_indices,
            'dataset': self.dataset,
        }


def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD',
                       feature_mode=FEATURE_MODE_MEASUREMENTS, train_split_ratio=0.8, seed=42):
    if (dataset == 'SMD'):
        dataset_obj = SMDSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'MSL'):
        dataset_obj = MSLSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SMAP'):
        dataset_obj = SMAPSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'PSM'):
        dataset_obj = PSMSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'ballistic'):
        dataset_obj = BallisticSegLoader(data_path, win_size, step, mode,
                                         feature_mode=feature_mode,
                                         train_split_ratio=train_split_ratio,
                                         seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset_obj,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
