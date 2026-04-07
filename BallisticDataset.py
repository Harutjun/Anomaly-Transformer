"""
Ballistic trajectory dataset implementation for PyTorch.
Handles loading, preprocessing, and batching of ballistic trajectory data.
Copied from TranAD project for Anomaly-Transformer adaptation.
"""
from typing import List, Dict, Optional, Tuple, Iterable
import torch
import numpy as np
from torch.utils.data import Dataset

from LoadTrajectoryData import load_and_process_files

# Feature mode constants
FEATURE_MODE_MEASUREMENTS = 'measurements'
FEATURE_MODE_FILTER_OUTPUTS = 'filter_outputs'
FEATURE_MODE_JOINT = 'joint'
FEATURE_MODE_CONTEXTUAL_FILTER = 'contextual_filter'
VALID_FEATURE_MODES = [FEATURE_MODE_MEASUREMENTS, FEATURE_MODE_FILTER_OUTPUTS, FEATURE_MODE_JOINT, FEATURE_MODE_CONTEXTUAL_FILTER]

# Constants
SCALING_FACTOR = 1e3
DEFAULT_MAX_LENGTH_MULTIPLIER = 100

# Feature dimensions per mode
# For contextual_filter: input=2D measurements, output=4D filter outputs
FEATURE_DIMS = {
    FEATURE_MODE_MEASUREMENTS: 2,
    FEATURE_MODE_FILTER_OUTPUTS: 4,
    FEATURE_MODE_JOINT: 6,
    FEATURE_MODE_CONTEXTUAL_FILTER: {'input': 2, 'output': 4},
}


class BallisticDataset(Dataset):
    """
    Dataset class for ballistic trajectory data.

    Supports loading from pre-processed data or directly from folder path.
    Includes preprocessing capabilities for scaling and padding sequences.

    Args:
        data: Pre-loaded trajectory data as list of dictionaries
        foldername: Path to folder containing trajectory files
        subset_filenames: Optional list of specific filenames to load

    Raises:
        ValueError: If neither data nor foldername is provided
    """

    def __init__(self, data: Optional[List[Dict]] = None, foldername: Optional[str] = None,
                 subset_filenames: Optional[Iterable[str]] = None) -> None:
        self.padded_len = None
        self.units = 'raw'  # Will be updated to 'km' after preprocessing

        if data is not None:
            self.data = data
        elif foldername is not None:
            self.data = load_and_process_files(foldername, subset_filenames=subset_filenames)
        else:
            raise ValueError("Either data or foldername must be provided")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict, torch.Tensor, str]:
        """
        Retrieve a single trajectory sample.

        Args:
            idx: Index of the data item to retrieve

        Returns:
            Tuple containing:
                - X: Measurement tensor
                - Y: State tensor
                - anomaly_idxs: Anomaly indices tensor
                - datalength: Length of valid data
                - time: Time tensor
                - generating_params: Dictionary of generating parameters
                - estimated_states: Estimated states tensor
                - source_filename: Original trajectory filename
        """
        sample = self.data[idx]
        return (
            sample['measurement'],
            sample['state'],
            sample['anomaly_idxs'],
            sample['traj_len'],
            sample['time'],
            sample['generating_params'],
            sample['estimated_states'],
            sample.get('source_filename', f'sample_{idx}')
        )

    def pre_process(self, max_length: Optional[int] = None) -> None:
        """
        Preprocess the dataset by scaling and padding sequences.

        Applies scaling factor and pads all sequences to the same length.
        Updates internal state to track preprocessing parameters.

        Args:
            max_length: Target length for padding. If None, computed from data.
        """
        if max_length is None:
            max_length = self._compute_max_length()

        for i in range(len(self.data)):
            self._process_sample(i, max_length)

        self.units = 'km'
        self.padded_len = max_length

    def _compute_max_length(self) -> int:
        """Compute maximum length rounded to nearest hundred."""
        max_len = max(len(self.data[k]['measurement'][0, :]) for k in range(len(self.data)))
        return (max_len // DEFAULT_MAX_LENGTH_MULTIPLIER + 1) * DEFAULT_MAX_LENGTH_MULTIPLIER

    def _process_sample(self, idx: int, max_length: int) -> None:
        """Process a single sample with scaling and padding."""
        sample = self.data[idx]
        measurement_length = sample['measurement'].shape[1]

        # Record the original (pre-padding) trajectory length
        sample['traj_len'] = int(measurement_length)

        # Scale measurements and states
        sample['measurement'] = sample['measurement'].float() / SCALING_FACTOR
        sample['state'] = sample['state'].float() / SCALING_FACTOR
        sample['estimated_states'] = sample['estimated_states'].float() / SCALING_FACTOR

        # Apply padding if needed
        if measurement_length < max_length:
            padding_size = max_length - measurement_length

            # Pad measurements and states
            for key in ['measurement', 'state', 'estimated_states']:
                padding = torch.zeros((sample[key].shape[0], padding_size))
                sample[key] = torch.cat((sample[key], padding), dim=1)

            # Pad anomaly indices
            padding = torch.zeros((1, padding_size))
            sample['anomaly_idxs'] = torch.cat((sample['anomaly_idxs'].unsqueeze(0), padding), dim=1)
            sample['anomaly_idxs'] = sample['anomaly_idxs'].squeeze(0)

            # Pad time
            padding = torch.zeros((1, padding_size))
            sample['time'] = torch.cat((sample['time'], padding), dim=1)
            sample['time'] = sample['time'].float()

    def get_features(self, idx: int, mode: str = FEATURE_MODE_MEASUREMENTS) -> torch.Tensor:
        """
        Extract features for a single trajectory based on mode.

        Args:
            idx: Index of the trajectory
            mode: Feature extraction mode ('measurements', 'filter_outputs', 'joint', 'contextual_filter')

        Returns:
            Feature tensor of shape [n_features, seq_len]
            For contextual_filter: returns 2D measurements only (input features)
        """
        if mode not in VALID_FEATURE_MODES:
            raise ValueError(f"Invalid feature mode '{mode}'. Must be one of {VALID_FEATURE_MODES}")

        sample = self.data[idx]

        if mode == FEATURE_MODE_MEASUREMENTS:
            # measurement has shape [2, seq_len] - 2D position measurements
            return sample['measurement'][:2, :]
        elif mode == FEATURE_MODE_FILTER_OUTPUTS:
            # estimated_states has shape [4, seq_len] - filter outputs (position + velocity)
            return sample['estimated_states'][:4, :]
        elif mode == FEATURE_MODE_JOINT:
            # Concatenate measurements and filter outputs [6, seq_len]
            measurements = sample['measurement'][:2, :]
            filter_outputs = sample['estimated_states'][:4, :]
            return torch.cat([measurements, filter_outputs], dim=0)
        elif mode == FEATURE_MODE_CONTEXTUAL_FILTER:
            # Contextual input uses measurements only to avoid target leakage
            return sample['measurement'][:2, :]

    def get_contextual_inputs_targets(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return measurements (inputs) and filter outputs (targets) for contextual_filter."""
        sample = self.data[idx]
        measurements = sample['measurement'][:2, :]
        filter_outputs = sample['estimated_states'][:4, :]
        return measurements, filter_outputs

    def get_all_features(self, mode: str = FEATURE_MODE_MEASUREMENTS) -> List[torch.Tensor]:
        """
        Extract features for all trajectories.

        Args:
            mode: Feature extraction mode

        Returns:
            List of feature tensors, each of shape [n_features, seq_len]
        """
        return [self.get_features(i, mode) for i in range(len(self.data))]

    def get_labels(self, idx: int) -> torch.Tensor:
        """
        Get anomaly labels for a trajectory.

        Args:
            idx: Index of the trajectory

        Returns:
            Label tensor of shape [seq_len] with 1 for anomaly, 0 otherwise
        """
        sample = self.data[idx]
        labels = sample['anomaly_idxs']
        # Handle both 1D and 2D label tensors
        if labels.dim() > 1:
            labels = labels.squeeze()
        return labels

    def get_trajectory_length(self, idx: int) -> int:
        """Get the actual (non-padded) length of a trajectory."""
        return int(self.data[idx]['traj_len'])

    def get_source_filename(self, idx: int) -> str:
        """Get the source filename for a trajectory."""
        return self.data[idx].get('source_filename', f'sample_{idx}')

    def get_estimated_states(self, idx: int) -> torch.Tensor:
        """Get filter outputs (estimated states) for a trajectory."""
        return self.data[idx]['estimated_states']

    def get_measurements(self, idx: int) -> torch.Tensor:
        """Get measurements for a trajectory."""
        return self.data[idx]['measurement']

    def get_time(self, idx: int) -> torch.Tensor:
        """Get time vector for a trajectory."""
        return self.data[idx]['time']

    def get_state(self, idx: int) -> torch.Tensor:
        """Get ground-truth state for a trajectory."""
        return self.data[idx]['state']

    def to_windows(
        self,
        window_size: int,
        mode: str = FEATURE_MODE_MEASUREMENTS,
        indices: Optional[List[int]] = None,
        step: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert trajectories to overlapping windows for Anomaly-Transformer.

        Windows are created WITHIN each trajectory (no cross-trajectory windows).
        For early timesteps (t < window_size), the window is padded by repeating first sample.

        Args:
            window_size: Size of the sliding window
            mode: Feature extraction mode
            indices: Optional list of trajectory indices to process. If None, process all.
            step: Step size for sliding window (default 1 for inference, can be larger for training)

        Returns:
            Tuple of:
                - windows: Tensor of shape [total_windows, window_size, n_features]
                - labels: Tensor of shape [total_windows] with anomaly labels
                - traj_ids: Tensor of shape [total_windows] with trajectory index for each window
                - timestep_indices: Tensor of shape [total_windows] with timestep index within trajectory
        """
        if indices is None:
            indices = list(range(len(self.data)))

        all_windows = []
        all_labels = []
        all_traj_ids = []
        all_timestep_indices = []

        for traj_idx in indices:
            # Get features: [n_features, seq_len]
            features = self.get_features(traj_idx, mode)
            traj_len = self.get_trajectory_length(traj_idx)
            labels = self.get_labels(traj_idx)

            # Transpose to [seq_len, n_features] for windowing
            features = features.T[:traj_len]  # Only use non-padded portion
            labels = labels[:traj_len]

            # Create windows
            for t in range(0, traj_len, step):
                if t >= window_size:
                    # Normal case: full window available
                    window = features[t - window_size:t]
                else:
                    # Early timesteps: pad by repeating first sample
                    padding = features[0].unsqueeze(0).repeat(window_size - t, 1)
                    if t > 0:
                        window = torch.cat([padding, features[0:t]], dim=0)
                    else:
                        window = padding

                all_windows.append(window)
                all_labels.append(labels[t])
                all_traj_ids.append(traj_idx)
                all_timestep_indices.append(t)

        # Stack all windows
        windows = torch.stack(all_windows)  # [total_windows, window_size, n_features]
        labels = torch.tensor(all_labels, dtype=torch.float32)
        traj_ids = torch.tensor(all_traj_ids, dtype=torch.long)
        timestep_indices = torch.tensor(all_timestep_indices, dtype=torch.long)

        return windows, labels, traj_ids, timestep_indices

    def to_windows_contextual(
        self,
        window_size: int,
        indices: Optional[List[int]] = None,
        step: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert trajectories to overlapping windows for contextual_filter mode.

        Returns:
            Tuple of:
                - input_windows: [total_windows, window_size, 2] measurements
                - target_windows: [total_windows, window_size, 4] filter outputs
                - labels: [total_windows] anomaly labels
                - traj_ids: [total_windows] trajectory index for each window
                - timestep_indices: [total_windows] timestep index within trajectory
        """
        if indices is None:
            indices = list(range(len(self.data)))

        input_windows = []
        target_windows = []
        all_labels = []
        all_traj_ids = []
        all_timestep_indices = []

        for traj_idx in indices:
            inputs, targets = self.get_contextual_inputs_targets(traj_idx)
            traj_len = self.get_trajectory_length(traj_idx)
            labels = self.get_labels(traj_idx)

            inputs = inputs.T[:traj_len]
            targets = targets.T[:traj_len]
            labels = labels[:traj_len]

            for t in range(0, traj_len, step):
                if t >= window_size:
                    input_window = inputs[t - window_size:t]
                    target_window = targets[t - window_size:t]
                else:
                    input_padding = inputs[0].unsqueeze(0).repeat(window_size - t, 1)
                    target_padding = targets[0].unsqueeze(0).repeat(window_size - t, 1)
                    if t > 0:
                        input_window = torch.cat([input_padding, inputs[0:t]], dim=0)
                        target_window = torch.cat([target_padding, targets[0:t]], dim=0)
                    else:
                        input_window = input_padding
                        target_window = target_padding

                input_windows.append(input_window)
                target_windows.append(target_window)
                all_labels.append(labels[t])
                all_traj_ids.append(traj_idx)
                all_timestep_indices.append(t)

        input_windows = torch.stack(input_windows)
        target_windows = torch.stack(target_windows)
        labels = torch.tensor(all_labels, dtype=torch.float32)
        traj_ids = torch.tensor(all_traj_ids, dtype=torch.long)
        timestep_indices = torch.tensor(all_timestep_indices, dtype=torch.long)

        return input_windows, target_windows, labels, traj_ids, timestep_indices


if __name__ == "__main__":
    # Example usage and testing
    DATASET_PATH = "./dataset/ballistic"

    try:
        # Initialize from folder path
        dataset = BallisticDataset(foldername=DATASET_PATH)
        print(f"Loaded {len(dataset)} trajectories")

        # Preprocess data
        dataset.pre_process()
        print(f"Preprocessed to max length: {dataset.padded_len}")

        # Test feature extraction
        for mode in VALID_FEATURE_MODES:
            features = dataset.get_features(0, mode)
            print(f"Mode '{mode}': features shape = {features.shape}")

        # Test windowing
        windows, labels, traj_ids, timestep_indices = dataset.to_windows(
            window_size=100, mode=FEATURE_MODE_MEASUREMENTS, indices=[0, 1]
        )
        print(f"Windows shape: {windows.shape}")
        print(f"Labels shape: {labels.shape}")

    except Exception as e:
        print(f"An error occurred: {e}")
