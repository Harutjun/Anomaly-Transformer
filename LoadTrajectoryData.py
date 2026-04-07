"""
Load and process ballistic trajectory data from MATLAB .mat files.
Copied from TranAD project for Anomaly-Transformer adaptation.
"""
import os
from typing import Iterable, Optional
from scipy.io import loadmat
import numpy as np
import torch


def _normalize_mat_filename(name: str) -> str:
    base = os.path.basename(name)
    return base if base.lower().endswith('.mat') else f"{base}.mat"


def extract_struct(mat_struct):
    """
    Recursively extracts fields from a MATLAB struct loaded with scipy.io.loadmat.
    """
    if isinstance(mat_struct, np.ndarray) and mat_struct.size == 1:
        # Extract single element if it's a MATLAB struct with (1, 1) shape
        if len(mat_struct.shape) == 1:
            mat_struct = mat_struct[0]
        else:
            mat_struct = mat_struct[0, 0]
    elif isinstance(mat_struct, np.ndarray):
        # For cell arrays or larger arrays
        return [extract_struct(el) for el in mat_struct]

    if isinstance(mat_struct, np.void):  # MATLAB struct
        return {name: extract_struct(mat_struct[name]) for name in mat_struct.dtype.names}
    else:
        return mat_struct


def load_and_process_files(foldername, subset_filenames: Optional[Iterable[str]] = None):
    """
    Load and process all .mat trajectory files from a folder.

    Args:
        foldername: Path to folder containing .mat files
        subset_filenames: Optional list of specific filenames to load

    Returns:
        List of dictionaries containing trajectory data
    """
    tensor_data = []
    if subset_filenames:
        seen = set()
        candidate_files = []
        for entry in subset_filenames:
            norm_name = _normalize_mat_filename(entry)
            if norm_name not in seen:
                candidate_files.append(norm_name)
                seen.add(norm_name)
    else:
        candidate_files = sorted(fname for fname in os.listdir(foldername) if fname.lower().endswith('.mat'))

    for filename in candidate_files:
        filepath = os.path.join(foldername, filename)
        if not os.path.isfile(filepath):
            continue
        data = loadmat(filepath)
        trajectory_data_dict = extract_struct(data['trajectory_data'])

        # Convert arrays to tensors
        time = torch.tensor(trajectory_data_dict['time'])
        state = torch.tensor(trajectory_data_dict['state'])
        measurement = torch.tensor(trajectory_data_dict['noisy_states'])
        anomaly_flags = {key: torch.tensor(trajectory_data_dict['anomaly_flags'][key] if key != 'anomaly_probs' else []) for key in trajectory_data_dict['anomaly_flags'].keys()}
        theta = torch.tensor(trajectory_data_dict['theta']).float()
        h = torch.tensor(trajectory_data_dict['h']).float()
        k = torch.tensor(trajectory_data_dict['k']).float()
        v0 = torch.tensor(trajectory_data_dict['v0']).float()
        estimated_states = torch.tensor(trajectory_data_dict['estimated_states'])
        flags = [key for key in trajectory_data_dict['anomaly_flags'].keys() if key != 'anomaly_probs' and key != 'bias_val']

        anomaly_rec = trajectory_data_dict['anomaly_records']
        anomaly_idxs = []
        [anomaly_idxs.append(anomaly_rec[flag][0]) if isinstance(anomaly_rec[flag], list) and len(anomaly_rec[flag]) > 0 else anomaly_idxs.append([anomaly_rec[flag]]) for flag in flags]
        filtered_list = [[sublist for sublist in inner_list if sublist] for inner_list in anomaly_idxs if
                         any(sublist for sublist in inner_list)]
        anomaly_idxs = np.concatenate(filtered_list if len(filtered_list) > 0 else [np.array([])])
        anomaly_idxs.sort()

        anomaly_idxs = torch.tensor(anomaly_idxs)
        anomaly_tensor = torch.zeros(time.shape[1])
        if len(anomaly_idxs) > 0:
            anomaly_tensor[anomaly_idxs.long() - 1] = 1
        generating_params = {'theta': theta, 'h': h, 'k': k, 'v0': v0}

        tensor_data.append({
            'traj_len': time.shape[1],
            'time': time,
            'state': state,
            'measurement': measurement,
            'anomaly_idxs': anomaly_tensor,
            'anomaly_flags': anomaly_flags,
            'anomaly_records': anomaly_rec,
            'source_filename': os.path.basename(filename),
            'generating_params': generating_params,
            'estimated_states': estimated_states
        })

    return tensor_data


if __name__ == "__main__":
    foldername = "./dataset/ballistic"
    tensor_data = load_and_process_files(foldername)
    print(f"Loaded {len(tensor_data)} trajectories")
