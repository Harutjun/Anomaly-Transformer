"""
Plot ballistic inference outputs against ground truth and anomaly scores.

This script loads a per-trajectory .npz produced by infer_ballistic.py and
plots model outputs vs ground truth, plus anomaly scores over time.
"""
import argparse
import json
import os
from typing import Optional, Tuple

import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from BallisticDataset import BallisticDataset


def _resolve_results_file(results_dir: str, result_file: Optional[str], trajectory: Optional[str]) -> str:
    if result_file:
        return result_file

    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    candidates = sorted(
        [os.path.join(results_dir, name) for name in os.listdir(results_dir) if name.endswith('.npz')]
    )
    if not candidates:
        raise FileNotFoundError(f"No .npz files found in: {results_dir}")

    if trajectory:
        normalized = trajectory if trajectory.lower().endswith('.mat') else f"{trajectory}.mat"
        for path in candidates:
            if os.path.basename(path).startswith(normalized):
                return path
        raise FileNotFoundError(f"No results for trajectory '{trajectory}' in {results_dir}")

    if len(candidates) == 1:
        return candidates[0]

    names = ", ".join(os.path.basename(p) for p in candidates[:8])
    raise ValueError(
        "Multiple result files found; specify --trajectory or --result_file. "
        f"Examples: {names}"
    )


def _load_ground_truth(data_folder: str, source_filename: str) -> Tuple[np.ndarray, np.ndarray]:
    dataset = BallisticDataset(foldername=data_folder, subset_filenames=[source_filename])
    dataset.pre_process()

    state = dataset.get_state(0).numpy()
    time = dataset.get_time(0).numpy().squeeze()

    return state, time


def _select_model_output(npz_data, model_output: str, traj_len: int) -> np.ndarray:
    if model_output == 'estimated_states':
        output = npz_data['Y_estimated_state']
    elif model_output == 'measurements':
        output = npz_data['X_measurements']
    elif model_output == 'reconstructions':
        if 'reconstructions' not in npz_data:
            raise KeyError("Results file does not contain 'reconstructions'.")
        output = npz_data['reconstructions']
    else:
        raise ValueError(f"Unknown model_output: {model_output}")

    if output.ndim == 3:
        raise ValueError("Windowed reconstructions found; per-timestep outputs are required.")

    # Normalize to [D, T] for downstream plotting.
    if output.shape[0] == traj_len and output.shape[1] != traj_len:
        output = output.T

    return output[:, :traj_len]


def _plot_timeseries(time_axis, model_output, ground_truth, scores, anomaly_flags, title, save_path, show,
                     measurements, estimated_states):
    model_output = model_output.T
    ground_truth = ground_truth.T
    measurements = measurements.T
    estimated_states = estimated_states.T

    # Align lengths to the shortest series to avoid shape mismatches.
    min_len = min(
        len(time_axis),
        ground_truth.shape[0],
        model_output.shape[0],
        measurements.shape[0],
        estimated_states.shape[0],
        len(scores),
        len(anomaly_flags),
    )
    time_axis = time_axis[:min_len]
    ground_truth = ground_truth[:min_len]
    model_output = model_output[:min_len]
    scores = scores[:min_len]
    anomaly_flags = anomaly_flags[:min_len]
    measurements = measurements[:min_len]
    estimated_states = estimated_states[:min_len]

    n_dims = max(
        ground_truth.shape[1],
        model_output.shape[1],
        measurements.shape[1],
        estimated_states.shape[1],
    )
    fig, axes = plt.subplots(n_dims + 1, 1, figsize=(12, 3 * (n_dims + 1)), sharex=True)

    color_map = {
        'ground_truth': 'tab:blue',
        'measurements': 'tab:orange',
        'filter_outputs': 'tab:green',
        'model_output': 'tab:red',
        'anomaly_score': 'tab:purple',
        'anomaly_flag': 'tab:red',
    }

    for dim in range(n_dims):
        axes[dim].plot(
            time_axis,
            ground_truth[:, dim],
            label='ground_truth',
            linewidth=1.5,
            color=color_map['ground_truth'],
        )
        if dim < measurements.shape[1]:
            axes[dim].plot(
                time_axis,
                measurements[:, dim],
                label='measurements',
                linewidth=1.0,
                color=color_map['measurements'],
            )
        if dim < estimated_states.shape[1]:
            axes[dim].plot(
                time_axis,
                estimated_states[:, dim],
                label='filter_outputs',
                linewidth=1.0,
                color=color_map['filter_outputs'],
            )
        if model_output.shape[1] > dim:
            axes[dim].plot(
                time_axis,
                model_output[:, dim],
                label='model_output',
                linewidth=1.0,
                alpha=0.7,
                color=color_map['model_output'],
            )
        axes[dim].set_ylabel(f"Dim {dim}")
        axes[dim].legend(loc='upper right')
        axes[dim].grid(True, alpha=0.3)

    axes[-1].plot(
        time_axis,
        scores,
        label='anomaly_score',
        color=color_map['anomaly_score'],
        linewidth=1.2,
    )
    anomaly_indices = np.where(anomaly_flags > 0.5)[0]
    if anomaly_indices.size > 0:
        axes[-1].scatter(
            time_axis[anomaly_indices],
            scores[anomaly_indices],
            color=color_map['anomaly_flag'],
            s=12,
            label='anomaly_flag',
        )
    axes[-1].set_ylabel("Score")
    axes[-1].set_xlabel("Time")
    axes[-1].legend(loc='upper right')
    axes[-1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot ballistic results vs ground truth and anomaly scores')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory containing .npz inference outputs')
    parser.add_argument('--result_file', type=str, default=None,
                        help='Path to a specific .npz file')
    parser.add_argument('--trajectory', type=str, default=None,
                        help='Trajectory filename (e.g., traj1.mat or traj1) to select from results_dir')
    parser.add_argument('--data_folder', type=str, default='./dataset/ballistic',
                        help='Path to folder containing .mat trajectory files')
    parser.add_argument('--ground_truth', type=str, default='state', choices=['state', 'measurements'],
                        help='Ground truth series to plot')
    parser.add_argument('--model_output', type=str, default='estimated_states',
                        choices=['estimated_states', 'measurements', 'reconstructions'],
                        help='Model output series to plot')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Optional path to save the figure (e.g., plots/traj1.png)')
    parser.add_argument('--no_show', action='store_true', help='Do not display the plot window')
    parser.add_argument('--title', type=str, default=None, help='Optional custom plot title')

    args = parser.parse_args()

    results_path = _resolve_results_file(args.results_dir, args.result_file, args.trajectory)
    npz_data = np.load(results_path, allow_pickle=True)

    traj_len = int(npz_data['traj_len'])
    source_filename = str(npz_data['source_filename'])
    scores = npz_data['scores'][:traj_len]
    anomaly_flags = npz_data['anomaly_flags'][:traj_len]
    measurements = npz_data['X_measurements']
    estimated_states = npz_data['Y_estimated_state']

    if args.ground_truth == 'state':
        ground_truth, time_axis = _load_ground_truth(args.data_folder, source_filename)
    else:
        ground_truth = npz_data['X_measurements']
        time_axis = np.arange(ground_truth.shape[1])

    config = {}
    if 'config' in npz_data:
        try:
            config = json.loads(str(npz_data['config']))
        except json.JSONDecodeError:
            config = {}

    feature_mode = str(npz_data.get('feature_mode', 'ballistic'))
    if feature_mode == 'contextual_filter' and args.model_output == 'estimated_states':
        if 'reconstructions' in npz_data:
            print("Warning: switching model_output to 'reconstructions' for contextual_filter.")
            args.model_output = 'reconstructions'
        else:
            print("Warning: plotting filter outputs, not model reconstructions. "
                  "Run inference with --save_reconstructions and replot for true model output.")

    ground_truth = ground_truth[:, :traj_len]
    measurements = measurements[:, :traj_len]
    estimated_states = estimated_states[:, :traj_len]
    model_output = _select_model_output(npz_data, args.model_output, traj_len)

    title = args.title
    if not title:
        title = f"{source_filename} | feature_mode={feature_mode}"

    _plot_timeseries(
        time_axis=time_axis,
        model_output=model_output,
        ground_truth=ground_truth,
        scores=scores,
        anomaly_flags=anomaly_flags,
        title=title,
        save_path=args.save_path,
        show=not args.no_show,
        measurements=measurements,
        estimated_states=estimated_states,
    )


if __name__ == '__main__':
    main()
