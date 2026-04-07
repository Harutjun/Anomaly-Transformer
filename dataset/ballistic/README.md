# Ballistic Dataset Directory

Place your trajectory `.mat` files here.

## Expected Format

Each `.mat` file should contain a `trajectory_data` struct with:
- `time`: Time vector
- `state`: True state vector
- `noisy_states`: Measurement vector (used as input)
- `estimated_states`: Filter output vector
- `anomaly_flags`: Dictionary of anomaly flags
- `anomaly_records`: Dictionary of anomaly records
- `theta`, `h`, `k`, `v0`: Generating parameters

## Usage

After placing files here, run:
```bash
python train_ballistic.py --data_folder ./dataset/ballistic --feature_mode measurements
```
