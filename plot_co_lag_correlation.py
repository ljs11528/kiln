import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CO_COL = 2
TIME_COL = 1


def time_to_seconds(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def read_co_series(data_dir: Path) -> tuple[Dict[float, List[np.ndarray]], int]:
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No csv files found in: {data_dir}")

    grouped_series: Dict[float, List[np.ndarray]] = {}
    total_samples = 0

    for csv_path in csv_files:
        df = pd.read_csv(csv_path, header=None)
        if df.shape[1] < 3:
            raise ValueError(f"{csv_path.name} must have at least 3 columns")
        if df.shape[0] < 2:
            continue

        co = pd.to_numeric(df.iloc[:, CO_COL], errors="coerce").to_numpy(dtype=np.float64)
        if np.isnan(co).any():
            raise ValueError(f"NaN found in CO column of {csv_path.name}")

        time_col = df.iloc[:, TIME_COL].astype(str)
        time_seconds = time_col.map(time_to_seconds).to_numpy(dtype=np.float64)
        time_diffs = np.diff(time_seconds)
        positive_diffs = time_diffs[time_diffs > 0]
        if positive_diffs.size == 0:
            raise ValueError(f"Invalid sampling interval in {csv_path.name}: no positive time diff found")
        dt = float(np.median(positive_diffs))

        grouped_series.setdefault(dt, []).append(co)
        total_samples += len(co)

    if not grouped_series:
        raise ValueError("No valid CO series found in csv files")

    return grouped_series, total_samples


def compute_autocorrelation(co_series_list: List[np.ndarray], max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    lags = []
    corrs = []

    for k in range(1, max_lag + 1):
        x_all = []
        y_all = []

        for co in co_series_list:
            if len(co) <= k:
                continue
            x_all.append(co[:-k])
            y_all.append(co[k:])

        if not x_all:
            continue

        x = np.concatenate(x_all, axis=0)
        y = np.concatenate(y_all, axis=0)
        corr = np.corrcoef(x, y)[0, 1]

        lags.append(k)
        corrs.append(corr)

    return np.array(lags, dtype=np.int64), np.array(corrs, dtype=np.float64)


def plot_autocorrelation(time_axis_min: np.ndarray, corrs: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(time_axis_min, corrs, linewidth=1.8)
    plt.xlabel("Prediction Horizon (minutes)")
    plt.ylabel("Autocorrelation")
    plt.title("CO Temporal Autocorrelation")
    plt.grid(alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def print_corr_at_minutes(minutes: float, dt: float, lags: np.ndarray, corrs: np.ndarray) -> None:
    lag = int(round((minutes * 60.0) / dt))
    match_idx = np.where(lags == lag)[0]
    if match_idx.size == 0:
        print(f"corr(CO_t, CO_t+{minutes:g}min) is unavailable")
        return
    corr = corrs[match_idx[0]]
    print(f"corr(CO_t, CO_t+{minutes:g}min) = {corr:.6f}")


def choose_sampling_interval(grouped_series: Dict[float, List[np.ndarray]], requested_dt: float | None) -> Tuple[float, List[np.ndarray]]:
    available = sorted(grouped_series.keys())
    if requested_dt is not None:
        matched = min(available, key=lambda dt: abs(dt - requested_dt))
        if abs(matched - requested_dt) > 1e-6:
            print(f"Requested sampling interval {requested_dt:.3f}s not found exactly, using closest {matched:.3f}s")
        return matched, grouped_series[matched]

    # Default to the finest interval so horizon resolution is highest.
    selected = available[0]
    return selected, grouped_series[selected]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CO autocorrelation vs prediction horizon")
    parser.add_argument("--data_dir", type=str, default="huizhuanyao_data")
    parser.add_argument("--out_file", type=str, default="outputs_hgnn_tt/co_autocorr_vs_horizon.png")
    parser.add_argument("--max_minutes", type=float, default=20.0)
    parser.add_argument("--sampling_interval_sec", type=float, default=None)
    args = parser.parse_args()

    if args.max_minutes <= 0:
        raise ValueError("max_minutes must be > 0")

    grouped_series, total_samples = read_co_series(Path(args.data_dir))
    dt, co_series_list = choose_sampling_interval(grouped_series, args.sampling_interval_sec)
    max_lag = int((args.max_minutes * 60.0) / dt)
    if max_lag < 1:
        raise ValueError("max_minutes is too small for the detected sampling interval")

    lags, corrs = compute_autocorrelation(co_series_list, max_lag=max_lag)
    if lags.size == 0:
        raise ValueError("No valid autocorrelation values were computed")

    time_axis_min = lags * dt / 60.0
    out_path = Path(args.out_file)
    plot_autocorrelation(time_axis_min, corrs, out_path)

    interval_counter = Counter({interval: len(series_list) for interval, series_list in grouped_series.items()})

    print(f"Total samples: {total_samples}")
    print(f"Available sampling intervals (seconds -> file_count): {dict(sorted(interval_counter.items()))}")
    print(f"Sampling interval (s): {dt:.3f}")
    print(f"Files used for this plot: {len(co_series_list)}")
    print(f"Computed horizon range (min): {time_axis_min[0]:.3f} .. {time_axis_min[-1]:.3f}")
    print_corr_at_minutes(1, dt, lags, corrs)
    print_corr_at_minutes(5, dt, lags, corrs)
    print_corr_at_minutes(10, dt, lags, corrs)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
