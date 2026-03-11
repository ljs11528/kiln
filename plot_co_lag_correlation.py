import argparse
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_pipeline import read_all_csv
from hgnn_config import COL_INDEX


def infer_base_interval_seconds(arrays: List[np.ndarray]) -> float:
    diffs = []
    for arr in arrays:
        if arr.shape[0] < 2:
            continue
        t = arr[:, COL_INDEX["time"]]
        d = np.diff(t)
        d = d[d > 0]
        if d.size > 0:
            diffs.append(d)

    if not diffs:
        return 1.0

    return float(np.median(np.concatenate(diffs)))


def compute_acf(arrays: List[np.ndarray], lags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    acf_values: List[float] = []
    pair_counts: List[int] = []

    for lag in lags:
        x_all = []
        y_all = []
        for arr in arrays:
            co = arr[:, COL_INDEX["co"]]
            if co.shape[0] <= lag:
                continue
            x_all.append(co[:-lag])
            y_all.append(co[lag:])

        if not x_all:
            acf_values.append(np.nan)
            pair_counts.append(0)
            continue

        x = np.concatenate(x_all, axis=0)
        y = np.concatenate(y_all, axis=0)
        acf_values.append(float(np.corrcoef(x, y)[0, 1]))
        pair_counts.append(int(x.shape[0]))

    return np.array(acf_values, dtype=np.float64), np.array(pair_counts, dtype=np.int64)


def plot_acf_vs_horizon(
    horizon_seconds: np.ndarray,
    acf_values: np.ndarray,
    ci_values: np.ndarray,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4.5))
    plt.plot(horizon_seconds, acf_values, marker="o", linewidth=1.6, markersize=4)
    plt.plot(
        horizon_seconds,
        ci_values,
        linestyle="--",
        linewidth=1.2,
        color="gray",
        label="95% CI (+/-1.96/sqrt(N_k))",
    )
    plt.plot(horizon_seconds, -ci_values, linestyle="--", linewidth=1.2, color="gray")

    valid_mask = ~np.isnan(acf_values)
    sig_mask = valid_mask & (np.abs(acf_values) > ci_values)
    plt.scatter(
        horizon_seconds[sig_mask],
        acf_values[sig_mask],
        color="crimson",
        s=26,
        zorder=4,
        label="Significant lag",
    )

    plt.axhline(0.0, color="black", linewidth=1)
    plt.title("Autocorrelation vs Horizon: corr(CO_t, CO_{t+k})")
    plt.xlabel("Horizon (seconds)")
    plt.ylabel("Correlation")
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot autocorrelation vs horizon: corr(CO_t, CO_{t+k})")
    parser.add_argument("--data_dir", type=str, default="huizhuanyao_data")
    parser.add_argument("--out_file", type=str, default="outputs_hgnn_tt/co_autocorr_vs_horizon.png")
    parser.add_argument("--min_horizon_sec", type=float, default=10.0)
    parser.add_argument("--max_horizon_sec", type=float, default=300.0)
    args = parser.parse_args()

    arrays = read_all_csv(Path(args.data_dir))
    base_dt = infer_base_interval_seconds(arrays)

    if args.min_horizon_sec <= 0 or args.max_horizon_sec <= 0:
        raise ValueError("min_horizon_sec and max_horizon_sec must be > 0")
    if args.max_horizon_sec < args.min_horizon_sec:
        raise ValueError("max_horizon_sec must be >= min_horizon_sec")

    min_lag = max(1, int(round(args.min_horizon_sec / base_dt)))
    max_lag = max(min_lag, int(round(args.max_horizon_sec / base_dt)))
    lags = np.arange(min_lag, max_lag + 1)

    acf_values, pair_counts = compute_acf(arrays, lags)
    horizon_seconds = lags.astype(np.float64) * base_dt
    ci_values = np.full_like(acf_values, np.nan, dtype=np.float64)
    valid_n = pair_counts > 0
    ci_values[valid_n] = 1.96 / np.sqrt(pair_counts[valid_n].astype(np.float64))

    out_path = Path(args.out_file)
    plot_acf_vs_horizon(horizon_seconds, acf_values, ci_values, out_path)

    sig_mask = (~np.isnan(acf_values)) & (np.abs(acf_values) > ci_values)
    sig_horizons = horizon_seconds[sig_mask]

    print(f"Base interval (seconds): {base_dt:.3f}")
    print(f"Lag range: {min_lag}..{max_lag}")
    print(f"Horizon range (seconds): {horizon_seconds[0]:.3f}..{horizon_seconds[-1]:.3f}")
    print(f"Significant lags: {int(sig_mask.sum())}/{len(lags)}")
    if sig_horizons.size > 0:
        print(
            "Significant horizons (seconds): "
            + ", ".join(f"{v:.1f}" for v in sig_horizons[:20])
            + (" ..." if sig_horizons.size > 20 else "")
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
