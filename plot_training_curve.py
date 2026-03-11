import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training convergence curves")
    parser.add_argument("--exp_dir", type=str, default="outputs_hgnn_tt")
    parser.add_argument("--history_file", type=str, default=None)
    parser.add_argument("--out_file", type=str, default="training_curve.png")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)

    history_path = exp_dir / (args.history_file or "training_history.json")
    if not history_path.exists():
        summary_path = exp_dir / "run_summary.json"
        if summary_path.exists() and args.history_file is None:
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            alt_name = summary.get("training_history_file")
            if alt_name:
                history_path = exp_dir / alt_name

    if not history_path.exists():
        raise FileNotFoundError(
            f"Cannot find training history file in {exp_dir}. "
            "Please retrain once with the updated training script."
        )

    with history_path.open("r", encoding="utf-8") as f:
        history = json.load(f)

    if not history:
        raise ValueError("Training history is empty.")

    epochs = [item["epoch"] for item in history]
    train_mse = [item["train_mse"] for item in history]
    val_rmse = [item["val_rmse"] for item in history]
    train_r2 = [item["train_r2"] for item in history]
    val_r2 = [item["val_r2"] for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_mse, marker="o", label="Train MSE")
    axes[0].plot(epochs, val_rmse, marker="s", label="Val RMSE")
    axes[0].set_title("Training Convergence")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Error")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_r2, marker="o", label="Train R2")
    axes[1].plot(epochs, val_r2, marker="s", label="Val R2")
    axes[1].set_title("R2 Over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("R2")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    out_path = exp_dir / args.out_file
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"History: {history_path}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
