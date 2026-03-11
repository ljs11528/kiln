import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_pipeline import StandardScaler, SequenceDataset, build_samples_per_file, make_node_inputs, read_all_csv, split_dataset
from hgnn_config import HYPEREDGES, NODE_FEATURES, NODE_ORDER
from hgnn_temporal_model import HGNNTemporalTransformer, build_incidence


def resolve_checkpoint(exp_dir: Path, checkpoint_arg: str | None) -> Path:
    if checkpoint_arg:
        ckpt = Path(checkpoint_arg)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt

    summary_path = exp_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"run_summary.json not found in {exp_dir}")

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    best_name = summary.get("best_checkpoint")
    if not best_name:
        raise KeyError("best_checkpoint missing in run_summary.json")

    ckpt = exp_dir / best_name
    if not ckpt.exists():
        raise FileNotFoundError(f"Best checkpoint does not exist: {ckpt}")
    return ckpt


def build_scaled_splits(data_dir: Path, train_args: Dict[str, float], feature_scaler: StandardScaler):
    arrays = read_all_csv(data_dir)
    X, y = build_samples_per_file(
        arrays,
        seq_len=int(train_args["seq_len"]),
        horizon=int(train_args["horizon"]),
    )
    splits = split_dataset(
        X,
        y,
        train_ratio=float(train_args["train_ratio"]),
        val_ratio=float(train_args["val_ratio"]),
    )

    for key in ["X_train", "X_val", "X_test"]:
        flat = splits[key].reshape(-1, splits[key].shape[-1])
        splits[key] = feature_scaler.transform(flat).reshape(splits[key].shape)
    return splits


def predict_scaled(model: HGNNTemporalTransformer, X_split: np.ndarray, batch_size: int) -> np.ndarray:
    ds = SequenceDataset(X_split, np.zeros((X_split.shape[0],), dtype=np.float32))
    preds = []
    for i in range(0, len(ds), batch_size):
        bx = ds.features[i : i + batch_size]
        node_inputs = make_node_inputs(bx, NODE_FEATURES)
        with torch.no_grad():
            pred = model(node_inputs)
        preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot prediction error distribution")
    parser.add_argument("--exp_dir", type=str, default="outputs_hgnn_tt")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="huizhuanyao_data")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--out_file", type=str, default="co_error_distribution.png")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    ckpt_path = resolve_checkpoint(exp_dir, args.checkpoint)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    train_args = checkpoint["args"]

    feature_scaler = StandardScaler(
        mean=np.array(checkpoint["feature_scaler_mean"], dtype=np.float32),
        std=np.array(checkpoint["feature_scaler_std"], dtype=np.float32),
    )
    target_scaler = StandardScaler(
        mean=np.array(checkpoint["target_scaler_mean"], dtype=np.float32),
        std=np.array(checkpoint["target_scaler_std"], dtype=np.float32),
    )

    splits = build_scaled_splits(Path(args.data_dir), train_args, feature_scaler)
    x_key = f"X_{args.split}"
    y_key = f"y_{args.split}"

    X_split = splits[x_key]
    y_true_raw = splits[y_key]

    incidence = build_incidence(NODE_ORDER, HYPEREDGES)
    node_input_dims = [len(NODE_FEATURES[name]) for name in NODE_ORDER]
    model = HGNNTemporalTransformer(
        node_input_dims=node_input_dims,
        incidence=incidence,
        d_model=int(train_args["d_model"]),
        nhead=int(train_args["nhead"]),
        num_transformer_layers=int(train_args["num_layers"]),
        ff_dim=int(train_args["ff_dim"]),
        dropout=float(train_args["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pred_scaled = predict_scaled(model, X_split, batch_size=args.batch_size)

    y_true = target_scaler.inverse_transform(y_true_raw.reshape(-1, 1)).reshape(-1)
    y_pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
    err = y_pred - y_true

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(err, bins=args.bins, alpha=0.8, edgecolor="black")
    axes[0].set_title(f"Error Distribution ({args.split})")
    axes[0].set_xlabel("Prediction Error (Pred - True)")
    axes[0].set_ylabel("Count")
    axes[0].grid(alpha=0.3)

    axes[1].boxplot(err, vert=True)
    axes[1].set_title("Error Boxplot")
    axes[1].set_ylabel("Prediction Error (Pred - True)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path = exp_dir / args.out_file
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    print(f"Checkpoint: {ckpt_path}")
    print(f"Split: {args.split}, n={len(err)}")
    print(f"Error mean={float(np.mean(err)):.6f}, std={float(np.std(err)):.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
