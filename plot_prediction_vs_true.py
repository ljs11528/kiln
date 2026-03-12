import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_pipeline import (
    StandardScaler,
    SequenceDataset,
    build_samples_per_file,
    make_node_inputs,
    read_all_csv,
    split_file_arrays,
)
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


def try_load_saved_val_predictions(exp_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    summary_path = exp_dir / "run_summary.json"
    if not summary_path.exists():
        return None

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    pred_file = summary.get("val_prediction_file")
    if not pred_file:
        return None

    pred_path = exp_dir / pred_file
    if not pred_path.exists():
        return None

    arr = np.load(pred_path)
    return arr["y_true"].reshape(-1), arr["y_pred"].reshape(-1)


def build_scaled_splits(data_dir: Path, train_args: Dict[str, float], feature_scaler: StandardScaler):
    arrays = read_all_csv(data_dir)
    file_splits = split_file_arrays(
        arrays,
        train_ratio=float(train_args["train_ratio"]),
        val_ratio=float(train_args["val_ratio"]),
    )
    X_train, y_train = build_samples_per_file(
        file_splits["train"],
        seq_len=int(train_args["seq_len"]),
        horizon=int(train_args["horizon"]),
    )
    X_val, y_val = build_samples_per_file(
        file_splits["val"],
        seq_len=int(train_args["seq_len"]),
        horizon=int(train_args["horizon"]),
    )
    X_test, y_test = build_samples_per_file(
        file_splits["test"],
        seq_len=int(train_args["seq_len"]),
        horizon=int(train_args["horizon"]),
    )

    splits = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

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
            pred = model(node_inputs, raw_sequence=bx)
        preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)


def resolve_co_history_steps(train_args: Dict[str, float]) -> int:
    if "co_history_steps" in train_args:
        return int(train_args["co_history_steps"])
    base_dt = float(train_args.get("base_interval_seconds", 10.0))
    hist_minutes = float(train_args.get("co_history_minutes", 10.0))
    return int(max(1, round((hist_minutes * 60.0) / base_dt)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Predicted vs True CO scatter")
    parser.add_argument("--exp_dir", type=str, default="outputs_hgnn_tt_tmp_valsave")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="huizhuanyao_data")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--max_points", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--out_file", type=str, default="co_scatter.png")
    parser.add_argument(
        "--prefer_saved_val_predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When split=val, prefer loading val predictions saved during training.",
    )
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    y_true = None
    y_pred = None
    data_source = ""

    if args.split == "val" and args.prefer_saved_val_predictions:
        saved = try_load_saved_val_predictions(exp_dir)
        if saved is not None:
            y_true, y_pred = saved
            data_source = "saved_validation_predictions"

    if y_true is None or y_pred is None:
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
            co_col_idx=2,
            co_history_steps=resolve_co_history_steps(train_args),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        pred_scaled = predict_scaled(model, X_split, batch_size=args.batch_size)
        y_true = y_true_raw.reshape(-1)
        y_pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
        data_source = "inference"

    n = min(args.max_points, y_true.shape[0])
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))

    min_v = float(min(y_true.min(), y_pred.min()))
    max_v = float(max(y_true.max(), y_pred.max()))

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=8, alpha=0.4)
    plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=2, label="Ideal")
    plt.title(f"Predicted vs True CO ({args.split})")
    plt.xlabel("True CO")
    plt.ylabel("Predicted CO")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = exp_dir / args.out_file
    plt.savefig(out_path, dpi=200)
    plt.close()

    if data_source == "saved_validation_predictions":
        print(f"Data source: {data_source}")
    else:
        print(f"Data source: {data_source}")
    print(f"Split: {args.split}, samples plotted: {n}")
    print(f"MAE={mae:.6f}, RMSE={rmse:.6f}, R2={r2:.6f}")

    print("y_true range:", y_true.min(), y_true.max())
    print("y_pred range:", y_pred.min(), y_pred.max())

    if data_source == "inference":
        print("raw y_true sample:", y_true_raw[:10])
        print("pred_scaled sample:", pred_scaled[:10])

        print("target_scaler mean:", target_scaler.mean)
        print("target_scaler std:", target_scaler.std)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
