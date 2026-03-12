import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_pipeline import (
    StandardScaler,
    build_samples_per_file,
    make_node_inputs,
    read_all_csv,
    split_file_arrays,
)
from hgnn_config import HYPEREDGES, NODE_FEATURES, NODE_ORDER
from hgnn_temporal_model import HGNNTemporalTransformer, build_incidence


def resolve_co_history_steps(train_args: Dict[str, float]) -> int:
    if "co_history_steps" in train_args:
        return int(train_args["co_history_steps"])
    base_dt = float(train_args.get("base_interval_seconds", 10.0))
    hist_minutes = float(train_args.get("co_history_minutes", 10.0))
    return int(max(1, round((hist_minutes * 60.0) / base_dt)))


def resolve_checkpoint(exp_dir: Path, checkpoint_arg: str | None) -> Path:
    if checkpoint_arg:
        ckpt = Path(checkpoint_arg)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt

    summary_path = exp_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"run_summary.json not found in {exp_dir}. Please provide --checkpoint explicitly."
        )

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    best_name = summary.get("best_checkpoint")
    if not best_name:
        raise KeyError("best_checkpoint missing in run_summary.json. Please provide --checkpoint.")

    ckpt = exp_dir / best_name
    if not ckpt.exists():
        raise FileNotFoundError(f"Best checkpoint listed in summary does not exist: {ckpt}")
    return ckpt


def load_split_inputs(
    data_dir: Path,
    train_args: Dict[str, float],
    feature_scaler: StandardScaler,
) -> Dict[str, np.ndarray]:
    file_arrays = read_all_csv(data_dir)
    file_splits = split_file_arrays(
        file_arrays,
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


def save_heatmap(
    matrix: np.ndarray,
    save_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    x_ticks: List[str] | None = None,
    y_ticks: List[str] | None = None,
) -> None:
    plt.figure(figsize=(8, 6))
    im = plt.imshow(matrix, aspect="auto", cmap="viridis")
    plt.colorbar(im)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if x_ticks is not None:
        plt.xticks(ticks=np.arange(len(x_ticks)), labels=x_ticks, rotation=45, ha="right")
    if y_ticks is not None:
        plt.yticks(ticks=np.arange(len(y_ticks)), labels=y_ticks)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize HGNN-Temporal attention as heatmaps")
    parser.add_argument("--exp_dir", type=str, default="outputs_hgnn_tt_tmp_valsave")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="huizhuanyao_data")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="outputs_hgnn_tt_tmp_valsave/attention_plots")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = resolve_checkpoint(exp_dir=exp_dir, checkpoint_arg=args.checkpoint)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    train_args = checkpoint["args"]

    feature_scaler = StandardScaler(
        mean=np.array(checkpoint["feature_scaler_mean"], dtype=np.float32),
        std=np.array(checkpoint["feature_scaler_std"], dtype=np.float32),
    )

    splits = load_split_inputs(
        data_dir=Path(args.data_dir),
        train_args=train_args,
        feature_scaler=feature_scaler,
    )

    x_key = f"X_{args.split}"
    X_split = splits[x_key]
    if X_split.shape[0] == 0:
        raise ValueError(f"Selected split {args.split} is empty.")

    sample_index = args.sample_index
    if sample_index < 0 or sample_index >= X_split.shape[0]:
        raise IndexError(f"sample_index {sample_index} out of range [0, {X_split.shape[0] - 1}]")

    sample_x = torch.tensor(X_split[sample_index : sample_index + 1], dtype=torch.float32)

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

    node_inputs = make_node_inputs(sample_x, NODE_FEATURES)
    with torch.no_grad():
        pred, attention_info = model(node_inputs, raw_sequence=sample_x, return_attention=True)

    temporal_attention = attention_info["temporal_attention"]
    node_attention = attention_info["node_attention"][0].squeeze(-1).cpu().numpy()
    time_attention = attention_info["time_attention"][0].squeeze(-1).cpu().numpy()
    hyperedge_weight = attention_info["hyperedge_weight"].cpu().numpy()

    # Temporal attention: one heatmap for each encoder layer, averaged over heads.
    for layer_idx, layer_attn in enumerate(temporal_attention):
        # layer_attn shape: [B, nhead, T, T]
        attn_mat = layer_attn[0].mean(dim=0).cpu().numpy()
        save_heatmap(
            matrix=attn_mat,
            save_path=out_dir / f"temporal_attention_layer_{layer_idx + 1}.png",
            title=f"Temporal Attention Layer {layer_idx + 1}",
            x_label="Key Time Step",
            y_label="Query Time Step",
        )

    save_heatmap(
        matrix=node_attention,
        save_path=out_dir / "node_attention.png",
        title="Node-level Attention",
        x_label="Node",
        y_label="Time Step",
        x_ticks=NODE_ORDER,
    )

    save_heatmap(
        matrix=np.expand_dims(time_attention, axis=0),
        save_path=out_dir / "time_attention.png",
        title="Temporal Pooling Attention",
        x_label="Time Step",
        y_label="Attention",
    )

    edge_labels = [f"E{i + 1}" for i in range(len(HYPEREDGES))]
    save_heatmap(
        matrix=np.expand_dims(hyperedge_weight, axis=0),
        save_path=out_dir / "hyperedge_weight.png",
        title="Learned Hyperedge Weights",
        x_label="Hyperedge",
        y_label="Weight",
        x_ticks=edge_labels,
    )

    pred_value = float(pred.item())
    print(f"Checkpoint: {ckpt_path}")
    print(f"Split: {args.split}, sample_index: {sample_index}")
    print(f"Predicted normalized CO: {pred_value:.6f}")
    print(f"Saved heatmaps to: {out_dir}")


if __name__ == "__main__":
    main()
