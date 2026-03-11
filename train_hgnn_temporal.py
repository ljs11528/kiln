import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_pipeline import (
    SequenceDataset,
    StandardScaler,
    build_samples_per_file,
    make_node_inputs,
    read_all_csv,
    split_dataset,
)
from hgnn_config import HYPEREDGES, NODE_FEATURES, NODE_ORDER
from hgnn_temporal_model import HGNNTemporalTransformer, build_incidence


def update_topk_checkpoints(
    topk_records: List[Dict[str, float]],
    checkpoint_record: Dict[str, float],
    topk: int,
) -> List[Dict[str, float]]:
    updated = topk_records + [checkpoint_record]
    updated.sort(key=lambda x: x["val_rmse"])
    return updated[:topk]


def infer_base_interval_seconds(file_arrays: List[np.ndarray]) -> float:
    diffs = []
    for arr in file_arrays:
        if arr.shape[0] < 2:
            continue
        t = arr[:, 1]
        d = np.diff(t)
        d = d[d > 0]
        if d.size > 0:
            diffs.append(d)

    if not diffs:
        return 1.0

    all_diffs = np.concatenate(diffs)
    return float(np.median(all_diffs))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_scaler: StandardScaler,
) -> Dict[str, float]:
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            node_inputs = [t.to(device) for t in make_node_inputs(bx, NODE_FEATURES)]
            pred = model(node_inputs)
            preds.append(pred.cpu().numpy())
            trues.append(by.cpu().numpy())

    pred_scaled = np.concatenate(preds)
    true_scaled = np.concatenate(trues)

    pred = target_scaler.inverse_transform(pred_scaled)
    true = target_scaler.inverse_transform(true_scaled)

    mse = float(np.mean((pred - true) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(pred - true)))
    mape = float(np.mean(np.abs((pred - true) / (true + 1e-8))) * 100.0)
    sst = float(np.sum((true - np.mean(true)) ** 2))
    sse = float(np.sum((true - pred) ** 2))
    r2 = 1.0 - (sse / (sst + 1e-12))

    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def main() -> None:
    parser = argparse.ArgumentParser(description="HGNN + Temporal Transformer for CO forecasting")
    parser.add_argument("--data_dir", type=str, default="huizhuanyao_data")
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--forecast_seconds",
        type=float,
        default=300.0,
        help="Forecast horizon in seconds. It will be converted to horizon steps using data interval.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--ff_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs_hgnn_tt")
    args = parser.parse_args()

    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_arrays = read_all_csv(data_dir)
    base_interval_seconds = infer_base_interval_seconds(file_arrays)
    if args.forecast_seconds is not None:
        if args.forecast_seconds <= 0:
            raise ValueError("forecast_seconds must be > 0")
        horizon_steps = max(1, int(round(args.forecast_seconds / base_interval_seconds)))
    else:
        horizon_steps = args.horizon

    args.horizon = int(horizon_steps)
    args.base_interval_seconds = float(base_interval_seconds)
    args.effective_forecast_seconds = float(args.horizon * base_interval_seconds)

    print(
        f"Using horizon_steps={args.horizon}, base_interval_seconds={args.base_interval_seconds:.3f}, "
        f"effective_forecast_seconds={args.effective_forecast_seconds:.3f}"
    )

    X, y = build_samples_per_file(file_arrays, seq_len=args.seq_len, horizon=args.horizon)
    splits = split_dataset(X, y, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    feature_scaler = StandardScaler.fit(splits["X_train"].reshape(-1, splits["X_train"].shape[-1]))
    target_scaler = StandardScaler.fit(splits["y_train"].reshape(-1, 1))

    for key in ["X_train", "X_val", "X_test"]:
        flat = splits[key].reshape(-1, splits[key].shape[-1])
        splits[key] = feature_scaler.transform(flat).reshape(splits[key].shape)

    for key in ["y_train", "y_val", "y_test"]:
        splits[key] = target_scaler.transform(splits[key].reshape(-1, 1)).reshape(-1)

    train_ds = SequenceDataset(splits["X_train"], splits["y_train"])
    val_ds = SequenceDataset(splits["X_val"], splits["y_val"])
    test_ds = SequenceDataset(splits["X_test"], splits["y_test"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    incidence = build_incidence(NODE_ORDER, HYPEREDGES)
    node_input_dims = [len(NODE_FEATURES[name]) for name in NODE_ORDER]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HGNNTemporalTransformer(
        node_input_dims=node_input_dims,
        incidence=incidence.to(device),
        d_model=args.d_model,
        nhead=args.nhead,
        num_transformer_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    topk = 5
    topk_records: List[Dict[str, float]] = []
    history_records: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)
            node_inputs = [t.to(device) for t in make_node_inputs(bx, NODE_FEATURES)]

            optimizer.zero_grad()
            pred = model(node_inputs)
            loss = criterion(pred, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * bx.size(0)

        train_loss = total_loss / len(train_ds)
        train_metrics = evaluate(model, train_loader, device, target_scaler)
        val_metrics = evaluate(model, val_loader, device, target_scaler)

        history_records.append(
            {
                "epoch": epoch,
                "train_mse": float(train_loss),
                "train_r2": float(train_metrics["r2"]),
                "val_rmse": float(val_metrics["rmse"]),
                "val_mae": float(val_metrics["mae"]),
                "val_mape": float(val_metrics["mape"]),
                "val_r2": float(val_metrics["r2"]),
            }
        )

        ckpt_name = f"epoch_{epoch:03d}_valrmse_{val_metrics['rmse']:.6f}.pt"
        ckpt_path = output_dir / ckpt_name
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "train_metrics": train_metrics,
                "node_order": NODE_ORDER,
                "node_features": NODE_FEATURES,
                "hyperedges": HYPEREDGES,
                "feature_scaler_mean": feature_scaler.mean,
                "feature_scaler_std": feature_scaler.std,
                "target_scaler_mean": target_scaler.mean,
                "target_scaler_std": target_scaler.std,
            },
            ckpt_path,
        )

        new_record = {
            "epoch": epoch,
            "val_rmse": float(val_metrics["rmse"]),
            "path": ckpt_path.name,
        }
        previous_names = {item["path"] for item in topk_records}
        topk_records = update_topk_checkpoints(topk_records, new_record, topk=topk)
        current_names = {item["path"] for item in topk_records}

        for removed_name in previous_names - current_names:
            removed_path = output_dir / removed_name
            if removed_path.exists():
                removed_path.unlink()

        if ckpt_path.name not in current_names and ckpt_path.exists():
            ckpt_path.unlink()

        print(
            f"Epoch {epoch:03d} | train_mse={train_loss:.6f} "
            f"| train_r2={train_metrics['r2']:.6f} "
            f"| val_rmse={val_metrics['rmse']:.6f} "
            f"| val_mae={val_metrics['mae']:.6f} "
            f"| val_r2={val_metrics['r2']:.6f}"
        )

    best_record = min(topk_records, key=lambda x: x["val_rmse"])
    best_path = output_dir / best_record["path"]
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, target_scaler)

    print("=" * 60)
    print("Best validation RMSE:", f"{best_record['val_rmse']:.6f}")
    print("Saved top-k checkpoints:", json.dumps(topk_records, ensure_ascii=True))
    print(
        "Test metrics:",
        json.dumps(
            {
                "rmse": round(test_metrics["rmse"], 6),
                "mae": round(test_metrics["mae"], 6),
                "mape_percent": round(test_metrics["mape"], 6),
                "r2": round(test_metrics["r2"], 6),
            },
            ensure_ascii=True,
        ),
    )

    history_path = output_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history_records, f, ensure_ascii=True, indent=2)

    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "best_val_rmse": best_record["val_rmse"],
                "best_checkpoint": best_record["path"],
                "topk_checkpoints": topk_records,
                "training_history_file": history_path.name,
                "test_metrics": test_metrics,
                "node_order": NODE_ORDER,
                "node_features": NODE_FEATURES,
                "hyperedges": HYPEREDGES,
            },
            f,
            ensure_ascii=True,
            indent=2,
        )

    print(f"Best model for testing: {best_path}")
    print(f"Saved history: {history_path}")
    print(f"Saved summary: {output_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
