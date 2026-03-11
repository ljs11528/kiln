from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from hgnn_config import COL_INDEX, NODE_ORDER


@dataclass
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "StandardScaler":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-8] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


class SequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


def read_all_csv(data_dir: Path) -> List[np.ndarray]:
    arrays = []
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No csv files found in: {data_dir}")

    for path in csv_files:
        df = pd.read_csv(path, header=None)
        if df.shape[1] != 20:
            raise ValueError(f"{path.name} has {df.shape[1]} cols, expected 20 after cleaning.")

        # Convert time text (HH:MM:SS) to seconds first, then write into a numeric dataframe.
        time_sec = pd.to_timedelta(df.iloc[:, COL_INDEX["time"]], errors="coerce").dt.total_seconds()
        df_num = df.apply(pd.to_numeric, errors="coerce")
        df_num.iloc[:, COL_INDEX["time"]] = time_sec

        arr = df_num.to_numpy(dtype=np.float32)
        if np.isnan(arr).any():
            raise ValueError(f"NaN found after numeric conversion in {path.name}")
        arrays.append(arr)
    return arrays


def build_samples_per_file(
    file_arrays: List[np.ndarray],
    seq_len: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x_list = []
    y_list = []

    for arr in file_arrays:
        n = arr.shape[0]
        for end in range(seq_len, n - horizon + 1):
            x_seq = arr[end - seq_len : end, :]
            y = arr[end + horizon - 1, COL_INDEX["co"]]
            x_list.append(x_seq)
            y_list.append(y)

    if not x_list:
        raise ValueError("No sequence samples generated. Check seq_len/horizon and file lengths.")

    X = np.stack(x_list, axis=0)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, np.ndarray]:
    n = X.shape[0]
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return {
        "X_train": X[:train_end],
        "y_train": y[:train_end],
        "X_val": X[train_end:val_end],
        "y_val": y[train_end:val_end],
        "X_test": X[val_end:],
        "y_test": y[val_end:],
    }


def make_node_inputs(batch_x: torch.Tensor, node_features: Dict[str, List[int]]) -> List[torch.Tensor]:
    node_inputs = []
    for node_name in NODE_ORDER:
        idxs = node_features[node_name]
        node_inputs.append(batch_x[:, :, idxs])
    return node_inputs
