"""Training dataset and sequence-window sampling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


SENSOR_DIM = 150


@dataclass(frozen=True)
class WindowRef:
    """Reference to one fixed-length window in one cached day."""

    day_index: int
    start: int


class CachedDay:
    """Memory-mapped arrays for one cached day."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.features = np.load(self.cache_dir / "features.npy", mmap_mode="r")
        self.sample_weight = np.load(self.cache_dir / "sample_weight.npy", mmap_mode="r")
        self.valid_target = np.load(self.cache_dir / "valid_target.npy", mmap_mode="r")

        target_sensor_path = self.cache_dir / "target_sensor.npy"
        self.target_sensor = (
            np.load(target_sensor_path, mmap_mode="r") if target_sensor_path.exists() else None
        )
        self.target_delta = np.load(self.cache_dir / "target_delta.npy", mmap_mode="r")

        target_mask_path = self.cache_dir / "target_mask.npy"
        fallback_mask_path = self.cache_dir / "loss_mask.npy"
        self.target_mask = np.load(
            target_mask_path if target_mask_path.exists() else fallback_mask_path, mmap_mode="r"
        )

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def target_slice(self, start: int, end: int) -> np.ndarray:
        """Return full-sensor targets for [start:end]."""
        if self.target_sensor is not None:
            return np.array(self.target_sensor[start:end], dtype=np.float32, copy=True)
        sensors = np.asarray(self.features[start:end, :SENSOR_DIM], dtype=np.float32)
        delta = np.asarray(self.target_delta[start:end], dtype=np.float32)
        return np.array(sensors + delta, dtype=np.float32, copy=True)


class SequenceWindowDataset(Dataset):
    """Cache-backed sliding-window dataset for teacher forcing and rollout stages."""

    def __init__(
        self,
        cache_root: str | Path,
        split: str,
        seq_len: int,
        stride: int,
        *,
        skip_empty_target_windows: bool = True,
        max_windows: int | None = None,
    ):
        self.cache_root = Path(cache_root)
        self.split = split
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")

        manifest_path = self.cache_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        day_entries = [entry for entry in manifest["dates"] if entry["split"] == split]
        if not day_entries:
            raise ValueError(f"No cache days found for split={split!r} in {manifest_path}")

        self.days = [CachedDay(entry["cache_dir"]) for entry in day_entries]
        self.day_entries = day_entries
        self.windows = self._build_windows(skip_empty_target_windows, max_windows)
        if not self.windows:
            raise ValueError(
                f"No windows available for split={split!r}, seq_len={seq_len}, stride={stride}"
            )

    def _build_windows(
        self, skip_empty_target_windows: bool, max_windows: int | None
    ) -> list[WindowRef]:
        windows: list[WindowRef] = []
        for day_index, day in enumerate(self.days):
            last_start = len(day) - self.seq_len
            if last_start < 0:
                continue
            for start in range(0, last_start + 1, self.stride):
                end = start + self.seq_len
                if skip_empty_target_windows and not bool(np.any(day.valid_target[start:end])):
                    continue
                windows.append(WindowRef(day_index=day_index, start=start))
                if max_windows is not None and len(windows) >= max_windows:
                    return windows
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ref = self.windows[index]
        day = self.days[ref.day_index]
        start = ref.start
        end = start + self.seq_len

        features = np.array(day.features[start:end], dtype=np.float32, copy=True)
        target = day.target_slice(start, end)
        target_mask = np.array(day.target_mask[start:end], dtype=np.float32, copy=True)
        sample_weight = np.array(day.sample_weight[start:end], dtype=np.float32, copy=True)

        return {
            "features": torch.from_numpy(features),
            "target": torch.from_numpy(target),
            "target_mask": torch.from_numpy(target_mask),
            "sample_weight": torch.from_numpy(sample_weight),
        }


def _stage_data_config(config: dict[str, Any], stage: dict[str, Any]) -> dict[str, Any]:
    data_config = dict(config.get("data", {}))
    data_config.update(stage.get("data", {}))
    return data_config


def make_dataloaders(config: dict[str, Any], stage: dict[str, Any] | None = None):
    """Create train/validation dataloaders from cached arrays."""
    stage = stage or {}
    data_config = _stage_data_config(config, stage)
    train_config = config.get("train", {})

    cache_root = data_config.get("cache_dir") or config.get("cache_dir")
    if cache_root is None:
        raise ValueError("cache_dir must be set in config.cache_dir or config.data.cache_dir")

    mode = stage.get("mode", "teacher_forcing")
    if mode == "rollout":
        seq_len = int(stage.get("context_len", train_config.get("seq_len", 1024))) + int(
            stage.get("rollout_steps", 128)
        )
    else:
        seq_len = int(stage.get("seq_len", train_config.get("seq_len", 1024)))

    batch_size = int(stage.get("batch_size", train_config.get("batch_size", 256)))
    num_workers = int(stage.get("num_workers", train_config.get("num_workers", 0)))
    pin_memory = bool(stage.get("pin_memory", train_config.get("pin_memory", False)))
    skip_empty = bool(data_config.get("skip_empty_target_windows", True))

    train_dataset = SequenceWindowDataset(
        cache_root,
        "train",
        seq_len,
        int(data_config.get("train_stride", data_config.get("stride", 512))),
        skip_empty_target_windows=skip_empty,
        max_windows=data_config.get("max_train_windows"),
    )
    val_dataset = SequenceWindowDataset(
        cache_root,
        "val",
        seq_len,
        int(data_config.get("val_stride", seq_len)),
        skip_empty_target_windows=skip_empty,
        max_windows=data_config.get("max_val_windows"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=bool(stage.get("shuffle", True)),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=bool(stage.get("drop_last", True)),
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(stage.get("val_batch_size", batch_size)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader
