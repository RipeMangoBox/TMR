"""src/data/motion_repr_dataset.py"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

MIN_STD_NORMALIZATION = 1.0e-3


def _load_ids(split_file: Path) -> list[str]:
    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file}")
    return [
        line.strip()
        for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_stat_vector(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing stats file: {path}")
    vector = np.load(path).astype(np.float32, copy=False)
    return vector.reshape(-1)


def _load_caption_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing text file: {path}")

    captions: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("#")
        caption = parts[0].strip() if parts else ""
        if caption:
            captions.append(caption)
    if not captions:
        raise ValueError(f"No valid captions found in {path}")
    return captions


def _load_motion_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing motion file: {path}")

    motion = np.load(path).astype(np.float32, copy=False)
    if motion.ndim == 3:
        return motion.reshape(motion.shape[0], -1)
    if motion.ndim == 2:
        return motion
    raise ValueError(f"Unsupported motion shape in {path}: {motion.shape}")


class MotionReprDataset(Dataset):
    """
    Generic motion-text dataset for motion representation ablation.
    """

    def __init__(
        self,
        motion_dir: str,
        text_dir: str,
        split_file: str,
        mean_path: str,
        std_path: str,
        max_motion_length: int = 224,
        text_encoder: str = "distilbert",
    ) -> None:
        self.motion_dir = Path(motion_dir).expanduser().resolve()
        self.text_dir = Path(text_dir).expanduser().resolve()
        self.split_file = Path(split_file).expanduser().resolve()
        self.mean = _load_stat_vector(Path(mean_path).expanduser().resolve())
        self.std = _load_stat_vector(Path(std_path).expanduser().resolve())
        self.safe_std = self.std.copy()
        tiny_mask = self.safe_std < MIN_STD_NORMALIZATION
        if np.any(tiny_mask):
            self.safe_std[tiny_mask] = 1.0
        self.max_motion_length = int(max_motion_length)
        self.text_encoder = text_encoder
        self.is_training = self.split_file.stem.lower() == "train"

        raw_names = _load_ids(self.split_file)
        self.caption_map: Dict[str, List[str]] = {}
        self.names: list[str] = []

        for name in raw_names:
            text_path = self.text_dir / f"{name}.txt"
            motion_path = self.motion_dir / f"{name}.npy"
            if not motion_path.exists():
                raise FileNotFoundError(f"Missing motion file: {motion_path}")
            captions = _load_caption_list(text_path)
            self.caption_map[name] = captions
            self.names.append(name)

        if not self.names:
            raise ValueError(f"No valid samples found for split: {self.split_file}")

        sample_motion = _load_motion_array(self.motion_dir / f"{self.names[0]}.npy")
        self.motion_dim = int(sample_motion.shape[1])
        if self.mean.shape[0] != self.motion_dim or self.std.shape[0] != self.motion_dim:
            raise ValueError(
                "Stats dimension mismatch: "
                f"motion_dim={self.motion_dim}, mean={self.mean.shape}, std={self.std.shape}"
            )

    def __len__(self) -> int:
        return len(self.names)

    def _select_caption(self, name: str) -> str:
        captions = self.caption_map[name]
        if self.is_training and len(captions) > 1:
            return random.choice(captions)
        return captions[0]

    def _normalize_motion(self, motion: np.ndarray) -> np.ndarray:
        normalized = (motion - self.mean[np.newaxis, :]) / self.safe_std[np.newaxis, :]
        return normalized.astype(np.float32, copy=False)

    def _crop_or_pad(self, motion: np.ndarray) -> tuple[np.ndarray, int]:
        motion_length = int(motion.shape[0])
        if motion_length >= self.max_motion_length:
            start = 0
            if self.is_training and motion_length > self.max_motion_length:
                start = random.randint(0, motion_length - self.max_motion_length)
            motion = motion[start : start + self.max_motion_length]
            return motion, self.max_motion_length

        pad_length = self.max_motion_length - motion_length
        padding = np.zeros((pad_length, motion.shape[1]), dtype=np.float32)
        motion = np.concatenate([motion, padding], axis=0)
        return motion, motion_length

    def __getitem__(self, index: int) -> Dict[str, object]:
        name = self.names[index]
        motion = _load_motion_array(self.motion_dir / f"{name}.npy")
        motion = self._normalize_motion(motion)
        motion, motion_length = self._crop_or_pad(motion)

        return {
            "motion": torch.from_numpy(motion).to(dtype=torch.float32),
            "motion_length": motion_length,
            "caption": self._select_caption(name),
            "name": name,
        }
