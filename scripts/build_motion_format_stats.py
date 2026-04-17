"""scripts/build_motion_format_stats.py"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm


DEFAULT_SCHEMAS = (
    "guo263",
    "pos66",
    "smpl_d135_recon",
    "hy201_recon",
    "kimodo_like_261",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-dimension motion normalization stats for each schema."
    )
    parser.add_argument(
        "--format-root",
        type=Path,
        required=True,
        help="Root produced by build_humanml3de_mp_motion_formats.py.",
    )
    parser.add_argument(
        "--schemas",
        nargs="+",
        default=DEFAULT_SCHEMAS,
        help=f"Schemas to process. Defaults to: {', '.join(DEFAULT_SCHEMAS)}.",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        required=True,
        help="Train split file. Stats are computed from these sample ids only.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory that will receive {schema}/Mean.npy and {schema}/Std.npy.",
    )
    return parser.parse_args()


def load_split_ids(split_file: Path) -> list[str]:
    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file}")
    return [
        line.strip()
        for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_motion_array(path: Path) -> np.ndarray:
    array = np.load(path).astype(np.float32, copy=False)
    if array.ndim == 3:
        return array.reshape(array.shape[0], -1)
    if array.ndim == 2:
        return array
    raise ValueError(f"Unsupported motion shape in {path}: {array.shape}")


def iter_motion_paths(schema_dir: Path, sample_ids: Iterable[str]) -> Iterable[Path]:
    for sample_id in sample_ids:
        motion_path = schema_dir / f"{sample_id}.npy"
        if not motion_path.exists():
            raise FileNotFoundError(f"Missing motion file: {motion_path}")
        yield motion_path


def compute_stats(
    paths: Iterable[Path],
    *,
    schema: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    path_list = list(paths)
    total_sum: np.ndarray | None = None
    total_sumsq: np.ndarray | None = None
    total_frames = 0

    for path in tqdm(
        path_list,
        desc=f"[{schema}] Accumulating frames",
        unit="file",
        leave=False,
    ):
        motion = load_motion_array(path)
        if motion.shape[0] == 0:
            continue

        motion64 = motion.astype(np.float64, copy=False)
        if total_sum is None:
            total_sum = np.zeros(motion64.shape[1], dtype=np.float64)
            total_sumsq = np.zeros(motion64.shape[1], dtype=np.float64)

        if motion64.shape[1] != total_sum.shape[0]:
            raise ValueError(
                f"Inconsistent feature dim in {path}: "
                f"expected {total_sum.shape[0]}, got {motion64.shape[1]}"
            )

        total_sum += motion64.sum(axis=0)
        total_sumsq += np.square(motion64).sum(axis=0)
        total_frames += int(motion64.shape[0])

    if total_sum is None or total_sumsq is None or total_frames == 0:
        raise ValueError("No valid motion frames were found while computing stats.")

    mean = total_sum / float(total_frames)
    variance = np.maximum(total_sumsq / float(total_frames) - np.square(mean), 0.0)
    std = np.sqrt(variance).clip(min=1.0e-6)
    return mean.astype(np.float32), std.astype(np.float32), total_frames


def main() -> None:
    args = parse_args()
    sample_ids = load_split_ids(args.split_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[build_motion_format_stats] "
        f"schemas={len(args.schemas)} samples={len(sample_ids)} output_dir={args.output_dir}"
    )

    for schema in tqdm(args.schemas, desc="Computing schema stats", unit="schema"):
        schema_dir = args.format_root / schema
        if not schema_dir.exists():
            raise FileNotFoundError(f"Missing schema directory: {schema_dir}")

        schema_paths = list(iter_motion_paths(schema_dir=schema_dir, sample_ids=sample_ids))
        mean, std, total_frames = compute_stats(
            schema_paths,
            schema=schema,
        )

        output_dir = args.output_dir / schema
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "Mean.npy", mean)
        np.save(output_dir / "Std.npy", std)
        print(
            f"[build_motion_format_stats] schema={schema} "
            f"files={len(schema_paths)} dim={mean.shape[0]} total_frames={total_frames} "
            f"to {output_dir}"
        )


if __name__ == "__main__":
    main()
