"""scripts/build_humanml3de_mp_motion_formats.py"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.guofeats.common.quaternion import qrot_np, quaternion_to_cont6d_np
from src.guofeats.common.skeleton import Skeleton
from src.guofeats.paramUtil import t2m_kinematic_chain, t2m_raw_offsets


FACE_JOINT_INDEX = [2, 1, 17, 16]
LEFT_FOOT_INDEX = [7, 10]
RIGHT_FOOT_INDEX = [8, 11]
ROOT_JOINT_INDEX = 0
ROOT_HEADING_AXIS = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
KIMODO_SMOOTH_KERNEL = np.array([1.0, 2.0, 1.0], dtype=np.float32)
DEFAULT_SCHEMAS = (
    "guo263",
    "pos66",
    "smpl_d135_recon",
    "hy201_recon",
    "kimodo_like_261",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export HumanML3D-E-MP motions into multiple schema variants. "
            "Rotation-heavy schemas are reconstructed from raw 22-joint positions "
            "via inverse kinematics because no native SMPL rotation files are "
            "present in the local HumanML3D-E-MP / HumanML3D-E / HumanML3D assets."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E-MP"),
        help="Single-root HumanML3D-E-MP dataset directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Destination root. Defaults to <dataset-root>/motion_formats.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "val", "test"),
        help="Which split txt files to use when enumerating sample ids.",
    )
    parser.add_argument(
        "--schemas",
        nargs="+",
        default=DEFAULT_SCHEMAS,
        help=f"Schemas to export. Defaults to: {', '.join(DEFAULT_SCHEMAS)}.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on the number of samples to export. 0 means all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-sample schema files.",
    )
    return parser.parse_args()


def _load_pickle_dict(path: Path) -> Dict[str, Dict]:
    packed = np.load(path, allow_pickle=True)
    if isinstance(packed, np.ndarray) and packed.shape == ():
        packed = packed.item()
    if not isinstance(packed, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(packed).__name__}")
    return {str(key): value for key, value in packed.items()}


def _load_split_ids(dataset_root: Path, split: str) -> List[str]:
    split_path = dataset_root / f"{split}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    return [
        line.strip()
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_guo_map(dataset_root: Path, splits: Iterable[str]) -> Dict[str, np.ndarray]:
    guo_map: Dict[str, np.ndarray] = {}
    split_list = list(splits)
    for split in tqdm(split_list, desc="Loading guo263 source", unit="split"):
        split_path = dataset_root / f"data_{split}.npy"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing packaged motion file: {split_path}")
        split_data = _load_pickle_dict(split_path)
        for sample_id, sample in split_data.items():
            motion = sample.get("motion")
            if motion is None:
                continue
            guo_map[str(sample_id)] = np.asarray(motion, dtype=np.float32)
    return guo_map


def _build_source_audit(dataset_root: Path) -> Dict[str, object]:
    humanml3de_root = dataset_root.parent / "HumanML3D-E"
    humanml3d_root = dataset_root.parent / "HumanML3D" / "HumanML3D"
    audit = {
        "humanml3de_mp": {
            "root": str(dataset_root),
            "native_rotation_files_present": False,
            "raw_joint_dir_present": (dataset_root / "new_joints").exists(),
            "packaged_motion_present": (dataset_root / "data_train.npy").exists(),
        },
        "humanml3de": {
            "root": str(humanml3de_root),
            "native_rotation_files_present": False,
            "packaged_motion_present": (humanml3de_root / "data_train.npy").exists(),
        },
        "humanml3d": {
            "root": str(humanml3d_root),
            "native_rotation_files_present": False,
            "available_motion_dirs": [
                name
                for name in (
                    "joints",
                    "new_joints",
                    "new_joint_vecs",
                    "new_joints_abs_3d",
                    "new_joint_vecs_abs_3d",
                )
                if (humanml3d_root / name).exists()
            ],
        },
    }
    return audit


def _make_skeleton() -> Skeleton:
    raw_offsets = torch.from_numpy(t2m_raw_offsets)
    return Skeleton(raw_offsets, t2m_kinematic_chain, "cpu")


def _load_positions(path: Path) -> np.ndarray:
    positions = np.load(path).astype(np.float32, copy=False)
    if positions.ndim != 3 or positions.shape[1:] != (22, 3):
        raise ValueError(f"Unexpected motion shape in {path}: {positions.shape}")
    return positions


def _estimate_rotations(
    positions: np.ndarray,
    skeleton: Skeleton,
) -> tuple[np.ndarray, np.ndarray]:
    quat = skeleton.inverse_kinematics_np(
        positions,
        FACE_JOINT_INDEX,
        smooth_forward=True,
    )
    cont6d = quaternion_to_cont6d_np(quat).astype(np.float32)
    return quat.astype(np.float32), cont6d


def _root_xz_velocity(root_pos: np.ndarray) -> np.ndarray:
    velocity = np.zeros((len(root_pos), 2), dtype=np.float32)
    if len(root_pos) > 1:
        velocity[1:] = root_pos[1:, [0, 2]] - root_pos[:-1, [0, 2]]
    return velocity


def _local_positions(positions: np.ndarray, root_quat: np.ndarray) -> np.ndarray:
    root_pos = positions[:, ROOT_JOINT_INDEX : ROOT_JOINT_INDEX + 1]
    relative = positions - root_pos
    repeated_quat = np.repeat(root_quat[:, None, :], positions.shape[1], axis=1)
    local = qrot_np(repeated_quat, relative)
    return local.astype(np.float32)


def _compute_forward_heading(positions: np.ndarray) -> np.ndarray:
    left_hip, right_hip, shoulder_r, shoulder_l = FACE_JOINT_INDEX
    across = (positions[:, right_hip] - positions[:, left_hip]) + (
        positions[:, shoulder_r] - positions[:, shoulder_l]
    )
    across_norm = np.linalg.norm(across, axis=-1, keepdims=True).clip(min=1.0e-8)
    across = across / across_norm
    forward = np.cross(
        np.repeat(ROOT_HEADING_AXIS, len(across), axis=0),
        across,
        axis=-1,
    )
    forward_norm = np.linalg.norm(forward[:, [0, 2]], axis=-1, keepdims=True).clip(
        min=1.0e-8
    )
    return (forward[:, [0, 2]] / forward_norm).astype(np.float32)


def _smooth_root_positions(root_pos: np.ndarray) -> np.ndarray:
    kernel = KIMODO_SMOOTH_KERNEL / KIMODO_SMOOTH_KERNEL.sum()
    padded = np.pad(root_pos, ((1, 1), (0, 0)), mode="edge")
    smoothed = np.empty_like(root_pos)
    for dim in range(root_pos.shape[1]):
        smoothed[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
    return smoothed.astype(np.float32)


def _joint_velocity(positions: np.ndarray) -> np.ndarray:
    velocity = np.zeros_like(positions, dtype=np.float32)
    if len(positions) > 1:
        velocity[1:] = positions[1:] - positions[:-1]
    return velocity


def _foot_contact(positions: np.ndarray, threshold: float = 0.002) -> np.ndarray:
    left = np.zeros((len(positions), 2), dtype=np.float32)
    right = np.zeros((len(positions), 2), dtype=np.float32)
    if len(positions) <= 1:
        return np.concatenate([left, right], axis=-1)

    left_speed = np.sum(
        np.square(positions[1:, LEFT_FOOT_INDEX] - positions[:-1, LEFT_FOOT_INDEX]),
        axis=-1,
    )
    right_speed = np.sum(
        np.square(positions[1:, RIGHT_FOOT_INDEX] - positions[:-1, RIGHT_FOOT_INDEX]),
        axis=-1,
    )
    left[1:] = (left_speed < threshold).astype(np.float32)
    right[1:] = (right_speed < threshold).astype(np.float32)
    return np.concatenate([left, right], axis=-1)


def _build_pos66(positions: np.ndarray) -> np.ndarray:
    return positions.reshape(len(positions), -1).astype(np.float32)


def _build_smpl_d135_recon(
    positions: np.ndarray,
    cont6d: np.ndarray,
) -> np.ndarray:
    root_pos = positions[:, ROOT_JOINT_INDEX]
    root_feat = np.concatenate(
        [
            cont6d[:, ROOT_JOINT_INDEX],
            _root_xz_velocity(root_pos),
            root_pos[:, 1:2],
        ],
        axis=-1,
    )
    joint_rot = cont6d[:, 1:].reshape(len(cont6d), -1)
    return np.concatenate([root_feat, joint_rot], axis=-1).astype(np.float32)


def _build_hy201_recon(
    positions: np.ndarray,
    quat: np.ndarray,
    cont6d: np.ndarray,
) -> np.ndarray:
    root_pos = positions[:, ROOT_JOINT_INDEX]
    local_joint_pos = _local_positions(positions, quat[:, ROOT_JOINT_INDEX]).reshape(
        len(positions),
        -1,
    )
    local_rot = cont6d[:, 1:].reshape(len(cont6d), -1)
    hy201 = np.concatenate(
        [
            root_pos,
            cont6d[:, ROOT_JOINT_INDEX],
            local_rot,
            local_joint_pos,
        ],
        axis=-1,
    )
    return hy201.astype(np.float32)


def _build_kimodo_like_261(
    positions: np.ndarray,
    cont6d: np.ndarray,
) -> np.ndarray:
    root_pos = positions[:, ROOT_JOINT_INDEX]
    non_root_pos = positions[:, 1:]
    kimodo_like = np.concatenate(
        [
            _smooth_root_positions(root_pos),
            _compute_forward_heading(positions),
            non_root_pos.reshape(len(positions), -1),
            _joint_velocity(non_root_pos).reshape(len(positions), -1),
            cont6d[:, 1:].reshape(len(cont6d), -1),
            _foot_contact(positions),
        ],
        axis=-1,
    )
    return kimodo_like.astype(np.float32)


def _schema_metadata() -> Dict[str, Dict[str, object]]:
    return {
        "guo263": {
            "dim": 263,
            "source": "packaged HumanML3D-E-MP data_<split>.npy",
            "native_rotation": False,
            "note": "Existing Guo/HumanML3D packaged feature, includes reconstructed 6D joint rotations inside the 263-d view.",
        },
        "pos66": {
            "dim": 66,
            "source": "flattened HumanML3D-E-MP new_joints",
            "native_rotation": False,
            "note": "Global 22x3 joint positions from MotionPatches raw-joint input.",
        },
        "smpl_d135_recon": {
            "dim": 135,
            "source": "IK reconstructed from HumanML3D-E-MP new_joints",
            "native_rotation": False,
            "note": "6D root orientation + root xz velocity + root height + 21x6D joint rotations. Reconstructed, not native SMPL parameters.",
        },
        "hy201_recon": {
            "dim": 201,
            "source": "IK reconstructed from HumanML3D-E-MP new_joints",
            "native_rotation": False,
            "note": "3D root translation + 6D root orientation + 21x6D local rotations + 22x3 root-frame local positions. Reconstructed, not native HY-Motion files.",
        },
        "kimodo_like_261": {
            "dim": 261,
            "source": "IK reconstructed from HumanML3D-E-MP new_joints",
            "native_rotation": False,
            "note": "Smoothed root position + 2D heading + 21x3 global joint positions + 21x3 velocities + 21x6D angles + 4D foot contact. Kimodo-like approximation aligned to 21 non-root joints.",
        },
    }


def _export_sample(
    sample_id: str,
    positions: np.ndarray,
    guo_motion: np.ndarray | None,
    schemas: Iterable[str],
    schema_dirs: Dict[str, Path],
    overwrite: bool,
    skeleton: Skeleton,
) -> Dict[str, List[int]]:
    exported_shapes: Dict[str, List[int]] = {}
    quat: np.ndarray | None = None
    cont6d: np.ndarray | None = None

    for schema in schemas:
        out_path = schema_dirs[schema] / f"{sample_id}.npy"
        if out_path.exists() and not overwrite:
            exported_shapes[schema] = list(np.load(out_path, mmap_mode="r").shape)
            continue

        if schema == "guo263":
            if guo_motion is None:
                raise KeyError(f"Missing guo263 packaged motion for sample_id={sample_id}")
            array = guo_motion
        elif schema == "pos66":
            array = _build_pos66(positions)
        else:
            if quat is None or cont6d is None:
                quat, cont6d = _estimate_rotations(positions, skeleton)

            if schema == "smpl_d135_recon":
                array = _build_smpl_d135_recon(positions, cont6d)
            elif schema == "hy201_recon":
                array = _build_hy201_recon(positions, quat, cont6d)
            elif schema == "kimodo_like_261":
                array = _build_kimodo_like_261(positions, cont6d)
            else:
                raise ValueError(f"Unsupported schema: {schema}")

        np.save(out_path, array.astype(np.float32, copy=False))
        exported_shapes[schema] = list(array.shape)

    return exported_shapes


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else (dataset_root / "motion_formats").resolve()
    )
    schemas = tuple(args.schemas)
    schema_meta = _schema_metadata()

    for schema in schemas:
        if schema not in schema_meta:
            raise ValueError(
                f"Unsupported schema={schema}. Expected one of {sorted(schema_meta)}."
            )

    if not (dataset_root / "new_joints").exists():
        raise FileNotFoundError(f"Missing new_joints under {dataset_root}")

    sample_ids: List[str] = []
    sample_to_split: Dict[str, str] = {}
    for split in args.splits:
        split_ids = _load_split_ids(dataset_root, split)
        for sample_id in split_ids:
            if sample_id not in sample_to_split:
                sample_to_split[sample_id] = split
                sample_ids.append(sample_id)

    if args.limit > 0:
        sample_ids = sample_ids[: args.limit]

    print(
        f"[build_humanml3de_mp_motion_formats] "
        f"samples={len(sample_ids)} schemas={len(schemas)} output_root={output_root}"
    )

    guo_map = _load_guo_map(dataset_root, args.splits)
    skeleton = _make_skeleton()

    output_root.mkdir(parents=True, exist_ok=True)
    schema_dirs = {}
    for schema in schemas:
        schema_dir = output_root / schema
        schema_dir.mkdir(parents=True, exist_ok=True)
        schema_dirs[schema] = schema_dir

    export_manifest_path = output_root / "export_manifest.jsonl"
    metadata_path = output_root / "metadata.json"

    exported_count = 0
    with export_manifest_path.open("w", encoding="utf-8") as manifest_f:
        progress = tqdm(
            sample_ids,
            desc="Exporting motion formats",
            unit="sample",
        )
        for sample_id in progress:
            motion_path = dataset_root / "new_joints" / f"{sample_id}.npy"
            positions = _load_positions(motion_path)
            guo_motion = guo_map.get(sample_id)
            exported_shapes = _export_sample(
                sample_id=sample_id,
                positions=positions,
                guo_motion=guo_motion,
                schemas=schemas,
                schema_dirs=schema_dirs,
                overwrite=args.overwrite,
                skeleton=skeleton,
            )
            record = {
                "sample_id": sample_id,
                "split": sample_to_split[sample_id],
                "source_motion_path": str(motion_path),
                "exported_shapes": exported_shapes,
            }
            manifest_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            exported_count += 1
            progress.set_postfix(
                split=sample_to_split[sample_id],
                sample_id=sample_id,
                refresh=False,
            )

    metadata = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "splits": list(args.splits),
        "sample_count": exported_count,
        "schemas": {schema: schema_meta[schema] for schema in schemas},
        "source_audit": _build_source_audit(dataset_root),
        "notes": [
            "HumanML3D-E-MP and HumanML3D-E do not expose native stored SMPL rotation files; only packaged 263-d motion is present in data_<split>.npy.",
            "The local HumanML3D root exposes joints/new_joints/new_joint_vecs assets but no native stored SMPL pose tensors. Rotation-heavy schemas exported here are reconstructed from 22-joint positions via inverse kinematics.",
            "Use manifest.jsonl under HumanML3D-E-MP for provenance back to the original HumanML3D raw-joint file and clip span when needed.",
        ],
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Exported {exported_count} samples to {output_root}")
    print(f"Manifest: {export_manifest_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
