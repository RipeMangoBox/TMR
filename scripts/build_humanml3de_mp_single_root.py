#!/usr/bin/env python3
"""
Build a single-root MotionPatches-friendly dataset from:
1) HumanML3D-E split / decomposed-event metadata
2) HumanML3D raw motion assets (`new_joints`)
3) HumanML3D raw text assets (`texts`)

The derived dataset root contains only samples that are fully usable by the
MotionPatches raw-joint pipeline, while preserving the original HumanML3D-E
263-d motion view for EventT2M-style loaders:
- `new_joints/<sample_id>.npy` (full motion or reconstructed subclip)
- `texts/<sample_id>.txt`      (captions from HumanML3D-E, all tagged as 0.0/0.0)
- `data_{train,val,test}.npy`  (filtered metadata with 263-d `motion` + `length` + `text`)
- `train.txt`, `val.txt`, `test.txt`, `all.txt`, `nsim_test.txt`
- `Mean_raw.npy`, `Std_raw.npy`
- `metadata.json`, `manifest.jsonl`

Notes:
- HumanML3D-E `motion` is 263-dim and is NOT used as training motion here.
- Prefixed keys like `V_000012` are reconstructed from raw HumanML3D motion by
  matching the HumanML3D-E caption against the base HumanML3D text file and
  applying the corresponding `f_tag/to_tag` span.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


PREFIX_RE = re.compile(r"^[A-Za-z]_(.+)$")


@dataclass(frozen=True)
class RawTextEntry:
    caption: str
    tokens: List[str]
    f_tag: float
    to_tag: float
    index: int


@dataclass(frozen=True)
class SpanChoice:
    start: int
    end: int
    votes: int
    token_votes: int
    len_diff: int
    caption: str
    line_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a single-root HumanML3D-E-MP derived dataset."
    )
    parser.add_argument(
        "--humanml3de-root",
        type=Path,
        default=Path("/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E"),
        help="Root containing HumanML3D-E data_{train,val,test}.npy files.",
    )
    parser.add_argument(
        "--humanml3d-root",
        type=Path,
        default=Path("/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D/HumanML3D"),
        help="Root containing HumanML3D raw assets: new_joints/, texts/, Mean_raw.npy, Std_raw.npy.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E-MP"),
        help="Destination root for the derived single-directory dataset.",
    )
    parser.add_argument(
        "--nsim-split",
        type=Path,
        default=Path("/home/ripemangobox/Coding/Github/Motion/TMR/MotionPatches-main/datasets/annotations/humanml3d/splits/nsim_test.txt"),
        help="Official HumanML3D nsim split file used to build nsim_test.txt.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frame rate used by HumanML3D raw motions.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="How to materialize full-motion files that can be reused directly.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild output-root if it already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on the number of samples per split (for smoke testing). 0 = full build.",
    )
    return parser.parse_args()


def ensure_empty_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root already exists: {path}. Pass --overwrite to rebuild."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def canonical_base_id(sample_id: str) -> str:
    match = PREFIX_RE.match(sample_id)
    return match.group(1) if match else sample_id


def is_prefixed_id(sample_id: str) -> bool:
    return PREFIX_RE.match(sample_id) is not None


def load_split_dict(path: Path) -> Dict[str, Dict]:
    packed = np.load(path, allow_pickle=True)
    if isinstance(packed, np.ndarray) and packed.shape == ():
        packed = packed.item()
    if not isinstance(packed, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(packed).__name__}")
    return {str(k): v for k, v in packed.items()}


def parse_raw_text_file(path: Path) -> List[RawTextEntry]:
    entries: List[RawTextEntry] = []
    with path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split("#")
            if len(parts) < 4:
                continue
            caption = parts[0].strip()
            tokens = [tok for tok in parts[1].split(" ") if tok]
            f_tag = float(parts[2]) if parts[2] else 0.0
            to_tag = float(parts[3]) if parts[3] else 0.0
            if math.isnan(f_tag):
                f_tag = 0.0
            if math.isnan(to_tag):
                to_tag = 0.0
            entries.append(
                RawTextEntry(
                    caption=caption,
                    tokens=tokens,
                    f_tag=f_tag,
                    to_tag=to_tag,
                    index=index,
                )
            )
    return entries


def raw_span_from_tags(
    entry: RawTextEntry,
    raw_motion_len: int,
    fps: int,
) -> Tuple[int, int]:
    if entry.f_tag == 0.0 and entry.to_tag == 0.0:
        return 0, raw_motion_len
    start = max(0, int(entry.f_tag * fps))
    end = int(entry.to_tag * fps)
    if end <= 0 or end > raw_motion_len:
        end = raw_motion_len
    end = max(start, min(end, raw_motion_len))
    return start, end


def choose_prefixed_span(
    sample_id: str,
    sample: Dict,
    raw_text_entries: Sequence[RawTextEntry],
    raw_motion_len: int,
    fps: int,
) -> Tuple[Optional[Tuple[int, int]], Dict[str, object]]:
    sample_len = int(sample.get("length", 0))
    span_stats: Dict[Tuple[int, int], Dict[str, object]] = {}

    for text_item in sample.get("text", []):
        if not isinstance(text_item, dict):
            continue
        caption = str(text_item.get("caption", "")).strip()
        tokens = [str(tok) for tok in text_item.get("tokens", []) if str(tok)]
        for raw_entry in raw_text_entries:
            if raw_entry.caption != caption:
                continue
            start, end = raw_span_from_tags(raw_entry, raw_motion_len=raw_motion_len, fps=fps)
            clip_len = end - start
            key = (start, end)
            stats = span_stats.setdefault(
                key,
                {
                    "start": start,
                    "end": end,
                    "votes": 0,
                    "token_votes": 0,
                    "len_diff": abs(clip_len - sample_len),
                    "caption": caption,
                    "line_index": raw_entry.index,
                },
            )
            stats["votes"] = int(stats["votes"]) + 1
            if tokens and raw_entry.tokens == tokens:
                stats["token_votes"] = int(stats["token_votes"]) + 1
            stats["len_diff"] = min(int(stats["len_diff"]), abs(clip_len - sample_len))

    if not span_stats:
        if sample_len == raw_motion_len:
            return (0, raw_motion_len), {
                "resolution": "prefixed_fallback_full_motion_no_caption_match",
                "votes": 0,
                "token_votes": 0,
                "len_diff": 0,
            }
        return None, {
            "resolution": "prefixed_no_caption_match",
            "votes": 0,
            "token_votes": 0,
            "len_diff": None,
        }

    ranked = sorted(
        (
            SpanChoice(
                start=int(stats["start"]),
                end=int(stats["end"]),
                votes=int(stats["votes"]),
                token_votes=int(stats["token_votes"]),
                len_diff=int(stats["len_diff"]),
                caption=str(stats["caption"]),
                line_index=int(stats["line_index"]),
            )
            for stats in span_stats.values()
        ),
        key=lambda item: (
            -item.votes,
            -item.token_votes,
            item.len_diff,
            item.start,
            item.end,
            item.line_index,
        ),
    )
    best = ranked[0]
    return (best.start, best.end), {
        "resolution": "prefixed_caption_matched",
        "votes": best.votes,
        "token_votes": best.token_votes,
        "len_diff": best.len_diff,
        "caption": best.caption,
        "line_index": best.line_index,
    }


def write_text_file(path: Path, text_entries: Sequence[Dict]) -> None:
    lines: List[str] = []
    for text_item in text_entries:
        if not isinstance(text_item, dict):
            continue
        caption = str(text_item.get("caption", "")).strip()
        tokens = " ".join(str(tok) for tok in text_item.get("tokens", []) if str(tok))
        if not caption:
            continue
        # The derived raw-joint clip already corresponds to this sample unit, so
        # all captions should describe the whole clip.
        lines.append(f"{caption}#{tokens}#0.0#0.0")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def materialize_full_motion(src: Path, dst: Path, link_mode: str) -> str:
    if dst.exists():
        dst.unlink()
    if link_mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy_fallback"
    if link_mode == "symlink":
        dst.symlink_to(src)
        return "symlink"
    shutil.copy2(src, dst)
    return "copy"


def compute_mean_std_streaming(motion_paths: Iterable[Path]) -> Tuple[np.ndarray, np.ndarray, int]:
    total_frames = 0
    sum_arr: Optional[np.ndarray] = None
    sumsq_arr: Optional[np.ndarray] = None

    for motion_path in motion_paths:
        motion = np.load(motion_path).astype(np.float64, copy=False)
        if motion.ndim != 3:
            raise ValueError(f"Unexpected motion shape in {motion_path}: {motion.shape}")
        if sum_arr is None:
            sum_arr = motion.sum(axis=0)
            sumsq_arr = np.square(motion).sum(axis=0)
        else:
            sum_arr += motion.sum(axis=0)
            sumsq_arr += np.square(motion).sum(axis=0)
        total_frames += int(motion.shape[0])

    if total_frames == 0 or sum_arr is None or sumsq_arr is None:
        raise RuntimeError("Cannot compute Mean_raw / Std_raw with zero total frames.")

    mean = sum_arr / float(total_frames)
    var = np.maximum(sumsq_arr / float(total_frames) - np.square(mean), 1e-12)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32), total_frames


def copy_small_file(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()

    humanml3de_root = args.humanml3de_root.expanduser().resolve()
    humanml3d_root = args.humanml3d_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    nsim_split = args.nsim_split.expanduser().resolve()

    for required in [
        humanml3de_root / "data_train.npy",
        humanml3de_root / "data_val.npy",
        humanml3de_root / "data_test.npy",
        humanml3d_root / "new_joints",
        humanml3d_root / "texts",
        humanml3d_root / "Mean_raw.npy",
        humanml3d_root / "Std_raw.npy",
        nsim_split,
    ]:
        if not required.exists():
            raise FileNotFoundError(f"Missing required input: {required}")

    ensure_empty_dir(output_root, overwrite=args.overwrite)
    new_joints_root = output_root / "new_joints"
    texts_root = output_root / "texts"
    new_joints_root.mkdir(parents=True, exist_ok=True)
    texts_root.mkdir(parents=True, exist_ok=True)

    split_names = ("train", "val", "test")
    raw_motion_root = humanml3d_root / "new_joints"
    raw_text_root = humanml3d_root / "texts"

    official_nsim = [
        line.strip()
        for line in nsim_split.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    metadata: Dict[str, object] = {
        "source": {
            "humanml3de_root": str(humanml3de_root),
            "humanml3d_root": str(humanml3d_root),
            "nsim_split": str(nsim_split),
            "fps": int(args.fps),
            "link_mode": args.link_mode,
        },
        "counts": {},
        "reconstruction": {},
        "dropped_reasons": {},
    }

    manifest_path = output_root / "manifest.jsonl"
    manifest_f = manifest_path.open("w", encoding="utf-8")

    derived_splits: Dict[str, Dict[str, Dict[str, object]]] = {}
    usable_split_ids: Dict[str, List[str]] = {}
    motion_paths_for_stats: List[Path] = []

    try:
        for split_name in split_names:
            split_data = load_split_dict(humanml3de_root / f"data_{split_name}.npy")
            split_items = list(split_data.items())
            if args.limit > 0:
                split_items = split_items[: args.limit]

            derived_dict: Dict[str, Dict[str, object]] = {}
            usable_ids: List[str] = []
            dropped_counter: Counter = Counter()
            reconstruction_counter: Counter = Counter()

            for sample_id, sample in split_items:
                base_id = canonical_base_id(sample_id)
                raw_motion_path = raw_motion_root / f"{base_id}.npy"
                raw_text_path = raw_text_root / f"{base_id}.txt"

                if not raw_motion_path.exists():
                    dropped_counter["missing_raw_motion"] += 1
                    continue
                if not raw_text_path.exists():
                    dropped_counter["missing_raw_text"] += 1
                    continue

                raw_motion = np.load(raw_motion_path)
                raw_text_entries = parse_raw_text_file(raw_text_path)

                if is_prefixed_id(sample_id):
                    span, span_info = choose_prefixed_span(
                        sample_id=sample_id,
                        sample=sample,
                        raw_text_entries=raw_text_entries,
                        raw_motion_len=int(raw_motion.shape[0]),
                        fps=int(args.fps),
                    )
                    if span is None:
                        dropped_counter[str(span_info["resolution"])] += 1
                        continue
                    start, end = span
                    derived_motion = raw_motion[start:end].astype(np.float32, copy=False)
                    if derived_motion.shape[0] == 0:
                        dropped_counter["empty_prefixed_clip"] += 1
                        continue
                    motion_out_path = new_joints_root / f"{sample_id}.npy"
                    np.save(motion_out_path, derived_motion)
                    materialization = "sliced_save"
                    reconstruction_counter[str(span_info["resolution"])] += 1
                    resolution = str(span_info["resolution"])
                else:
                    start, end = 0, int(raw_motion.shape[0])
                    motion_out_path = new_joints_root / f"{sample_id}.npy"
                    materialization = materialize_full_motion(
                        src=raw_motion_path,
                        dst=motion_out_path,
                        link_mode=args.link_mode,
                    )
                    reconstruction_counter["base_full_motion"] += 1
                    resolution = "base_full_motion"

                derived_len = int(end - start)
                if derived_len <= 0:
                    dropped_counter["non_positive_length"] += 1
                    continue

                event_motion = sample.get("motion")
                if event_motion is None:
                    dropped_counter["missing_event_motion"] += 1
                    continue
                event_motion = np.asarray(event_motion)
                event_motion_len = int(event_motion.shape[0]) if event_motion.ndim >= 1 else 0
                event_length = int(sample.get("length", event_motion_len))
                if event_motion_len <= 0 or event_length <= 0:
                    dropped_counter["invalid_event_motion_length"] += 1
                    continue

                text_out_path = texts_root / f"{sample_id}.txt"
                write_text_file(text_out_path, sample.get("text", []))

                usable_ids.append(sample_id)
                motion_paths_for_stats.append(motion_out_path)

                # `data_{split}.npy` is the EventT2M / HumanML3D-E view: preserve
                # the original 263-d motion package so HumanML3DEventDataset can
                # read the same single-root dataset directly.
                derived_dict[sample_id] = {
                    "motion": event_motion,
                    "length": event_length,
                    "text": sample.get("text", []),
                }

                if event_motion_len == derived_len:
                    reconstruction_counter["length_match_event_vs_raw"] += 1
                else:
                    reconstruction_counter["length_mismatch_event_vs_raw"] += 1

                manifest_record = {
                    "split": split_name,
                    "sample_id": sample_id,
                    "base_id": base_id,
                    "prefixed": is_prefixed_id(sample_id),
                    "raw_motion_path": str(raw_motion_path),
                    "derived_motion_path": str(motion_out_path),
                    "text_path": str(text_out_path),
                    "resolution": resolution,
                    "materialization": materialization,
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "raw_motion_length": int(raw_motion.shape[0]),
                    "derived_motion_length": int(derived_len),
                    "event_motion_length": int(event_motion_len),
                    "event_length": int(event_length),
                    "event_vs_raw_length_match": bool(event_motion_len == derived_len),
                    "num_captions": len(sample.get("text", [])),
                }
                manifest_f.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")

            derived_splits[split_name] = derived_dict
            usable_split_ids[split_name] = usable_ids
            metadata["counts"][split_name] = {
                "nominal_total": len(split_items),
                "usable_total": len(usable_ids),
                "dropped_total": len(split_items) - len(usable_ids),
            }
            metadata["reconstruction"][split_name] = dict(reconstruction_counter)
            metadata["dropped_reasons"][split_name] = dict(dropped_counter)

            np.save(output_root / f"data_{split_name}.npy", derived_dict, allow_pickle=True)
            (output_root / f"{split_name}.txt").write_text(
                "".join(f"{sample_id}\n" for sample_id in usable_ids),
                encoding="utf-8",
            )

        all_ids = usable_split_ids["train"] + usable_split_ids["val"] + usable_split_ids["test"]
        (output_root / "all.txt").write_text(
            "".join(f"{sample_id}\n" for sample_id in all_ids),
            encoding="utf-8",
        )

        usable_test = set(usable_split_ids["test"])
        nsim_ids = [sample_id for sample_id in official_nsim if sample_id in usable_test]
        (output_root / "nsim_test.txt").write_text(
            "".join(f"{sample_id}\n" for sample_id in nsim_ids),
            encoding="utf-8",
        )

        mean_raw, std_raw, total_frames = compute_mean_std_streaming(motion_paths_for_stats)
        np.save(output_root / "Mean_raw.npy", mean_raw)
        np.save(output_root / "Std_raw.npy", std_raw)

        metadata["counts"]["all"] = {
            "usable_total": len(all_ids),
        }
        metadata["counts"]["nsim_test"] = {
            "official_total": len(official_nsim),
            "usable_overlap": len(nsim_ids),
        }
        metadata["stats"] = {
            "total_frames": total_frames,
            "mean_raw_shape": list(mean_raw.shape),
            "std_raw_shape": list(std_raw.shape),
        }
        metadata["notes"] = [
            "texts/*.txt are regenerated from HumanML3D-E text entries and always use 0.0/0.0 tags because each derived motion file already represents the exact sample unit.",
            "data_{split}.npy preserves the original HumanML3D-E 263-d motion view and can be read directly by HumanML3DEventDataset / EventT2M-style loaders.",
            "MotionPatches training should use new_joints/ + texts/ as the raw-joint source; data_{split}.npy is the event-aware packaged view.",
            "Prefixed keys are reconstructed from raw HumanML3D motion by matching the HumanML3D-E caption against the base HumanML3D text file.",
        ]

        (output_root / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        manifest_f.close()

    print(f"Built derived dataset root: {output_root}")
    for split_name in split_names:
        counts = metadata["counts"][split_name]
        print(
            f"{split_name}: usable={counts['usable_total']} / nominal={counts['nominal_total']} "
            f"(dropped={counts['dropped_total']})"
        )
    print(
        f"nsim_test usable overlap: {metadata['counts']['nsim_test']['usable_overlap']} / "
        f"{metadata['counts']['nsim_test']['official_total']}"
    )
    print(f"Mean_raw / Std_raw written to: {output_root}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
