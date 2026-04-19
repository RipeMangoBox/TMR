"""scripts/diagnose_humanml3de_decompose.py"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


PARALLEL_CUE_PATTERNS = (
    r"\bwhile\b",
    r"\bduring\b",
    r"\bmeanwhile\b",
    r"\bsimultaneously\b",
    r"\bat the same time\b",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose HumanML3D-E decomposed-event quality for TAMR D0.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E"),
        help="Directory containing data_train.npy / data_val.npy / data_test.npy.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "val", "test"),
        help="Splits to include.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the summary as JSON.",
    )
    return parser.parse_args()


def load_split(path: Path) -> Dict[str, Dict]:
    packed = np.load(path, allow_pickle=True)
    if isinstance(packed, np.ndarray) and packed.shape == ():
        packed = packed.item()
    if not isinstance(packed, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(packed).__name__}")
    return {str(key): value for key, value in packed.items()}


def has_parallel_cue(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in PARALLEL_CUE_PATTERNS)


def normalize_events(text_entry: Dict) -> List[str]:
    decomposed = text_entry.get("decomposed")
    if not isinstance(decomposed, list):
        return []

    events: List[str] = []
    for item in decomposed:
        if isinstance(item, dict):
            caption = str(item.get("caption", "")).strip()
        else:
            caption = str(item).strip()
        if caption:
            events.append(caption)
    return events


def summarize_caption_lengths(lengths: Iterable[int]) -> Dict[str, float]:
    values = list(lengths)
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}

    array = np.asarray(values, dtype=np.float32)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def diagnose_split(samples: Dict[str, Dict]) -> Dict[str, object]:
    caption_counter = 0
    decomposed_counter = 0
    k1_counter = 0
    k2_or_more_counter = 0
    k2_or_more_without_parallel_counter = 0
    k2_or_more_with_parallel_counter = 0
    event_count_hist = Counter()
    event_token_lengths: List[int] = []

    for sample in samples.values():
        text_entries = sample.get("text", [])
        if not isinstance(text_entries, list):
            continue

        for text_entry in text_entries:
            if not isinstance(text_entry, dict):
                continue

            caption = str(text_entry.get("caption", "")).strip()
            if not caption:
                continue

            caption_counter += 1
            events = normalize_events(text_entry)
            if not events:
                continue

            decomposed_counter += 1
            event_count = len(events)
            event_count_hist[event_count] += 1
            event_token_lengths.extend(len(event.split()) for event in events)

            if event_count == 1:
                k1_counter += 1
                continue

            k2_or_more_counter += 1
            if has_parallel_cue(caption):
                k2_or_more_with_parallel_counter += 1
            else:
                k2_or_more_without_parallel_counter += 1

    no_parallel_ratio_total = 0.0
    if caption_counter > 0:
        no_parallel_ratio_total = k2_or_more_without_parallel_counter / caption_counter

    parallel_ratio_k2 = 0.0
    if k2_or_more_counter > 0:
        parallel_ratio_k2 = k2_or_more_with_parallel_counter / k2_or_more_counter

    return {
        "num_captions": caption_counter,
        "num_with_decomposed": decomposed_counter,
        "decomposed_coverage": decomposed_counter / max(caption_counter, 1),
        "k1_count": k1_counter,
        "k1_ratio": k1_counter / max(caption_counter, 1),
        "k2_or_more_count": k2_or_more_counter,
        "k2_or_more_ratio": k2_or_more_counter / max(caption_counter, 1),
        "k2_or_more_with_parallel_count": k2_or_more_with_parallel_counter,
        "k2_or_more_without_parallel_count": k2_or_more_without_parallel_counter,
        "k2_parallel_ratio": parallel_ratio_k2,
        "k2_without_parallel_ratio_total": no_parallel_ratio_total,
        "event_count_hist": dict(sorted(event_count_hist.items())),
        "event_token_length": summarize_caption_lengths(event_token_lengths),
    }


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()

    summary: Dict[str, object] = {
        "dataset_root": str(dataset_root),
        "parallel_cue_mode": "conservative_explicit_markers",
        "splits": {},
    }

    merged_samples: Dict[str, Dict] = {}
    for split in args.splits:
        split_path = dataset_root / f"data_{split}.npy"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        samples = load_split(split_path)
        summary["splits"][split] = diagnose_split(samples)
        merged_samples.update({f"{split}:{k}": v for k, v in samples.items()})

    summary["overall"] = diagnose_split(merged_samples)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
