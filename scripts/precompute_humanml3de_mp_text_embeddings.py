#!/usr/bin/env python3
"""Precompute reusable text embeddings for HumanML3D-E-MP captions and events."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import orjson
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.text import (
    save_sent_embeddings_from_texts,
    save_token_embeddings_from_texts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute token and sentence embeddings for HumanML3D-E-MP "
            "whole captions plus decomposed event captions."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../datasets/HumanML3D-E-MP"),
    )
    parser.add_argument(
        "--token-model",
        default="distilbert-base-uncased",
    )
    parser.add_argument(
        "--sentence-model",
        default="sentence-transformers/all-mpnet-base-v2",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Encoding batch size used for both token and sentence models.",
    )
    parser.add_argument(
        "--limit-records",
        type=int,
        default=0,
        help="Only process the first N caption records for smoke testing.",
    )
    return parser.parse_args()


def _load_captions_jsonl(path: Path, limit_records: int = 0) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing captions file: {path}")

    total_records = sum(1 for _ in path.open("r", encoding="utf-8"))
    records = []
    with path.open("rb") as handle:
        for line_idx, line in enumerate(
            tqdm(handle, total=total_records, desc="Reading captions.jsonl", unit="line")
        ):
            if limit_records > 0 and line_idx >= limit_records:
                break
            line = line.strip()
            if not line:
                continue
            records.append(orjson.loads(line))
    return records


def _collect_texts(records: list[dict]) -> tuple[list[str], Counter]:
    texts = []
    stats = Counter()
    for record in tqdm(records, desc="Collecting captions/events", unit="record"):
        whole_caption = str(record.get("whole_caption", "")).strip()
        if whole_caption:
            texts.append(whole_caption)
            stats["whole_caption"] += 1

        decomposed = record.get("decomposed")
        if not isinstance(decomposed, list):
            continue
        for event_item in decomposed:
            if not isinstance(event_item, dict):
                continue
            event_caption = str(event_item.get("caption", "")).strip()
            if not event_caption:
                continue
            texts.append(event_caption)
            stats["event_caption"] += 1
    return texts, stats


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    captions_path = dataset_root / "captions.jsonl"

    records = _load_captions_jsonl(
        captions_path,
        limit_records=args.limit_records,
    )
    texts, stats = _collect_texts(records)
    unique_texts = list(dict.fromkeys(texts))

    print(
        "[precompute_humanml3de_mp_text_embeddings] "
        f"dataset_root={dataset_root}"
    )
    print(
        "[precompute_humanml3de_mp_text_embeddings] "
        f"records={len(records)} whole_captions={stats['whole_caption']} "
        f"event_captions={stats['event_caption']} unique_texts={len(unique_texts)}"
    )

    save_token_embeddings_from_texts(
        str(dataset_root),
        unique_texts,
        modelname=args.token_model,
        device=args.device,
        batch_size=args.batch_size,
        progress_desc="Encoding HumanML3D-E-MP token embeddings",
    )
    save_sent_embeddings_from_texts(
        str(dataset_root),
        unique_texts,
        modelname=args.sentence_model,
        device=args.device,
        batch_size=args.batch_size,
        progress_desc="Encoding HumanML3D-E-MP sentence embeddings",
    )

    print(
        "[precompute_humanml3de_mp_text_embeddings] "
        "Done. Future runs can reuse these files until the dataset texts or model names change."
    )


if __name__ == "__main__":
    main()
