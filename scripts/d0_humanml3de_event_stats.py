#!/usr/bin/env python3
"""
D0 statistics for Stage4.1 corrected real-data rerun:
HumanML3D-E event decomposition diagnostics and quantitative gates.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.humanml3de_event import extract_event_captions, load_humanml3de_split


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

OVERLAP_RULES = {
    "while": re.compile(r"\bwhile\b", flags=re.IGNORECASE),
    "meanwhile": re.compile(r"\bmeanwhile\b", flags=re.IGNORECASE),
    "simultaneous": re.compile(r"\bsimultaneous(?:ly)?\b", flags=re.IGNORECASE),
    "at_same_time": re.compile(
        r"\bat\s+(?:the\s+)?same\s+time\b", flags=re.IGNORECASE
    ),
    "during": re.compile(r"\bduring\b", flags=re.IGNORECASE),
    "concurrent": re.compile(r"\bconcurrent(?:ly)?\b", flags=re.IGNORECASE),
}


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def detect_overlap_cues(events: Iterable[str]) -> List[str]:
    merged = " ".join(event for event in events if event)
    hits = [name for name, pattern in OVERLAP_RULES.items() if pattern.search(merged)]
    return sorted(hits)


def ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def f4(value: float) -> float:
    return round(float(value), 4)


def summarize_array(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": f4(arr.mean()),
        "median": f4(float(np.median(arr))),
        "p25": f4(float(np.percentile(arr, 25))),
        "p75": f4(float(np.percentile(arr, 75))),
    }


def summarize_text_count_counter(counter: Counter) -> Dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items())}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run D0 statistics on corrected HumanML3D-E decomposed captions."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E"),
        help="Canonical directory containing the six trusted HumanML3D-E npy files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("RUN_DIR/stage4_1_realdata_e50_b128_d0"),
        help="Directory to save JSON/CSV/Markdown outputs.",
    )
    parser.add_argument(
        "--report-date",
        type=str,
        default=str(date.today()),
        help="Date string used in output filenames.",
    )
    parser.add_argument(
        "--max-overlap-examples",
        type=int,
        default=20,
        help="Maximum number of overlap examples to keep in report.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    main_splits = ["train", "val", "test"]
    condition_files = [
        "data_test_condition2.npy",
        "data_test_condition3.npy",
        "data_test_condition4.npy",
    ]
    for split in main_splits:
        split_path = dataset_root / f"data_{split}.npy"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")
    for name in condition_files:
        path = dataset_root / name
        if not path.exists():
            raise FileNotFoundError(f"Missing condition subset file: {path}")

    per_split = {
        split: {
            "entries": 0,
            "captions_total": 0,
            "captions_valid": 0,
            "captions_invalid": 0,
            "captions_with_decomposed": 0,
            "text_count_counter": Counter(),
            "event_source_counter": Counter(),
            "k_counter": Counter(),
            "overlap_count": 0,
        }
        for split in main_splits
    }

    invalid_records: List[Dict[str, str]] = []
    overlap_records: List[Dict[str, str]] = []
    caption_avg_event_words: List[float] = []
    event_words: List[float] = []
    caption_avg_frames_per_event: List[float] = []

    k_counter = Counter()
    event_source_counter = Counter()
    caption_style_counter = Counter()
    text_count_counter = Counter()

    total_samples = 0
    total_captions = 0
    valid_captions = 0

    for split in main_splits:
        data = load_humanml3de_split(dataset_root / f"data_{split}.npy")
        per_split[split]["entries"] = len(data)
        total_samples += len(data)

        for sample_id, sample in data.items():
            texts = sample.get("text", [])
            per_split[split]["text_count_counter"][len(texts)] += 1
            text_count_counter[len(texts)] += 1
            motion_len = int(sample.get("length", 0))

            for text_item in texts:
                if not isinstance(text_item, dict):
                    continue

                total_captions += 1
                per_split[split]["captions_total"] += 1

                caption = str(text_item.get("caption", "")).strip()
                if caption.lower().startswith("action 1:"):
                    caption_style_counter["action_marker"] += 1
                else:
                    caption_style_counter["plain_caption"] += 1

                if isinstance(text_item.get("decomposed"), list) and text_item.get(
                    "decomposed"
                ):
                    per_split[split]["captions_with_decomposed"] += 1

                events, event_source = extract_event_captions(
                    text_item, strict_event_parse=False
                )
                if not events:
                    per_split[split]["captions_invalid"] += 1
                    invalid_records.append(
                        {
                            "split": split,
                            "sample_id": str(sample_id),
                            "reason": event_source,
                            "caption": caption,
                        }
                    )
                    continue

                valid_captions += 1
                per_split[split]["captions_valid"] += 1
                per_split[split]["event_source_counter"][event_source] += 1
                event_source_counter[event_source] += 1

                k = len(events)
                k_counter[k] += 1
                per_split[split]["k_counter"][k] += 1

                unit_word_counts = [count_words(event) for event in events]
                event_words.extend(float(word_count) for word_count in unit_word_counts)
                caption_avg_event_words.append(float(np.mean(unit_word_counts)))
                if motion_len > 0:
                    caption_avg_frames_per_event.append(float(motion_len) / float(k))

                cues = detect_overlap_cues(events)
                if cues:
                    per_split[split]["overlap_count"] += 1
                    if len(overlap_records) < args.max_overlap_examples:
                        overlap_records.append(
                            {
                                "split": split,
                                "sample_id": str(sample_id),
                                "cues": ",".join(cues),
                                "caption": caption,
                                "events": events,
                            }
                        )

    k1_count = k_counter.get(1, 0)
    kge2_count = sum(v for k, v in k_counter.items() if k >= 2)
    overlap_count = sum(v["overlap_count"] for v in per_split.values())
    invalid_count = len(invalid_records)

    k1_ratio = ratio(k1_count, valid_captions)
    kge2_ratio = ratio(kge2_count, valid_captions)
    overlap_ratio = ratio(overlap_count, valid_captions)
    decomposed_caption_count = sum(
        split_stats["captions_with_decomposed"] for split_stats in per_split.values()
    )

    gate_1_triggered = kge2_ratio < 0.40
    gate_2_triggered = k1_ratio > 0.60
    gate_3_triggered = overlap_ratio > 0.15

    if gate_1_triggered:
        gate_path = (
            "DATA-GATE NO-GO: corrected real data fail the K>=2 threshold; "
            "downgrade Stage4.1 to Stage0-2 + minimal evidence head."
        )
    else:
        gate_path = (
            "DATA-GATE GO: corrected real data support launching D1 frozen "
            "minimal event-time head."
        )

    full_test = load_humanml3de_split(dataset_root / "data_test.npy")
    full_test_keys = set(full_test.keys())
    condition_subset_stats = {}
    for name in condition_files:
        subset = load_humanml3de_split(dataset_root / name)
        subset_text_counter = Counter()
        subset_k_counter = Counter()
        overlap_keys = sum(1 for keyid in subset if keyid in full_test_keys)
        for sample in subset.values():
            texts = sample.get("text", [])
            subset_text_counter[len(texts)] += 1
            for text_item in texts:
                if not isinstance(text_item, dict):
                    continue
                events, _event_source = extract_event_captions(
                    text_item, strict_event_parse=False
                )
                if events:
                    subset_k_counter[len(events)] += 1

        condition_name = name.replace("data_test_", "").replace(".npy", "")
        condition_subset_stats[condition_name] = {
            "entries": len(subset),
            "overlap_with_data_test": overlap_keys,
            "text_count_distribution": summarize_text_count_counter(subset_text_counter),
            "event_count_distribution": summarize_text_count_counter(subset_k_counter),
        }

    result = {
        "meta": {
            "report_type": "data_audit_only",
            "training_dependency": "none",
            "reliability_scope": (
                "D0 is computed directly from the six trusted HumanML3D-E npy "
                "files and is not affected by whether D1/D1.5/D2a/D2b training "
                "finished full epochs."
            ),
            "gate_scope": (
                "D0 only decides whether corrected real data justify launching D1; "
                "it is not a model-performance verdict."
            ),
            "dataset_root": str(dataset_root),
            "splits": main_splits,
            "total_samples": total_samples,
            "total_captions": total_captions,
            "valid_captions": valid_captions,
            "invalid_captions": invalid_count,
            "event_extraction_coverage": f4(ratio(valid_captions, total_captions)),
            "captions_with_canonical_decomposed": decomposed_caption_count,
            "canonical_decomposed_coverage": f4(
                ratio(decomposed_caption_count, total_captions)
            ),
            "caption_style_breakdown": dict(caption_style_counter),
            "event_source_breakdown": dict(event_source_counter),
            "motion_level_text_count_distribution": summarize_text_count_counter(
                text_count_counter
            ),
            "overlap_rulebook": list(OVERLAP_RULES.keys()),
        },
        "split_entry_counts": {
            split: per_split[split]["entries"] for split in main_splits
        },
        "condition_subset_stats": condition_subset_stats,
        "k_distribution": {
            str(k): {
                "count": int(count),
                "ratio_over_valid": f4(ratio(count, valid_captions)),
            }
            for k, count in sorted(k_counter.items())
        },
        "key_ratios": {
            "k_eq_1": {"count": k1_count, "ratio_over_valid": f4(k1_ratio)},
            "k_ge_2": {"count": kge2_count, "ratio_over_valid": f4(kge2_ratio)},
            "overlap_rule_based": {
                "count": overlap_count,
                "ratio_over_valid": f4(overlap_ratio),
            },
        },
        "event_length_words": {
            "caption_avg_event_words": summarize_array(caption_avg_event_words),
            "event_word_count": summarize_array(event_words),
        },
        "event_length_frames_proxy": {
            "caption_avg_frames_per_event": summarize_array(caption_avg_frames_per_event)
        },
        "per_split": {
            split: {
                "entries": per_split[split]["entries"],
                "captions_total": per_split[split]["captions_total"],
                "captions_valid": per_split[split]["captions_valid"],
                "captions_invalid": per_split[split]["captions_invalid"],
                "captions_with_decomposed": per_split[split][
                    "captions_with_decomposed"
                ],
                "overlap_count": per_split[split]["overlap_count"],
                "text_count_distribution": summarize_text_count_counter(
                    per_split[split]["text_count_counter"]
                ),
                "event_source_breakdown": dict(per_split[split]["event_source_counter"]),
                "k_distribution": {
                    str(k): int(v)
                    for k, v in sorted(per_split[split]["k_counter"].items())
                },
            }
            for split in main_splits
        },
        "gates": {
            "gate_1": {
                "condition": "K>=2 ratio < 40%",
                "value": f4(kge2_ratio),
                "triggered": gate_1_triggered,
            },
            "gate_2": {
                "condition": "K=1 ratio > 60%",
                "value": f4(k1_ratio),
                "triggered": gate_2_triggered,
            },
            "gate_3": {
                "condition": "rule-based overlap ratio > 15%",
                "value": f4(overlap_ratio),
                "triggered": gate_3_triggered,
            },
            "recommendation": gate_path,
        },
        "samples": {
            "invalid_examples": invalid_records[:20],
            "overlap_examples": overlap_records,
        },
    }

    tag = args.report_date
    json_path = output_dir / f"{tag}_d0_realdata_event_stats.json"
    csv_path = output_dir / f"{tag}_d0_realdata_k_distribution.csv"
    md_path = output_dir / f"{tag}_d0_realdata_report.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["K", "count", "ratio_over_valid"])
        for k, count in sorted(k_counter.items()):
            writer.writerow([k, count, f"{ratio(count, valid_captions):.6f}"])

    gate_1_text = "TRIGGERED" if gate_1_triggered else "NOT_TRIGGERED"
    gate_2_text = "TRIGGERED" if gate_2_triggered else "NOT_TRIGGERED"
    gate_3_text = "TRIGGERED" if gate_3_triggered else "NOT_TRIGGERED"

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# D0: HumanML3D-E Corrected Real-Data Audit\n\n")
        f.write("- This report supersedes the earlier D0 report that was generated from the wrong HumanML3D-E source.\n")
        f.write("- Report type: **data audit / launch gate only**\n")
        f.write(
            "- This report is computed directly from the six trusted HumanML3D-E `.npy` files and does not use any training checkpoint.\n"
        )
        f.write(
            "- Reliability boundary: the statistics below remain valid even if D1/D1.5/D2a/D2b have not finished full training epochs.\n"
        )
        f.write(
            "- Scope boundary: D0 only determines whether corrected real data justify launching D1; it is not evidence that Stage4.1 already won.\n"
        )
        f.write(f"- Canonical dataset root: `{dataset_root}`\n")
        f.write(f"- Total entries across train/val/test: **{total_samples}**\n")
        f.write(f"- Total captions across train/val/test: **{total_captions}**\n")
        f.write(f"- Valid event extractions: **{valid_captions}**\n")
        f.write(
            f"- Event extraction coverage: **{ratio(valid_captions, total_captions) * 100:.2f}%**\n"
        )
        f.write(
            f"- Canonical `text[].decomposed` coverage: **{ratio(decomposed_caption_count, total_captions) * 100:.2f}%**\n"
        )
        f.write(f"- Invalid captions: **{invalid_count}**\n\n")

        f.write("## Split Entry Counts\n\n")
        f.write("| Split | Entries |\n")
        f.write("|---|---:|\n")
        for split in main_splits:
            f.write(f"| {split} | {per_split[split]['entries']} |\n")
        f.write("\n")

        f.write("## Motion-Level Text Count Distribution\n\n")
        f.write("| #texts per motion | Count |\n")
        f.write("|---:|---:|\n")
        for text_count, count in sorted(text_count_counter.items()):
            f.write(f"| {text_count} | {count} |\n")
        f.write("\n")

        f.write("## Caption Structure Summary\n\n")
        f.write(
            f"- Plain natural-language captions: **{caption_style_counter.get('plain_caption', 0)}**\n"
        )
        f.write(
            f"- `action i:` marker captions: **{caption_style_counter.get('action_marker', 0)}**\n"
        )
        f.write(
            f"- Event source breakdown: **{dict(event_source_counter)}**\n\n"
        )

        f.write("## Conditioned Test Files\n\n")
        f.write("| File | Entries | Overlap with `data_test.npy` |\n")
        f.write("|---|---:|---:|\n")
        for condition_name, stats in condition_subset_stats.items():
            f.write(
                f"| {condition_name} | {stats['entries']} | {stats['overlap_with_data_test']} |\n"
            )
        f.write("\n")

        f.write("## K Distribution\n\n")
        f.write("| K | Count | Ratio over valid |\n")
        f.write("|---:|---:|---:|\n")
        for k, count in sorted(k_counter.items()):
            f.write(f"| {k} | {count} | {ratio(count, valid_captions) * 100:.2f}% |\n")
        f.write("\n")

        f.write("## Key Ratios for D0 Gates\n\n")
        f.write(
            f"- `K>=2` ratio: **{kge2_ratio * 100:.2f}%** ({kge2_count}/{valid_captions})\n"
        )
        f.write(f"- `K=1` ratio: **{k1_ratio * 100:.2f}%** ({k1_count}/{valid_captions})\n")
        f.write(
            f"- Rule-based overlap ratio: **{overlap_ratio * 100:.2f}%** "
            f"({overlap_count}/{valid_captions})\n\n"
        )

        words_stats = result["event_length_words"]["caption_avg_event_words"]
        frames_stats = result["event_length_frames_proxy"]["caption_avg_frames_per_event"]
        f.write("## Event Length Summary\n\n")
        f.write(
            "- Caption-level average event length (words): "
            f"mean **{words_stats['mean']}**, median **{words_stats['median']}**, "
            f"p25/p75 **{words_stats['p25']} / {words_stats['p75']}**\n"
        )
        f.write(
            "- Caption-level average event length (frames proxy = motion_length / K): "
            f"mean **{frames_stats['mean']}**, median **{frames_stats['median']}**, "
            f"p25/p75 **{frames_stats['p25']} / {frames_stats['p75']}**\n\n"
        )

        f.write("## D0 Quantitative Gates\n\n")
        f.write(
            f"1. Gate-1 (`K>=2` ratio < 40%): **{gate_1_text}** "
            f"(value={kge2_ratio * 100:.2f}%)\n"
        )
        f.write(
            f"2. Gate-2 (`K=1` ratio > 60%): **{gate_2_text}** "
            f"(value={k1_ratio * 100:.2f}%)\n"
        )
        f.write(
            f"3. Gate-3 (rule-based overlap ratio > 15%): **{gate_3_text}** "
            f"(value={overlap_ratio * 100:.2f}%)\n\n"
        )
        f.write(f"- Recommendation: **{gate_path}**\n\n")

        if overlap_records:
            f.write("## Rule-Based Overlap Examples\n\n")
            for sample in overlap_records:
                f.write(
                    f"- [{sample['split']}] `{sample['sample_id']}` "
                    f"(cues={sample['cues']}): {sample['caption']}\n"
                )
            f.write("\n")

        if invalid_records:
            f.write("## Invalid Event Extraction Examples\n\n")
            for sample in invalid_records[:20]:
                f.write(
                    f"- [{sample['split']}] `{sample['sample_id']}` "
                    f"(reason={sample['reason']}): {sample['caption']}\n"
                )
            f.write("\n")

        f.write("## Caveat\n\n")
        f.write(
            "This corrected rerun uses canonical `text[].decomposed[].caption` as the "
            "primary event structure and only keeps caption parsing as a fallback. "
            "Training completeness matters for D1-D3 model comparisons, but not for "
            "this D0 data-audit report.\n"
        )

    print(f"[D0] total_captions={total_captions} valid={valid_captions} invalid={invalid_count}")
    print(
        f"[D0] split_entries=train:{per_split['train']['entries']} "
        f"val:{per_split['val']['entries']} test:{per_split['test']['entries']}"
    )
    print(
        f"[D0] K>=2 ratio={kge2_ratio * 100:.2f}% | "
        f"K=1 ratio={k1_ratio * 100:.2f}%"
    )
    print(f"[D0] overlap ratio={overlap_ratio * 100:.2f}%")
    print(f"[D0] gate1={gate_1_text} gate2={gate_2_text} gate3={gate_3_text}")
    print(f"[D0] recommendation={gate_path}")
    print(f"[D0] wrote: {json_path}")
    print(f"[D0] wrote: {csv_path}")
    print(f"[D0] wrote: {md_path}")


if __name__ == "__main__":
    main()
