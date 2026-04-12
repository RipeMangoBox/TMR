#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_PREFIX = "stage4_1_realdata_e50_b128"
PROTOCOLS = ["normal", "threshold_0.95", "nsim", "guo"]
QUICK_METRICS = [
    "t2m/R01",
    "t2m/R05",
    "t2m/R10",
    "m2t/R01",
    "m2t/R05",
    "m2t/R10",
    "t2m/MedR",
    "m2t/MedR",
]
PRIMARY_SCORE_PROTOCOLS = ["normal", "nsim"]
PRIMARY_SCORE_METRICS = ["t2m/R01", "m2t/R01", "t2m/R05", "m2t/R05"]
TRUSTED_DATA_FILES = [
    "/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E/data_train.npy",
    "/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E/data_val.npy",
    "/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E/data_test.npy",
    "/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E/data_test_condition2.npy",
    "/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E/data_test_condition3.npy",
    "/home/ripemangobox/Coding/Github/Motion/datasets/HumanML3D-E/data_test_condition4.npy",
]


def build_stage_specs(run_prefix: str) -> List[Tuple[str, str, Path]]:
    return [
        ("d1", "D1", REPO_ROOT / "RUN_DIR" / f"{run_prefix}_d1"),
        ("d1_5", "D1.5", REPO_ROOT / "RUN_DIR" / f"{run_prefix}_d1_5"),
        ("d2a", "D2a", REPO_ROOT / "RUN_DIR" / f"{run_prefix}_d2a"),
        ("d2b", "D2b", REPO_ROOT / "RUN_DIR" / f"{run_prefix}_d2b"),
    ]


def default_d0_dir(run_prefix: str) -> Path:
    return REPO_ROOT / "RUN_DIR" / f"{run_prefix}_d0"


def default_d3_dir(run_prefix: str) -> Path:
    return REPO_ROOT / "RUN_DIR" / f"{run_prefix}_d3"


def find_latest_file(directory: Path, pattern: str) -> Optional[Path]:
    matches = sorted(directory.glob(pattern))
    if not matches:
        return None
    return matches[-1]


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def maybe_load_metrics(run_dir: Path) -> Dict[str, Optional[Dict]]:
    metric_dir = run_dir / "contrastive_metrics"
    metrics: Dict[str, Optional[Dict]] = {}
    for protocol in PROTOCOLS:
        metric_path = metric_dir / f"{protocol}.yaml"
        metrics[protocol] = load_yaml(metric_path) if metric_path.exists() else None
    return metrics


def maybe_load_config(run_dir: Path) -> Optional[Dict]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return None
    return load_json(config_path)


def find_last_checkpoint(run_dir: Path) -> Optional[Path]:
    direct = sorted(run_dir.glob("**/checkpoints/last.ckpt"))
    if direct:
        return direct[0]
    versioned = sorted(run_dir.glob("**/checkpoints/last-v*.ckpt"))
    if versioned:
        return versioned[0]
    return None


def format_metric(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def compute_primary_score(metrics: Dict[str, Optional[Dict]]) -> Optional[float]:
    values: List[float] = []
    for protocol in PRIMARY_SCORE_PROTOCOLS:
        metric_dict = metrics.get(protocol)
        if not metric_dict:
            continue
        for metric_name in PRIMARY_SCORE_METRICS:
            value = metric_dict.get(metric_name)
            if isinstance(value, (int, float)):
                values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def stage_status(config: Optional[Dict], metrics: Dict[str, Optional[Dict]]) -> str:
    if any(metrics.get(protocol) for protocol in PROTOCOLS):
        return "evaluated"
    if config is not None:
        return "trained_pending_eval"
    return "pending"


def stage_summary_record(stage_key: str, stage_label: str, run_dir: Path) -> Dict:
    config = maybe_load_config(run_dir)
    metrics = maybe_load_metrics(run_dir)
    primary_score = compute_primary_score(metrics)
    return {
        "stage_key": stage_key,
        "stage_label": stage_label,
        "run_dir": str(run_dir),
        "config": config,
        "metrics": metrics,
        "status": stage_status(config, metrics),
        "primary_score": primary_score,
        "last_checkpoint": str(find_last_checkpoint(run_dir)) if find_last_checkpoint(run_dir) else None,
        "has_last_weights": (run_dir / "last_weights").exists()
        and any((run_dir / "last_weights").glob("*.pt")),
    }


def render_stage_metric_table(metrics: Dict[str, Optional[Dict]]) -> List[str]:
    lines = [
        "| Protocol | t2m/R01 | t2m/R05 | t2m/R10 | m2t/R01 | m2t/R05 | m2t/R10 | t2m/MedR | m2t/MedR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for protocol in PROTOCOLS:
        metric_dict = metrics.get(protocol)
        row = [protocol]
        for metric_name in QUICK_METRICS:
            row.append(format_metric(metric_dict.get(metric_name) if metric_dict else None))
        lines.append("| " + " | ".join(row) + " |")
    return lines


def render_stage_report(record: Dict, report_date: str) -> None:
    config = record["config"]
    if config is None:
        return

    run_dir = Path(record["run_dir"])
    stage_key = record["stage_key"]
    stage_label = record["stage_label"]
    output_path = run_dir / f"{report_date}_{stage_key}_realdata_report.md"

    data_cfg = config.get("data", {})
    dataloader_cfg = config.get("dataloader", {})
    trainer_cfg = config.get("trainer", {})

    lines = [
        f"# {stage_label}: Corrected Real-Data Rerun",
        "",
        f"- Stage status: **{record['status']}**",
        f"- Run dir: `{run_dir}`",
        f"- Dataset root: `{data_cfg.get('dataset_root', '-')}`",
        f"- Batch size: `{dataloader_cfg.get('batch_size', '-')}`",
        f"- Num workers: `{dataloader_cfg.get('num_workers', '-')}`",
        f"- Max epochs: `{trainer_cfg.get('max_epochs', '-')}`",
        f"- Last checkpoint: `{record['last_checkpoint'] or '-'}`",
        f"- last_weights present: **{'yes' if record['has_last_weights'] else 'no'}**",
        f"- Primary retrieval score (normal+nsim R@1/R@5 mean): **{format_metric(record['primary_score'])}**",
        "",
        "## Retrieval Metrics",
        "",
        *render_stage_metric_table(record["metrics"]),
        "",
    ]

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def render_comparison_table(records: List[Dict]) -> List[str]:
    header = [
        "Stage",
        "Status",
        "PrimaryScore",
        "normal t2m/R01",
        "normal m2t/R01",
        "nsim t2m/R01",
        "nsim m2t/R01",
        "normal t2m/R05",
        "normal m2t/R05",
        "nsim t2m/R05",
        "nsim m2t/R05",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for record in records:
        normal = record["metrics"].get("normal") or {}
        nsim = record["metrics"].get("nsim") or {}
        row = [
            record["stage_label"],
            record["status"],
            format_metric(record["primary_score"]),
            format_metric(normal.get("t2m/R01")),
            format_metric(normal.get("m2t/R01")),
            format_metric(nsim.get("t2m/R01")),
            format_metric(nsim.get("m2t/R01")),
            format_metric(normal.get("t2m/R05")),
            format_metric(normal.get("m2t/R05")),
            format_metric(nsim.get("t2m/R05")),
            format_metric(nsim.get("m2t/R05")),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return lines


def choose_recommendation(records: List[Dict]) -> Tuple[str, Optional[Dict]]:
    evaluated = [record for record in records if record["primary_score"] is not None]
    if not evaluated:
        return "Pending: wait for corrected D1/D1.5/D2a/D2b retrieval metrics.", None

    best = max(evaluated, key=lambda item: item["primary_score"])
    if best["stage_key"] == "d2b":
        return "Go Phase 2 with D2b", best
    if best["stage_key"] == "d2a":
        return "Keep D2a", best
    return "Go D3", best


def build_d3_summary(
    records: List[Dict],
    d0_json: Optional[Dict],
    output_dir: Path,
    report_date: str,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    recommendation, best_record = choose_recommendation(records)
    summary_json = {
        "report_date": report_date,
        "trusted_data_files": TRUSTED_DATA_FILES,
        "d0": d0_json,
        "stages": records,
        "recommendation": recommendation,
        "winner": best_record["stage_label"] if best_record else None,
    }

    json_path = output_dir / f"{report_date}_d3_stage4_1_closure_summary.json"
    md_path = output_dir / f"{report_date}_d3_stage4_1_closure_summary.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    d0_meta = (d0_json or {}).get("meta", {})
    d0_gates = (d0_json or {}).get("gates", {})

    lines = [
        "# D3: Stage4.1 Corrected Real-Data Closure Summary",
        "",
        "- This closure supersedes the earlier Stage4.1 chain because the previous D0-D3 results were generated from the wrong HumanML3D-E source.",
        "- Corrected trusted HumanML3D-E source:",
    ]
    lines.extend([f"  - `{path}`" for path in TRUSTED_DATA_FILES])
    lines.extend(
        [
            "",
            "## D0 Gate Recap",
            "",
            f"- D0 scope: **{d0_meta.get('report_type', 'data_audit_only')}**",
            f"- D0 training dependency: **{d0_meta.get('training_dependency', 'none')}**",
            f"- Total train/val/test entries: **{d0_meta.get('total_samples', '-')}**",
            f"- Total captions: **{d0_meta.get('total_captions', '-')}**",
            f"- Canonical decomposed coverage: **{format_metric(d0_meta.get('canonical_decomposed_coverage'))}**",
            f"- Gate recommendation: **{d0_gates.get('recommendation', 'pending')}**",
            "- Interpretation: D0 is a corrected real-data audit and launch gate only; it does not depend on whether downstream training finished full epochs.",
            "",
            "## nsim Note",
            "",
            "- `nsim_test` now prefers the official TMR split file and uses the corrected data overlap.",
            "- Current overlap is `97/100`; missing keyids are `001052`, `008340`, and `M010392`.",
            "",
            "## Stage Comparison",
            "",
            *render_comparison_table(records),
            "",
            "## Recommendation",
            "",
            f"- Final recommendation: **{recommendation}**",
            f"- Winner under the current corrected evidence: **{best_record['stage_label'] if best_record else '-'}**",
            "",
            "## Per-Stage Docs",
            "",
        ]
    )

    for record in records:
        stage_doc = Path(record["run_dir"]) / f"{report_date}_{record['stage_key']}_realdata_report.md"
        if stage_doc.exists():
            lines.append(f"- {record['stage_label']}: `{stage_doc}`")
        else:
            lines.append(f"- {record['stage_label']}: pending")

    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate corrected real-data Stage4.1 per-stage reports and D3 closure summary."
    )
    parser.add_argument(
        "--run-prefix",
        default=DEFAULT_RUN_PREFIX,
        help="RunDir prefix, e.g. stage4_1_realdata or stage4_1_realdata_e50_b128.",
    )
    parser.add_argument(
        "--report-date",
        default=str(date.today()),
        help="Date string used in output file names.",
    )
    parser.add_argument(
        "--d0-dir",
        type=Path,
        default=None,
        help="Directory containing corrected D0 outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save the D3 closure summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    d0_dir = args.d0_dir or default_d0_dir(args.run_prefix)
    output_dir = args.output_dir or default_d3_dir(args.run_prefix)
    stage_specs = build_stage_specs(args.run_prefix)

    d0_json_path = find_latest_file(d0_dir, "*_d0_realdata_event_stats.json")
    d0_json = load_json(d0_json_path) if d0_json_path else None

    records = [
        stage_summary_record(stage_key, stage_label, run_dir)
        for stage_key, stage_label, run_dir in stage_specs
    ]

    for record in records:
        render_stage_report(record, args.report_date)

    json_path, md_path = build_d3_summary(
        records=records,
        d0_json=d0_json,
        output_dir=output_dir,
        report_date=args.report_date,
    )

    print(f"[summary] wrote D3 JSON: {json_path}")
    print(f"[summary] wrote D3 Markdown: {md_path}")
    for record in records:
        stage_doc = Path(record["run_dir"]) / f"{args.report_date}_{record['stage_key']}_realdata_report.md"
        if stage_doc.exists():
            print(f"[summary] wrote {record['stage_label']} report: {stage_doc}")


if __name__ == "__main__":
    main()
