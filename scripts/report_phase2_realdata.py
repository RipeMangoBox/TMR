#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import yaml


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
DEFAULT_BASE_DIR = "RUN_DIR/stage4_1_realdata_e50_b128_d2b"


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def maybe_load_config(run_dir: Path) -> Optional[Dict]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return None
    return load_json(config_path)


def maybe_load_metrics(run_dir: Path) -> Dict[str, Optional[Dict]]:
    metric_dir = run_dir / "contrastive_metrics"
    metrics: Dict[str, Optional[Dict]] = {}
    for protocol in PROTOCOLS:
        metric_path = metric_dir / f"{protocol}.yaml"
        metrics[protocol] = load_yaml(metric_path) if metric_path.exists() else None
    return metrics


def find_last_checkpoint(run_dir: Path) -> Optional[Path]:
    direct = sorted(run_dir.glob("**/checkpoints/last.ckpt"))
    if direct:
        return direct[0]
    versioned = sorted(run_dir.glob("**/checkpoints/last-v*.ckpt"))
    if versioned:
        return versioned[0]
    return None


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


def format_metric(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def format_delta(current: Optional[float], base: Optional[float]) -> str:
    if current is None or base is None:
        return "-"
    delta = current - base
    return f"{delta:+.2f}"


def stage_status(config: Optional[Dict], metrics: Dict[str, Optional[Dict]]) -> str:
    if any(metrics.get(protocol) for protocol in PROTOCOLS):
        return "evaluated"
    if config is not None:
        return "trained_pending_eval"
    return "pending"


def render_metric_table(metrics: Dict[str, Optional[Dict]]) -> List[str]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a Phase 2 real-data report.")
    parser.add_argument("--report-date", default=str(date.today()))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    base_dir = Path(args.base_dir)

    run_dir.mkdir(parents=True, exist_ok=True)

    current_config = maybe_load_config(run_dir)
    current_metrics = maybe_load_metrics(run_dir)
    current_status = stage_status(current_config, current_metrics)
    current_score = compute_primary_score(current_metrics)
    last_ckpt = find_last_checkpoint(run_dir)

    base_config = maybe_load_config(base_dir)
    base_metrics = maybe_load_metrics(base_dir)
    base_score = compute_primary_score(base_metrics)

    model_cfg = (current_config or {}).get("model", {})
    dataloader_cfg = (current_config or {}).get("dataloader", {})
    trainer_cfg = (current_config or {}).get("trainer", {})

    output_path = run_dir / f"{args.report_date}_phase2_realdata_report.md"
    lines = [
        "# Phase2 Real-Data Report",
        "",
        f"- Experiment run dir: `{run_dir}`",
        f"- Status: **{current_status}**",
        f"- Base corrected winner: `{base_dir}`",
        f"- Warm start: `{model_cfg.get('warm_start_weights_dir', '-')}`",
        f"- Batch size: `{dataloader_cfg.get('batch_size', '-')}`",
        f"- Num workers: `{dataloader_cfg.get('num_workers', '-')}`",
        f"- Max epochs: `{trainer_cfg.get('max_epochs', '-')}`",
        f"- Last checkpoint: `{last_ckpt or '-'}`",
        f"- Motion encoder frozen: `{model_cfg.get('freeze_motion_backbone', '-')}`",
        f"- Text encoder frozen: `{model_cfg.get('freeze_text_encoder', '-')}`",
        f"- Motion decoder frozen: `{model_cfg.get('freeze_motion_decoder', '-')}`",
        "",
        "## Retrieval-First Gate Snapshot",
        "",
        f"- Current PrimaryScore: **{format_metric(current_score)}**",
        f"- Base D2b PrimaryScore: **{format_metric(base_score)}**",
        f"- Delta vs D2b: **{format_delta(current_score, base_score)}**",
        "- PrimaryScore is the mean of normal+nsim `t2m/m2t` R@1 and R@5.",
        "",
        "## Current Metrics",
        "",
        *render_metric_table(current_metrics),
        "",
        "## Base D2b Metrics",
        "",
        *render_metric_table(base_metrics),
        "",
    ]

    if current_score is None:
        lines.extend(
            [
                "## Note",
                "",
                "- Retrieval metrics are not complete yet, so the gate stays pending.",
                "",
            ]
        )

    if base_config is None:
        lines.extend(
            [
                "## Warning",
                "",
                "- Base D2b config was not found; verify the Stage4.1 winner directory before trusting the comparison.",
                "",
            ]
        )

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
