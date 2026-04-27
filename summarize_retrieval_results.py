from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import yaml
from linked_codebases import EVENTT2M_ROOT, MOTIONPATCHES_ROOT

PROTOCOLS = ["normal", "threshold_0.95", "nsim", "guo"]
METRIC_ORDER = [
    "t2m/R01",
    "t2m/R02",
    "t2m/R03",
    "t2m/R05",
    "t2m/R10",
    "t2m/MedR",
    "m2t/R01",
    "m2t/R02",
    "m2t/R03",
    "m2t/R05",
    "m2t/R10",
    "m2t/MedR",
]
DEFAULT_NAMES = ["TMR", "MotionPatches", "EventT2M"]
DEFAULT_PATHS = [
    str(Path("RUN_DIR/contrastive_metrics")),
    str(MOTIONPATCHES_ROOT / "checkpoints" / "pretrained" / "HumanML3D" / "contrastive_metrics"),
    str(EVENTT2M_ROOT / "checkpoints" / "pretrained" / "HumanML3D" / "eval"),
]
DEFAULT_OUTPUT = "retrieval_results_summary.md"
SUMMARY_PROTOCOL_FILES = {
    "TMR": {
        "normal": ["normal.yaml"],
        "threshold_0.95": ["threshold_0.95.yaml"],
        "nsim": ["nsim.yaml"],
        "guo": ["guo.yaml"],
    },
    "MotionPatches": {
        "normal": ["retrieval_normal.yaml", "normal.yaml"],
        "threshold_0.95": ["retrieval_threshold_0.95.yaml", "threshold_0.95.yaml"],
        "nsim": ["retrieval_nsim.yaml", "nsim.yaml"],
        "guo": ["retrieval_guo.yaml", "guo.yaml"],
    },
    "EventT2M": {
        "normal": ["normal.yaml", "retrieval_normal.yaml"],
        "threshold_0.95": ["threshold_0.95.yaml", "retrieval_threshold_0.95.yaml"],
        "nsim": ["nsim.yaml", "retrieval_nsim.yaml"],
        "guo": ["guo.yaml", "retrieval_guo.yaml"],
    },
}
NATIVE_PROTOCOLS = {
    "TMR": {"normal", "threshold_0.95", "nsim", "guo"},
    "MotionPatches": {"normal", "guo"},
    "EventT2M": {"normal"},
}
NATIVE_PROTOCOL_FILES = {
    "TMR": {
        "normal": ["normal.yaml"],
        "threshold_0.95": ["threshold_0.95.yaml"],
        "nsim": ["nsim.yaml"],
        "guo": ["guo.yaml"],
    },
    "MotionPatches": {
        "normal": ["normal.yaml"],
        "guo": ["guo.yaml"],
    },
    "EventT2M": {
        "normal": ["E-native_normal.yaml", "native_normal.yaml"],
    },
}


def load_yaml(path: Path) -> Dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def resolve_metric_file(eval_dir: Path, candidates: List[str]) -> Path:
    for candidate in candidates:
        direct_path = eval_dir / candidate
        if direct_path.exists():
            return direct_path

        recursive_matches = sorted(eval_dir.glob(f"**/{candidate}"))
        if recursive_matches:
            return recursive_matches[0]

    joined = ", ".join(candidates)
    raise FileNotFoundError(f"Missing file under {eval_dir}: one of [{joined}]")


def format_value(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def collect_results(repo_names: List[str], eval_dirs: List[Path]) -> Dict[str, Dict[str, Dict[str, float]]]:
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for repo_name, eval_dir in zip(repo_names, eval_dirs):
        protocol_map: Dict[str, Dict[str, float]] = {}
        file_map = SUMMARY_PROTOCOL_FILES.get(repo_name)
        if file_map is None:
            raise KeyError(f"Missing summary protocol file mapping for repo: {repo_name}")
        for protocol in PROTOCOLS:
            protocol_path = resolve_metric_file(eval_dir, file_map[protocol])
            protocol_map[protocol] = load_yaml(protocol_path)
        results[repo_name] = protocol_map
    return results


def format_repo_value(repo_name: str, protocol: str, value) -> str:
    text = format_value(value)
    if protocol in NATIVE_PROTOCOLS.get(repo_name, set()):
        return f"{text} *"
    return text


def render_protocol_table(protocol: str, repo_names: List[str], results: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    header = ["Metric", *repo_names]
    separator = ["---"] * len(header)
    lines = [
        f"## {protocol}",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for metric in METRIC_ORDER:
        row = [metric]
        for repo_name in repo_names:
            value = results[repo_name][protocol].get(metric)
            row.append(format_repo_value(repo_name, protocol, value))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def render_overview_table(repo_names: List[str], results: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    header = ["Protocol", "Metric", *repo_names]
    separator = ["---"] * len(header)
    lines = [
        "## Quick view",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    quick_metrics = ["t2m/R01", "t2m/R05", "t2m/R10", "t2m/MedR", "m2t/R01", "m2t/R05", "m2t/R10", "m2t/MedR"]
    for protocol in PROTOCOLS:
        for metric in quick_metrics:
            row = [protocol, metric]
            for repo_name in repo_names:
                row.append(format_repo_value(repo_name, protocol, results[repo_name][protocol].get(metric)))
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def render_native_reference(repo_names: List[str], eval_dirs: List[Path]) -> str:
    lines = [
        "## Native evaluation markers",
        "",
        "- `*` 表示该单元格对应仓库原生支持的评测协议。",
        "- 未标 `*` 的结果表示为了与 TMR 对齐而补充的 retrieval-style 评测。",
        "",
        "| Repo | Native protocols | Native files |",
        "| --- | --- | --- |",
    ]
    for repo_name, eval_dir in zip(repo_names, eval_dirs):
        native_protocols = sorted(NATIVE_PROTOCOLS.get(repo_name, set()), key=PROTOCOLS.index)
        native_files = NATIVE_PROTOCOL_FILES.get(repo_name, {})
        file_desc = ", ".join(
            f"`{protocol}` → {', '.join(f'`{name}`' for name in native_files[protocol])}"
            for protocol in native_protocols
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    repo_name,
                    ", ".join(f"`{protocol}`" for protocol in native_protocols),
                    file_desc or "-",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def build_summary(repo_names: List[str], eval_dirs: List[Path], results: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    lines = [
        "# Retrieval Results Summary",
        "",
        "## Evaluation directories",
        "",
    ]
    for repo_name, eval_dir in zip(repo_names, eval_dirs):
        lines.append(f"- **{repo_name}**: `{eval_dir}`")
    lines.append("")
    lines.append(render_native_reference(repo_names, eval_dirs))
    lines.append(render_overview_table(repo_names, results))
    for protocol in PROTOCOLS:
        lines.append(render_protocol_table(protocol, repo_names, results))
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize retrieval YAML metrics from three repositories.")
    parser.add_argument(
        "--eval-dir",
        dest="eval_dirs",
        action="append",
        help="Path to a contrastive_metrics directory. Provide exactly three times.",
    )
    parser.add_argument(
        "--name",
        dest="names",
        action="append",
        help="Display name for each eval dir. Provide three times in the same order as --eval-dir.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output markdown file path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_names = args.names or DEFAULT_NAMES
    eval_dirs_raw = args.eval_dirs or DEFAULT_PATHS

    if len(repo_names) != 3:
        raise ValueError("Please provide exactly three --name values or use defaults.")
    if len(eval_dirs_raw) != 3:
        raise ValueError("Please provide exactly three --eval-dir values or use defaults.")

    root = Path(__file__).resolve().parent
    eval_dirs = [Path(p) if Path(p).is_absolute() else root / p for p in eval_dirs_raw]
    results = collect_results(repo_names, eval_dirs)
    summary = build_summary(repo_names, eval_dirs, results)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path.write_text(summary, encoding="utf-8")

    print(f"Saved summary to {output_path}")


if __name__ == "__main__":
    main()
