#!/usr/bin/env python3
"""Launch TMR HumanML3D-E-MP motion-representation comparison runs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_ALIASES = {
    "guo263": "emp_guo263",
    "pos66": "emp_pos66",
    "kimodo261": "emp_kimodo_like_261",
    "kimodo_like_261": "emp_kimodo_like_261",
    "smpl135": "emp_smpl_d135_recon",
    "smpl_d135_recon": "emp_smpl_d135_recon",
    "hy201": "emp_hy201_recon",
    "hy201_recon": "emp_hy201_recon",
    "hml272": "emp_hml272",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run TMR training/retrieval on HumanML3D-E-MP with event-decomposed text "
            "and schema-specific motion representations."
        )
    )
    parser.add_argument(
        "--schemas",
        nargs="+",
        default=["guo263", "pos66", "kimodo261", "smpl135", "hy201", "hml272"],
    )
    parser.add_argument("--model", default="tmr_d1")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--retrieval-batch-size", type=int, default=256)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "humanml3d_e_mp_motion_repr",
    )
    parser.add_argument(
        "--train-override",
        action="append",
        default=[],
        help="Extra Hydra override passed to train.py. Repeatable.",
    )
    parser.add_argument(
        "--retrieval-override",
        action="append",
        default=[],
        help="Extra Hydra override passed to retrieval.py. Repeatable.",
    )
    parser.add_argument(
        "--skip-retrieval",
        action="store_true",
        help="Only train and skip the post-train retrieval evaluation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def resolve_motion_loader(schema: str) -> str:
    try:
        return SCHEMA_ALIASES[schema]
    except KeyError as exc:
        supported = ", ".join(sorted(SCHEMA_ALIASES))
        raise ValueError(f"Unsupported schema '{schema}'. Supported: {supported}") from exc


def run_command(command: list[str], dry_run: bool) -> None:
    print("[tmr-hml3de-mp]", " ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def build_train_command(args: argparse.Namespace, schema: str, run_dir: Path) -> list[str]:
    motion_loader = resolve_motion_loader(schema)
    command = [
        sys.executable,
        "train.py",
        "data=humanml3d_e_mp",
        f"data/motion_loader={motion_loader}",
        f"model={args.model}",
        f"run_dir={run_dir}",
        f"dataloader.batch_size={args.batch_size}",
        f"dataloader.num_workers={args.num_workers}",
        f"seed={args.seed}",
    ]
    command.append(f"trainer.max_epochs={args.epochs}")
    command.extend(args.train_override)
    return command


def build_retrieval_command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "retrieval.py",
        f"run_dir={run_dir}",
        f"device={args.device}",
        f"batch_size={args.retrieval_batch_size}",
        "protocol=all",
    ]
    command.extend(args.retrieval_override)
    return command


def main() -> None:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)

    for schema in args.schemas:
        run_dir = args.run_root / args.model / schema
        train_command = build_train_command(args, schema, run_dir)
        run_command(train_command, args.dry_run)

        if args.skip_retrieval:
            continue

        retrieval_command = build_retrieval_command(args, run_dir)
        run_command(retrieval_command, args.dry_run)


if __name__ == "__main__":
    main()
