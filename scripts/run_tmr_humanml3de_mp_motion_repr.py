#!/usr/bin/env python3
"""Launch TMR HumanML3D-E-MP motion-representation comparison runs.

Supports multiple experiment stages:
  warmstart    - vanilla TMR 500ep, produces per-rep warm-start weights
  finetune_d2b - D2b finetune 50ep from warm-start (frozen text encoder)
  finetune_p2a - P2a finetune 50ep from warm-start (trainable text encoder)
  scratch      - P2a scratch 500ep, no warm-start
  all          - warmstart -> finetune_d2b -> finetune_p2a (sequential)
"""

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

STAGE_CONFIGS = {
    "warmstart": {"model": "tmr", "epochs": 500},
    "finetune_d2b": {"model": "tmr_d2b_ft", "epochs": 50},
    "finetune_p2a": {"model": "tmr_p2a_ft", "epochs": 50},
    "scratch": {"model": "tmr_p2a_scratch", "epochs": 500},
}

VALID_STAGES = list(STAGE_CONFIGS.keys()) + ["all"]


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
        default=["guo263", "kimodo261"],
    )
    parser.add_argument(
        "--stage",
        choices=VALID_STAGES,
        default="all",
        help="Experiment stage to run.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--retrieval-batch-size", type=int, default=256)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "humanml3d_e_mp_motion_repr_server",
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


def warmstart_weights_dir(run_root: Path, schema: str) -> Path:
    """Return the last_weights path produced by the warmstart stage."""
    return run_root / "tmr" / schema / "last_weights"


def build_train_command(
    *,
    model: str,
    epochs: int,
    schema: str,
    run_dir: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
    extra_overrides: list[str],
    warm_start_dir: Path | None = None,
) -> list[str]:
    motion_loader = resolve_motion_loader(schema)
    command = [
        sys.executable,
        "train.py",
        "data=humanml3d_e_mp",
        f"data/motion_loader={motion_loader}",
        f"model={model}",
        f"run_dir={run_dir}",
        f"dataloader.batch_size={batch_size}",
        f"dataloader.num_workers={num_workers}",
        f"seed={seed}",
        f"trainer.max_epochs={epochs}",
    ]
    if warm_start_dir is not None:
        command.append(f"model.warm_start_weights_dir={warm_start_dir}")
    command.extend(extra_overrides)
    return command


def build_retrieval_command(
    *, run_dir: Path, device: str, retrieval_batch_size: int, extra_overrides: list[str]
) -> list[str]:
    command = [
        sys.executable,
        "retrieval.py",
        f"run_dir={run_dir}",
        f"device={device}",
        f"batch_size={retrieval_batch_size}",
        "protocol=all",
    ]
    command.extend(extra_overrides)
    return command


def run_stage(
    stage_name: str, args: argparse.Namespace, schema: str
) -> None:
    cfg = STAGE_CONFIGS[stage_name]
    model = cfg["model"]
    epochs = cfg["epochs"]
    run_dir = args.run_root / model / schema

    warm_start_dir: Path | None = None
    if stage_name in ("finetune_d2b", "finetune_p2a"):
        warm_start_dir = warmstart_weights_dir(args.run_root, schema)
        if not args.dry_run and not warm_start_dir.exists():
            print(
                f"[tmr-hml3de-mp] SKIP {stage_name}/{schema}: "
                f"warm-start not found at {warm_start_dir}"
            )
            return

    print(f"\n{'='*60}")
    print(f"[tmr-hml3de-mp] stage={stage_name} schema={schema} model={model} epochs={epochs}")
    print(f"{'='*60}\n")

    train_cmd = build_train_command(
        model=model,
        epochs=epochs,
        schema=schema,
        run_dir=run_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        extra_overrides=args.train_override,
        warm_start_dir=warm_start_dir,
    )
    run_command(train_cmd, args.dry_run)

    if args.skip_retrieval:
        return

    retrieval_cmd = build_retrieval_command(
        run_dir=run_dir,
        device=args.device,
        retrieval_batch_size=args.retrieval_batch_size,
        extra_overrides=args.retrieval_override,
    )
    run_command(retrieval_cmd, args.dry_run)


def main() -> None:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)

    if args.stage == "all":
        stages = ["warmstart", "finetune_d2b", "finetune_p2a"]
    else:
        stages = [args.stage]

    for schema in args.schemas:
        for stage in stages:
            run_stage(stage, args, schema)


if __name__ == "__main__":
    main()
