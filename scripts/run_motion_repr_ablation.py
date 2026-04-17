"""scripts/run_motion_repr_ablation.py"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.motion_repr_dataset import MotionReprDataset
from src.model.metrics import all_contrastive_metrics
from src.model.motion_repr_baseline import DistilBertTokenCache, MotionReprBaseline


DEFAULT_SCHEMAS = (
    "guo263",
    "pos66",
    "smpl_d135_recon",
    "hy201_recon",
    "kimodo_like_261",
)


def default_data_root() -> Path:
    fallback_root = Path(
        Path.home()
        / "Coding"
        / "Github"
        / "Motion"
        / "datasets"
        / "HumanML3D-E-MP"
    )
    return Path(os.environ.get("MOTIONPATCHES_DATA_ROOT", str(fallback_root)))


def parse_args() -> argparse.Namespace:
    data_root = default_data_root()
    parser = argparse.ArgumentParser(
        description="Run motion representation ablation over multiple motion schemas."
    )
    parser.add_argument(
        "--schemas",
        nargs="+",
        default=DEFAULT_SCHEMAS,
        help=f"Schemas to train. Defaults to: {', '.join(DEFAULT_SCHEMAS)}.",
    )
    parser.add_argument(
        "--format-root",
        type=Path,
        default=data_root / "motion_formats",
        help="Root containing per-schema motion directories.",
    )
    parser.add_argument(
        "--stats-root",
        type=Path,
        default=data_root / "motion_format_stats",
        help="Root containing per-schema Mean.npy / Std.npy.",
    )
    parser.add_argument(
        "--text-dir",
        type=Path,
        default=data_root / "texts",
        help="Directory containing caption txt files.",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=data_root,
        help="Directory containing train.txt / val.txt / test.txt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Experiment output directory.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-motion-length", type=int, default=224)
    parser.add_argument("--max-text-length", type=int, default=64)
    parser.add_argument("--text-encode-batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    dataset: MotionReprDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    drop_last: bool,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    def seed_worker(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=generator,
        worker_init_fn=seed_worker,
    )


def move_motion_batch(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    motion = batch["motion"].to(device=device, dtype=torch.float32, non_blocking=True)
    motion_length = batch["motion_length"].to(device=device, dtype=torch.long)
    return motion, motion_length


def build_schema_datasets(args: argparse.Namespace, schema: str) -> dict[str, MotionReprDataset]:
    motion_dir = args.format_root / schema
    mean_path = args.stats_root / schema / "Mean.npy"
    std_path = args.stats_root / schema / "Std.npy"

    datasets = {}
    for split in ("train", "val", "test"):
        datasets[split] = MotionReprDataset(
            motion_dir=str(motion_dir),
            text_dir=str(args.text_dir),
            split_file=str(args.split_dir / f"{split}.txt"),
            mean_path=str(mean_path),
            std_path=str(std_path),
            max_motion_length=args.max_motion_length,
        )
    return datasets


def build_model(args: argparse.Namespace, motion_input_dim: int, device: torch.device) -> MotionReprBaseline:
    model = MotionReprBaseline(
        motion_input_dim=motion_input_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_size=args.ff_size,
        dropout=args.dropout,
        temperature=args.temperature,
        max_text_length=args.max_text_length,
    )
    return model.to(device)


def train_one_epoch(
    model: MotionReprBaseline,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    text_cache: DistilBertTokenCache,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        motion, motion_length = move_motion_batch(batch, device)
        text_emb, text_length = text_cache.encode(batch["caption"])

        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            motion=motion,
            motion_length=motion_length,
            text_emb=text_emb,
            text_length=text_length,
        )
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        num_batches += 1

    if num_batches == 0:
        return 0.0
    return running_loss / float(num_batches)


@torch.inference_mode()
def evaluate_retrieval(
    model: MotionReprBaseline,
    loader: DataLoader,
    text_cache: DistilBertTokenCache,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_motion_emb: list[torch.Tensor] = []
    all_text_emb: list[torch.Tensor] = []

    for batch in tqdm(loader, desc="Eval", leave=False):
        motion, motion_length = move_motion_batch(batch, device)
        text_emb, text_length = text_cache.encode(batch["caption"])
        motion_latent = model.encode_motion(motion, motion_length)
        text_latent = model.encode_text(text_emb=text_emb, text_length=text_length)
        all_motion_emb.append(motion_latent.cpu())
        all_text_emb.append(text_latent.cpu())

    if not all_motion_emb or not all_text_emb:
        raise ValueError("Evaluation loader produced no batches.")

    motion_matrix = torch.cat(all_motion_emb, dim=0)
    text_matrix = torch.cat(all_text_emb, dim=0)
    sim_matrix = (text_matrix @ motion_matrix.transpose(0, 1)).cpu().numpy()

    metrics = all_contrastive_metrics(sim_matrix)
    metrics["PrimaryScore"] = round(
        (
            metrics["t2m/R01"]
            + metrics["m2t/R01"]
            + metrics["t2m/R05"]
            + metrics["m2t/R05"]
        )
        / 4.0,
        2,
    )
    return metrics


def save_checkpoint(
    path: Path,
    model: MotionReprBaseline,
    schema: str,
    epoch: int,
    best_val_r1: float,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": schema,
        "epoch": epoch,
        "best_val_r1": best_val_r1,
        "metrics": metrics,
        "model_state_dict": model.state_dict(),
    }
    torch.save(payload, path)


def run_schema(
    args: argparse.Namespace,
    schema: str,
    text_cache: DistilBertTokenCache,
    device: torch.device,
) -> dict[str, Any]:
    print(f"\n[run_motion_repr_ablation] schema={schema}")
    seed_everything(args.seed)

    datasets = build_schema_datasets(args, schema)
    train_loader = make_loader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
        drop_last=len(datasets["train"]) >= args.batch_size,
        pin_memory=device.type == "cuda",
    )
    val_loader = make_loader(
        datasets["val"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    test_loader = make_loader(
        datasets["test"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args, motion_input_dim=datasets["train"].motion_dim, device=device)
    model.set_text_embedder(text_cache)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
    )

    schema_output_dir = args.output_dir / schema
    best_ckpt_path = schema_output_dir / "best_model.pt"
    best_val_r1 = -float("inf")
    best_epoch = -1
    best_val_metrics: dict[str, float] | None = None
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            text_cache=text_cache,
            device=device,
        )
        val_metrics = evaluate_retrieval(
            model=model,
            loader=val_loader,
            text_cache=text_cache,
            device=device,
        )
        val_r1 = 0.5 * (val_metrics["t2m/R01"] + val_metrics["m2t/R01"])
        scheduler.step()

        print(
            f"  epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_R1={val_r1:.2f} primary={val_metrics['PrimaryScore']:.2f}"
        )

        if val_r1 > best_val_r1:
            best_val_r1 = val_r1
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            bad_epochs = 0
            save_checkpoint(
                path=best_ckpt_path,
                model=model,
                schema=schema,
                epoch=epoch,
                best_val_r1=best_val_r1,
                metrics=val_metrics,
            )
            continue

        bad_epochs += 1
        if bad_epochs >= args.patience:
            print(f"  early_stop epoch={epoch:03d} patience={args.patience}")
            break

    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_retrieval(
        model=model,
        loader=test_loader,
        text_cache=text_cache,
        device=device,
    )

    metrics_payload = {
        "schema": schema,
        "motion_dim": datasets["train"].motion_dim,
        "best_epoch": best_epoch,
        "best_val_r1": round(float(best_val_r1), 2),
        "val_metrics": best_val_metrics or {},
        "test_metrics": test_metrics,
    }
    schema_output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = schema_output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"  finished schema={schema} best_epoch={best_epoch} "
        f"test_primary={test_metrics['PrimaryScore']:.2f}"
    )
    return metrics_payload


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    text_cache = DistilBertTokenCache(
        device=device,
        max_length=args.max_text_length,
        encode_batch_size=args.text_encode_batch_size,
    )

    results: dict[str, Any] = {}
    for schema in args.schemas:
        results[schema] = run_schema(
            args=args,
            schema=schema,
            text_cache=text_cache,
            device=device,
        )

    comparison_table = {
        "schemas": args.schemas,
        "results": results,
        "ranking": sorted(
            (
                {
                    "schema": schema,
                    "PrimaryScore": payload["test_metrics"]["PrimaryScore"],
                    "t2m/R01": payload["test_metrics"]["t2m/R01"],
                    "m2t/R01": payload["test_metrics"]["m2t/R01"],
                }
                for schema, payload in results.items()
            ),
            key=lambda item: item["PrimaryScore"],
            reverse=True,
        ),
    }
    comparison_path = args.output_dir / "comparison_table.json"
    comparison_path.write_text(
        json.dumps(comparison_table, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved comparison table to {comparison_path}")


if __name__ == "__main__":
    main()
