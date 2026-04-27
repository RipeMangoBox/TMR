#!/usr/bin/env python3
"""Compute vanilla TMR retrieval and ceiling statistics."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.prepare  # noqa: F401
from src.config import read_config
from src.data.collate import collate_text_motion
from src.load import _resolve_ckpt_path, load_model_from_cfg
from src.model.metrics import all_contrastive_metrics


DEFAULT_RUN_DIR = Path("models/tmr_humanml3d_guoh3dfeats")
DEFAULT_OUTPUT = Path(
    "/data/Life Me/Obsidian Respository/ResearchWY/paperIDEAs/TAMR/"
    "vanilla_tmr_humanml3d_ceiling_stats.json"
)
TEMPORAL_CUE_RE = re.compile(
    r"\b(and then|then|before|after|while|simultaneously|at the same time|meanwhile)\b",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute vanilla TMR retrieval and ceiling stats on HumanML3D."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--ckpt-name", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "nsim_test"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def chunked_indices(size: int, batch_size: int):
    for begin in range(0, size, batch_size):
        yield slice(begin, min(size, begin + batch_size))


def compute_rank(sim_row: np.ndarray, gt_index: int) -> float:
    gt_score = float(sim_row[gt_index])
    better = int(np.sum(sim_row > gt_score))
    tied = int(np.sum(sim_row == gt_score))
    return better + max(tied - 1, 0) / 2.0


def cols_to_metrics(cols: np.ndarray) -> dict[str, float]:
    if cols.size == 0:
        return {"R01": 0.0, "R05": 0.0, "R10": 0.0, "MedR": float("inf")}
    metrics = {
        "R01": 100.0 * float(np.mean(cols < 1)),
        "R05": 100.0 * float(np.mean(cols < 5)),
        "R10": 100.0 * float(np.mean(cols < 10)),
        "MedR": float(np.median(cols) + 1.0),
    }
    return {key: round(value, 4) for key, value in metrics.items()}


def compute_ceiling(sim_matrix: np.ndarray, row_indices: np.ndarray, topk_values):
    gt_ranks = np.asarray(
        [compute_rank(sim_matrix[row_idx], int(row_idx)) for row_idx in row_indices],
        dtype=np.float64,
    )
    metrics = {}
    for topk in topk_values:
        metrics[f"ceiling@{topk}"] = round(100.0 * float(np.mean(gt_ranks < topk)), 4)
    metrics["count"] = int(row_indices.size)
    return metrics, gt_ranks


def compute_subset_metrics(sim_matrix: np.ndarray, row_indices: np.ndarray):
    cols = np.asarray(
        [compute_rank(sim_matrix[row_idx], int(row_idx)) for row_idx in row_indices],
        dtype=np.float64,
    )
    metrics = cols_to_metrics(cols)
    metrics["count"] = int(row_indices.size)
    return metrics


def collect_features(model, dataset, batch_size: int, device: str):
    keyids = []
    texts = []
    sent_embs = []
    text_latents = []
    motion_latents = []

    for batch_slice in tqdm(
        list(chunked_indices(len(dataset), batch_size)),
        desc="Collecting vanilla TMR features",
    ):
        batch_items = [dataset[idx] for idx in range(batch_slice.start, batch_slice.stop)]
        batch = collate_text_motion(batch_items, device=device)
        with torch.inference_mode():
            t_latent = model.encode(batch["text_x_dict"], modality="text", sample_mean=True)
            m_latent = model.encode(batch["motion_x_dict"], modality="motion", sample_mean=True)

        keyids.extend(batch["keyid"])
        texts.extend(batch["text"])
        sent_embs.append(batch["sent_emb"].cpu())
        text_latents.append(t_latent.cpu())
        motion_latents.append(m_latent.cpu())

    return {
        "keyids": keyids,
        "texts": texts,
        "sent_embs": torch.cat(sent_embs, dim=0),
        "text_latents": torch.cat(text_latents, dim=0),
        "motion_latents": torch.cat(motion_latents, dim=0),
    }


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cfg = read_config(str(args.run_dir))
    ckpt_name = args.ckpt_name or cfg.ckpt
    try:
        resolved_ckpt_path = _resolve_ckpt_path(str(args.run_dir), ckpt_name)
    except FileNotFoundError:
        resolved_ckpt_path = f"{args.run_dir}/{ckpt_name}_weights (extracted weights only)"
    model = load_model_from_cfg(cfg, ckpt_name=ckpt_name, device=args.device, eval_mode=True)
    dataset = instantiate(cfg.data, split=args.split)

    features = collect_features(model, dataset, args.batch_size, args.device)
    text_latents = F.normalize(features["text_latents"].float(), dim=-1)
    motion_latents = F.normalize(features["motion_latents"].float(), dim=-1)
    sim_matrix = (text_latents @ motion_latents.T).cpu().numpy()
    sent_embs = features["sent_embs"].cpu().numpy()

    normal_metrics = all_contrastive_metrics(sim_matrix)
    threshold_metrics = all_contrastive_metrics(
        sim_matrix,
        sent_embs,
        threshold=cfg.model.threshold_selfsim_metrics,
    )
    overall_indices = np.arange(len(features["texts"]))
    overall_ceiling, overall_ranks = compute_ceiling(sim_matrix, overall_indices, [10, 20, 50, 100])
    temporal_indices = np.asarray(
        [idx for idx, text in enumerate(features["texts"]) if TEMPORAL_CUE_RE.search(text)],
        dtype=np.int64,
    )

    temporal_results = {
        "definition": "Caption contains one of {then, before, after, while, simultaneously, at the same time, meanwhile}. This is a proxy subset, not equivalent to HumanML3D-E decomposed K>=2.",
        "fraction_of_dataset": round(float(len(temporal_indices) / len(features["texts"])), 6),
        "metrics_against_full_gallery": compute_subset_metrics(sim_matrix, temporal_indices),
        "ceiling_against_full_gallery": compute_ceiling(sim_matrix, temporal_indices, [10, 20, 50, 100])[0],
    }

    results = {
        "meta": {
            "run_dir": str(args.run_dir),
            "ckpt": ckpt_name,
            "resolved_ckpt_path": resolved_ckpt_path,
            "split": args.split,
            "device": args.device,
            "dataset_target": cfg.data._target_,
            "dataset_path": str(getattr(cfg.data, "path", "")),
            "dataset_size": len(features["texts"]),
            "motion_rep": "guoh3dfeats",
            "note": "HumanML3D original annotations do not contain HumanML3D-E style decomposed events, so S3-style event-order diagnostics are not directly available here.",
        },
        "overall_retrieval_metrics": {
            "normal": normal_metrics,
            f"threshold_{cfg.model.threshold_selfsim_metrics}": threshold_metrics,
        },
        "overall_ceiling": overall_ceiling,
        "temporal_cue_proxy_subset": temporal_results,
        "rank_distribution": {
            "mean_rank": round(float(np.mean(overall_ranks) + 1.0), 4),
            "median_rank": round(float(np.median(overall_ranks) + 1.0), 4),
            "p90_rank": round(float(np.percentile(overall_ranks, 90) + 1.0), 4),
        },
    }

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
