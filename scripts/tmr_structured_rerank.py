#!/usr/bin/env python3
"""Phase 1 R1 verification for vanilla TMR on HumanML3D-E-MP."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from tqdm import tqdm

from linked_codebases import RESEARCHWY_ROOT

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.prepare  # noqa: F401
from src.config import read_config
from src.data.collate import collate_text_motion_event, collate_x_dict
from src.load import _resolve_ckpt_path, load_model_from_cfg
from src.model.event_grounded.ordered_matching import OrderedMatchingModule


DEFAULT_RUN_DIR = Path("outputs/humanml3d_e_mp_motion_repr_server/tmr/guo263")
DEFAULT_OUTPUT = RESEARCHWY_ROOT / "paperIDEAs" / "TAMR" / "r1_verification_tmr_humanml3de_mp.json"
DEFAULT_LAMBDAS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
OVERLAP_RE = re.compile(
    r"\b(while|simultaneously|simultaneous|meanwhile|concurrently|at the same time)\b",
    flags=re.IGNORECASE,
)


@dataclass
class FeatureBank:
    keyids: list[str]
    captions: list[str]
    event_texts: list[list[str]]
    event_counts: np.ndarray
    overlap_mask: np.ndarray
    motion_lengths: np.ndarray
    text_latents: torch.Tensor
    motion_latents: torch.Tensor
    event_embs: torch.Tensor
    event_mask: torch.Tensor
    segment_tokens: torch.Tensor
    segment_mask: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Phase 1 R1 gates with vanilla TMR on HumanML3D-E-MP."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--ckpt-name", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "nsim_test"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window-min-frames", type=int, default=100)
    parser.add_argument("--s0-max-samples", type=int, default=256)
    parser.add_argument("--segment-count", type=int, default=14)
    parser.add_argument("--rerank-top-k", type=int, default=100)
    parser.add_argument(
        "--event-encode-mode",
        default="independent",
        choices=["independent", "prefix", "context"],
    )
    parser.add_argument(
        "--score-norm",
        default="zscore",
        choices=["none", "zscore", "minmax"],
    )
    parser.add_argument(
        "--lambda-s",
        type=float,
        nargs="+",
        default=DEFAULT_LAMBDAS,
        help="Structured-score weights. lambda_g is always 1 - lambda_s.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def build_event_strings(caption: str, events: Sequence[str], mode: str) -> list[str]:
    if mode == "independent":
        return [str(event) for event in events]
    if mode == "prefix":
        total = len(events)
        return [f"Event {idx + 1} of {total}: {event}" for idx, event in enumerate(events)]
    if mode == "context":
        all_events = " ; ".join(
            f"Event {idx + 1}: {event}" for idx, event in enumerate(events)
        )
        total = len(events)
        return [
            (
                f"Full caption: {caption}. "
                f"All events: {all_events}. "
                f"Focus event {idx + 1} of {total}: {event}"
            )
            for idx, event in enumerate(events)
        ]
    raise ValueError(f"Unsupported event mode: {mode}")


def has_overlap_cue(caption: str) -> bool:
    return bool(OVERLAP_RE.search(caption))


def chunked_indices(size: int, batch_size: int) -> Iterable[slice]:
    for begin in range(0, size, batch_size):
        yield slice(begin, min(size, begin + batch_size))


def normalize_scores(scores: torch.Tensor, mode: str, eps: float = 1.0e-6) -> torch.Tensor:
    if mode == "none":
        return scores
    if mode == "zscore":
        std = scores.std(unbiased=False)
        if float(std) < eps:
            return torch.zeros_like(scores)
        return (scores - scores.mean()) / std
    if mode == "minmax":
        min_val = scores.min()
        max_val = scores.max()
        if float(max_val - min_val) < eps:
            return torch.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)
    raise ValueError(f"Unsupported score norm: {mode}")


def segment_temporal_tokens_single(
    temporal_tokens: torch.Tensor,
    valid_length: int,
    num_segments: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    latent_dim = temporal_tokens.shape[-1]
    segment_tokens = temporal_tokens.new_zeros((num_segments, latent_dim))
    segment_mask = torch.zeros((num_segments,), dtype=torch.bool, device=temporal_tokens.device)

    valid_length = min(int(valid_length), int(temporal_tokens.shape[0]))
    if valid_length <= 0:
        return segment_tokens, segment_mask

    temporal_tokens = temporal_tokens[:valid_length]
    for segment_idx in range(num_segments):
        start = math.floor(segment_idx * valid_length / num_segments)
        end = math.floor((segment_idx + 1) * valid_length / num_segments)
        if end <= start:
            continue
        segment_tokens[segment_idx] = temporal_tokens[start:end].mean(dim=0)
        segment_mask[segment_idx] = True
    return segment_tokens, segment_mask


def segment_temporal_tokens_batch(
    temporal_tokens: torch.Tensor,
    lengths: Sequence[int],
    num_segments: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, _max_len, latent_dim = temporal_tokens.shape
    segment_tokens = temporal_tokens.new_zeros((batch_size, num_segments, latent_dim))
    segment_mask = torch.zeros((batch_size, num_segments), dtype=torch.bool, device=temporal_tokens.device)
    for batch_idx, length in enumerate(lengths):
        seg_tokens, seg_mask = segment_temporal_tokens_single(
            temporal_tokens[batch_idx],
            int(length),
            num_segments,
        )
        segment_tokens[batch_idx] = seg_tokens
        segment_mask[batch_idx] = seg_mask
    return segment_tokens, segment_mask


def encode_event_batch(
    *,
    model,
    token_embedder,
    captions: Sequence[str],
    event_texts_batch: Sequence[Sequence[str]],
    event_mode: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(event_texts_batch)
    max_events = max((len(events) for events in event_texts_batch), default=0)
    latent_dim = model.motion_encoder.projection.out_features
    event_embs = torch.zeros((batch_size, max_events, latent_dim), device=device)
    event_mask = torch.zeros((batch_size, max_events), dtype=torch.bool, device=device)

    flat_texts: list[str] = []
    sample_ids: list[int] = []
    slot_ids: list[int] = []
    for sample_idx, (caption, events) in enumerate(zip(captions, event_texts_batch)):
        texts = build_event_strings(caption, events, event_mode)
        for slot_idx, text in enumerate(texts):
            flat_texts.append(text)
            sample_ids.append(sample_idx)
            slot_ids.append(slot_idx)
            event_mask[sample_idx, slot_idx] = True

    if not flat_texts:
        return event_embs, event_mask

    event_x_dicts = token_embedder(flat_texts)
    if isinstance(event_x_dicts, dict):
        event_x_dicts = [event_x_dicts]
    collated = collate_x_dict(event_x_dicts, device=device)
    with torch.inference_mode():
        flat_event_embs = model.encode(collated, modality="text", sample_mean=True)

    sample_idx_tensor = torch.tensor(sample_ids, dtype=torch.long, device=device)
    slot_idx_tensor = torch.tensor(slot_ids, dtype=torch.long, device=device)
    event_embs[sample_idx_tensor, slot_idx_tensor] = flat_event_embs
    return event_embs, event_mask


def collect_feature_bank(
    *,
    model,
    dataset,
    batch_size: int,
    device: str,
    segment_count: int,
    event_mode: str,
) -> FeatureBank:
    keyids: list[str] = []
    captions: list[str] = []
    event_texts: list[list[str]] = []
    event_counts: list[int] = []
    overlap_mask: list[bool] = []
    motion_lengths: list[int] = []
    text_latents: list[torch.Tensor] = []
    motion_latents: list[torch.Tensor] = []
    event_embs: list[torch.Tensor] = []
    event_masks: list[torch.Tensor] = []
    segment_tokens: list[torch.Tensor] = []
    segment_masks: list[torch.Tensor] = []
    max_events_dataset = max(
        len(dataset._select_text_item(sample, training=False)["events"])
        for sample in dataset.samples
    )

    for batch_slice in tqdm(
        list(chunked_indices(len(dataset), batch_size)),
        desc="Collecting global/event/segment features",
    ):
        batch_items = [dataset[idx] for idx in range(batch_slice.start, batch_slice.stop)]
        batch = collate_text_motion_event(batch_items, device=device)

        with torch.inference_mode():
            text_latent = model.encode(batch["text_x_dict"], modality="text", sample_mean=True)
            motion_latent = model.encode(
                batch["motion_x_dict"], modality="motion", sample_mean=True
            )
            _motion_tokens, temporal = model.motion_encoder(
                batch["motion_x_dict"], return_temporal=True
            )
            seg_tokens, seg_mask = segment_temporal_tokens_batch(
                temporal,
                batch["motion_x_dict"]["length"],
                segment_count,
            )
            evt_embs, evt_mask = encode_event_batch(
                model=model,
                token_embedder=dataset.text_to_token_emb,
                captions=batch["text"],
                event_texts_batch=batch["event_texts"],
                event_mode=event_mode,
                device=device,
            )
            if evt_embs.shape[1] < max_events_dataset:
                pad_events = max_events_dataset - evt_embs.shape[1]
                evt_embs = F.pad(evt_embs, (0, 0, 0, pad_events))
                evt_mask = F.pad(evt_mask, (0, pad_events))

        keyids.extend(batch["keyid"])
        captions.extend(batch["text"])
        event_texts.extend([list(events) for events in batch["event_texts"]])
        event_counts.extend(len(events) for events in batch["event_texts"])
        overlap_mask.extend(has_overlap_cue(text) for text in batch["text"])
        motion_lengths.extend(int(length) for length in batch["motion_x_dict"]["length"])

        text_latents.append(text_latent.cpu())
        motion_latents.append(motion_latent.cpu())
        event_embs.append(evt_embs.cpu())
        event_masks.append(evt_mask.cpu())
        segment_tokens.append(seg_tokens.cpu())
        segment_masks.append(seg_mask.cpu())

    return FeatureBank(
        keyids=keyids,
        captions=captions,
        event_texts=event_texts,
        event_counts=np.asarray(event_counts, dtype=np.int64),
        overlap_mask=np.asarray(overlap_mask, dtype=bool),
        motion_lengths=np.asarray(motion_lengths, dtype=np.int64),
        text_latents=torch.cat(text_latents, dim=0),
        motion_latents=torch.cat(motion_latents, dim=0),
        event_embs=torch.cat(event_embs, dim=0),
        event_mask=torch.cat(event_masks, dim=0),
        segment_tokens=torch.cat(segment_tokens, dim=0),
        segment_mask=torch.cat(segment_masks, dim=0),
    )


def one_sided_sign_test_p(num_positive: int, num_total: int) -> float:
    if num_total <= 0:
        return 1.0
    if num_total <= 200:
        tail = 0
        for count in range(num_positive, num_total + 1):
            tail += math.comb(num_total, count)
        return tail / float(2**num_total)

    mean = 0.5 * num_total
    std = math.sqrt(0.25 * num_total)
    z = (num_positive - 0.5 - mean) / max(std, 1.0e-12)
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a[None], b[None], dim=-1).item())


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


def compute_subset_metrics(sim_matrix: np.ndarray, row_indices: np.ndarray) -> dict[str, float]:
    cols = np.asarray(
        [compute_rank(sim_matrix[row_idx], int(row_idx)) for row_idx in row_indices],
        dtype=np.float64,
    )
    metrics = cols_to_metrics(cols)
    metrics["count"] = int(row_indices.size)
    return metrics


def compute_topk_ceiling(
    sim_matrix: np.ndarray,
    row_indices: np.ndarray,
    topk_values: Sequence[int],
) -> dict[str, float]:
    gt_ranks = np.asarray(
        [compute_rank(sim_matrix[row_idx], int(row_idx)) for row_idx in row_indices],
        dtype=np.float64,
    )
    metrics = {}
    for topk in topk_values:
        metrics[f"ceiling@{topk}"] = round(100.0 * float(np.mean(gt_ranks < topk)), 4)
    return metrics


def evaluate_s0(
    *,
    model,
    dataset,
    device: str,
    min_frames: int,
    max_samples: int,
) -> dict[str, float | int | bool]:
    candidate_indices = []
    for idx in range(len(dataset)):
        motion_length = int(dataset.samples[idx]["length"])
        if motion_length >= min_frames:
            candidate_indices.append(idx)
        if len(candidate_indices) >= max_samples:
            break

    margins: list[float] = []
    adj_scores: list[float] = []
    far_scores: list[float] = []
    for idx in tqdm(candidate_indices, desc="R1-S0 temporal-window sanity"):
        item = dataset[idx]
        motion = item["motion_x_dict"]["x"]
        length = int(item["motion_x_dict"]["length"])
        window = length // 2
        if window <= 1:
            continue

        starts = [0, max((length - window) // 2, 0), max(length - window, 0)]
        if len(set(starts)) < 3:
            continue

        windows = [{"x": motion[start : start + window], "length": window} for start in starts]
        batch = collate_x_dict(windows, device=device)
        with torch.inference_mode():
            _tokens, temporal = model.motion_encoder(batch, return_temporal=True)

        pooled = []
        for row_idx in range(temporal.shape[0]):
            pooled.append(temporal[row_idx, : window].mean(dim=0))

        adj = cosine_similarity(pooled[0], pooled[1])
        far = cosine_similarity(pooled[0], pooled[2])
        adj_scores.append(adj)
        far_scores.append(far)
        margins.append(adj - far)

    wins = sum(margin > 0 for margin in margins)
    total = len(margins)
    p_value = one_sided_sign_test_p(wins, total)
    mean_adj = float(np.mean(adj_scores)) if adj_scores else 0.0
    mean_far = float(np.mean(far_scores)) if far_scores else 0.0
    return {
        "samples": total,
        "mean_adjacent_cosine": round(mean_adj, 6),
        "mean_far_cosine": round(mean_far, 6),
        "mean_margin": round(float(np.mean(margins)) if margins else 0.0, 6),
        "adjacent_gt_far_ratio": round(float(wins / total) if total else 0.0, 6),
        "sign_test_p_one_sided": round(float(p_value), 8),
        "gate_pass": bool(total > 0 and mean_adj > mean_far and p_value < 0.05),
    }


def evaluate_s1(
    *,
    model,
    dataset,
    batch_size: int,
    device: str,
    event_mode: str,
) -> dict[str, float | int | bool]:
    diag_scores: list[float] = []
    off_scores: list[float] = []
    margins: list[float] = []

    for batch_slice in tqdm(
        list(chunked_indices(len(dataset), batch_size)),
        desc="R1-S1 event-segment alignment",
    ):
        batch_items = [dataset[idx] for idx in range(batch_slice.start, batch_slice.stop)]
        batch = collate_text_motion_event(batch_items, device=device)
        valid_rows = [idx for idx, events in enumerate(batch["event_texts"]) if len(events) >= 2]
        if not valid_rows:
            continue

        with torch.inference_mode():
            _tokens, temporal = model.motion_encoder(batch["motion_x_dict"], return_temporal=True)
            event_embs, _event_mask = encode_event_batch(
                model=model,
                token_embedder=dataset.text_to_token_emb,
                captions=batch["text"],
                event_texts_batch=batch["event_texts"],
                event_mode=event_mode,
                device=device,
            )

        for row_idx in valid_rows:
            events = batch["event_texts"][row_idx]
            n_events = len(events)
            seg_tokens, seg_mask = segment_temporal_tokens_single(
                temporal[row_idx],
                int(batch["motion_x_dict"]["length"][row_idx]),
                n_events,
            )
            if int(seg_mask.sum().item()) < n_events:
                continue

            event_vecs = event_embs[row_idx, :n_events]
            segment_vecs = seg_tokens[:n_events]
            sim = F.normalize(event_vecs, dim=-1) @ F.normalize(segment_vecs, dim=-1).T
            diag_mean = float(torch.diagonal(sim).mean().item())
            off_mask = ~torch.eye(n_events, dtype=torch.bool, device=sim.device)
            off_mean = float(sim[off_mask].mean().item())

            diag_scores.append(diag_mean)
            off_scores.append(off_mean)
            margins.append(diag_mean - off_mean)

    wins = sum(margin > 0 for margin in margins)
    total = len(margins)
    return {
        "samples": total,
        "mean_diag_cosine": round(float(np.mean(diag_scores)) if diag_scores else 0.0, 6),
        "mean_offdiag_cosine": round(float(np.mean(off_scores)) if off_scores else 0.0, 6),
        "mean_margin": round(float(np.mean(margins)) if margins else 0.0, 6),
        "diag_gt_off_ratio": round(float(wins / total) if total else 0.0, 6),
        "gate_pass": bool(total > 0 and np.mean(diag_scores) > np.mean(off_scores)),
    }


def reverse_event_order(event_embs: torch.Tensor, event_mask: torch.Tensor) -> torch.Tensor:
    reversed_embs = event_embs.clone()
    for row_idx in range(event_embs.shape[0]):
        count = int(event_mask[row_idx].sum().item())
        if count <= 1:
            continue
        reversed_embs[row_idx, :count] = event_embs[row_idx, :count].flip(0)
    return reversed_embs


def unordered_matching_score(
    event_embs: torch.Tensor,
    segment_embs: torch.Tensor,
    event_mask: torch.Tensor,
    segment_mask: torch.Tensor,
) -> torch.Tensor:
    event_norm = F.normalize(event_embs, dim=-1)
    segment_norm = F.normalize(segment_embs, dim=-1)
    sims = torch.einsum("bed,msd->bmes", event_norm, segment_norm)
    valid = event_mask[:, None, :, None] & segment_mask[None, :, None, :]
    sims = sims.masked_fill(~valid, -1.0e4)
    best_per_event = sims.max(dim=-1).values
    best_per_event = best_per_event.masked_fill(~event_mask[:, None, :], 0.0)
    counts = event_mask.sum(dim=-1).clamp_min(1).to(best_per_event.dtype).unsqueeze(1)
    return best_per_event.sum(dim=-1) / counts


def evaluate_s3(bank: FeatureBank, matcher: OrderedMatchingModule, device: str) -> dict[str, float | int | bool]:
    valid_indices = np.where((bank.event_counts >= 2) & (~bank.overlap_mask))[0]
    ordered_scores: list[float] = []
    reversed_scores: list[float] = []
    margins: list[float] = []

    for idx in tqdm(valid_indices.tolist(), desc="R1-S3 reverse-order sanity"):
        event_embs = bank.event_embs[idx : idx + 1].to(device)
        event_mask = bank.event_mask[idx : idx + 1].to(device)
        reversed_embs = reverse_event_order(event_embs, event_mask)
        segment_tokens = bank.segment_tokens[idx : idx + 1].to(device)
        segment_mask = bank.segment_mask[idx : idx + 1].to(device)

        with torch.inference_mode():
            ordered = matcher.compute_score_matrix(
                event_embs=event_embs,
                segment_embs=segment_tokens,
                event_mask=event_mask,
                segment_mask=segment_mask,
            ).squeeze()
            reversed_order = matcher.compute_score_matrix(
                event_embs=reversed_embs,
                segment_embs=segment_tokens,
                event_mask=event_mask,
                segment_mask=segment_mask,
            ).squeeze()

        ordered_value = float(ordered.item())
        reversed_value = float(reversed_order.item())
        ordered_scores.append(ordered_value)
        reversed_scores.append(reversed_value)
        margins.append(ordered_value - reversed_value)

    wins = sum(margin > 0 for margin in margins)
    total = len(margins)
    return {
        "samples": total,
        "mean_forward_score": round(float(np.mean(ordered_scores)) if ordered_scores else 0.0, 6),
        "mean_reverse_score": round(float(np.mean(reversed_scores)) if reversed_scores else 0.0, 6),
        "mean_margin": round(float(np.mean(margins)) if margins else 0.0, 6),
        "forward_gt_reverse_ratio": round(float(wins / total) if total else 0.0, 6),
        "gate_pass": bool(total > 0 and (wins / total) > 0.6),
    }


def rerank_topk_rows(
    *,
    bank: FeatureBank,
    matcher: OrderedMatchingModule,
    device: str,
    top_k: int,
    lambda_values: Sequence[float],
    score_norm: str,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float | int]]]:
    global_sim = (
        F.normalize(bank.text_latents.float(), dim=-1)
        @ F.normalize(bank.motion_latents.float(), dim=-1).T
    ).cpu().numpy()

    reranked = {f"{lambda_s:.4f}": global_sim.copy() for lambda_s in lambda_values}
    num_queries = global_sim.shape[0]
    effective_topk = min(int(top_k), int(global_sim.shape[1]))

    for query_idx in tqdm(range(num_queries), desc="R1-S4 structured rerank"):
        if bank.event_counts[query_idx] < 2:
            continue

        row = global_sim[query_idx]
        topk_idx = np.argpartition(-row, effective_topk - 1)[:effective_topk]
        topk_idx = topk_idx[np.argsort(-row[topk_idx])]
        original_ranked_scores = np.sort(row[topk_idx])[::-1]

        event_embs = bank.event_embs[query_idx : query_idx + 1].to(device)
        event_mask = bank.event_mask[query_idx : query_idx + 1].to(device)
        candidate_segments = bank.segment_tokens[topk_idx].to(device)
        candidate_mask = bank.segment_mask[topk_idx].to(device)

        with torch.inference_mode():
            if bank.overlap_mask[query_idx]:
                structured = unordered_matching_score(
                    event_embs=event_embs,
                    segment_embs=candidate_segments,
                    event_mask=event_mask,
                    segment_mask=candidate_mask,
                ).squeeze(0)
            else:
                structured = matcher.compute_score_matrix(
                    event_embs=event_embs,
                    segment_embs=candidate_segments,
                    event_mask=event_mask,
                    segment_mask=candidate_mask,
                ).squeeze(0)

        global_topk = torch.from_numpy(row[topk_idx]).to(device=device, dtype=structured.dtype)
        norm_global = normalize_scores(global_topk, score_norm)
        norm_structured = normalize_scores(structured, score_norm)

        for lambda_s in lambda_values:
            lambda_key = f"{lambda_s:.4f}"
            if lambda_s == 0.0:
                continue
            fused_local = (1.0 - lambda_s) * norm_global + lambda_s * norm_structured
            rerank_order = topk_idx[torch.argsort(fused_local, descending=True).cpu().numpy()]
            reranked[lambda_key][query_idx, rerank_order] = original_ranked_scores

    ordered_subset = np.where(bank.event_counts >= 2)[0]
    diagnostics = {}
    for lambda_s in lambda_values:
        lambda_key = f"{lambda_s:.4f}"
        metrics_all = compute_subset_metrics(reranked[lambda_key], np.arange(num_queries))
        metrics_k2 = compute_subset_metrics(reranked[lambda_key], ordered_subset)
        diagnostics[lambda_key] = {
            "all": metrics_all,
            "k_ge_2": metrics_k2,
        }
    return reranked, diagnostics


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cfg = read_config(str(args.run_dir))
    ckpt_name = args.ckpt_name or cfg.ckpt
    resolved_ckpt_path = _resolve_ckpt_path(str(args.run_dir), ckpt_name)
    model = load_model_from_cfg(cfg, ckpt_name=ckpt_name, device=args.device, eval_mode=True)
    dataset = instantiate(cfg.data, split=args.split)

    matcher = OrderedMatchingModule().to(args.device)

    s0 = evaluate_s0(
        model=model,
        dataset=dataset,
        device=args.device,
        min_frames=args.window_min_frames,
        max_samples=args.s0_max_samples,
    )
    s1 = evaluate_s1(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        device=args.device,
        event_mode=args.event_encode_mode,
    )

    bank = collect_feature_bank(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        device=args.device,
        segment_count=args.segment_count,
        event_mode=args.event_encode_mode,
    )

    global_sim = (
        F.normalize(bank.text_latents.float(), dim=-1)
        @ F.normalize(bank.motion_latents.float(), dim=-1).T
    ).cpu().numpy()
    k_ge_2_indices = np.where(bank.event_counts >= 2)[0]

    s2 = {
        "samples_k_ge_2": int(k_ge_2_indices.size),
        **compute_topk_ceiling(global_sim, k_ge_2_indices, [10, 20, 50, 100]),
    }
    s2["gate_pass"] = bool(s2["ceiling@100"] > 80.0)

    s3 = evaluate_s3(bank=bank, matcher=matcher, device=args.device)

    _reranked, s4_diagnostics = rerank_topk_rows(
        bank=bank,
        matcher=matcher,
        device=args.device,
        top_k=args.rerank_top_k,
        lambda_values=args.lambda_s,
        score_norm=args.score_norm,
    )

    base_key = "0.0000"
    best_lambda = base_key
    best_gain = -float("inf")
    baseline_r1 = s4_diagnostics[base_key]["k_ge_2"]["R01"]
    for lambda_key, metrics in s4_diagnostics.items():
        gain = metrics["k_ge_2"]["R01"] - baseline_r1
        if gain > best_gain:
            best_gain = gain
            best_lambda = lambda_key

    s4 = {
        "baseline_lambda": base_key,
        "baseline_k_ge_2_r1": baseline_r1,
        "best_lambda": best_lambda,
        "best_gain_r1_pp": round(float(best_gain), 4),
        "best_metrics": s4_diagnostics[best_lambda],
        "all_lambda_metrics": s4_diagnostics,
        "gate_pass": bool(best_gain > 2.0),
    }

    results = {
        "meta": {
            "run_dir": str(args.run_dir),
            "split": args.split,
            "device": args.device,
            "ckpt": ckpt_name,
            "resolved_ckpt_path": resolved_ckpt_path,
            "event_encode_mode": args.event_encode_mode,
            "score_norm": args.score_norm,
            "segment_count": args.segment_count,
            "rerank_top_k": args.rerank_top_k,
            "lambda_s": [float(value) for value in args.lambda_s],
            "dataset_size": len(dataset),
            "k_ge_2_count": int(k_ge_2_indices.size),
            "explicit_overlap_count": int(bank.overlap_mask.sum()),
            "vanilla_tmr_motion_rep": str(cfg.data.motion_rep),
            "roadmap_phase0_expected_motion_rep": "pos66",
            "roadmap_mismatch": bool(str(cfg.data.motion_rep) != "pos66"),
        },
        "R1-S0": s0,
        "R1-S1": s1,
        "R1-S2": s2,
        "R1-S3": s3,
        "R1-S4": s4,
    }

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
