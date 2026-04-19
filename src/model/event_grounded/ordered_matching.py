"""src/model/event_grounded/ordered_matching.py"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class OrderedMatchingModule(nn.Module):
    """Order-preserving event-to-segment matching via dynamic programming."""

    def __init__(
        self,
        *,
        eps: float = 1.0e-6,
        invalid_score: float = -1.0e4,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.invalid_score = invalid_score

    def _pairwise_similarity(self, event_embs: Tensor, segment_embs: Tensor) -> Tensor:
        event_norm = F.normalize(event_embs, dim=-1, eps=self.eps)
        segment_norm = F.normalize(segment_embs, dim=-1, eps=self.eps)
        return torch.einsum("bed,msd->bmes", event_norm, segment_norm)

    def compute_score_matrix(
        self,
        event_embs: Tensor,
        segment_embs: Tensor,
        event_mask: Tensor,
        segment_mask: Tensor | None = None,
    ) -> Tensor:
        if event_embs.ndim != 3 or segment_embs.ndim != 3:
            raise ValueError("event_embs and segment_embs must have shape [B, N, D].")

        batch_text, max_events, _ = event_embs.shape
        batch_motion, max_segments, _ = segment_embs.shape
        if segment_mask is None:
            segment_mask = torch.ones(
                (batch_motion, max_segments),
                dtype=torch.bool,
                device=segment_embs.device,
            )
        else:
            segment_mask = segment_mask.to(device=segment_embs.device, dtype=torch.bool)
        event_mask = event_mask.to(device=event_embs.device, dtype=torch.bool)

        if max_events == 0 or max_segments == 0:
            return event_embs.new_full((batch_text, batch_motion), self.invalid_score)

        sims = self._pairwise_similarity(event_embs, segment_embs)
        valid_sim = event_mask[:, None, :, None] & segment_mask[None, :, None, :]
        sims = sims.masked_fill(~valid_sim, self.invalid_score)

        dp = sims.new_full(
            (batch_text, batch_motion, max_events, max_segments),
            self.invalid_score,
        )

        base_valid = event_mask[:, None, 0].unsqueeze(-1) & segment_mask.unsqueeze(0)
        dp[:, :, 0, :] = torch.where(base_valid, sims[:, :, 0, :], dp[:, :, 0, :])

        for event_idx in range(1, max_events):
            best_prev = torch.cummax(dp[:, :, event_idx - 1, :], dim=-1).values
            best_prev = torch.roll(best_prev, shifts=1, dims=-1)
            best_prev[..., 0] = self.invalid_score

            current = sims[:, :, event_idx, :] + best_prev
            current_valid = (
                event_mask[:, None, event_idx].unsqueeze(-1)
                & segment_mask.unsqueeze(0)
            )
            fill_value = current.new_full(current.shape, self.invalid_score)
            dp[:, :, event_idx, :] = torch.where(current_valid, current, fill_value)

        event_counts = event_mask.sum(dim=-1)
        segment_counts = segment_mask.sum(dim=-1)
        last_event_idx = (event_counts - 1).clamp_min(0)
        gather_idx = last_event_idx.view(batch_text, 1, 1, 1).expand(
            -1,
            batch_motion,
            1,
            max_segments,
        )
        final_paths = dp.gather(dim=2, index=gather_idx).squeeze(2)
        best_scores = final_paths.max(dim=-1).values

        score_matrix = best_scores / event_counts.clamp_min(1).to(best_scores.dtype).unsqueeze(1)
        feasible = (event_counts > 0).unsqueeze(1) & (
            segment_counts.unsqueeze(0) >= event_counts.unsqueeze(1)
        )
        fill_value = score_matrix.new_full(score_matrix.shape, self.invalid_score)
        return torch.where(feasible, score_matrix, fill_value)

    def forward(
        self,
        event_embs: Tensor,
        segment_embs: Tensor,
        event_mask: Tensor,
        segment_mask: Tensor | None = None,
    ) -> Tensor:
        if event_embs.shape[0] != segment_embs.shape[0]:
            raise ValueError(
                "OrderedMatchingModule.forward expects aligned batches with equal size."
            )
        score_matrix = self.compute_score_matrix(
            event_embs=event_embs,
            segment_embs=segment_embs,
            event_mask=event_mask,
            segment_mask=segment_mask,
        )
        return torch.diagonal(score_matrix)
