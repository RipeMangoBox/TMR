"""src/model/event_grounded/loss.py"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .ordered_matching import OrderedMatchingModule


class EventGroundedContrastiveLoss(nn.Module):
    """Structured InfoNCE with an optional K=1 global auxiliary branch."""

    def __init__(
        self,
        matching_module: OrderedMatchingModule | None = None,
        *,
        temperature: float = 0.07,
        learnable_temperature: bool = False,
        min_temperature: float = 1.0e-4,
        enable_k1_global_fallback: bool = True,
        k1_global_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.matching_module = matching_module or OrderedMatchingModule()
        self.learnable_temperature = learnable_temperature
        self.min_temperature = min_temperature
        self.enable_k1_global_fallback = enable_k1_global_fallback
        self.k1_global_weight = k1_global_weight

        if learnable_temperature:
            self.log_temperature = nn.Parameter(
                torch.tensor(math.log(temperature), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "fixed_temperature",
                torch.tensor(float(temperature), dtype=torch.float32),
                persistent=False,
            )

    @property
    def temperature(self) -> Tensor:
        if self.learnable_temperature:
            return self.log_temperature.exp().clamp_min(self.min_temperature)
        return self.fixed_temperature.clamp_min(self.min_temperature)

    @staticmethod
    def _resolve_segment_mask(segment_embs: Tensor, segment_mask: Tensor | None) -> Tensor:
        if segment_mask is None:
            shape = segment_embs.shape[:2]
            return torch.ones(shape, dtype=torch.bool, device=segment_embs.device)
        return segment_mask.to(device=segment_embs.device, dtype=torch.bool)

    @staticmethod
    def _masked_mean_pool(embs: Tensor, mask: Tensor) -> Tensor:
        weights = mask.to(dtype=embs.dtype).unsqueeze(-1)
        pooled = (embs * weights).sum(dim=1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return pooled / denom

    def compute_score_matrix(
        self,
        event_embs: Tensor,
        segment_embs: Tensor,
        event_mask: Tensor,
        segment_mask: Tensor | None = None,
        *,
        return_details: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        event_mask = event_mask.to(device=event_embs.device, dtype=torch.bool)
        resolved_segment_mask = self._resolve_segment_mask(segment_embs, segment_mask)

        structured_score_matrix = self.matching_module.compute_score_matrix(
            event_embs=event_embs,
            segment_embs=segment_embs,
            event_mask=event_mask,
            segment_mask=resolved_segment_mask,
        )
        event_counts = event_mask.sum(dim=-1)
        k1_fallback_mask = event_counts == 1

        if not return_details:
            return structured_score_matrix

        return {
            "structured_score_matrix": structured_score_matrix,
            "k1_fallback_mask": k1_fallback_mask,
            "segment_mask": resolved_segment_mask,
        }

    def forward(
        self,
        event_embs: Tensor,
        segment_embs: Tensor,
        event_mask: Tensor,
        segment_mask: Tensor | None = None,
        *,
        return_score_matrix: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        if event_embs.shape[0] != segment_embs.shape[0]:
            raise ValueError(
                "EventGroundedContrastiveLoss.forward expects paired batches with "
                "equal batch size. Use compute_score_matrix for rectangular retrieval."
            )

        score_outputs = self.compute_score_matrix(
            event_embs=event_embs,
            segment_embs=segment_embs,
            event_mask=event_mask,
            segment_mask=segment_mask,
            return_details=True,
        )
        score_matrix = score_outputs["structured_score_matrix"]
        logits = score_matrix / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_t2m = F.cross_entropy(logits, labels)
        loss_m2t = F.cross_entropy(logits.transpose(0, 1), labels)
        structured_loss = 0.5 * (loss_t2m + loss_m2t)

        k1_fallback_mask = score_outputs["k1_fallback_mask"]
        k1_global_count = int(k1_fallback_mask.sum().item())
        k1_global_loss = logits.new_zeros(())
        k1_global_score_matrix = score_matrix.new_empty((0, 0))

        if self.enable_k1_global_fallback and k1_global_count >= 2:
            resolved_segment_mask = score_outputs["segment_mask"]
            global_event_embs = self._masked_mean_pool(
                event_embs[k1_fallback_mask],
                event_mask[k1_fallback_mask],
            )
            global_segment_embs = self._masked_mean_pool(
                segment_embs[k1_fallback_mask],
                resolved_segment_mask[k1_fallback_mask],
            )
            global_event_embs = F.normalize(global_event_embs, dim=-1)
            global_segment_embs = F.normalize(global_segment_embs, dim=-1)
            k1_global_score_matrix = global_event_embs @ global_segment_embs.transpose(0, 1)
            k1_global_logits = k1_global_score_matrix / self.temperature
            k1_labels = torch.arange(k1_global_logits.shape[0], device=logits.device)
            k1_loss_t2m = F.cross_entropy(k1_global_logits, k1_labels)
            k1_loss_m2t = F.cross_entropy(k1_global_logits.transpose(0, 1), k1_labels)
            k1_global_loss = 0.5 * (k1_loss_t2m + k1_loss_m2t)

        loss = structured_loss + self.k1_global_weight * k1_global_loss

        if not return_score_matrix:
            return loss

        return {
            "loss": loss,
            "score_matrix": score_matrix,
            "structured_score_matrix": score_matrix,
            "k1_fallback_mask": k1_fallback_mask,
            "structured_loss": structured_loss,
            "k1_global_loss": k1_global_loss,
            "k1_global_weight": score_matrix.new_tensor(self.k1_global_weight),
            "k1_global_count": score_matrix.new_tensor(k1_global_count, dtype=torch.long),
            "k1_global_score_matrix": k1_global_score_matrix,
            "logits": logits,
            "temperature": self.temperature.detach(),
        }
