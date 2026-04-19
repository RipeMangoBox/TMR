"""src/model/event_grounded/retriever.py"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .event_text_encoder import EventTextEncoder
from .loss import EventGroundedContrastiveLoss
from .ordered_matching import OrderedMatchingModule
from .temporal_segment_encoder import TemporalSegmentEncoder


class EventGroundedRetriever(nn.Module):
    """Temporal-first event-grounded motion-text retriever."""

    def __init__(
        self,
        *,
        motion_input_dim: int = 263,
        text_input_dim: int = 768,
        latent_dim: int = 256,
        max_events: int = 6,
        max_segments: int | None = None,
        text_ff_size: int = 512,
        text_num_layers: int = 2,
        text_num_heads: int = 4,
        motion_ff_size: int = 512,
        motion_num_layers: int = 2,
        motion_num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        text_pooler: str = "attention",
        segment_pooler: str = "attention",
        segmentation_strategy: str = "auto",
        temperature: float = 0.07,
        learnable_temperature: bool = False,
        enable_k1_global_fallback: bool = True,
        k1_global_weight: float = 1.0,
        matching_eps: float = 1.0e-6,
        matching_invalid_score: float = -1.0e4,
        event_text_encoder: nn.Module | None = None,
        temporal_segment_encoder: nn.Module | None = None,
        ordered_matching: OrderedMatchingModule | None = None,
        contrastive_loss: EventGroundedContrastiveLoss | None = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.max_events = max_events
        self.max_segments = max_segments or max_events

        self.event_text_encoder = event_text_encoder or EventTextEncoder(
            input_dim=text_input_dim,
            latent_dim=latent_dim,
            ff_size=text_ff_size,
            num_layers=text_num_layers,
            num_heads=text_num_heads,
            dropout=dropout,
            activation=activation,
            pooler=text_pooler,
        )
        self.temporal_segment_encoder = temporal_segment_encoder or TemporalSegmentEncoder(
            input_dim=motion_input_dim,
            latent_dim=latent_dim,
            max_segments=self.max_segments,
            ff_size=motion_ff_size,
            num_layers=motion_num_layers,
            num_heads=motion_num_heads,
            dropout=dropout,
            activation=activation,
            pooler=segment_pooler,
            segmentation_strategy=segmentation_strategy,
        )
        self.ordered_matching = ordered_matching or OrderedMatchingModule(
            eps=matching_eps,
            invalid_score=matching_invalid_score,
        )
        self.contrastive_loss = contrastive_loss or EventGroundedContrastiveLoss(
            matching_module=self.ordered_matching,
            temperature=temperature,
            learnable_temperature=learnable_temperature,
            enable_k1_global_fallback=enable_k1_global_fallback,
            k1_global_weight=k1_global_weight,
        )

    @staticmethod
    def _get_event_text_x_dict(batch: Dict[str, Any]) -> Dict[str, Any]:
        event_x_dict = batch.get("event_text_x_dict")
        if isinstance(event_x_dict, dict):
            return event_x_dict

        event_x_dict = batch.get("event_text_x_dicts")
        if isinstance(event_x_dict, dict):
            return event_x_dict
        raise KeyError("Batch must contain collated event_text_x_dict.")

    @staticmethod
    def _to_index_tensor(
        value: Tensor | Sequence[int] | None,
        *,
        device: torch.device,
    ) -> Tensor | None:
        if value is None:
            return None
        if isinstance(value, Tensor):
            return value.to(device=device, dtype=torch.long)
        return torch.as_tensor(value, device=device, dtype=torch.long)

    @staticmethod
    def _resolve_event_frame_spans(batch: Dict[str, Any]) -> Tensor | None:
        for key in ("event_frame_spans", "event_spans"):
            value = batch.get(key)
            if isinstance(value, Tensor):
                return value

        start = batch.get("event_f_tag")
        end = batch.get("event_to_tag")
        if isinstance(start, Tensor) and isinstance(end, Tensor):
            return torch.stack((start, end), dim=-1)
        return None

    def encode_events(
        self,
        event_text_x_dict: Dict[str, Any],
        event_mask: Tensor,
        event_sample_idx: Tensor | Sequence[int] | None = None,
        event_slot_idx: Tensor | Sequence[int] | None = None,
    ) -> Tensor:
        flat_event_embs = self.event_text_encoder(event_text_x_dict)
        event_mask = event_mask.to(device=flat_event_embs.device, dtype=torch.bool)
        batch_size, max_events = event_mask.shape
        event_embs = flat_event_embs.new_zeros((batch_size, max_events, self.latent_dim))

        if flat_event_embs.shape[0] == 0:
            return event_embs

        sample_idx = self._to_index_tensor(event_sample_idx, device=flat_event_embs.device)
        slot_idx = self._to_index_tensor(event_slot_idx, device=flat_event_embs.device)
        if sample_idx is None or slot_idx is None:
            raise ValueError("event_sample_idx and event_slot_idx are required.")

        event_embs[sample_idx, slot_idx] = flat_event_embs
        event_embs = event_embs * event_mask.unsqueeze(-1).to(dtype=event_embs.dtype)
        return event_embs

    def encode_motion_with_metadata(
        self,
        motion_x_dict: Dict[str, Any],
        *,
        event_mask: Tensor | None = None,
        event_frame_spans: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        return self.temporal_segment_encoder(
            motion_x_dict,
            event_frame_spans=event_frame_spans,
            event_mask=event_mask,
        )

    def encode_motion(
        self,
        motion_x_dict: Dict[str, Any],
        *,
        event_mask: Tensor | None = None,
        event_frame_spans: Tensor | None = None,
    ) -> Tensor:
        outputs = self.encode_motion_with_metadata(
            motion_x_dict,
            event_mask=event_mask,
            event_frame_spans=event_frame_spans,
        )
        return outputs["segment_embs"]

    def compute_score(
        self,
        event_embs: Tensor,
        segment_embs: Tensor,
        event_mask: Tensor,
        segment_mask: Tensor | None = None,
    ) -> Tensor:
        """Return pure order-preserving retrieval scores."""
        return self.ordered_matching.compute_score_matrix(
            event_embs=event_embs,
            segment_embs=segment_embs,
            event_mask=event_mask,
            segment_mask=segment_mask,
        )

    def compute_structured_score(
        self,
        event_embs: Tensor,
        segment_embs: Tensor,
        event_mask: Tensor,
        segment_mask: Tensor | None = None,
    ) -> Tensor:
        return self.compute_score(
            event_embs=event_embs,
            segment_embs=segment_embs,
            event_mask=event_mask,
            segment_mask=segment_mask,
        )

    def forward(
        self,
        batch: Dict[str, Any],
        *,
        return_outputs: bool = False,
    ) -> Tensor | Dict[str, Tensor]:
        event_mask = batch["event_mask"].to(dtype=torch.bool)
        event_text_x_dict = self._get_event_text_x_dict(batch)
        event_embs = self.encode_events(
            event_text_x_dict=event_text_x_dict,
            event_mask=event_mask,
            event_sample_idx=batch.get("event_sample_idx"),
            event_slot_idx=batch.get("event_slot_idx"),
        )

        event_frame_spans = self._resolve_event_frame_spans(batch)
        motion_outputs = self.encode_motion_with_metadata(
            motion_x_dict=batch["motion_x_dict"],
            event_mask=event_mask,
            event_frame_spans=event_frame_spans,
        )
        loss_outputs = self.contrastive_loss(
            event_embs=event_embs,
            segment_embs=motion_outputs["segment_embs"],
            event_mask=event_mask,
            segment_mask=motion_outputs["segment_mask"],
            return_score_matrix=True,
        )
        if not return_outputs:
            return loss_outputs["loss"]

        pair_scores = torch.diagonal(loss_outputs["score_matrix"])
        return {
            "loss": loss_outputs["loss"],
            "score_matrix": loss_outputs["score_matrix"],
            "structured_score_matrix": loss_outputs["structured_score_matrix"],
            "k1_fallback_mask": loss_outputs["k1_fallback_mask"],
            "structured_loss": loss_outputs["structured_loss"],
            "k1_global_loss": loss_outputs["k1_global_loss"],
            "k1_global_weight": loss_outputs["k1_global_weight"],
            "k1_global_count": loss_outputs["k1_global_count"],
            "k1_global_score_matrix": loss_outputs["k1_global_score_matrix"],
            "logits": loss_outputs["logits"],
            "temperature": loss_outputs["temperature"],
            "event_embs": event_embs,
            "segment_embs": motion_outputs["segment_embs"],
            "segment_mask": motion_outputs["segment_mask"],
            "segment_spans": motion_outputs["segment_spans"],
            "pair_scores": pair_scores,
        }
