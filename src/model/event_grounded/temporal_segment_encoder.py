"""src/model/event_grounded/temporal_segment_encoder.py"""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from ..actor import PositionalEncoding


HUMANML3D_JOINT_NAMES = (
    "root",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)

HUMANML3D_JOINT_GROUPS = {
    "torso": (0, 3, 6, 9, 12, 15),
    "left_arm": (13, 16, 18, 20),
    "right_arm": (14, 17, 19, 21),
    "left_leg": (1, 4, 7, 10),
    "right_leg": (2, 5, 8, 11),
}


def _lengths_to_mask(
    lengths: Sequence[int] | Tensor,
    max_len: int,
    *,
    device: torch.device,
) -> Tensor:
    if isinstance(lengths, Tensor):
        lengths_tensor = lengths.to(device=device, dtype=torch.long)
    else:
        lengths_tensor = torch.as_tensor(lengths, device=device, dtype=torch.long)

    if lengths_tensor.numel() == 0:
        return torch.zeros((0, max_len), dtype=torch.bool, device=device)

    frame_ids = torch.arange(max_len, device=device).unsqueeze(0)
    return frame_ids < lengths_tensor.unsqueeze(1)


def _resolve_lengths(motion_x_dict: Dict[str, Tensor | Sequence[int]]) -> Tensor:
    frames = motion_x_dict["x"]
    if not isinstance(frames, Tensor):
        raise TypeError("motion_x_dict['x'] must be a tensor.")

    lengths = motion_x_dict.get("length")
    if lengths is not None:
        if isinstance(lengths, Tensor):
            return lengths.to(device=frames.device, dtype=torch.long)
        return torch.as_tensor(lengths, device=frames.device, dtype=torch.long)

    mask = motion_x_dict.get("mask")
    if isinstance(mask, Tensor):
        return mask.to(device=frames.device, dtype=torch.long).sum(dim=-1)

    batch = frames.shape[0]
    return torch.full(
        (batch,),
        frames.shape[1],
        dtype=torch.long,
        device=frames.device,
    )


def _resolve_frame_mask(motion_x_dict: Dict[str, Tensor | Sequence[int]]) -> Tensor:
    frames = motion_x_dict["x"]
    if not isinstance(frames, Tensor):
        raise TypeError("motion_x_dict['x'] must be a tensor.")

    mask = motion_x_dict.get("mask")
    if isinstance(mask, Tensor):
        return mask.to(device=frames.device, dtype=torch.bool)

    lengths = _resolve_lengths(motion_x_dict)
    return _lengths_to_mask(lengths, frames.shape[1], device=frames.device)


class TemporalSegmentEncoder(nn.Module):
    """Encode motion features into padded temporal segment embeddings."""

    def __init__(
        self,
        input_dim: int = 263,
        latent_dim: int = 256,
        max_segments: int = 6,
        ff_size: int = 512,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        pooler: str = "attention",
        segmentation_strategy: str = "auto",
    ) -> None:
        super().__init__()
        if pooler not in {"attention", "mean"}:
            raise ValueError(f"Unsupported pooler: {pooler}")
        if segmentation_strategy not in {"auto", "fixed", "event_aligned"}:
            raise ValueError(
                f"Unsupported segmentation strategy: {segmentation_strategy}"
            )

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.max_segments = max_segments
        self.pooler = pooler
        self.segmentation_strategy = segmentation_strategy

        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.input_norm = nn.LayerNorm(latent_dim)
        self.position = PositionalEncoding(
            latent_dim,
            dropout=dropout,
            batch_first=True,
        )

        if num_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                activation=activation,
                batch_first=True,
            )
            self.encoder: nn.Module | None = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
        else:
            self.encoder = None

        self.output_norm = nn.LayerNorm(latent_dim)
        self.segment_attn = nn.Linear(latent_dim, 1) if pooler == "attention" else None

    def _build_fixed_segments(self, lengths: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = lengths.shape[0]
        device = lengths.device
        spans = torch.zeros(
            (batch_size, self.max_segments, 2),
            dtype=torch.long,
            device=device,
        )
        segment_mask = torch.zeros(
            (batch_size, self.max_segments),
            dtype=torch.bool,
            device=device,
        )

        if self.max_segments <= 0:
            return spans, segment_mask

        for slot in range(self.max_segments):
            starts = torch.div(lengths * slot, self.max_segments, rounding_mode="floor")
            ends = torch.div(
                lengths * (slot + 1),
                self.max_segments,
                rounding_mode="floor",
            )
            spans[:, slot, 0] = starts
            spans[:, slot, 1] = torch.minimum(ends, lengths)
            segment_mask[:, slot] = spans[:, slot, 1] > spans[:, slot, 0]
        return spans, segment_mask

    def _build_event_segments(
        self,
        lengths: Tensor,
        event_frame_spans: Tensor,
        event_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        batch_size = lengths.shape[0]
        device = lengths.device
        spans = torch.zeros(
            (batch_size, self.max_segments, 2),
            dtype=torch.long,
            device=device,
        )
        segment_mask = torch.zeros(
            (batch_size, self.max_segments),
            dtype=torch.bool,
            device=device,
        )

        usable_slots = min(self.max_segments, event_frame_spans.shape[1])
        if usable_slots == 0:
            return spans, segment_mask

        clamped = event_frame_spans[:, :usable_slots].to(device=device, dtype=torch.long)
        starts = torch.clamp(clamped[..., 0], min=0)
        ends = torch.clamp(clamped[..., 1], min=0)
        starts = torch.minimum(starts, lengths.unsqueeze(1))
        ends = torch.minimum(torch.maximum(ends, starts), lengths.unsqueeze(1))

        spans[:, :usable_slots, 0] = starts
        spans[:, :usable_slots, 1] = ends
        segment_mask[:, :usable_slots] = ends > starts
        if event_mask is not None:
            segment_mask[:, :usable_slots] &= event_mask[:, :usable_slots].to(
                device=device,
                dtype=torch.bool,
            )
        return spans, segment_mask

    def _build_segment_layout(
        self,
        lengths: Tensor,
        event_frame_spans: Tensor | None,
        event_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        fixed_spans, fixed_mask = self._build_fixed_segments(lengths)
        wants_event = self.segmentation_strategy in {"auto", "event_aligned"}
        has_event_spans = event_frame_spans is not None
        if not wants_event or not has_event_spans:
            return fixed_spans, fixed_mask

        event_spans, event_segment_mask = self._build_event_segments(
            lengths=lengths,
            event_frame_spans=event_frame_spans,
            event_mask=event_mask,
        )
        if self.segmentation_strategy == "event_aligned":
            return event_spans, event_segment_mask

        use_event = event_segment_mask.any(dim=-1)
        fixed_spans[use_event] = event_spans[use_event]
        fixed_mask[use_event] = event_segment_mask[use_event]
        return fixed_spans, fixed_mask

    def _pool_segments(
        self,
        frame_features: Tensor,
        frame_mask: Tensor,
        segment_spans: Tensor,
        segment_mask: Tensor,
    ) -> Tensor:
        batch_size, num_frames, _ = frame_features.shape
        num_segments = segment_spans.shape[1]
        frame_ids = torch.arange(num_frames, device=frame_features.device).view(1, 1, -1)
        starts = segment_spans[..., 0].unsqueeze(-1)
        ends = segment_spans[..., 1].unsqueeze(-1)
        segment_frame_mask = (frame_ids >= starts) & (frame_ids < ends)
        segment_frame_mask &= frame_mask.unsqueeze(1)
        segment_frame_mask &= segment_mask.unsqueeze(-1)

        if self.pooler == "mean":
            weights = segment_frame_mask.to(dtype=frame_features.dtype).unsqueeze(-1)
            pooled = (weights * frame_features.unsqueeze(1)).sum(dim=2)
            denom = weights.sum(dim=2).clamp_min(1.0)
            pooled = pooled / denom
        else:
            assert self.segment_attn is not None
            logits = self.segment_attn(frame_features).squeeze(-1)
            logits = logits.unsqueeze(1).expand(batch_size, num_segments, num_frames)
            logits = logits.masked_fill(~segment_frame_mask, -1.0e4)
            weights = torch.softmax(logits, dim=-1)
            weights = weights * segment_frame_mask.to(dtype=frame_features.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
            pooled = torch.einsum("bst,btd->bsd", weights, frame_features)

        pooled = pooled * segment_mask.unsqueeze(-1).to(dtype=pooled.dtype)
        return pooled

    def forward(
        self,
        motion_x_dict: Dict[str, Tensor | Sequence[int]],
        *,
        event_frame_spans: Tensor | None = None,
        event_mask: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        frames = motion_x_dict["x"]
        if not isinstance(frames, Tensor):
            raise TypeError("motion_x_dict['x'] must be a tensor.")
        if frames.ndim != 3:
            raise ValueError("motion_x_dict['x'] must have shape [B, T, C].")

        frames = frames.to(dtype=torch.float32)
        lengths = _resolve_lengths(motion_x_dict)
        frame_mask = _resolve_frame_mask(motion_x_dict)

        hidden = self.input_proj(frames)
        hidden = self.input_norm(hidden)
        hidden = self.position(hidden)
        if self.encoder is not None:
            hidden = self.encoder(hidden, src_key_padding_mask=~frame_mask)
        hidden = self.output_norm(hidden)
        hidden = hidden * frame_mask.unsqueeze(-1).to(dtype=hidden.dtype)

        segment_spans, segment_mask = self._build_segment_layout(
            lengths=lengths,
            event_frame_spans=event_frame_spans,
            event_mask=event_mask,
        )
        segment_embs = self._pool_segments(
            frame_features=hidden,
            frame_mask=frame_mask,
            segment_spans=segment_spans,
            segment_mask=segment_mask,
        )
        return {
            "segment_embs": segment_embs,
            "segment_mask": segment_mask,
            "segment_spans": segment_spans,
        }
