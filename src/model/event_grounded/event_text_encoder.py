"""src/model/event_grounded/event_text_encoder.py"""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from ..actor import PositionalEncoding


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


def _resolve_token_mask(x_dict: Dict[str, Tensor | Sequence[int]]) -> Tensor:
    tokens = x_dict["x"]
    if not isinstance(tokens, Tensor):
        raise TypeError("event_text_x_dict['x'] must be a tensor.")

    mask = x_dict.get("mask")
    if isinstance(mask, Tensor):
        return mask.to(device=tokens.device, dtype=torch.bool)

    lengths = x_dict.get("length")
    if lengths is None:
        shape = tokens.shape[:2]
        return torch.ones(shape, dtype=torch.bool, device=tokens.device)

    return _lengths_to_mask(lengths, tokens.shape[1], device=tokens.device)


class EventTextEncoder(nn.Module):
    """Encode flattened event token embeddings into event-level embeddings."""

    def __init__(
        self,
        input_dim: int = 768,
        latent_dim: int = 256,
        ff_size: int = 512,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        pooler: str = "attention",
    ) -> None:
        super().__init__()
        if pooler not in {"attention", "mean"}:
            raise ValueError(f"Unsupported pooler: {pooler}")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.pooler = pooler

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
        self.attn_pool = nn.Linear(latent_dim, 1) if pooler == "attention" else None

    def _attention_pool(self, hidden: Tensor, mask: Tensor) -> Tensor:
        assert self.attn_pool is not None
        logits = self.attn_pool(hidden).squeeze(-1)
        logits = logits.masked_fill(~mask, -1.0e4)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * mask.to(dtype=hidden.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        return torch.einsum("nl,nld->nd", weights, hidden)

    @staticmethod
    def _mean_pool(hidden: Tensor, mask: Tensor) -> Tensor:
        weights = mask.to(dtype=hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return pooled / denom

    def forward(self, event_text_x_dict: Dict[str, Tensor | Sequence[int]]) -> Tensor:
        tokens = event_text_x_dict["x"]
        if not isinstance(tokens, Tensor):
            raise TypeError("event_text_x_dict['x'] must be a tensor.")
        if tokens.ndim != 3:
            raise ValueError(
                "event_text_x_dict['x'] must have shape [N_evt, L, C]."
            )
        if tokens.shape[0] == 0:
            return tokens.new_empty((0, self.latent_dim))

        tokens = tokens.to(dtype=torch.float32)
        mask = _resolve_token_mask(event_text_x_dict)

        hidden = self.input_proj(tokens)
        hidden = self.input_norm(hidden)
        hidden = self.position(hidden)
        if self.encoder is not None:
            hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        hidden = self.output_norm(hidden)

        if self.pooler == "attention":
            return self._attention_pool(hidden, mask)
        return self._mean_pool(hidden, mask)
