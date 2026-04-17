"""src/model/motion_repr_baseline.py"""

from __future__ import annotations

import os
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .actor import PositionalEncoding


def lengths_to_mask(
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


def masked_mean_pool(hidden: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(dtype=hidden.dtype).unsqueeze(-1)
    pooled = (hidden * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return pooled / denom


class GlobalSequenceEncoder(nn.Module):
    """Linear + PE + TransformerEncoder + masked mean pool."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_size: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)

        self.input_proj = nn.Linear(self.input_dim, self.latent_dim)
        self.input_norm = nn.LayerNorm(self.latent_dim)
        self.position = PositionalEncoding(
            self.latent_dim,
            dropout=dropout,
            batch_first=True,
        )

        if num_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.latent_dim,
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

        self.output_norm = nn.LayerNorm(self.latent_dim)

    def forward(self, x: Tensor, lengths: Sequence[int] | Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B, T, C] input, got {tuple(x.shape)}")

        x = x.to(dtype=torch.float32)
        mask = lengths_to_mask(lengths, x.shape[1], device=x.device)

        hidden = self.input_proj(x)
        hidden = self.input_norm(hidden)
        hidden = self.position(hidden)
        if self.encoder is not None:
            hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        hidden = self.output_norm(hidden)
        hidden = hidden * mask.unsqueeze(-1).to(dtype=hidden.dtype)
        pooled = masked_mean_pool(hidden, mask)
        return F.normalize(pooled, dim=-1)


class DistilBertTokenCache:
    """Frozen DistilBERT token embedding cache shared across schema runs."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        device: str | torch.device = "cpu",
        max_length: int = 64,
        encode_batch_size: int = 64,
        cache_dtype: torch.dtype = torch.float16,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.max_length = int(max_length)
        self.encode_batch_size = int(encode_batch_size)
        self.cache_dtype = cache_dtype
        self._tokenizer = None
        self._model = None
        self._hidden_size: int | None = None
        self._cache: dict[str, tuple[Tensor, int]] = {}

    def to(self, device: str | torch.device) -> "DistilBertTokenCache":
        self.device = torch.device(device)
        if self._model is not None:
            self._model.to(self.device)
        return self

    def _load_components(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            from transformers import AutoModel, AutoTokenizer
            from transformers import logging as hf_logging
        except ImportError as exc:
            raise ImportError(
                "transformers is required for DistilBERT text encoding. "
                "Please install transformers before running motion_repr_ablation."
            ) from exc

        hf_logging.set_verbosity_error()
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad = False
        self._model.to(self.device)
        self._hidden_size = int(self._model.config.hidden_size)

    @property
    def hidden_size(self) -> int:
        self._load_components()
        assert self._hidden_size is not None
        return self._hidden_size

    @torch.no_grad()
    def _encode_missing(self, captions: list[str]) -> None:
        if not captions:
            return

        self._load_components()
        assert self._model is not None
        assert self._tokenizer is not None

        for start in range(0, len(captions), self.encode_batch_size):
            batch_captions = captions[start : start + self.encode_batch_size]
            encoded = self._tokenizer(
                batch_captions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            output = self._model(**encoded)
            hidden = output.last_hidden_state
            attention_mask = encoded["attention_mask"]

            for idx, caption in enumerate(batch_captions):
                length = int(attention_mask[idx].sum().item())
                token_emb = hidden[idx, :length].detach().to(
                    device="cpu",
                    dtype=self.cache_dtype,
                )
                self._cache[caption] = (token_emb.contiguous(), length)

    @torch.no_grad()
    def encode(self, captions: Sequence[str] | str) -> tuple[Tensor, Tensor]:
        if isinstance(captions, str):
            captions = [captions]
        captions = list(captions)
        self._load_components()

        if not captions:
            empty_x = torch.empty(
                (0, 0, self.hidden_size),
                device=self.device,
                dtype=torch.float32,
            )
            empty_len = torch.empty((0,), device=self.device, dtype=torch.long)
            return empty_x, empty_len

        uncached: list[str] = []
        seen: set[str] = set()
        for caption in captions:
            if caption in self._cache or caption in seen:
                continue
            seen.add(caption)
            uncached.append(caption)
        self._encode_missing(uncached)

        lengths = torch.as_tensor(
            [self._cache[caption][1] for caption in captions],
            device=self.device,
            dtype=torch.long,
        )
        max_len = int(lengths.max().item())
        batch = torch.zeros(
            (len(captions), max_len, self.hidden_size),
            device=self.device,
            dtype=torch.float32,
        )
        for idx, caption in enumerate(captions):
            token_emb, length = self._cache[caption]
            batch[idx, :length] = token_emb.to(device=self.device, dtype=torch.float32)
        return batch, lengths


class MotionReprBaseline(nn.Module):
    """
    Global contrastive motion-text retrieval baseline for representation ablation.
    """

    def __init__(
        self,
        motion_input_dim: int,
        text_input_dim: int = 768,
        latent_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_size: int = 512,
        dropout: float = 0.1,
        temperature: float = 0.07,
        text_model_name: str = "distilbert-base-uncased",
        max_text_length: int = 64,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")

        self.motion_encoder = GlobalSequenceEncoder(
            input_dim=motion_input_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_size=ff_size,
            dropout=dropout,
        )
        self.text_encoder = GlobalSequenceEncoder(
            input_dim=text_input_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_size=ff_size,
            dropout=dropout,
        )
        self.register_buffer(
            "temperature",
            torch.tensor(float(temperature), dtype=torch.float32),
            persistent=False,
        )
        self.text_model_name = text_model_name
        self.max_text_length = int(max_text_length)
        self.text_embedder: DistilBertTokenCache | None = None

    def set_text_embedder(self, text_embedder: DistilBertTokenCache) -> None:
        self.text_embedder = text_embedder

    def _resolve_text_inputs(
        self,
        text_emb: Tensor | None,
        text_length: Sequence[int] | Tensor | None,
        captions: Sequence[str] | None,
        *,
        device: torch.device,
    ) -> tuple[Tensor, Sequence[int] | Tensor]:
        if text_emb is not None and text_length is not None:
            return text_emb.to(device=device, dtype=torch.float32), text_length
        if captions is None:
            raise ValueError(
                "Provide either (text_emb, text_length) or captions for text encoding."
            )
        if self.text_embedder is None:
            self.text_embedder = DistilBertTokenCache(
                model_name=self.text_model_name,
                device=device,
                max_length=self.max_text_length,
            )
        else:
            self.text_embedder.to(device)
        return self.text_embedder.encode(captions)

    def encode_motion(
        self,
        motion: Tensor,
        motion_length: Sequence[int] | Tensor,
    ) -> Tensor:
        return self.motion_encoder(motion, motion_length)

    def encode_text(
        self,
        text_emb: Tensor | None = None,
        text_length: Sequence[int] | Tensor | None = None,
        captions: Sequence[str] | None = None,
        *,
        device: torch.device | None = None,
    ) -> Tensor:
        target_device = device or (
            text_emb.device if text_emb is not None else next(self.parameters()).device
        )
        text_emb, text_length = self._resolve_text_inputs(
            text_emb=text_emb,
            text_length=text_length,
            captions=captions,
            device=target_device,
        )
        return self.text_encoder(text_emb, text_length)

    def compute_similarity(
        self,
        motion_emb: Tensor,
        text_emb: Tensor,
    ) -> Tensor:
        return text_emb @ motion_emb.transpose(0, 1)

    def compute_loss(self, similarity: Tensor) -> Tensor:
        logits = similarity / self.temperature.clamp_min(1.0e-6)
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_t2m = F.cross_entropy(logits, labels)
        loss_m2t = F.cross_entropy(logits.transpose(0, 1), labels)
        return 0.5 * (loss_t2m + loss_m2t)

    def forward(
        self,
        motion: Tensor,
        motion_length: Sequence[int] | Tensor,
        text_emb: Tensor | None = None,
        text_length: Sequence[int] | Tensor | None = None,
        captions: Sequence[str] | None = None,
    ) -> dict[str, Tensor]:
        motion_emb = self.encode_motion(motion, motion_length)
        text_emb = self.encode_text(
            text_emb=text_emb,
            text_length=text_length,
            captions=captions,
            device=motion.device,
        )
        similarity = self.compute_similarity(motion_emb=motion_emb, text_emb=text_emb)
        loss = self.compute_loss(similarity)
        return {
            "loss": loss,
            "similarity": similarity,
            "motion_emb": motion_emb,
            "text_emb": text_emb,
            "temperature": self.temperature.detach(),
        }
