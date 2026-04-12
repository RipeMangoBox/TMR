from __future__ import annotations

from contextlib import nullcontext
from typing import Dict

import torch

from .tmr_d2b import TMRD2bFullMotionEncoder


class TMRP2aFullMotionTextEncoder(TMRD2bFullMotionEncoder):
    """Phase 2 candidate: continue from D2b and unfreeze the text encoder."""

    def _grad_context(self, trainable: bool):
        return nullcontext() if trainable else torch.no_grad()

    def compute_loss(self, batch: Dict, return_all: bool = False):
        text_x_dict = batch["text_x_dict"]
        motion_x_dict = batch["motion_x_dict"]
        event_text_x_dict = batch["event_text_x_dict"]
        sent_emb = batch["sent_emb"]
        sid = batch["event_sample_idx"]
        event_mask = batch["event_mask"]

        motion_mask = motion_x_dict["mask"]
        motion_trainable = not self.freeze_motion_backbone
        text_trainable = not self.freeze_text_encoder

        with self._grad_context(motion_trainable):
            motion_outputs = self._encode_motion_with_temporal(motion_x_dict)
            motion_temporal = motion_outputs["motion_temporal"]
            m_latents = motion_outputs["motion_latents"]

        if not motion_trainable:
            motion_temporal = motion_temporal.detach()
            m_latents = m_latents.detach()

        with self._grad_context(text_trainable):
            t_latents = self.encode(
                text_x_dict, modality="text", sample_mean=True
            )
            event_latents = self.encode(
                event_text_x_dict, modality="text", sample_mean=True
            )

        if not text_trainable:
            t_latents = t_latents.detach()
            event_latents = event_latents.detach()

        loss_global = self.global_contrastive_loss_fn(t_latents, m_latents, sent_emb)
        event_losses = self._masked_event_infonce(
            event_latents=event_latents,
            motion_temporal=motion_temporal,
            motion_mask=motion_mask,
            sid=sid,
        )
        loss_evt_align = event_losses["loss_evt_align"]

        weighted_global = self.lmd.get("global", 0.0) * loss_global
        weighted_evt_align = self.lmd.get("evt_align", 1.0) * loss_evt_align
        total_loss = weighted_global + weighted_evt_align

        losses = {
            "global": loss_global,
            "evt_align": loss_evt_align,
            "evt_align_acc": event_losses["evt_align_acc"],
            "avg_events_per_caption": event_mask.to(dtype=torch.float32).sum(1).mean(),
            "loss": total_loss,
        }

        if return_all:
            return losses, t_latents, m_latents
        return losses
