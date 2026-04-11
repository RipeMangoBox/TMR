from pathlib import Path
from typing import Dict, Optional

import torch
from torch import Tensor

from .tmr_d1 import TMRD1FrozenMinimalHead


class TMRD2aLastTwoMotionBlocks(TMRD1FrozenMinimalHead):
    def __init__(
        self,
        *args,
        unfreeze_motion_last_n_blocks: int = 2,
        warm_start_event_head: bool = True,
        **kwargs,
    ) -> None:
        self.unfreeze_motion_last_n_blocks = unfreeze_motion_last_n_blocks
        self.warm_start_event_head = warm_start_event_head
        super().__init__(*args, **kwargs)
        self.motion_encoder.seqTransEncoder.enable_nested_tensor = False
        self.motion_encoder.seqTransEncoder.use_nested_tensor = False

    def _load_warm_start(
        self, warm_start_weights_dir: Optional[str], warm_start_ckpt: Optional[str]
    ) -> None:
        if warm_start_weights_dir:
            weights_dir = Path(warm_start_weights_dir)
            weight_paths = {
                "motion_encoder": weights_dir / "motion_encoder.pt",
                "text_encoder": weights_dir / "text_encoder.pt",
                "motion_decoder": weights_dir / "motion_decoder.pt",
            }
            if self.warm_start_event_head:
                weight_paths.update(
                    {
                        "event_proj_e": weights_dir / "event_proj_e.pt",
                        "event_proj_t": weights_dir / "event_proj_t.pt",
                    }
                )

            for module_name, module_path in weight_paths.items():
                if module_path.exists():
                    getattr(self, module_name).load_state_dict(
                        torch.load(module_path, map_location="cpu"), strict=True
                    )

        if warm_start_ckpt:
            ckpt_state = torch.load(warm_start_ckpt, map_location="cpu")
            state_dict = ckpt_state.get("state_dict", ckpt_state)
            prefixes = ["motion_encoder", "text_encoder", "motion_decoder"]
            if self.warm_start_event_head:
                prefixes.extend(["event_proj_e", "event_proj_t"])
            for prefix in prefixes:
                self._load_module_from_state(getattr(self, prefix), state_dict, prefix)

    def _apply_freeze(self) -> None:
        for param in self.motion_encoder.parameters():
            param.requires_grad = False

        encoder_layers = list(self.motion_encoder.seqTransEncoder.layers)
        if self.unfreeze_motion_last_n_blocks < 0:
            raise ValueError("unfreeze_motion_last_n_blocks must be >= 0")
        if self.unfreeze_motion_last_n_blocks > len(encoder_layers):
            raise ValueError(
                "Requested more motion encoder blocks than available: "
                f"{self.unfreeze_motion_last_n_blocks} > {len(encoder_layers)}"
            )

        if self.unfreeze_motion_last_n_blocks > 0:
            for layer in encoder_layers[-self.unfreeze_motion_last_n_blocks :]:
                for param in layer.parameters():
                    param.requires_grad = True

        if self.freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        if self.freeze_motion_decoder:
            for param in self.motion_decoder.parameters():
                param.requires_grad = False

        self._enforce_frozen_eval_mode()

    def _enforce_frozen_eval_mode(self) -> None:
        if self.freeze_text_encoder:
            self.text_encoder.eval()
        if self.freeze_motion_decoder:
            self.motion_decoder.eval()

        # Keep frozen motion blocks deterministic while allowing the last
        # trainable blocks to remain in train mode during D2a fine-tuning.
        self.motion_encoder.train(self.training)
        self.motion_encoder.projection.eval()
        self.motion_encoder.sequence_pos_encoding.eval()

        encoder_layers = list(self.motion_encoder.seqTransEncoder.layers)
        split_idx = len(encoder_layers) - self.unfreeze_motion_last_n_blocks
        for idx, layer in enumerate(encoder_layers):
            layer.train(self.training and idx >= split_idx)

    def compute_loss(self, batch: Dict, return_all=False):
        text_x_dict = batch["text_x_dict"]
        motion_x_dict = batch["motion_x_dict"]
        event_text_x_dict = batch["event_text_x_dict"]
        sent_emb = batch["sent_emb"]
        sid = batch["event_sample_idx"]
        event_mask = batch["event_mask"]

        motion_mask = motion_x_dict["mask"]

        motion_outputs = self._encode_motion_with_temporal(motion_x_dict)
        motion_temporal = motion_outputs["motion_temporal"]
        m_latents = motion_outputs["motion_latents"]

        with torch.no_grad():
            t_latents = self.encode(
                text_x_dict, modality="text", sample_mean=True
            ).detach()
            event_latents = self.encode(
                event_text_x_dict, modality="text", sample_mean=True
            ).detach()

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
