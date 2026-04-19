import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .losses import InfoNCE_with_filtering
from .metrics import all_contrastive_metrics
from .temos import TEMOS


logger = logging.getLogger(__name__)


def _get_sim_matrix(x: Tensor, y: Tensor) -> Tensor:
    x_logits = F.normalize(x, dim=-1)
    y_logits = F.normalize(y, dim=-1)
    return x_logits @ y_logits.T


class TMRD1FrozenMinimalHead(TEMOS):
    def __init__(
        self,
        motion_encoder: nn.Module,
        text_encoder: nn.Module,
        motion_decoder: nn.Module,
        vae: bool,
        fact: Optional[float] = None,
        sample_mean: Optional[bool] = True,
        lmd: Dict = {"global": 0.1, "evt_align": 1.0},
        lr: float = 1e-4,
        temperature: float = 0.1,
        threshold_selfsim: float = 0.80,
        threshold_selfsim_metrics: float = 0.95,
        event_align_tau: float = 0.1,
        warm_start_weights_dir: Optional[str] = None,
        warm_start_ckpt: Optional[str] = None,
        freeze_motion_backbone: bool = True,
        freeze_text_encoder: bool = True,
        freeze_motion_decoder: bool = True,
    ) -> None:
        super().__init__(
            motion_encoder=motion_encoder,
            text_encoder=text_encoder,
            motion_decoder=motion_decoder,
            vae=vae,
            fact=fact,
            sample_mean=sample_mean,
            lmd=lmd,
            lr=lr,
        )

        latent_dim = motion_encoder.projection.out_features
        self.event_proj_e = nn.Linear(latent_dim, latent_dim)
        self.event_proj_t = nn.Linear(latent_dim, latent_dim)

        self.event_align_tau = event_align_tau
        self.global_contrastive_loss_fn = InfoNCE_with_filtering(
            temperature=temperature, threshold_selfsim=threshold_selfsim
        )
        self.threshold_selfsim_metrics = threshold_selfsim_metrics

        self.freeze_motion_backbone = freeze_motion_backbone
        self.freeze_text_encoder = freeze_text_encoder
        self.freeze_motion_decoder = freeze_motion_decoder

        self._load_warm_start(
            warm_start_weights_dir=warm_start_weights_dir, warm_start_ckpt=warm_start_ckpt
        )
        self._apply_freeze()

        self.validation_step_t_latents = []
        self.validation_step_m_latents = []
        self.validation_step_sent_emb = []

    def _load_module_state(
        self, module: nn.Module, state_dict: Dict[str, Tensor], module_name: str
    ) -> bool:
        target_state = module.state_dict()
        loadable_state = {}
        skipped_keys = []
        for key, value in state_dict.items():
            target_value = target_state.get(key)
            if target_value is None:
                continue
            if tuple(target_value.shape) != tuple(value.shape):
                skipped_keys.append(
                    f"{key}: ckpt{tuple(value.shape)} != model{tuple(target_value.shape)}"
                )
                continue
            loadable_state[key] = value

        if not loadable_state:
            if state_dict:
                logger.warning(
                    "Warm-start skipped %s because no parameter shapes matched.", module_name
                )
            return False

        missing_keys, unexpected_keys = module.load_state_dict(
            loadable_state, strict=False
        )
        skipped_count = len(skipped_keys)
        if skipped_count or missing_keys or unexpected_keys:
            logger.info(
                "Warm-start loaded %s/%s tensors into %s; skipped_shape=%s missing=%s unexpected=%s",
                len(loadable_state),
                len(target_state),
                module_name,
                skipped_count,
                len(missing_keys),
                len(unexpected_keys),
            )
            if skipped_count:
                logger.info(
                    "Warm-start shape mismatches for %s (first 8): %s",
                    module_name,
                    skipped_keys[:8],
                )
        return True

    def _load_module_from_state(self, module: nn.Module, state: Dict, prefix: str) -> bool:
        module_state = {}
        direct_prefix = f"{prefix}."
        model_prefix = f"model.{prefix}."
        for key, value in state.items():
            if key.startswith(model_prefix):
                module_state[key[len(model_prefix) :]] = value
            elif key.startswith(direct_prefix):
                module_state[key[len(direct_prefix) :]] = value
        if not module_state:
            return False
        return self._load_module_state(module, module_state, prefix)

    def _load_warm_start(
        self, warm_start_weights_dir: Optional[str], warm_start_ckpt: Optional[str]
    ) -> None:
        if warm_start_weights_dir:
            weights_dir = Path(warm_start_weights_dir)
            motion_encoder_path = weights_dir / "motion_encoder.pt"
            text_encoder_path = weights_dir / "text_encoder.pt"
            motion_decoder_path = weights_dir / "motion_decoder.pt"

            if motion_encoder_path.exists():
                self._load_module_state(
                    self.motion_encoder,
                    torch.load(motion_encoder_path, map_location="cpu"),
                    "motion_encoder",
                )
            if text_encoder_path.exists():
                self._load_module_state(
                    self.text_encoder,
                    torch.load(text_encoder_path, map_location="cpu"),
                    "text_encoder",
                )
            if motion_decoder_path.exists():
                self._load_module_state(
                    self.motion_decoder,
                    torch.load(motion_decoder_path, map_location="cpu"),
                    "motion_decoder",
                )

        if warm_start_ckpt:
            ckpt_state = torch.load(warm_start_ckpt, map_location="cpu")
            state_dict = ckpt_state.get("state_dict", ckpt_state)
            self._load_module_from_state(self.motion_encoder, state_dict, "motion_encoder")
            self._load_module_from_state(self.text_encoder, state_dict, "text_encoder")
            self._load_module_from_state(self.motion_decoder, state_dict, "motion_decoder")

    def _apply_freeze(self) -> None:
        if self.freeze_motion_backbone:
            for param in self.motion_encoder.parameters():
                param.requires_grad = False
        if self.freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        if self.freeze_motion_decoder:
            for param in self.motion_decoder.parameters():
                param.requires_grad = False

        self._enforce_frozen_eval_mode()

    def _enforce_frozen_eval_mode(self) -> None:
        if self.freeze_motion_backbone:
            self.motion_encoder.eval()
        if self.freeze_text_encoder:
            self.text_encoder.eval()
        if self.freeze_motion_decoder:
            self.motion_decoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._enforce_frozen_eval_mode()
        return self

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters left in D1 model.")
        return {"optimizer": torch.optim.AdamW(params=trainable_params, lr=self.lr)}

    def _encode_motion_with_temporal(self, motion_x_dict: Dict) -> Dict[str, Tensor]:
        encoded_tokens, motion_temporal = self.motion_encoder(
            motion_x_dict, return_temporal=True
        )
        if self.vae:
            motion_mu, _motion_logvar = encoded_tokens.unbind(1)
            motion_latents = motion_mu
        else:
            (motion_latents,) = encoded_tokens.unbind(1)
        return {"motion_temporal": motion_temporal, "motion_latents": motion_latents}

    def _masked_event_infonce(
        self, event_latents: Tensor, motion_temporal: Tensor, motion_mask: Tensor, sid: Tensor
    ) -> Dict[str, Tensor]:
        if event_latents.numel() == 0:
            zero = motion_temporal.new_zeros(())
            return {"loss_evt_align": zero, "evt_align_acc": zero}

        proj_event = self.event_proj_e(event_latents)
        proj_temporal = self.event_proj_t(motion_temporal)

        temporal_for_events = proj_temporal[sid]  # [N, T, D]
        mask_for_events = motion_mask[sid]  # [N, T]

        scores = torch.einsum("nd,ntd->nt", proj_event, temporal_for_events)
        scores = scores.masked_fill(~mask_for_events, -1e4)
        attn = torch.softmax(scores, dim=-1)

        # Keep pooling and contrastive matching in the same projected space.
        pooled_motion = torch.einsum("nt,ntd->nd", attn, temporal_for_events)

        z = F.normalize(pooled_motion, dim=-1)
        e = F.normalize(proj_event, dim=-1)
        logits = z @ e.T / self.event_align_tau

        n_events = logits.shape[0]
        diag = torch.eye(n_events, dtype=torch.bool, device=logits.device)
        allow = (sid[:, None] != sid[None, :]) | diag
        logits = logits.masked_fill(~allow, -1e4)

        targets = torch.arange(n_events, device=logits.device)
        loss_evt_align = F.cross_entropy(logits, targets)
        evt_align_acc = (logits.argmax(dim=-1) == targets).to(dtype=logits.dtype).mean()
        return {"loss_evt_align": loss_evt_align, "evt_align_acc": evt_align_acc}

    def compute_loss(self, batch: Dict, return_all=False):
        text_x_dict = batch["text_x_dict"]
        motion_x_dict = batch["motion_x_dict"]
        event_text_x_dict = batch["event_text_x_dict"]
        sent_emb = batch["sent_emb"]
        sid = batch["event_sample_idx"]
        event_mask = batch["event_mask"]

        motion_mask = motion_x_dict["mask"]

        with torch.no_grad():
            motion_outputs = self._encode_motion_with_temporal(motion_x_dict)
            motion_temporal = motion_outputs["motion_temporal"].detach()
            m_latents = motion_outputs["motion_latents"].detach()
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

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        bs = len(batch["motion_x_dict"]["x"])
        losses = self.compute_loss(batch)
        for loss_name in sorted(losses):
            self.log(
                f"train_{loss_name}",
                losses[loss_name],
                on_epoch=True,
                on_step=True,
                batch_size=bs,
            )
        return losses["loss"]

    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
        bs = len(batch["motion_x_dict"]["x"])
        losses, t_latents, m_latents = self.compute_loss(batch, return_all=True)

        self.validation_step_t_latents.append(t_latents)
        self.validation_step_m_latents.append(m_latents)
        self.validation_step_sent_emb.append(batch["sent_emb"])

        for loss_name in sorted(losses):
            self.log(
                f"val_{loss_name}",
                losses[loss_name],
                on_epoch=True,
                on_step=True,
                batch_size=bs,
            )
        return losses["loss"]

    def on_validation_epoch_end(self):
        if not self.validation_step_t_latents:
            return

        t_latents = torch.cat(self.validation_step_t_latents)
        m_latents = torch.cat(self.validation_step_m_latents)
        sent_emb = torch.cat(self.validation_step_sent_emb)

        sim_matrix = _get_sim_matrix(t_latents, m_latents).cpu().numpy()
        contrastive_metrics = all_contrastive_metrics(
            sim_matrix,
            emb=sent_emb.cpu().numpy(),
            threshold=self.threshold_selfsim_metrics,
        )

        for metric_name, metric_value in sorted(contrastive_metrics.items()):
            self.log(
                f"val_{metric_name}_epoch",
                metric_value,
                on_epoch=True,
                on_step=False,
            )

        self.validation_step_t_latents.clear()
        self.validation_step_m_latents.clear()
        self.validation_step_sent_emb.clear()
