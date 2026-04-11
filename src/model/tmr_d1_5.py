from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from .tmr_d1 import TMRD1FrozenMinimalHead


class TMRD15FrozenMinimalHeadUniformPool(TMRD1FrozenMinimalHead):
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

        # D1.5 keeps the exact D1 contrastive objective and only swaps
        # attention pooling for uniform temporal averaging under the mask.
        pooled_motion = (
            temporal_for_events * mask_for_events.unsqueeze(-1)
        ).sum(dim=1) / mask_for_events.sum(dim=-1, keepdim=True).clamp_min(1).to(
            temporal_for_events.dtype
        )

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
