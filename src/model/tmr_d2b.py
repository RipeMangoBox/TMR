from .tmr_d2a import TMRD2aLastTwoMotionBlocks


class TMRD2bFullMotionEncoder(TMRD2aLastTwoMotionBlocks):
    def __init__(self, *args, warm_start_event_head: bool = True, **kwargs) -> None:
        super().__init__(
            *args,
            unfreeze_motion_last_n_blocks=0,
            warm_start_event_head=warm_start_event_head,
            **kwargs,
        )

    def _apply_freeze(self) -> None:
        for param in self.motion_encoder.parameters():
            param.requires_grad = not self.freeze_motion_backbone

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

        self.motion_encoder.train(self.training and not self.freeze_motion_backbone)
        if self.freeze_motion_backbone:
            self.motion_encoder.eval()
