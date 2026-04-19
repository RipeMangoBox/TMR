import logging

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback


logger = logging.getLogger(__name__)


class ProgressLogger(Callback):
    def __init__(self, precision: int = 2):
        self.precision = precision
        self._last_logged_step = -1

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule, **kwargs):
        logger.info("Training started")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule, **kwargs):
        logger.info("Training done")

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule, **kwargs
    ):
        if trainer.sanity_checking:
            logger.info("Sanity checking ok.")

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
        **kwargs,
    ):
        log_every = int(getattr(trainer, "log_every_n_steps", 0) or 0)
        if log_every <= 0:
            return

        global_step = int(trainer.global_step)
        if global_step <= 0 or global_step == self._last_logged_step:
            return
        if global_step % log_every != 0:
            return

        self._last_logged_step = global_step
        total_batches = getattr(trainer, "num_training_batches", "?")
        losses_dict = trainer.callback_metrics
        loss_val = losses_dict.get("train_loss_step")
        evt_align_val = losses_dict.get("train_evt_align_step")
        global_val = losses_dict.get("train_global_step")

        metrics = []
        if loss_val is not None:
            metrics.append(f"loss {float(loss_val):.{self.precision}e}")
        if global_val is not None:
            metrics.append(f"global {float(global_val):.{self.precision}e}")
        if evt_align_val is not None:
            metrics.append(f"evt_align {float(evt_align_val):.{self.precision}e}")

        metrics_str = "  ".join(metrics) if metrics else "metrics pending"
        logger.info(
            f"Epoch {trainer.current_epoch} step {global_step}/{total_batches}: {metrics_str}"
        )

    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule, **kwargs
    ):
        metric_format = f"{{:.{self.precision}e}}"
        line = f"Epoch {trainer.current_epoch}"
        metrics_str = []

        losses_dict = trainer.callback_metrics

        def is_contrastive_metrics(name: str) -> bool:
            return "t2m" in name or "m2t" in name

        extra_metric_names = {"evt_align_acc", "avg_events_per_caption"}

        parsed_metrics = []
        for metric_name in losses_dict.keys():
            if metric_name.startswith("train_"):
                split = "train"
            elif metric_name.startswith("val_"):
                split = "val"
            else:
                continue

            suffix = "_epoch"
            if not metric_name.endswith(suffix):
                continue

            prefix = f"{split}_"
            name = metric_name[len(prefix) : -len(suffix)]
            if not (
                name in pl_module.lmd
                or name == "loss"
                or name in extra_metric_names
                or is_contrastive_metrics(name)
            ):
                continue
            parsed_metrics.append((metric_name, split, name))

        # Natual order for contrastive
        letters = "0123456789"
        mapping = str.maketrans(letters, letters[::-1])

        def sort_losses(item):
            _, split, name = item
            if is_contrastive_metrics(name):
                # put them at the end
                name = "a" + name.translate(mapping)
            return (name, split)

        parsed_metrics = sorted(parsed_metrics, key=sort_losses, reverse=True)
        for metric_name, split, name in parsed_metrics:
            metric = losses_dict[metric_name].item()

            if is_contrastive_metrics(name):
                if "len" in name:
                    metric = str(int(metric))
                elif "MedR" in name:
                    metric = str(int(metric * 100) / 100) + "%"
                else:
                    metric = str(int(metric * 100) / 100) + "%"
            else:
                metric = metric_format.format(metric)

            if split == "train":
                mname = name
            else:
                mname = f"v_{name}"

            metric = f"{mname} {metric}"
            metrics_str.append(metric)

        if len(metrics_str) == 0:
            return

        line = line + ": " + "  ".join(metrics_str)
        logger.info(line)
