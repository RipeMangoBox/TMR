import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from src.config import read_config, save_config
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pytorch_lightning
import shutil

logger = logging.getLogger(__name__)

import os
import shutil
import pytorch_lightning as pl


class CopyConfigCallback(pl.Callback):
    def __init__(self, config_paths):
        super().__init__()
        self.config_paths = config_paths if isinstance(config_paths, list) else [config_paths]

    def on_train_start(self, trainer, pl_module):
        log_dir = trainer.log_dir
        if log_dir is None:
            print("Warning: log_dir is None, skipping config copy.")
            return

        for config_path in self.config_paths:
            if not os.path.exists(config_path):
                print(f"Warning: Config path {config_path} not found!")
                continue

            if os.path.isfile(config_path):
                # 是文件：直接复制到 log_dir
                shutil.copy2(config_path, log_dir)
                print(f"Copied file: {config_path} → {log_dir}")
            elif os.path.isdir(config_path):
                # 是目录：复制整个目录到 log_dir 下（保留目录名）
                dir_name = os.path.basename(config_path)
                dst_dir = os.path.join(log_dir, dir_name)
                if os.path.exists(dst_dir):
                    # 可选：如果已存在，可以选择覆盖或跳过
                    shutil.rmtree(dst_dir)  # 删除已有目录（确保干净复制）
                shutil.copytree(config_path, dst_dir)
                print(f"Copied directory: {config_path} → {dst_dir}")
            else:
                print(f"Warning: {config_path} is neither a file nor a directory.")

@hydra.main(config_path="configs", config_name="train_fd", version_base="1.3")
def train(cfg: DictConfig):
    # Resuming if needed
    ckpt = None
    if cfg.resume_dir is not None:
        assert cfg.ckpt is not None
        ckpt = cfg.ckpt
        cfg = read_config(cfg.resume_dir)
        logger.info("Resuming training")
        logger.info(f"The config is loaded from: \n{cfg.resume_dir}")
    else:
        config_path = save_config(cfg)
        logger.info("Training script")
        logger.info(f"The config can be found here: \n{config_path}")

    import src.prepare  # noqa
    import pytorch_lightning as pl

    pl.seed_everything(cfg.seed)

    logger.info("Loading the dataloaders")
    train_dataset = instantiate(cfg.data, split="train")
    val_dataset = instantiate(cfg.data, split="val")

    train_dataloader = instantiate(
        cfg.dataloader,
        dataset=train_dataset,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
    )

    val_dataloader = instantiate(
        cfg.dataloader,
        dataset=val_dataset,
        collate_fn=val_dataset.collate_fn,
        shuffle=False,
    )

    logger.info("Loading the model")
    model = instantiate(cfg.model)

    _hydra_path = f"{os.path.dirname(config_path)}/.hydra"
    copy_config_callback = CopyConfigCallback([config_path, _hydra_path])

    logger.info("Training")
    trainer = instantiate(cfg.trainer)
    trainer.callbacks.append(copy_config_callback)
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt)


if __name__ == "__main__":
    train()
