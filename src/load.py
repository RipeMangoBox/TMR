import os
from pathlib import Path
from omegaconf import DictConfig
import logging
import hydra

from src.config import read_config

logger = logging.getLogger(__name__)


def _version_index(path: Path) -> int:
    for part in path.parts:
        if not part.startswith("version_"):
            continue
        try:
            return int(part.split("_", 1)[1])
        except (IndexError, ValueError):
            return -1
    return -1


def _select_best_ckpt(matches):
    return max(
        matches,
        key=lambda path: (_version_index(path), path.stat().st_mtime_ns, str(path)),
    )


def _resolve_ckpt_path(run_dir, ckpt_name="last"):
    default_path = os.path.join(run_dir, f"logs/checkpoints/{ckpt_name}.ckpt")
    if os.path.exists(default_path):
        return default_path

    run_dir_path = Path(run_dir)
    direct_matches = sorted(run_dir_path.glob(f"**/checkpoints/{ckpt_name}.ckpt"))
    if direct_matches:
        return str(_select_best_ckpt(direct_matches))

    versioned_matches = sorted(run_dir_path.glob(f"**/checkpoints/{ckpt_name}-v*.ckpt"))
    if versioned_matches:
        return str(_select_best_ckpt(versioned_matches))

    raise FileNotFoundError(
        f"Could not find checkpoint '{ckpt_name}.ckpt' under run_dir={run_dir}"
    )


def _source_meta_path(extracted_path):
    return os.path.join(extracted_path, ".source_ckpt")


def _has_valid_extracted_weights(extracted_path):
    if not os.path.exists(extracted_path):
        return False
    return any(fname.endswith(".pt") for fname in os.listdir(extracted_path))


def _extracted_weights_match_ckpt(extracted_path, ckpt_path):
    meta_path = _source_meta_path(extracted_path)
    if not os.path.exists(meta_path):
        return False
    with open(meta_path, "r") as f:
        recorded = f.read().strip()
    return recorded == str(Path(ckpt_path).resolve())


# split the lightning checkpoint into
# seperate state_dict modules for faster loading
def extract_ckpt(run_dir, ckpt_name="last"):
    import torch

    ckpt_path = _resolve_ckpt_path(run_dir, ckpt_name)
    ckpt_path = str(Path(ckpt_path).resolve())

    extracted_path = os.path.join(run_dir, f"{ckpt_name}_weights")
    os.makedirs(extracted_path, exist_ok=True)
    for fname in os.listdir(extracted_path):
        if fname.endswith(".pt"):
            os.remove(os.path.join(extracted_path, fname))

    new_path_template = os.path.join(extracted_path, "{}.pt")
    ckpt_dict = torch.load(ckpt_path)
    state_dict = ckpt_dict["state_dict"]
    module_names = list(set([x.split(".")[0] for x in state_dict.keys()]))

    # should be ['motion_encoder', 'text_encoder', 'motion_decoder'] for example
    for module_name in module_names:
        path = new_path_template.format(module_name)
        sub_state_dict = {
            ".".join(x.split(".")[1:]): y.cpu()
            for x, y in state_dict.items()
            if x.split(".")[0] == module_name
        }
        torch.save(sub_state_dict, path)

    with open(_source_meta_path(extracted_path), "w") as f:
        f.write(ckpt_path)


def load_model(run_dir, **params):
    # Load last config
    cfg = read_config(run_dir)
    cfg.run_dir = run_dir
    return load_model_from_cfg(cfg, **params)


def load_model_from_cfg(cfg, ckpt_name="last", device="cpu", eval_mode=True):
    import src.prepare  # noqa
    import torch

    run_dir = cfg.run_dir
    model = hydra.utils.instantiate(cfg.model)

    # Loading modules one by one
    # motion_encoder / text_encoder / text_decoder
    pt_path = os.path.join(run_dir, f"{ckpt_name}_weights")
    ckpt_path = None
    try:
        ckpt_path = _resolve_ckpt_path(run_dir, ckpt_name)
    except FileNotFoundError:
        if not _has_valid_extracted_weights(pt_path):
            raise

    needs_extract = not _has_valid_extracted_weights(pt_path)
    if ckpt_path is not None and not needs_extract:
        needs_extract = not _extracted_weights_match_ckpt(pt_path, ckpt_path)

    if needs_extract:
        logger.info("The extracted model is not found. Split into submodules..")
        extract_ckpt(run_dir, ckpt_name)

    assert os.path.exists(pt_path) and len(os.listdir(pt_path)) > 0
    for fname in os.listdir(pt_path):
        module_name, ext = os.path.splitext(fname)

        if ext != ".pt":
            continue

        module = getattr(model, module_name, None)
        if module is None:
            continue

        module_path = os.path.join(pt_path, fname)
        state_dict = torch.load(module_path)
        module.load_state_dict(state_dict)
        logger.info(f"    {module_name} loaded")

    logger.info("Loading previous checkpoint done")
    model = model.to(device)
    logger.info(f"Put the model on {device}")
    if eval_mode:
        model = model.eval()
        logger.info("Put the model in eval mode")
    return model


@hydra.main(version_base=None, config_path="../configs", config_name="load_model")
def hydra_load_model(cfg: DictConfig) -> None:
    run_dir = cfg.run_dir
    ckpt_name = cfg.ckpt
    device = cfg.device
    eval_mode = cfg.eval_mode
    return load_model(run_dir, ckpt_name, device, eval_mode)


if __name__ == "__main__":
    hydra_load_model()
