import torch

from typing import List, Dict, Optional
from torch import Tensor
from torch.utils.data import default_collate


def length_to_mask(length, device: torch.device = None) -> Tensor:
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length, device=device)

    max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    return mask


def collate_tensor_with_padding(batch: List[Tensor]) -> Tensor:
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate_x_dict(lst_x_dict: List, *, device: Optional[str] = None) -> Dict:
    x = collate_tensor_with_padding([x_dict["x"] for x_dict in lst_x_dict])
    if device is not None:
        x = x.to(device)
    length = [x_dict["length"] for x_dict in lst_x_dict]
    mask = length_to_mask(length, device=x.device)
    batch = {"x": x, "length": length, "mask": mask}
    return batch


def collate_text_motion(lst_elements: List, *, device: Optional[str] = None) -> Dict:
    one_el = lst_elements[0]
    keys = one_el.keys()

    x_dict_keys = [key for key in keys if "x_dict" in key]
    other_keys = [key for key in keys if "x_dict" not in key]

    batch = {key: default_collate([x[key] for x in lst_elements]) for key in other_keys}
    for key, val in batch.items():
        if isinstance(val, torch.Tensor) and device is not None:
            batch[key] = val.to(device)

    for key in x_dict_keys:
        batch[key] = collate_x_dict([x[key] for x in lst_elements], device=device)
    return batch


def collate_text_motion_event(
    lst_elements: List, *, device: Optional[str] = None
) -> Dict:
    one_el = lst_elements[0]
    keys = one_el.keys()

    # Keep nested event keys out of default_collate (variable lengths).
    skip_keys = {"event_text_x_dicts", "event_texts"}
    x_dict_keys = [
        key
        for key in keys
        if "x_dict" in key and key not in {"event_text_x_dict", "event_text_x_dicts"}
    ]
    other_keys = [
        key for key in keys if "x_dict" not in key and key not in skip_keys
    ]

    batch = {key: default_collate([x[key] for x in lst_elements]) for key in other_keys}
    for key, val in batch.items():
        if isinstance(val, torch.Tensor) and device is not None:
            batch[key] = val.to(device)

    for key in x_dict_keys:
        batch[key] = collate_x_dict([x[key] for x in lst_elements], device=device)

    bs = len(lst_elements)
    max_events = max(len(x["event_text_x_dicts"]) for x in lst_elements)
    event_mask = torch.zeros((bs, max_events), dtype=torch.bool)

    event_x_dicts = []
    event_sample_idx = []
    event_slot_idx = []
    for sample_idx, element in enumerate(lst_elements):
        sample_event_x = element["event_text_x_dicts"]
        sample_event_text = element["event_texts"]
        k = len(sample_event_x)
        if k > 0:
            event_mask[sample_idx, :k] = True
        for slot_idx, x_dict in enumerate(sample_event_x):
            event_x_dicts.append(x_dict)
            event_sample_idx.append(sample_idx)
            event_slot_idx.append(slot_idx)

    if event_x_dicts:
        batch["event_text_x_dict"] = collate_x_dict(event_x_dicts, device=device)
        sample_idx_tensor = torch.tensor(event_sample_idx, dtype=torch.long)
        slot_idx_tensor = torch.tensor(event_slot_idx, dtype=torch.long)
    else:
        batch["event_text_x_dict"] = {
            "x": torch.empty(0, 1, batch["text_x_dict"]["x"].shape[-1]),
            "length": [],
            "mask": torch.empty(0, 1, dtype=torch.bool),
        }
        sample_idx_tensor = torch.empty(0, dtype=torch.long)
        slot_idx_tensor = torch.empty(0, dtype=torch.long)

    if device is not None:
        event_mask = event_mask.to(device)
        sample_idx_tensor = sample_idx_tensor.to(device)
        slot_idx_tensor = slot_idx_tensor.to(device)

    batch["event_mask"] = event_mask
    batch["event_sample_idx"] = sample_idx_tensor
    batch["event_slot_idx"] = slot_idx_tensor
    batch["event_texts"] = [x["event_texts"] for x in lst_elements]
    return batch
