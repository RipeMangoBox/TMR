import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from .collate import collate_text_motion_event


MARKER_RE = re.compile(r"\b(action|condition)\s*(\d+)\s*:", flags=re.IGNORECASE)
NULL_CONDITIONS = {"none", "null", "n/a", "na", "nil"}
DEFAULT_NSIM_SPLIT = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "annotations"
    / "humanml3d"
    / "splits"
    / "nsim_test.txt"
)


def _clean_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    return str(text).strip(" \t\r\n")


def _normalize_condition(text: Optional[str]) -> Optional[str]:
    cleaned = _clean_text(text).strip(".;,:")
    if not cleaned:
        return None
    if cleaned.lower() in NULL_CONDITIONS:
        return None
    return cleaned


def parse_decomposed_events_from_caption(caption: str) -> List[str]:
    markers = list(MARKER_RE.finditer(caption))
    if not markers:
        return []

    buckets: Dict[int, Dict[str, str]] = {}
    for i, marker in enumerate(markers):
        kind = marker.group(1).lower()
        idx = int(marker.group(2))
        begin = marker.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(caption)
        value = caption[begin:end].strip(" \t\r\n.;,:")
        buckets.setdefault(idx, {})
        buckets[idx][kind] = value

    action_indices = sorted(
        idx for idx, content in buckets.items() if content.get("action", "").strip()
    )
    if not action_indices:
        return []

    k = max(action_indices)
    events = []
    for idx in range(1, k + 1):
        content = buckets.get(idx, {})
        action = content.get("action", "").strip()
        if not action:
            return []
        condition = _normalize_condition(content.get("condition"))
        if condition:
            events.append(f"{action} {condition}".strip())
        else:
            events.append(action)
    return events


def load_humanml3de_split(split_path: Path) -> Dict[str, Dict]:
    raw_data = np.load(split_path, allow_pickle=True)
    if isinstance(raw_data, np.ndarray) and raw_data.shape == ():
        raw_data = raw_data.item()

    if isinstance(raw_data, dict):
        return raw_data

    if isinstance(raw_data, np.ndarray):
        samples: Dict[str, Dict] = {}
        for idx, sample in enumerate(raw_data):
            if not isinstance(sample, dict):
                raise TypeError(
                    f"Unsupported sample type in {split_path}: {type(sample).__name__}"
                )
            keyid = (
                sample.get("keyid")
                or sample.get("name")
                or sample.get("id")
                or f"{split_path.stem}_{idx:06d}"
            )
            samples[str(keyid)] = sample
        return samples

    raise TypeError(
        f"Unsupported container type in {split_path}: {type(raw_data).__name__}"
    )


def extract_event_captions(
    text_entry: Dict, strict_event_parse: bool = False
) -> Tuple[List[str], str]:
    decomposed = text_entry.get("decomposed")
    events = []
    if isinstance(decomposed, list):
        for event_item in decomposed:
            if isinstance(event_item, dict):
                caption = _clean_text(event_item.get("caption"))
            else:
                caption = _clean_text(event_item)
            if caption:
                events.append(caption)
    if events:
        return events, "decomposed"

    caption = _clean_text(text_entry.get("caption"))
    regex_events = parse_decomposed_events_from_caption(caption)
    if regex_events:
        return regex_events, "caption_regex"

    if strict_event_parse:
        return [], "strict_parse_failed"
    if caption:
        return [caption], "caption_fallback"
    return [], "empty_caption"


class HumanML3DEventDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        text_to_sent_emb,
        text_to_token_emb,
        split: str = "train",
        preload: bool = False,
        strict_event_parse: bool = False,
        nsim_subset_size: int = 100,
        nsim_min_similarity: Optional[float] = 0.75,
        nsim_split_path: Optional[str] = None,
    ):
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.split = split
        self.is_training = split == "train"
        self.text_to_sent_emb = text_to_sent_emb
        self.text_to_token_emb = text_to_token_emb
        self.strict_event_parse = strict_event_parse
        self.nsim_subset_size = nsim_subset_size
        self.nsim_min_similarity = nsim_min_similarity
        self.nsim_split_path = (
            Path(nsim_split_path).expanduser().resolve()
            if nsim_split_path is not None
            else DEFAULT_NSIM_SPLIT
        )
        self.collate_fn = collate_text_motion_event

        load_split = "test" if split == "nsim_test" else split
        split_path = self.dataset_root / f"data_{load_split}.npy"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")

        raw_data = load_humanml3de_split(split_path)
        self.samples = []
        dropped = 0
        event_source_counter = Counter()

        for keyid, sample in raw_data.items():
            motion = sample.get("motion")
            length = sample.get("length")
            text_entries = sample.get("text", [])
            if motion is None or length is None or not isinstance(text_entries, list):
                dropped += 1
                continue

            normalized_texts = []
            for text_entry in text_entries:
                if not isinstance(text_entry, dict):
                    continue
                caption = _clean_text(text_entry.get("caption"))
                if not caption:
                    continue
                events, event_source = extract_event_captions(
                    text_entry, strict_event_parse=self.strict_event_parse
                )
                if not events:
                    continue
                normalized_texts.append(
                    {
                        "caption": caption,
                        "events": events,
                        "event_source": event_source,
                    }
                )
                event_source_counter[event_source] += 1

            if not normalized_texts:
                dropped += 1
                continue

            self.samples.append(
                {
                    "keyid": str(keyid),
                    "motion": motion,
                    "length": int(length),
                    "texts": normalized_texts,
                }
            )

        if split == "nsim_test":
            self.samples = self._resolve_nsim_subset(self.samples)

        self.keyids = [sample["keyid"] for sample in self.samples]
        self.samples_by_keyid = {sample["keyid"]: sample for sample in self.samples}
        self.nfeats = int(self.samples[0]["motion"].shape[-1]) if self.samples else 0

        print(
            "[HumanML3DEventDataset] "
            f"split={split} dataset_root={self.dataset_root} "
            f"samples={len(self.samples)} dropped={dropped} "
            f"event_sources={dict(event_source_counter)}"
        )

        if preload:
            iterator = tqdm(
                self.samples, desc=f"Preloading HumanML3D-E ({load_split}) embeddings"
            )
            for sample in iterator:
                for text_item in sample["texts"]:
                    _ = self.text_to_token_emb(text_item["caption"])
                    _ = self.text_to_sent_emb(text_item["caption"])
                    _ = self.text_to_token_emb(text_item["events"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        return self._build_event_sample(sample, training=self.is_training)

    def _select_text_item(self, sample: Dict, training: bool = False) -> Dict:
        texts = sample["texts"]
        if training and len(texts) > 1:
            return texts[int(np.random.randint(len(texts)))]
        return texts[0]

    def _build_event_sample(self, sample: Dict, training: bool = False) -> Dict:
        text_item = self._select_text_item(sample, training=training)
        caption = text_item["caption"]
        events = text_item["events"]

        motion = torch.from_numpy(sample["motion"]).to(torch.float)
        motion_x_dict = {"x": motion, "length": int(sample["length"])}

        text_x_dict = self.text_to_token_emb(caption)
        sent_emb = self.text_to_sent_emb(caption)

        event_text_x_dicts = self.text_to_token_emb(events)
        if isinstance(event_text_x_dicts, dict):
            event_text_x_dicts = [event_text_x_dicts]

        output = {
            "motion_x_dict": motion_x_dict,
            "text_x_dict": text_x_dict,
            "text": caption,
            "event_texts": events,
            "event_text_x_dicts": event_text_x_dicts,
            "keyid": sample["keyid"],
            "sent_emb": sent_emb,
        }
        return output

    def load_keyid(self, keyid: str) -> Dict:
        sample = self.samples_by_keyid[keyid]
        text_item = self._select_text_item(sample, training=False)
        caption = text_item["caption"]

        motion = torch.from_numpy(sample["motion"]).to(torch.float)
        motion_x_dict = {"x": motion, "length": int(sample["length"])}
        text_x_dict = self.text_to_token_emb(caption)
        sent_emb = self.text_to_sent_emb(caption)

        return {
            "motion_x_dict": motion_x_dict,
            "text_x_dict": text_x_dict,
            "text": caption,
            "keyid": sample["keyid"],
            "sent_emb": sent_emb,
        }

    def _resolve_nsim_subset(self, samples: List[Dict]) -> List[Dict]:
        subset = self._build_nsim_subset_from_split_file(samples)
        if subset is not None:
            return subset
        return self._build_nsim_like_subset(samples)

    def _build_nsim_subset_from_split_file(
        self, samples: List[Dict]
    ) -> Optional[List[Dict]]:
        if self.nsim_split_path is None or not self.nsim_split_path.exists():
            return None

        requested_keyids = [
            line.strip()
            for line in self.nsim_split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not requested_keyids:
            return None

        sample_map = {sample["keyid"]: sample for sample in samples}
        subset = [sample_map[keyid] for keyid in requested_keyids if keyid in sample_map]
        missing = len(requested_keyids) - len(subset)
        if not subset:
            return None

        print(
            "[HumanML3DEventDataset] nsim_test -> official split subset "
            f"size={len(subset)}/{len(requested_keyids)} "
            f"missing={missing} path={self.nsim_split_path}"
        )
        return subset

    def _build_nsim_like_subset(self, samples: List[Dict]) -> List[Dict]:
        if len(samples) <= 1:
            return samples

        subset_size = min(self.nsim_subset_size, len(samples))
        captions = [sample["texts"][0]["caption"] for sample in samples]
        sent_emb_list = self.text_to_sent_emb(captions)
        if isinstance(sent_emb_list, torch.Tensor):
            sent_emb = sent_emb_list
        else:
            sent_emb = torch.stack(sent_emb_list, dim=0)

        sent_emb = F.normalize(sent_emb.float(), dim=-1)
        sim = sent_emb @ sent_emb.T
        sim.fill_diagonal_(-1.0)
        max_neighbor = sim.max(dim=1).values

        candidate_idx = torch.arange(len(samples))
        if self.nsim_min_similarity is not None:
            mask = max_neighbor >= float(self.nsim_min_similarity)
            candidate_idx = candidate_idx[mask]
            if len(candidate_idx) == 0:
                candidate_idx = torch.arange(len(samples))

        candidate_scores = max_neighbor[candidate_idx]
        topk = torch.topk(
            candidate_scores,
            k=min(subset_size, len(candidate_scores)),
            largest=True,
            sorted=True,
        ).indices
        selected_idx = candidate_idx[topk].tolist()

        subset = [samples[idx] for idx in selected_idx]
        mean_neighbor = (
            float(max_neighbor[selected_idx].mean().item()) if selected_idx else 0.0
        )
        print(
            "[HumanML3DEventDataset] nsim_test -> nsim-like subset "
            f"size={len(subset)}/{len(samples)}, min_sim={self.nsim_min_similarity}, "
            f"mean_max_neighbor={mean_neighbor:.4f}"
        )
        return subset
