import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _cfg_get(config: Any, key: str, default=None):
    if config is None:
        return default
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _load_stat_vector(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Missing stats file: {path}")
    vector = np.load(path).astype(np.float32, copy=False)
    return vector.reshape(-1)


def _load_motion_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing motion file: {path}")
    motion = np.load(path).astype(np.float32, copy=False)
    if motion.ndim == 3:
        return motion.reshape(motion.shape[0], -1)
    if motion.ndim == 2:
        return motion
    raise ValueError(f"Unsupported motion shape in {path}: {motion.shape}")


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
        motion_loader: Optional[Any] = None,
        motion_dir: Optional[str] = None,
        mean_path: Optional[str] = None,
        std_path: Optional[str] = None,
        motion_rep: Optional[str] = None,
        strict_motion_length: bool = False,
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
        motion_dir = motion_dir or _cfg_get(motion_loader, "motion_dir")
        mean_path = mean_path or _cfg_get(motion_loader, "mean_path")
        std_path = std_path or _cfg_get(motion_loader, "std_path")
        motion_rep = motion_rep or _cfg_get(motion_loader, "schema")
        configured_nfeats = _cfg_get(motion_loader, "nfeats")
        self.motion_dir = (
            Path(motion_dir).expanduser().resolve() if motion_dir is not None else None
        )
        self.mean = _load_stat_vector(
            Path(mean_path).expanduser().resolve() if mean_path is not None else None
        )
        self.std = _load_stat_vector(
            Path(std_path).expanduser().resolve() if std_path is not None else None
        )
        self.motion_rep = motion_rep or "packaged_motion"
        self.strict_motion_length = strict_motion_length
        self.uses_external_motion = self.motion_dir is not None
        self.nsim_subset_size = nsim_subset_size
        self.nsim_min_similarity = nsim_min_similarity
        if nsim_split_path is not None:
            self.nsim_split_path = Path(nsim_split_path).expanduser().resolve()
        else:
            # Prefer a dataset-local nsim split when present so derived single-root
            # datasets remain self-contained. Fall back to the repo default split.
            dataset_local_nsim = self.dataset_root / "nsim_test.txt"
            self.nsim_split_path = (
                dataset_local_nsim.resolve()
                if dataset_local_nsim.exists()
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
            if length is None or not isinstance(text_entries, list):
                dropped += 1
                continue
            if not self.uses_external_motion and motion is None:
                dropped += 1
                continue
            if self.uses_external_motion:
                motion_path = self.motion_dir / f"{keyid}.npy"
                if not motion_path.exists():
                    raise FileNotFoundError(
                        f"Missing external motion for keyid={keyid}: {motion_path}"
                    )
                motion_payload = None
            else:
                motion_payload = np.asarray(motion, dtype=np.float32)
                if motion_payload.ndim == 3:
                    motion_payload = motion_payload.reshape(motion_payload.shape[0], -1)
                elif motion_payload.ndim != 2:
                    raise ValueError(
                        f"Unsupported packaged motion shape for keyid={keyid}: "
                        f"{motion_payload.shape}"
                    )
                length = int(motion_payload.shape[0])

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
                    "motion": motion_payload,
                    "length": int(length),
                    "texts": normalized_texts,
                }
            )

        if split == "nsim_test":
            self.samples = self._resolve_nsim_subset(self.samples)

        self.keyids = [sample["keyid"] for sample in self.samples]
        self.samples_by_keyid = {sample["keyid"]: sample for sample in self.samples}
        self.nfeats = 0
        if self.samples:
            inferred_nfeats = self._load_motion_array_by_keyid(self.samples[0]["keyid"]).shape[-1]
            if configured_nfeats is not None and int(configured_nfeats) != inferred_nfeats:
                raise ValueError(
                    "Configured nfeats does not match loaded motion representation: "
                    f"configured={configured_nfeats}, inferred={inferred_nfeats}, "
                    f"motion_rep={self.motion_rep}"
                )
            self.nfeats = inferred_nfeats

        print(
            "[HumanML3DEventDataset] "
            f"split={split} dataset_root={self.dataset_root} "
            f"samples={len(self.samples)} dropped={dropped} "
            f"motion_rep={self.motion_rep} nfeats={self.nfeats} "
            f"external_motion={self.uses_external_motion} "
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

        motion_x_dict = self._build_motion_x_dict(sample)

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

        motion_x_dict = self._build_motion_x_dict(sample)
        text_x_dict = self.text_to_token_emb(caption)
        sent_emb = self.text_to_sent_emb(caption)

        return {
            "motion_x_dict": motion_x_dict,
            "text_x_dict": text_x_dict,
            "text": caption,
            "keyid": sample["keyid"],
            "sent_emb": sent_emb,
        }

    def _normalize_motion(self, motion: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return motion.astype(np.float32, copy=False)
        if self.mean.shape[0] != motion.shape[-1] or self.std.shape[0] != motion.shape[-1]:
            raise ValueError(
                "Stats dimension mismatch for motion representation: "
                f"motion={motion.shape[-1]}, mean={self.mean.shape}, std={self.std.shape}, "
                f"motion_rep={self.motion_rep}"
            )
        normalized = (motion - self.mean[np.newaxis, :]) / np.clip(
            self.std[np.newaxis, :], a_min=1.0e-12, a_max=None
        )
        return normalized.astype(np.float32, copy=False)

    def _load_motion_array_by_keyid(self, keyid: str) -> np.ndarray:
        if self.uses_external_motion:
            motion = _load_motion_array(self.motion_dir / f"{keyid}.npy")
            return self._normalize_motion(motion)
        sample = self.samples_by_keyid[keyid]
        return np.asarray(sample["motion"], dtype=np.float32)

    def _build_motion_x_dict(self, sample: Dict) -> Dict[str, torch.Tensor | int]:
        motion = self._load_motion_array_by_keyid(sample["keyid"])
        motion_length = int(motion.shape[0])
        if self.strict_motion_length and motion_length != int(sample["length"]):
            raise ValueError(
                "Motion length mismatch for event sample: "
                f"keyid={sample['keyid']}, split_length={sample['length']}, "
                f"loaded_length={motion_length}, motion_rep={self.motion_rep}"
            )
        motion_tensor = torch.from_numpy(motion).to(torch.float)
        return {"x": motion_tensor, "length": motion_length}

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
