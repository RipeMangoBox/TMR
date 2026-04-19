"""scripts/smoke_test_event_grounded.py"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.event_grounded import EventGroundedRetriever


def build_fake_batch(
    *,
    batch_size: int = 5,
    max_frames: int = 32,
    max_events: int = 4,
    motion_dim: int = 263,
    text_dim: int = 768,
    token_len: int = 7,
) -> dict:
    torch.manual_seed(7)

    motion_lengths = torch.tensor([32, 27, 19, 15, 12], dtype=torch.long)
    motion_x = torch.randn(batch_size, max_frames, motion_dim)
    motion_mask = (
        torch.arange(max_frames).unsqueeze(0) < motion_lengths.unsqueeze(1)
    )
    motion_x = motion_x * motion_mask.unsqueeze(-1)

    event_counts = torch.tensor([4, 3, 2, 1, 1], dtype=torch.long)
    event_mask = torch.arange(max_events).unsqueeze(0) < event_counts.unsqueeze(1)

    event_sample_idx = []
    event_slot_idx = []
    event_lengths = []
    event_spans = torch.zeros(batch_size, max_events, 2, dtype=torch.long)

    flat_events = []
    for sample_idx in range(batch_size):
        length = int(motion_lengths[sample_idx].item())
        count = int(event_counts[sample_idx].item())
        for slot_idx in range(count):
            event_sample_idx.append(sample_idx)
            event_slot_idx.append(slot_idx)
            event_len = token_len - (slot_idx % 2)
            event_lengths.append(event_len)

            tokens = torch.randn(token_len, text_dim)
            tokens[event_len:] = 0
            flat_events.append(tokens)

            start = (length * slot_idx) // count
            end = (length * (slot_idx + 1)) // count
            event_spans[sample_idx, slot_idx, 0] = start
            event_spans[sample_idx, slot_idx, 1] = max(start + 1, end)

    event_x = torch.stack(flat_events, dim=0)
    event_mask_flat = (
        torch.arange(token_len).unsqueeze(0)
        < torch.tensor(event_lengths, dtype=torch.long).unsqueeze(1)
    )

    return {
        "motion_x_dict": {
            "x": motion_x,
            "length": motion_lengths,
            "mask": motion_mask,
        },
        "event_text_x_dict": {
            "x": event_x,
            "length": event_lengths,
            "mask": event_mask_flat,
        },
        "event_mask": event_mask,
        "event_sample_idx": torch.tensor(event_sample_idx, dtype=torch.long),
        "event_slot_idx": torch.tensor(event_slot_idx, dtype=torch.long),
        "event_frame_spans": event_spans,
    }


def main() -> None:
    batch = build_fake_batch()
    model = EventGroundedRetriever(
        motion_input_dim=263,
        text_input_dim=768,
        latent_dim=256,
        max_events=4,
        max_segments=4,
        segmentation_strategy="auto",
        temperature=0.07,
    )

    event_aligned_outputs = model(batch, return_outputs=True)
    fixed_window_batch = dict(batch)
    fixed_window_batch.pop("event_frame_spans")
    fixed_window_outputs = model(fixed_window_batch, return_outputs=True)
    structured_scores = model.compute_score(
        event_embs=event_aligned_outputs["event_embs"],
        segment_embs=event_aligned_outputs["segment_embs"],
        event_mask=batch["event_mask"],
        segment_mask=event_aligned_outputs["segment_mask"],
    )

    assert torch.isfinite(event_aligned_outputs["loss"])
    assert torch.isfinite(fixed_window_outputs["loss"])
    assert int(event_aligned_outputs["k1_fallback_mask"].sum().item()) == 2
    assert int(event_aligned_outputs["k1_global_count"].item()) == 2
    assert torch.isfinite(event_aligned_outputs["k1_global_loss"])
    assert torch.allclose(
        structured_scores,
        event_aligned_outputs["structured_score_matrix"],
    )
    assert torch.allclose(
        event_aligned_outputs["score_matrix"],
        event_aligned_outputs["structured_score_matrix"],
    )

    print("event-aligned loss:", float(event_aligned_outputs["loss"]))
    print("fixed-window loss:", float(fixed_window_outputs["loss"]))
    print("score_matrix shape:", tuple(event_aligned_outputs["score_matrix"].shape))
    print("event_embs shape:", tuple(event_aligned_outputs["event_embs"].shape))
    print("segment_embs shape:", tuple(event_aligned_outputs["segment_embs"].shape))
    print("structured_loss:", float(event_aligned_outputs["structured_loss"]))
    print("k1_global_loss:", float(event_aligned_outputs["k1_global_loss"]))
    print("k1_fallback_mask:", event_aligned_outputs["k1_fallback_mask"].int())
    print("event-aligned segment_mask:", event_aligned_outputs["segment_mask"].int())
    print("fixed-window segment_mask:", fixed_window_outputs["segment_mask"].int())
    print("event-aligned segment_spans:", event_aligned_outputs["segment_spans"])
    print("fixed-window segment_spans:", fixed_window_outputs["segment_spans"])


if __name__ == "__main__":
    main()
