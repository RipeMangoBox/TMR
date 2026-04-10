---
name: tamr-research
description: Provides formal context for the TAMR (Temporal-Aware Motion-Text Retrieval) research project. Use when the user asks about TAMR design, TMR codebase modifications, EventT2M integration, HumanML3D-E dataset, temporal-aware contrastive learning, motion-text retrieval experiments, or related tasks.
---

# TAMR Research Context

TAMR = Temporal-Aware Motion-text Retrieval. The goal is to extend the TMR retrieval backbone with temporal awareness for event ordering, grounding, and temporal constraints.

## 1. Codebase Layout

```
TMR/                              # Main workspace (this repo, fork of Mathux/TMR)
├── train.py                      # TMR training entry (Hydra + PyTorch Lightning)
├── retrieval.py                  # Retrieval evaluation entry
├── extract.py / encode_*.py      # Checkpoint extraction & dataset encoding
├── src/
│   ├── model/
│   │   ├── temos.py              # Base VAE: motion_encoder + text_encoder + motion_decoder
│   │   ├── tmr.py                # TMR = TEMOS + InfoNCE contrastive loss
│   │   ├── actor.py              # ACTORStyleEncoder/Decoder
│   │   ├── losses.py             # Contrastive loss
│   │   └── text_encoder.py       # Text embedding pipeline
│   ├── data/
│   │   ├── humanml3d.py          # HumanML3D dataset class
│   │   └── text.py               # Text loading & tokenization
│   └── config.py                 # Hydra structured configs
├── configs/                      # Hydra YAML configs
├── datasets/                     # Annotation splits & text files
├── stats/                        # mean.pt / std.pt normalization
├── EventT2M-codes-main/          # Event-T2M reference implementation
└── Codex/                        # Research notes & paper analyses
```

## 2. TMR Architecture

TMR is a VAE + contrastive dual-tower retrieval model:

- **Motion encoder**: `ACTORStyleEncoder` maps 263-dim Guo features to a 256-dim latent.
- **Text encoder**: `ACTORStyleEncoder` maps DistilBERT token features to a 256-dim latent.
- **Motion decoder**: `ACTORStyleDecoder` reconstructs motion from the latent.
- **Contrastive objective**: `InfoNCE_with_filtering` computes symmetric retrieval loss with false-negative filtering.
- **Retrieval**: `get_sim_matrix()` uses cosine similarity of normalized latents.

Key property: baseline TMR does not explicitly model temporal ordering.

## 3. EventT2M Reference

EventT2M decomposes text into sub-events for motion generation and serves as a temporal modeling reference.

```
EventT2M-codes-main/
├── src/
│   ├── train.py / eval.py
│   ├── models/
│   │   ├── event_final.py
│   │   └── nets/
│   │       ├── event_final.py
│   │       └── text_encoder.py
│   ├── data/
│   │   ├── hml3d_event.py
│   │   └── humanml/dataset.py
│   └── tools/
│       ├── data_decompose.py
│       └── data_preprocess_decomposed.py
├── TMR_model_wrapper.py
└── configs/
```

Key architecture: EventT2M uses `MiniConformer` blocks with cross-attention to decomposed event embeddings, enabling explicit event-level conditioning.

## 4. HumanML3D-E Dataset

Event-annotated extension of HumanML3D. Expected structure:

```
dataset/HumanML3D/
├── {train,val,test}.txt
├── Mean.npy / Std.npy
├── new_joint_vecs/*.npy
├── texts/*.txt
├── texts_decomposed/*.txt
├── data_{train,val,test}.npy
└── data_test_condition{2,3,4}.npy
```

`texts_decomposed/` stores decomposed event captions aligned to original captions.

## 5. TAMR Design

### Module A: Event-Aware Text Encoding
- Decompose text into an ordered event list with inter-event relations.
- Encode each event independently and fuse relation-aware representations.

### Module B: Temporal-Aware Contrastive Learning
- Combine global InfoNCE with event-level alignment loss and temporal hard negative loss.
- Temporal hard negatives include ordering shuffle, parallel-to-sequential conversion, negation, causal reversal, duration modification, and sync-to-async conversion.
- Use normalized timeline tokens for temporal alignment.

### Module C: Unified Retrieval-Localization Head
- Retrieval uses global cosine similarity for ranking.
- Localization uses event-level alignment for temporal boundary prediction.

### Loss
```
L = α·L_global_infonce + β·L_event_align + γ·L_temporal_hardneg + δ·L_localization
```

## 6. Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| R@1/5/10 | Standard retrieval |
| CAR@K | Chronologically accurate retrieval |
| TAR@K | Temporal-constraint-aware retrieval |
| IoU@0.5/0.7, mIoU | Temporal grounding accuracy |

## 7. Key Baselines

| Asset | Role | Repo |
|-------|------|------|
| TMR | Main backbone | github.com/Mathux/TMR |
| ChronAccRet | Ordering negatives code | github.com/line/ChronAccRet |
| Event-T2M | Event decomposition + HumanML3D-E | github.com/tjswodud/EventT2M-codes |
| FineMotion | Temporal annotation source | github.com/CVI-SZU/FineMotion |
| LaMP | Comparison baseline | github.com/gentlefress/LaMP |
| PST | Spatial fine-grained alignment baseline | Not open-sourced |

## 8. Positioning Against PST

PST focuses on spatial fine-grained alignment, while TAMR focuses on temporal fine-grained alignment, including event ordering and grounding. The two directions are complementary rather than equivalent.
