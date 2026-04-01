---
name: tamr-research
description: Provides context for the TAMR (Temporal-Aware Motion-Text Retrieval) research project. Use when the user asks about TAMR design, TMR codebase modifications, EventT2M integration, HumanML3D-E dataset, temporal-aware contrastive learning, motion-text retrieval experiments, or any task related to this research line.
---

# TAMR Research Context

TAMR = Temporal-Aware Motion-text Retrieval. Goal: add temporal awareness (event ordering, grounding, temporal constraints) to the TMR retrieval backbone. Target venue: NeurIPS 2026 / ICLR 2027.

## Research Plan

Full design doc: `Codex/paperIDEAs/TAMR/2026-03-31_tamr-backbone-data-pipeline-design.md`

## 1. Codebase Layout

```
TMR/                              # Main workspace (this repo, fork of Mathux/TMR)
├── train.py                      # TMR training entry (Hydra + PyTorch Lightning)
├── retrieval.py                  # 4 retrieval evaluation protocols
├── extract.py / encode_*.py      # Checkpoint extraction & dataset encoding
├── src/
│   ├── model/
│   │   ├── temos.py              # Base VAE: motion_encoder + text_encoder + motion_decoder
│   │   ├── tmr.py                # TMR = TEMOS + InfoNCE contrastive loss
│   │   ├── actor.py              # ACTORStyleEncoder/Decoder (Transformer, 6L/4H/1024FFN)
│   │   ├── losses.py             # InfoNCE_with_filtering (threshold=0.80)
│   │   └── text_encoder.py       # TextToEmb: distilbert (tokens) + all-mpnet (sentences)
│   ├── data/
│   │   ├── humanml3d.py          # HumanML3D dataset class
│   │   └── text.py               # Text loading & tokenization
│   └── config.py                 # Hydra structured configs
├── configs/                      # Hydra YAML configs
├── datasets/                     # Annotation splits & text files
├── stats/                        # mean.pt / std.pt normalization
├── EventT2M-codes-main/          # Event-T2M reference implementation (ICLR'26)
└── Codex/                        # Research notes & paper analyses
```

## 2. TMR Architecture (Backbone)

VAE + contrastive dual-tower:

- **Motion encoder**: `ACTORStyleEncoder` — 263-dim Guo features → 256-dim latent (mu + logvar tokens, Transformer 6L/4H)
- **Text encoder**: `ACTORStyleEncoder` — 768-dim DistilBERT tokens → 256-dim latent
- **Motion decoder**: `ACTORStyleDecoder` — latent z → reconstructed motion
- **Contrastive**: `InfoNCE_with_filtering` — symmetric cross-entropy on cosine sim / τ=0.1, filters false negatives via sentence embedding self-similarity
- **Loss**: `L = recons(1.0) + latent(1e-5) + kl(1e-5) + contrastive(0.1)`
- **Retrieval**: `get_sim_matrix()` → cosine similarity of L2-normalized latents

Key: TMR has **zero temporal awareness** — treats motion as bag-of-features, cannot distinguish event ordering (proven by ChroAccRet ECCV'24).

## 3. EventT2M Reference (`EventT2M-codes-main/`)

ICLR'26 work. Decomposes text into sub-events for motion generation (not retrieval).

```
EventT2M-codes-main/
├── src/
│   ├── train.py / eval.py        # Hydra entry points
│   ├── models/
│   │   ├── event_final.py        # EventMotionGeneration (LightningModule)
│   │   └── nets/
│   │       ├── event_final.py    # EventT2M denoiser (MixedModule + MiniConformer)
│   │       └── text_encoder.py   # TMRWrapperEncoder / CLIP / RoBERTa
│   ├── data/
│   │   ├── hml3d_event.py        # HumanML3DDataModule
│   │   └── humanml/dataset.py    # T2MDataset with decomposed events
│   └── tools/
│       ├── data_decompose.py     # Gemini 2.5 Flash event decomposition
│       └── data_preprocess_decomposed.py  # Build data_{split}.npy
├── TMR_model_wrapper.py          # Wraps TMR for text encoding
└── configs/                      # Hydra configs (44 files)
```

Key architecture: `EventT2M` denoiser uses `MiniConformer` blocks with cross-attention to decomposed event embeddings. Events enter as separate K/V in attention, giving explicit per-event conditioning.

## 4. HumanML3D-E Dataset

Event-annotated extension of HumanML3D. Expected structure:

```
dataset/HumanML3D/
├── {train,val,test}.txt          # Split files (one motion ID per line)
├── Mean.npy / Std.npy            # Z-normalization stats [263]
├── new_joint_vecs/*.npy          # Motion features [T, 263]
├── texts/*.txt                   # Original captions: "caption#POS#f_tag#to_tag"
├── texts_decomposed/*.txt        # ★ Event decomposition (LLM-generated)
├── data_{train,val,test}.npy     # Pre-processed dicts
└── data_test_condition{2,3,4}.npy  # Stratified by event count
```

`texts_decomposed/` format: blocks separated by blank lines, each block = decomposed events for one original caption. Each line: `sub_caption#POS_tokens#f_tag#to_tag`.

Download: [Google Drive](https://drive.google.com/drive/folders/19mPyYV8j1vnfJ6W9tZX9758JtDUQpYop) or build via `data_decompose.py` + `data_preprocess_decomposed.py`.

## 5. TAMR Design (3 New Modules on TMR)

### Module A: Event-Aware Text Encoding
- LLM decomposes text → ordered event list with inter-event relations
- TMR text encoder encodes each event independently + learnable relation embeddings

### Module B: Temporal-Aware Contrastive Learning
- Global InfoNCE (TMR original) + event-level alignment loss + temporal hard negative loss
- 6 temporal constraint negatives: ordering shuffle, parallel→sequential, negation, causal reversal, duration modification, sync→async
- Timestamp tokens: motion timeline normalized to [0, 1000]

### Module C: Unified Retrieval-Localization Head
- Retrieval: global cosine → top-K
- Localization: event-level alignment matrix → anchor-free temporal boundary prediction

### Loss
```
L = α·L_global_infonce + β·L_event_align + γ·L_temporal_hardneg + δ·L_localization
    (α=1.0, β=0.5, γ=0.3, δ=0.2)
```

### Training Stages
1. Stage 0: Reproduce TMR baseline (1-2 days)
2. Stage 1: Freeze TMR, train Module C only (1-2 days)
3. Stage 2: Unfreeze last 2 encoder layers, add Module B (2-3 days)
4. Stage 3: Full fine-tune, joint retrieval+localization+temporal (3-5 days)

## 6. Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| R@1/5/10 | Standard retrieval (TMR baseline) |
| CAR@K | Chronologically-accurate retrieval (ChroAccRet) |
| TAR@K (new) | Temporal-constraint-aware retrieval (ours) |
| IoU@0.5/0.7, mIoU | Temporal grounding accuracy |

## 7. Key Baselines & Assets

| Asset | Role | Repo |
|-------|------|------|
| TMR (ICCV'23, 293⭐) | Main backbone | github.com/Mathux/TMR |
| ChronAccRet (ECCV'24) | Ordering negatives code | github.com/line/ChronAccRet |
| Event-T2M (ICLR'26) | Event decomposition + HumanML3D-E | github.com/tjswodud/EventT2M-codes |
| FineMotion (ICCV'25) | Temporal annotation source (BPMSD) | github.com/CVI-SZU/FineMotion |
| LaMP (ICLR'25) | Comparison baseline | github.com/gentlefress/LaMP |
| PST (arXiv'26) | Competitor — spatial fine-grained (orthogonal to TAMR's temporal) | Not open-sourced |

## 8. Data Pipeline Plan

```
FineMotion BPMSD (442K 0.5s segments)
  → semantic aggregation → event sequences [event_1(t1), ..., event_n(tn)]
  → inter-event relation inference (ordering/parallel/causal)
  → 6 types of temporal constraint negatives
  → quality control (human subset + 10% spot check)
```

Primary dataset: HumanML3D (14,616 sequences). Event evaluation: HumanML3D-E. Auxiliary: KIT-ML.

## 9. PST Competition Note

PST occupies "spatial fine-grained alignment" (joint→segment→holistic). TAMR occupies "temporal fine-grained" (event ordering + grounding). They are orthogonal:
- PST's STI is permutation-invariant → cannot sense event order
- PST has no temporal grounding capability
- TAMR paper must explicitly discuss PST and argue orthogonality

## 10. Timeline

```
Week 1:   TMR reproduce + ChronAccRet CAR verification
Week 2:   FineMotion event-level data construction
Week 3-4: TAMR 3-module implementation + staged training
Week 5-6: Evaluation + ablation experiments
Week 7:   Paper writing
```
