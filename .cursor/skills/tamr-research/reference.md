# TAMR Technical Reference

Detailed technical reference for TAMR implementation. Read this when working on specific modules.

## TMR Model Internals

### ACTORStyleEncoder (src/model/actor.py)
```python
# Input: (batch, seq_len, input_dim) + lengths
# 1. Linear(input_dim, latent_dim)
# 2. Prepend learnable tokens: 1 (deterministic) or 2 (VAE: mu + logvar)
# 3. Sinusoidal positional encoding
# 4. nn.TransformerEncoder(nhead=4, dim_feedforward=1024, num_layers=6, activation=GELU)
# 5. Extract prepended token outputs → mu, logvar (or just z)
# Output: distribution or deterministic latent [batch, 256]
```

### InfoNCE_with_filtering (src/model/losses.py)
```python
# sim_matrix = cosine_sim(text_latents, motion_latents) / temperature  # τ=0.1
# For false negative filtering:
#   sent_emb = sentence_transformer(texts)  # all-mpnet-base-v2
#   self_sim = cosine_sim(sent_emb, sent_emb)
#   mask = (self_sim > threshold)  # threshold=0.80
#   sim_matrix[mask] = -inf  (except diagonal)
# loss = (cross_entropy(sim, labels_t2m) + cross_entropy(sim.T, labels_m2t)) / 2
```

### TMR Training Config Defaults
- latent_dim: 256
- text_model: distilbert-base-uncased (tokens), sentence-transformers/all-mpnet-base-v2 (sentences)
- motion_features: 263-dim Guo (22 joints)
- optimizer: AdamW, lr=1e-4
- loss weights: recons=1.0, latent=1e-5, kl=1e-5, contrastive=0.1
- contrastive threshold: 0.80, temperature: 0.1

### TMR Retrieval Protocols (retrieval.py)
1. **All**: full test set, all-vs-all similarity
2. **Dissect**: per-sample retrieval within subsets
3. **Threshold**: filter by text similarity threshold (e.g., 0.95)
4. **NSim**: normalized similarity scoring

## EventT2M Model Internals

### EventT2M Denoiser (src/models/nets/event_final.py)
```python
class EventT2M(nn.Module):
    # Input: motion [B, T, 263], mask [B, T], timestep [B], text [B, 512],
    #        decomposed_embed [B, max_events, 512], decomposed_mask [B, max_events]
    # Architecture:
    #   1. Linear(263, base_dim=256)
    #   2. Sinusoidal pos encoding + timestep embedding
    #   3. Prepend time token
    #   4. N StageBlocks (default 4, all dim=256):
    #      StageBlock = LocalModule → MixedModule → LocalModule
    #      MixedModule:
    #        - Patch motion into segments
    #        - Conv downsample
    #        - Gated text fusion (text_embed * gate + motion)
    #        - MiniConformer with cross-attention to decomposed events
    #        - Upsample back
    #   5. Linear(base_dim, 263)
```

### MiniConformer Block
```
FFN(half-step) → SelfAttention → CrossAttention(to events) → DepthwiseConv → FFN(half-step)
```
All with pre-norm (LayerNorm) and residual connections.

### Data Preprocessing (data_preprocess_decomposed.py)
```python
# Output: data_{split}.npy = dict of:
# {
#   "motion_id": {
#     "motion": np.array [T, 263],
#     "length": int,
#     "text": [
#       {
#         "caption": "full caption text",
#         "tokens": "POS/tagged/tokens",
#         "decomposed": [
#           {"caption": "sub-event 1", "tokens": "POS/tagged"},
#           {"caption": "sub-event 2", "tokens": "POS/tagged"},
#         ]
#       },
#       ...  # multiple captions per motion
#     ]
#   }
# }
```

## Ablation Experiment IDs

| ID | Config | Purpose |
|----|--------|---------|
| A1 | w/o event decomposition (global align only) | Event decomposition gain |
| A2 | w/o temporal hard negatives | Temporal negative gain |
| A3 | w/o localization loss | Joint training benefit |
| A4 | ordering-only vs 6 constraints | Extended constraint gain |
| A5 | global InfoNCE vs event-level align | Fine-grained alignment gain |
| A6 | w/o timestamp tokens | Explicit time discretization gain |
| A7 | frozen TMR vs fine-tuned TMR | Transfer effectiveness |

## Comparison Table (from PST paper, HumanML3D All protocol)

| Method | T2M R@1 | T2M R@5 | T2M R@10 | MedR↓ |
|--------|---------|---------|----------|-------|
| TMR | 8.92 | 22.06 | 33.37 | 25 |
| MotionPatch | 10.80 | 26.72 | 38.02 | 19 |
| PST | 12.45 | 33.65 | 48.22 | 10 |
| PST++ | 13.83 | 34.82 | 49.15 | 10 |
