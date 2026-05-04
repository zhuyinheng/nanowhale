# SmolDeepSeek-V4 100M — Project Summary

## Goal

Train a small (~100M parameter) language model from scratch using the **DeepSeek-V4 architecture**, then fine-tune it for chat and upload to the Hugging Face Hub.

## What Was Done

### Session 1 (previous)

- Adapted DeepSeek-V4 modeling code (`modeling_deepseek_v4.py`, `configuration_deepseek_v4.py`) for a small ~110M param config
- Created training configs (`debug.yaml`, `main_100m.yaml`, `fallback_under_1b.yaml`)
- Reused the DeepSeek-V4 tokenizer (vocab_size=129,280)
- Wrote all scripts: `train_pretrain.py`, `train_sft.py`, `eval_smoke.py`, `chat.py`, `upload_to_hub.py`
- Ran debug pretraining (50 steps) and main pretraining (5,000 steps on FineWeb-Edu, ~2.6B tokens)
- Applied `torch.compile` for 1.77× speedup (122ms → 72ms/step)
- Saved checkpoints at steps 4000, 4500, 5000, and `final/`

### Session 2 (this session)

1. **Fixed pretrained checkpoint** — The saved weights had `_orig_mod.` prefix from `torch.compile`. Stripped this prefix from all 270 keys in `model.safetensors` across all checkpoints (final, 4000, 4500, 5000).

2. **Evaluated pretrained model** — Perplexity of 13.62 on held-out English text. Generations were incoherent multilingual text (expected — the model barely beats uniform distribution over the 129K vocab).

3. **Ran SFT on SmolTalk** — Fine-tuned on [HuggingFaceTB/smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) (460K conversations) for 3,000 steps. Had to solve a NaN issue: `from_pretrained` was re-initializing weights due to custom architecture quirks, so switched to manual `load_state_dict`. Also switched from bf16 to fp32 training since the Hyper-Connections + MoE architecture produced NaN in bf16 at this small scale.

4. **Evaluated SFT model** — Perplexity improved to 12.90. The model learned conversational structure (paragraphs, explanations, lists) but hallucinates facts, which is expected for a 110M model.

5. **Uploaded both models to Hub** with custom code (`trust_remote_code=True`), model cards, and fixed imports (relative + absolute fallback for Hub compatibility).

6. **Validated Hub loading** — Both models load from Hub via `AutoConfig` + `load_state_dict` and generate text.

## Artifacts

### Hub Repositories

| Repo | Description | Link |
|---|---|---|
| **cmpatino/smol-deepseek-v4-100m-pretrain** | Pretrained base model (5K steps on FineWeb-Edu) | [Hub](https://huggingface.co/cmpatino/smol-deepseek-v4-100m-pretrain) |
| **cmpatino/smol-deepseek-v4-100m** | SFT chat model (3K steps on SmolTalk) | [Hub](https://huggingface.co/cmpatino/smol-deepseek-v4-100m) |

Each repo contains:
- `model.safetensors` (~441MB)
- `config.json` (with `auto_map` for `AutoConfig`/`AutoModelForCausalLM`)
- `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`
- `configuration_deepseek_v4.py`, `modeling_deepseek_v4.py` (custom code)
- `README.md` (model card)

### Local Checkpoints

```
checkpoints/
├── pretrain_100m/
│   ├── checkpoint-4000/
│   ├── checkpoint-4500/
│   ├── checkpoint-5000/
│   └── final/           ← pretrained model (uploaded)
└── sft/
    ├── checkpoint-2000/
    ├── checkpoint-2500/
    ├── checkpoint-3000/
    └── final/           ← SFT model (uploaded)
```

### Source Files

| File | Purpose |
|---|---|
| `modeling_deepseek_v4.py` | DeepSeek-V4 model implementation (MLA, MoE, Hyper-Connections, MTP) |
| `configuration_deepseek_v4.py` | Model config class |
| `configs/main_100m.yaml` | Training hyperparameters for the 100M model |
| `scripts/train_pretrain.py` | Pretraining script (SFTTrainer on FineWeb-Edu) |
| `scripts/train_sft.py` | SFT script (SFTTrainer on SmolTalk) |
| `scripts/eval_smoke.py` | Smoke evaluation (perplexity, generation) |
| `scripts/chat.py` | Interactive chat |
| `scripts/upload_to_hub.py` | Hub upload utility |

## Architecture

The model implements these DeepSeek-V4 features at miniature scale:

| Component | Config |
|---|---|
| Parameters | ~110M (41M embeddings + 69M non-embedding) |
| Hidden size | 320 |
| Layers | 8 |
| Attention | 8 heads, 1 KV head (MQA), head_dim=96 (32 RoPE + 64 NoPE) |
| MLA | q_lora_rank=160, o_groups=2, o_lora_rank=80 |
| MoE | 4 routed + 1 shared expert, top-2 routing, SwiGLU FFN (dim 640) |
| Routing | sqrtsoftplus scoring, noaux_tc method |
| Hyper-Connections | hc_mult=4, Sinkhorn routing (2 iterations) |
| MTP | 1 next-token prediction layer |
| Vocab | 129,280 (DeepSeek-V4 tokenizer) |
| Context | 2,048 tokens |

## Training Metrics

### Pretraining (5,000 steps)

| Metric | Value |
|---|---|
| Dataset | FineWeb-Edu (streaming) |
| Tokens seen | ~2.6B |
| Final loss | ~5.3 |
| Token accuracy | 33.8% |
| Hardware | 1× H100 80GB, bf16 |

### SFT (3,000 steps)

| Metric | Start | End |
|---|---|---|
| Train loss | 15.41 | 10.22 |
| Eval loss | 2.873 | 2.607 |
| Token accuracy | 36.2% | 48.5% |
| Entropy | 3.95 | 2.81 |

Eval loss trajectory: 2.873 → 2.717 → 2.654 → 2.622 → 2.609 → 2.607 (converged, no overfitting).

### Perplexity Comparison (held-out English text)

| Model | Loss | Perplexity |
|---|---|---|
| Pretrained | 2.612 | 13.62 |
| SFT | 2.558 | 12.90 |

## Known Issues & Notes

- **`from_pretrained` weight mismatch**: The custom architecture causes `from_pretrained` to re-initialize some weights. Use manual `load_state_dict` instead (documented in model cards).
- **bf16 NaN**: The model produces NaN in bf16 inference/training. Use fp32. This is likely due to the Hyper-Connections architecture at small scale producing values that overflow bf16 range.
- **Large vocab vs small model**: The 129K vocab embedding table consumes 37% of all parameters, leaving limited capacity for language modeling. This is the main reason for high perplexity.
- **Import compatibility**: `modeling_deepseek_v4.py` uses a try/except for relative vs absolute import to work both locally and from Hub.
