"""Export a tiny version of the nanowhale (DeepSeek-V4 100M) model to ONNX so
Netron can render the *computation graph* (not just the weights).

Why a tiny version? The full 100M model's safetensors (442 MB) only contains
tensor data — Netron can list the weights but cannot draw a graph from it.
ONNX bundles both the topology (every op + every tensor connection) and the
weights, so Netron renders an interactive compute graph. We use the same
topology as the real model but shrink vocab + seqlen so the export is fast
and the resulting file stays a few MB (committable to the repo / GitHub Pages).

Output:
    scripts/viz/nanowhale.onnx

Notes:
- bf16 NaNs at this scale (see README), so we always export in fp32.
- The MoE forward contains data-dependent control flow
  (`if counts[i] == 0: continue`, `torch.where(indices == i)`) which the new
  Dynamo exporter cannot handle. We replace MoE.forward with an equivalent
  *static* version that runs every expert on every token (weight-masked sum).
  Mathematically identical when no expert is skipped; the resulting graph shows
  the gate + every routed expert + the shared expert connected in parallel,
  which is what we actually want to visualize.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn

from configuration_deepseek_v4 import DeepseekV4Config
from modeling_deepseek_v4 import DeepseekV4ForCausalLM, DeepseekV4MoE


def _static_moe_forward(self: DeepseekV4MoE, x: torch.Tensor) -> torch.Tensor:
    """Export-friendly MoE: run every expert on every token, weight by route mask.

    Equivalent to the original sparse routing when every selected (token, expert)
    pair gets multiplied by the same gate weight. Eliminates the data-dependent
    `if counts[i] == 0: continue` and `torch.where` that torch.export rejects.
    """
    shape = x.shape
    x_flat = x.view(-1, self.hidden_size)
    weights, indices = self.gate(x_flat)  # [N, topk], [N, topk]
    y = torch.zeros_like(x_flat, dtype=torch.float32)
    for i in range(self.n_routed_experts):
        mask = (indices == i).to(weights.dtype)  # [N, topk]
        w_token = (weights * mask).sum(dim=-1, keepdim=True).float()  # [N, 1]
        out = self.experts[i](x_flat).float()  # [N, hidden]
        y = y + w_token * out
    y = y + self.shared_expert(x_flat).float()
    return y.to(x.dtype).view(shape)


def build_tiny_model() -> DeepseekV4ForCausalLM:
    """Same architectural *skeleton* as main_100m.yaml (8 layers, 4+1 experts,
    hc_mult=4, MLA), but with every dim shrunk so the ONNX file stays a few MB.

    The point is to visualize the graph topology, not to run real inference, so
    smaller hidden/intermediate dims are fine.
    """
    cfg = DeepseekV4Config(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=1,
        moe_intermediate_size=128,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        q_lora_rank=32,
        head_dim=24,
        qk_rope_head_dim=8,
        o_groups=2,
        o_lora_rank=16,
        hc_mult=4,
        hc_sinkhorn_iters=2,
        hc_eps=1.0e-6,
        num_hash_layers=0,
        swiglu_limit=0.0,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        topk_method="noaux_tc",
        norm_topk_prob=True,
        max_position_embeddings=64,
        rms_norm_eps=1.0e-6,
        rope_theta=10000.0,
        initializer_range=0.02,
        tie_word_embeddings=False,
        attention_bias=False,
        attention_dropout=0.0,
        compress_ratios=[0] * 9,
    )
    model = DeepseekV4ForCausalLM(cfg)
    model.eval()
    return model


class LogitsOnly(nn.Module):
    """Wrap so ONNX export sees a single tensor input + single tensor output."""

    def __init__(self, model: DeepseekV4ForCausalLM):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, return_dict=True, use_cache=False)
        return out.logits


def main() -> None:
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "nanowhale.onnx")

    torch.manual_seed(0)
    DeepseekV4MoE.forward = _static_moe_forward  # patch before building
    model = build_tiny_model().float()
    wrapped = LogitsOnly(model).eval()

    input_ids = torch.randint(0, 256, (1, 8), dtype=torch.long)

    with torch.no_grad():
        _ = wrapped(input_ids)  # sanity check before export

    n_params = sum(p.numel() for p in wrapped.parameters())
    print(f"exporting tiny model ({n_params/1e6:.2f}M params) to {out_path} ...")
    torch.onnx.export(
        wrapped,
        (input_ids,),
        out_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamo=True,
        dynamic_shapes={"input_ids": {0: "batch", 1: "seq"}},
        opset_version=18,
        external_data=False,  # inline weights into the single .onnx file
    )

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"wrote {out_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
