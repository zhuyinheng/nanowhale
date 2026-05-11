"""Export nanowhale to ONNX so Netron can render the full compute graph.

We construct the model with the real main_100m.yaml shape (vocab_size=129280)
on CPU and use torch.onnx.export with export_params=False so the resulting file
contains only the graph structure (no weight values). That keeps the file under
a few MB — small enough to commit and host on GitHub Pages.

Caveats baked in:
  * MoE dynamic routing (`for i in range(n_routed_experts): if counts[i]==0: continue`)
    is traced. We use a long-enough sequence + favourable seed so all 4 experts
    receive at least one token, so every expert node appears in the graph.
  * HC Sinkhorn iterations (hc_sinkhorn_iters=2) are unrolled.
  * scaled_dot_product_attention is emitted as the ONNX Attention op (opset >= 14).
"""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from configuration_deepseek_v4 import DeepseekV4Config
from modeling_deepseek_v4 import DeepseekV4ForCausalLM


ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(ROOT, "nanowhale.onnx")


def build_model():
    cfg = DeepseekV4Config(
        vocab_size=129280,
        hidden_size=320,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=1,
        moe_intermediate_size=640,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        q_lora_rank=160,
        head_dim=96,
        qk_rope_head_dim=32,
        o_groups=2,
        o_lora_rank=80,
        hc_mult=4,
        hc_sinkhorn_iters=2,
        hc_eps=1.0e-6,
        num_hash_layers=0,
        swiglu_limit=0.0,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        topk_method="noaux_tc",
        norm_topk_prob=True,
        max_position_embeddings=2048,
        rms_norm_eps=1.0e-6,
        rope_theta=10000.0,
        initializer_range=0.02,
        tie_word_embeddings=False,
        attention_bias=False,
        attention_dropout=0.0,
        compress_ratios=[0] * 9,
    )
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(cfg)
    model.eval()
    return model, cfg


def main():
    model, cfg = build_model()

    # Long-enough seq so all 4 experts are hit by top-2 routing.
    bsz, seq = 2, 64
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq), dtype=torch.long)

    # Sanity: forward once.
    with torch.no_grad():
        _ = model(input_ids=input_ids)

    # We use the legacy (TorchScript-trace based) exporter via dynamo=False.
    # The newer torch.export-based path refuses data-dependent control flow in
    # DeepseekV4MoE.forward (`if counts[i] == 0: continue`). For visualization
    # we just need the trace-time graph.
    torch.onnx.export(
        model,
        (input_ids,),
        OUT_PATH,
        input_names=["input_ids"],
        output_names=["logits"],
        opset_version=17,
        export_params=False,         # graph structure only
        do_constant_folding=False,
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        dynamo=False,
    )
    size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
    print(f"wrote {OUT_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
