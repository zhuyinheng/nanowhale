"""Generate a torchview graph of the nanowhale (DeepSeek-V4 100M) model.

We build a *tiny* version of the model (small vocab, short seq) so torchview can
trace it on CPU within a reasonable amount of time, but keep the architectural
shape (num_hidden_layers=8, hc_mult=4, MoE 4+1, etc.) identical to the real
100M model so the visual structure matches.

Outputs:
    scripts/viz/nanowhale_arch.svg
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from torchview import draw_graph

from configuration_deepseek_v4 import DeepseekV4Config
from modeling_deepseek_v4 import DeepseekV4ForCausalLM


def build_model():
    # Match main_100m.yaml topology, but shrink vocab + seqlen so tracing is fast.
    cfg = DeepseekV4Config(
        vocab_size=512,                    # tiny vocab (real: 129280) for tracing only
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
        max_position_embeddings=64,
        rms_norm_eps=1.0e-6,
        rope_theta=10000.0,
        initializer_range=0.02,
        tie_word_embeddings=False,
        attention_bias=False,
        attention_dropout=0.0,
        compress_ratios=[0] * 9,
    )
    return DeepseekV4ForCausalLM(cfg)


def render(model, input_ids, depth, name, out_path):
    graph = draw_graph(
        model,
        input_data=input_ids,
        device="cpu",
        depth=depth,
        expand_nested=True,
        graph_name=name,
        save_graph=False,
        hide_module_functions=False,
        hide_inner_tensors=True,
        roll=True,
    )
    g = graph.visual_graph
    g.format = "svg"
    g.render(out_path, cleanup=True)
    print(f"wrote {out_path}.svg")


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    model = build_model()
    model.eval()
    input_ids = torch.randint(0, 256, (1, 8), dtype=torch.long)

    # Detailed view (every nn.Module up to depth=4)
    render(model, input_ids, depth=4,
           name="nanowhale (DeepSeek-V4 ~110M) — detailed",
           out_path=os.path.join(out_dir, "nanowhale_arch"))

    # Overview (top-level blocks only)
    render(model, input_ids, depth=2,
           name="nanowhale (DeepSeek-V4 ~110M) — overview",
           out_path=os.path.join(out_dir, "nanowhale_overview"))


if __name__ == "__main__":
    main()
