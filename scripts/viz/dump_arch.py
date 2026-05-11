"""Dump the nanowhale (DeepSeek-V4 ~110M) module tree to JSON.

We instantiate the model with the *real* main_100m.yaml shape (vocab_size=129280)
because parameter counts are derived purely from `nn.Parameter.numel()` and don't
require GPU or training.

Output: scripts/viz/arch.json
{
    "config": { ... model config dict ... },
    "tree": {
        "name": "model",
        "type": "DeepseekV4ForCausalLM",
        "params": int,
        "shape": null | [dims],   # for nn.Parameter leaves
        "children": [ ... recursive ... ]
    }
}
"""

import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn

from configuration_deepseek_v4 import DeepseekV4Config
from modeling_deepseek_v4 import DeepseekV4ForCausalLM


def real_config():
    # Matches configs/main_100m.yaml.
    return DeepseekV4Config(
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


def module_tree(module: nn.Module, name: str = "model") -> dict:
    """Recursive description of a module: its children (sub-modules) and its
    own parameters (nn.Parameter leaves). We include parameter shapes so the
    viewer can show tensor sizes."""
    direct_params = []
    for pname, p in module.named_parameters(recurse=False):
        direct_params.append({
            "name": pname,
            "type": "Parameter",
            "params": p.numel(),
            "shape": list(p.shape),
            "children": [],
        })

    children = []
    for cname, child in module.named_children():
        children.append(module_tree(child, cname))

    total = sum(c["params"] for c in children) + sum(c["params"] for c in direct_params)
    return {
        "name": name,
        "type": module.__class__.__name__,
        "params": total,
        "shape": None,
        "children": children + direct_params,
    }


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = real_config()
    # meta device avoids allocating the 41M-param embedding in RAM.
    with torch.device("meta"):
        model = DeepseekV4ForCausalLM(cfg)

    tree = module_tree(model, "model")

    out = {
        "config": {k: v for k, v in cfg.to_dict().items()
                   if not k.startswith("_") and isinstance(v, (int, float, str, bool, list, type(None)))},
        "tree": tree,
        "summary": {
            "total_params": tree["params"],
            "num_hidden_layers": cfg.num_hidden_layers,
            "hidden_size": cfg.hidden_size,
            "vocab_size": cfg.vocab_size,
            "hc_mult": cfg.hc_mult,
            "n_routed_experts": cfg.n_routed_experts,
            "n_shared_experts": cfg.n_shared_experts,
            "num_experts_per_tok": cfg.num_experts_per_tok,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "head_dim": cfg.head_dim,
            "qk_rope_head_dim": cfg.qk_rope_head_dim,
            "q_lora_rank": cfg.q_lora_rank,
            "o_lora_rank": cfg.o_lora_rank,
            "o_groups": cfg.o_groups,
            "moe_intermediate_size": cfg.moe_intermediate_size,
            "max_position_embeddings": cfg.max_position_embeddings,
        },
    }

    out_path = os.path.join(out_dir, "arch.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")
    print(f"total params: {tree['params']:,}")


if __name__ == "__main__":
    main()
