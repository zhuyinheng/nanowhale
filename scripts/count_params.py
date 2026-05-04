"""Count parameters for various DeepSeek-V4 config sizes."""
import sys
sys.path.insert(0, '.')
import torch
from configuration_deepseek_v4 import DeepseekV4Config
from modeling_deepseek_v4 import DeepseekV4ForCausalLM


def count_params(config, name=""):
    model = DeepseekV4ForCausalLM(config)
    total = sum(p.numel() for p in model.parameters())
    embed = sum(p.numel() for n, p in model.named_parameters() if 'embed' in n or 'lm_head' in n)
    non_embed = total - embed
    
    print(f"\n{'='*60}")
    print(f"Config: {name}")
    print(f"{'='*60}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  num_key_value_heads: {config.num_key_value_heads}")
    print(f"  head_dim: {config.head_dim}")
    print(f"  qk_rope_head_dim: {config.qk_rope_head_dim}")
    print(f"  q_lora_rank: {config.q_lora_rank}")
    print(f"  o_groups: {config.o_groups}")
    print(f"  o_lora_rank: {config.o_lora_rank}")
    print(f"  moe_intermediate_size: {config.moe_intermediate_size}")
    print(f"  n_routed_experts: {config.n_routed_experts}")
    print(f"  num_experts_per_tok: {config.num_experts_per_tok}")
    print(f"  hc_mult: {config.hc_mult}")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  max_position_embeddings: {config.max_position_embeddings}")
    print(f"  tie_word_embeddings: {config.tie_word_embeddings}")
    print(f"  ---")
    print(f"  Total params: {total:,} ({total/1e6:.1f}M)")
    print(f"  Embedding params: {embed:,} ({embed/1e6:.1f}M)")
    print(f"  Non-embedding params: {non_embed:,} ({non_embed/1e6:.1f}M)")
    
    # Breakdown by component
    attn_params = sum(p.numel() for n, p in model.named_parameters() if 'attn' in n and 'hc' not in n)
    ffn_params = sum(p.numel() for n, p in model.named_parameters() if 'ffn' in n and 'hc' not in n)
    hc_params = sum(p.numel() for n, p in model.named_parameters() if 'hc_' in n)
    norm_params = sum(p.numel() for n, p in model.named_parameters() if 'norm' in n)
    
    print(f"  Attention params: {attn_params:,} ({attn_params/1e6:.1f}M)")
    print(f"  FFN/MoE params: {ffn_params:,} ({ffn_params/1e6:.1f}M)")
    print(f"  HC params: {hc_params:,} ({hc_params/1e6:.1f}M)")
    print(f"  Norm params: {norm_params:,} ({norm_params/1e6:.1f}M)")
    
    del model
    return total


# Debug config (~400K params)
debug_config = DeepseekV4Config(
    vocab_size=1024, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
    num_key_value_heads=1, moe_intermediate_size=128, n_routed_experts=4,
    num_experts_per_tok=2, q_lora_rank=32, head_dim=32, qk_rope_head_dim=16,
    o_groups=2, o_lora_rank=16, hc_mult=4, num_hash_layers=0,
    max_position_embeddings=128, compress_ratios=[0, 0, 0],
)

# Target ~100M - iterate to find the right config
# Key insight: with MoE, most params are in experts. N_experts * 3 * hidden * inter_dim per layer
# With 8 experts, 3*h*inter = 3*384*768 = 884K per expert, * 8 = 7M per layer
# Plus attention, HC, norms

# Let's try various configs
configs_to_try = [
    # (name, hidden, layers, heads, kv_heads, inter, experts, topk, q_lora, head_dim, rope_dim, o_groups, o_lora)
    ("debug",       64,   2,  4,  1,  128,   4, 2,  32,  32,  16, 2,  16),
    ("~50M",       256,   6,  8,  1,  512,   4, 2, 128, 128,  32, 2,  64),
    ("~100M_v1",   384,   8,  8,  1,  768,   4, 2, 192, 128,  32, 2,  96),
    ("~100M_v2",   384,  10,  8,  1,  768,   8, 2, 192, 128,  32, 2,  96),
    ("~100M_v3",   512,   8, 16,  1,  1024,  4, 2, 256, 128,  32, 4, 128),
    ("~200M",      512,  12, 16,  1,  1024,  8, 2, 256, 128,  32, 4, 128),
    ("~500M",      768,  16, 16,  1,  1536,  8, 2, 384, 256,  64, 4, 192),
    ("~800M",     1024,  16, 16,  1,  2048,  8, 2, 512, 256,  64, 4, 256),
]

for name, hidden, layers, heads, kv, inter, experts, topk, q_lr, hd, rope, og, olr in configs_to_try:
    config = DeepseekV4Config(
        vocab_size=129280,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv,
        moe_intermediate_size=inter,
        n_routed_experts=experts,
        num_experts_per_tok=topk,
        q_lora_rank=q_lr,
        head_dim=hd,
        qk_rope_head_dim=rope,
        o_groups=og,
        o_lora_rank=olr,
        hc_mult=4,
        num_hash_layers=0,
        swiglu_limit=0.0,
        max_position_embeddings=2048,
        compress_ratios=[0] * (layers + 1),
        tie_word_embeddings=False,
    )
    count_params(config, name)
