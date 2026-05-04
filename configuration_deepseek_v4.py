"""DeepSeek-V4 model configuration.

Adapted from the DeepSeek-V4 inference config (deepseek-ai/DeepSeek-V4-Pro)
and the HF Transformers DeepSeek-V3 config for HF compatibility.

Key V4-specific features vs V3:
- Hyper-Connections (HC): multi-copy hidden states with Sinkhorn routing
- Compressed Sparse Attention (CSA): compression + sliding window + sparse indexing
- New MoE routing: sqrtsoftplus scoring, hash-based routing for first layers
- Large head_dim (512), o_groups/o_lora_rank for grouped output projection
- No kv_lora_rank (replaced by compress_ratios)
- No v_head_dim/qk_nope_head_dim (replaced by head_dim)
"""

from transformers.configuration_utils import PretrainedConfig


class DeepseekV4Config(PretrainedConfig):
    model_type = "deepseek_v4"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=129280,
        hidden_size=4096,
        num_hidden_layers=43,
        num_attention_heads=64,
        num_key_value_heads=1,
        # MoE
        moe_intermediate_size=2048,
        n_routed_experts=256,
        n_shared_experts=1,
        num_experts_per_tok=6,
        norm_topk_prob=True,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        topk_method="noaux_tc",
        num_hash_layers=3,
        swiglu_limit=10.0,
        # MLA / Attention
        q_lora_rank=1024,
        head_dim=512,
        qk_rope_head_dim=64,
        o_groups=8,
        o_lora_rank=1024,
        sliding_window=128,
        # Compression
        compress_ratios=None,
        compress_rope_theta=160000.0,
        # Index attention
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
        # Hyper-Connections
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        # MTP
        num_nextn_predict_layers=1,
        # Standard
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads

        # MoE
        self.moe_intermediate_size = moe_intermediate_size
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.topk_method = topk_method
        self.num_hash_layers = num_hash_layers
        self.swiglu_limit = swiglu_limit

        # Attention
        self.q_lora_rank = q_lora_rank
        self.head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.nope_head_dim = head_dim - qk_rope_head_dim
        self.o_groups = o_groups
        self.o_lora_rank = o_lora_rank
        self.sliding_window = sliding_window

        # Compression
        if compress_ratios is None:
            # Default: no compression for small models
            compress_ratios = [0] * (num_hidden_layers + 1)
        self.compress_ratios = compress_ratios
        self.compress_rope_theta = compress_rope_theta

        # Index attention
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk

        # Hyper-Connections
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps

        # MTP
        self.num_nextn_predict_layers = num_nextn_predict_layers

        # Standard
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
