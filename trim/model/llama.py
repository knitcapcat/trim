"""
Inference-only Llama model definition.

Maps to: vllm/model_executor/models/llama.py (500+ lines)

This is a simplified Llama adapted for efficient inference. Key differences
from HuggingFace's LlamaForCausalLM:

  1. NO BATCH DIMENSION
     HF:   [batch_size, seq_len, hidden_size]
     trim: [total_tokens, hidden_size]
     All sequences in the batch are concatenated into one flat tensor.
     This avoids padding waste and matches how attention kernels work.

  2. FUSED PROJECTIONS
     HF has separate q_proj, k_proj, v_proj → vLLM fuses into qkv_proj
     HF has separate gate_proj, up_proj   → vLLM fuses into gate_up_proj
     Fewer kernel launches, better GPU utilization.

  3. EXTERNAL ATTENTION LAYER
     HF computes attention internally and manages past_key_values.
     vLLM delegates to a separate Attention layer that handles paged
     KV cache read/write and dispatches to flash_attn kernels.

  4. EXPLICIT RESIDUAL PASSING
     vLLM passes residual as a separate tensor between layers.
     This enables fused add+norm kernels (Phase 2 optimization).
     Phase 1 uses simple add, but we keep the interface.

  5. WEIGHT LOADING
     HuggingFace checkpoints have q_proj, k_proj, v_proj as separate tensors.
     We fuse them into qkv_proj on load via stacked_params_mapping.
     Similarly gate_proj + up_proj → gate_up_proj.

References:
  LlamaMLP         → vllm LlamaMLP (fused gate_up + SiLU + down)
  LlamaAttention   → vllm LlamaAttention (QKV + RoPE + Attention + O)
  LlamaDecoderLayer→ vllm LlamaDecoderLayer (pre-norm, explicit residual)
  LlamaModel       → vllm LlamaModel (embed + layers + norm)
  LlamaForCausalLM → vllm LlamaForCausalLM (model + lm_head)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from trim.attention.attention import Attention, AttentionMetadata

if TYPE_CHECKING:
    from trim.config import ModelConfig


# ---------------------------------------------------------------------------
# RMSNorm → vllm/model_executor/layers/layernorm.py::RMSNorm
#
# vLLM uses a fused CUDA kernel for performance. We use pure PyTorch.
# The fused version also supports in-place add+norm (residual fusion)
# which we'll add in Phase 2.
#
# RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * weight
# Unlike LayerNorm, RMSNorm does NOT subtract the mean — simpler and faster.
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(x.dtype)


# ---------------------------------------------------------------------------
# RotaryEmbedding → vllm/model_executor/layers/rotary_embedding.py
#
# vLLM's get_rope() supports many variants (NTK-aware, YaRN, dynamic, …).
# We implement standard Llama RoPE only.
#
# RoPE encodes position by ROTATING pairs of dimensions:
#   For each pair (x_2i, x_{2i+1}):
#     x_2i'     = x_2i * cos(θ) - x_{2i+1} * sin(θ)
#     x_{2i+1}' = x_{2i+1} * cos(θ) + x_2i * sin(θ)
#   where θ_i = position / (rope_theta ^ (2i / head_dim))
#
# Properties:
#   - Relative position aware: attention(q_m, k_n) depends only on (m-n)
#   - No learnable parameters
#   - Applied to Q and K, but NOT to V
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float = 500000.0,
    ) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim

    def forward(
        self,
        positions: torch.Tensor,     # [total_tokens]
        q: torch.Tensor,             # [total_tokens, num_heads * head_dim]
        k: torch.Tensor,             # [total_tokens, num_kv_heads * head_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # positions → frequency matrix
        # [total_tokens] x [head_dim/2] → [total_tokens, head_dim/2]
        freqs = torch.outer(positions.float(), self.inv_freq)
        cos = freqs.cos()  # [total_tokens, head_dim/2]
        sin = freqs.sin()  # [total_tokens, head_dim/2]

        q = self._apply_rotary(q, cos, sin)
        k = self._apply_rotary(k, cos, sin)
        return q, k

    @staticmethod
    def _apply_rotary(
        x: torch.Tensor,       # [total_tokens, num_heads * head_dim]
        cos: torch.Tensor,     # [total_tokens, head_dim/2]
        sin: torch.Tensor,
    ) -> torch.Tensor:
        # Reshape to [..., num_heads, head_dim] to apply per-head rotation
        head_dim = cos.shape[-1] * 2
        *batch_shape, dim = x.shape
        num_heads = dim // head_dim
        x = x.view(*batch_shape, num_heads, head_dim)

        x1 = x[..., : head_dim // 2]
        x2 = x[..., head_dim // 2 :]

        # Broadcast cos/sin: [total_tokens, 1, head_dim/2]
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

        out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return out.view(*batch_shape, dim)


# ---------------------------------------------------------------------------
# LlamaMLP → vllm LlamaMLP
#
# SwiGLU activation: out = down_proj(SiLU(gate) * up)
#
# vLLM fuses gate_proj and up_proj into one gate_up_proj linear,
# then splits the output. Fewer kernel launches = faster.
#
# HuggingFace equivalent:
#   gate = self.gate_proj(x)       ─┐
#   up   = self.up_proj(x)         ─┤ fused → gate_up_proj(x) → split
#   out  = self.down_proj(SiLU(gate) * up)
# ---------------------------------------------------------------------------

class LlamaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        # Fused: projects hidden → [gate, up] in one matmul
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)               # [tokens, 2 * intermediate]
        gate, up = gate_up.chunk(2, dim=-1)           # each [tokens, intermediate]
        x = F.silu(gate) * up                         # SwiGLU
        x = self.down_proj(x)                         # [tokens, hidden]
        return x


# ---------------------------------------------------------------------------
# LlamaAttention → vllm LlamaAttention
#
# vLLM version uses:
#   - QKVParallelLinear: fused Q/K/V projection (supports TP sharding)
#   - RowParallelLinear: output projection (supports TP all-reduce)
#   - get_rope(): flexible rotary embedding
#   - Attention: dispatches to flash_attn / paged_attn backends
#
# Our Phase 1 version uses standard nn.Linear (no TP) and our own
# RotaryEmbedding + Attention.
# ---------------------------------------------------------------------------

class LlamaAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        # Fused QKV projection: hidden → [Q, K, V] in one matmul
        # Q: num_heads * head_dim, K: num_kv_heads * head_dim, V: same
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(
            self.hidden_size,
            self.q_size + 2 * self.kv_size,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.rotary_emb = RotaryEmbedding(
            self.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.attn = Attention(self.num_heads, self.head_dim, self.num_kv_heads)

    def forward(
        self,
        positions: torch.Tensor,        # [total_tokens]
        hidden_states: torch.Tensor,     # [total_tokens, hidden_size]
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # 1. Fused QKV projection
        qkv = self.qkv_proj(hidden_states)  # [total_tokens, q_size + 2*kv_size]
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # 2. Rotary positional embeddings (applied to Q and K, not V)
        q, k = self.rotary_emb(positions, q, k)

        # 3. Attention with paged KV cache
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)

        # 4. Output projection
        output = self.o_proj(attn_output)   # [total_tokens, hidden_size]
        return output


# ---------------------------------------------------------------------------
# LlamaDecoderLayer → vllm LlamaDecoderLayer
#
# Pre-norm transformer block with explicit residual passing.
#
# vLLM passes residual separately so that fused kernels can do
# add+norm in one pass (e.g., `hidden, residual = layernorm(x, residual)`).
# Phase 1 does it sequentially, but we keep the same interface.
#
# Flow:
#   residual = hidden_states
#   hidden = layernorm(hidden_states)
#   hidden = self_attn(hidden)
#   hidden = hidden + residual           ← residual connection
#   residual = hidden
#   hidden = post_attn_layernorm(hidden)
#   hidden = mlp(hidden)
#   hidden = hidden + residual           ← residual connection
# ---------------------------------------------------------------------------

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = LlamaAttention(config, layer_idx)
        self.mlp = LlamaMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # Self attention with pre-norm and residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, kv_cache, attn_metadata)
        hidden_states = residual + hidden_states

        # MLP with pre-norm and residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# LlamaModel → vllm LlamaModel
#
# vLLM wraps this with @support_torch_compile which enables:
#   - Custom Dynamo backend (VllmBackend) to split the graph at attention ops
#   - Piecewise compilation for multi-shape dispatch
#   - CUDA Graph capture
# We skip this in Phase 1 (eager mode).
#
# vLLM also has PP (Pipeline Parallelism) support with PPMissingLayer
# and make_layers(). We use a simple ModuleList.
# ---------------------------------------------------------------------------

class LlamaModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,          # [total_tokens]
        positions: torch.Tensor,           # [total_tokens]
        kv_caches: list[torch.Tensor],     # one per layer
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)  # [total_tokens, hidden_size]

        for i, layer in enumerate(self.layers):
            hidden_states = layer(positions, hidden_states, kv_caches[i], attn_metadata)

        hidden_states = self.norm(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# LlamaForCausalLM → vllm LlamaForCausalLM
#
# Top-level model: LlamaModel + lm_head projection → logits.
#
# vLLM separates forward() and compute_logits() because:
#   - In PP, only the last rank runs lm_head
#   - The logits processor handles gathering and scaling
# We keep the separation for clarity.
#
# load_weights() handles the HF → trim weight mapping:
#   HF q_proj, k_proj, v_proj → fused qkv_proj
#   HF gate_proj, up_proj     → fused gate_up_proj
# ---------------------------------------------------------------------------

class LlamaForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Returns hidden_states (not logits). Call compute_logits() separately."""
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits."""
        return self.lm_head(hidden_states)

    def load_weights(self, weights: dict[str, torch.Tensor]) -> None:
        """
        Load HuggingFace checkpoint weights into our fused model.

        Key mapping:
          model.layers.{i}.self_attn.q_proj.weight  ─┐
          model.layers.{i}.self_attn.k_proj.weight  ─┼→ self_attn.qkv_proj.weight
          model.layers.{i}.self_attn.v_proj.weight  ─┘
          model.layers.{i}.mlp.gate_proj.weight  ─┐
          model.layers.{i}.mlp.up_proj.weight    ─┘→ mlp.gate_up_proj.weight

        This is what vLLM's stacked_params_mapping does:
          (".qkv_proj", ".q_proj", "q"),
          (".qkv_proj", ".k_proj", "k"),
          (".qkv_proj", ".v_proj", "v"),
          (".gate_up_proj", ".gate_proj", 0),
          (".gate_up_proj", ".up_proj", 1),
        """
        params = dict(self.named_parameters())

        for name, weight in weights.items():
            if "rotary_emb.inv_freq" in name:
                continue

            # --- Fuse q_proj + k_proj + v_proj → qkv_proj ---
            if ".q_proj.weight" in name:
                target = name.replace(".q_proj.weight", ".qkv_proj.weight")
                q_size = self.config.num_attention_heads * self.config.head_dim
                params[target].data[:q_size].copy_(weight)
                continue
            if ".k_proj.weight" in name:
                target = name.replace(".k_proj.weight", ".qkv_proj.weight")
                q_size = self.config.num_attention_heads * self.config.head_dim
                kv_size = self.config.num_key_value_heads * self.config.head_dim
                params[target].data[q_size : q_size + kv_size].copy_(weight)
                continue
            if ".v_proj.weight" in name:
                target = name.replace(".v_proj.weight", ".qkv_proj.weight")
                q_size = self.config.num_attention_heads * self.config.head_dim
                kv_size = self.config.num_key_value_heads * self.config.head_dim
                params[target].data[q_size + kv_size :].copy_(weight)
                continue

            # --- Fuse gate_proj + up_proj → gate_up_proj ---
            if ".gate_proj.weight" in name:
                target = name.replace(".gate_proj.weight", ".gate_up_proj.weight")
                mid = self.config.intermediate_size
                params[target].data[:mid].copy_(weight)
                continue
            if ".up_proj.weight" in name:
                target = name.replace(".up_proj.weight", ".gate_up_proj.weight")
                mid = self.config.intermediate_size
                params[target].data[mid:].copy_(weight)
                continue

            # --- Direct 1:1 mapping ---
            if name in params:
                params[name].data.copy_(weight)
