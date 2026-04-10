"""
Paged Attention Layer.

Maps to:
  vllm/v1/attention/backend.py            — AttentionBackend 抽象接口
  vllm/v1/attention/backends/flash_attn.py — FlashAttentionImpl.forward()

vLLM 有一套复杂的 backend 抽象（FlashAttention, FlashInfer, PallasAttention…），
支持 FP8 KV cache, cascade attention, sliding window, MLA 等。
我们只实现最核心的路径：FlashAttention + 分页 KV cache。

=== 核心概念 ===

1. KV Cache 布局 (vLLM flash_attn backend):
   kv_cache shape: [2, num_blocks, block_size , num_kv_heads, head_dim]
                    ^                ^
                    K和V两份          每个block存block_size个token的KV

   unbind(0) 后:
     key_cache:   [num_blocks, block_size, num_kv_heads, head_dim]
     value_cache: [num_blocks, block_size, num_kv_heads, head_dim]

2. slot_mapping — 告诉我们"新 token 的 KV 写到哪里":
   slot = physical_block_idx * block_size + offset_within_block
   例如 block_size=16, slot=35 → block 2, offset 3

3. block_tables — 告诉 attention kernel"去哪里读历史 KV":
   block_tables[seq_i] = [block_5, block_2, block_9, ...]
   seq_i 的第 0~15 个 token 的 KV 在 block_5
   seq_i 的第 16~31 个 token 的 KV 在 block_2
   ...

4. 两步操作:
   Step A: write_kv_cache — 把本步新 K/V 写入 cache (用 slot_mapping)
   Step B: flash_attn — Q 读取 cache 中的全部 K/V 计算 attention (用 block_tables)

   vLLM 对应:
     Step A → FlashAttentionImpl.do_kv_cache_update()
              内部调用 reshape_and_cache_flash() CUDA kernel
     Step B → FlashAttentionImpl.forward()
              内部调用 flash_attn_varlen_func(block_table=...)

=== flash_attn_varlen_func 关键参数 ===

  q:            [total_q_tokens, num_heads, head_dim]
  k, v:         key_cache / value_cache (paged, 通过 block_table 寻址)
  cu_seqlens_q: [batch_size + 1], Q 的累积序列长度
                例如 3 个请求 query 长度 [5, 1, 1] → [0, 5, 6, 7]
  seqused_k:    [batch_size], 每个请求 KV cache 中的有效 token 数
  max_seqlen_q: 最大 query 长度
  max_seqlen_k: 最大 KV 长度
  block_table:  [batch_size, max_num_blocks], paged KV 的 block 索引
  causal:       True (decoder-only 模型是 causal attention)

  flash-attn >= 2.6 直接支持 block_table 参数，一个函数搞定
  prefill 和 decode 的区别仅在于 query 长度不同:
    prefill: query 长度 = prompt 长度 (多个 token)
    decode:  query 长度 = 1 (只有新生成的 token)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from flash_attn import flash_attn_varlen_func


# ---------------------------------------------------------------------------
# AttentionMetadata → vllm/v1/attention/backend.py::CommonAttentionMetadata
#
# vLLM 的 CommonAttentionMetadata 有 20+ 字段（包括 cascade attention,
# DCP, encoder_seq_lens, logits_indices, is_prefilling 等）。
# 我们只保留 flash_attn_varlen_func 必需的字段。
#
# 由 GPUModelRunner._prepare_inputs() 在每个 step 构建。
# ---------------------------------------------------------------------------

@dataclass
class AttentionMetadata:
    """Everything the attention kernel needs about the current batch."""

    # --- flash_attn_varlen_func 必需参数 ---

    # Q 的累积序列长度: [batch_size + 1]
    # 例如 3 个请求 query 长度 [5, 1, 1] → [0, 5, 6, 7]
    # 含义: 请求 0 的 Q 是 q[0:5], 请求 1 的 Q 是 q[5:6], ...
    query_start_loc: torch.Tensor      # [batch_size + 1], int32

    # KV cache 中每个请求的有效 token 数: [batch_size]
    # 含义: 请求 i 需要 attend 到 seq_lens[i] 个历史 token
    seq_lens: torch.Tensor             # [batch_size], int32

    max_query_len: int                 # 本 batch 最大 query 长度
    max_seq_len: int                   # 本 batch 最大 KV 长度

    # --- 分页 KV cache 参数 ---

    # 每个请求的 block table: [batch_size, max_num_blocks]
    # block_tables[i, j] = 请求 i 的第 j 个逻辑 block 对应的物理 block 编号
    block_tables: torch.Tensor         # [batch_size, max_num_blocks], int32

    # 每个新 token 在 KV cache 中的写入位置: [num_new_tokens]
    # slot = physical_block_idx * block_size + offset_within_block
    slot_mapping: torch.Tensor         # [num_new_tokens], int64

    num_actual_tokens: int = 0         # 本 step 实际处理的 token 总数


# ---------------------------------------------------------------------------
# Attention → vllm/v1/attention/backends/flash_attn.py::FlashAttentionImpl
#
# vLLM 的实现额外处理: FP8 KV quantization, cascade attention,
# sliding window, decode context parallelism, alibi slopes, logit softcap...
# 我们只实现核心路径。
#
# 关键洞察: vLLM v1 + flash-attn >= 2.6 后，prefill 和 decode 用
# **同一个** flash_attn_varlen_func 调用，区别仅在于 query 长度。
# 不再需要 v0 那样分开处理 prefill / decode 两个 codepath。
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """
    Paged attention layer used inside each LlamaDecoderLayer.

    每一层 transformer 都有一个 Attention 实例。
    模型调用: attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
    """

    def __init__(self, num_heads: int, head_dim: int, num_kv_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.scale = head_dim ** -0.5

    def forward(
        self,
        q: torch.Tensor,            # [num_tokens, num_heads * head_dim]
        k: torch.Tensor,            # [num_tokens, num_kv_heads * head_dim]
        v: torch.Tensor,            # [num_tokens, num_kv_heads * head_dim]
        kv_cache: torch.Tensor,     # [2, num_blocks, block_size, num_kv_heads, head_dim]
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """
        Returns: [num_tokens, num_heads * head_dim]

        Two-step process:
          A) Write new K/V → paged cache (at slot_mapping positions)
          B) Compute attention: Q reads ALL K/V from cache (via block_tables)
        """
        num_tokens = q.shape[0]

        # Reshape from [tokens, heads*dim] → [tokens, heads, dim]
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(num_tokens, self.num_kv_heads, self.head_dim)

        # ============================================================
        # Step A: Write new K/V to paged cache
        #
        # vLLM 对应: FlashAttentionImpl.do_kv_cache_update()
        #   内部调用 reshape_and_cache_flash() 自定义 CUDA kernel
        #   我们用纯 PyTorch 实现同样的逻辑
        #
        # kv_cache: [2, num_blocks, block_size, num_kv_heads, head_dim]
        #   kv_cache[0] = key_cache
        #   kv_cache[1] = value_cache
        #
        # slot_mapping: [num_tokens], 每个值 = block_idx * block_size + offset
        # 把 cache reshape 成 flat view 就可以用 slot_mapping 直接索引
        # ============================================================
        key_cache = kv_cache[0]    # [num_blocks, block_size, num_kv_heads, head_dim]
        value_cache = kv_cache[1]

        num_blocks, block_size, num_kv_heads, head_dim = key_cache.shape
        # Flatten block 维度: [num_blocks * block_size, num_kv_heads, head_dim]
        key_cache_flat = key_cache.view(-1, num_kv_heads, head_dim)
        value_cache_flat = value_cache.view(-1, num_kv_heads, head_dim)

        # 用 slot_mapping 作为索引，把新的 K/V scatter 写入 cache
        slot_mapping = attn_metadata.slot_mapping
        key_cache_flat[slot_mapping] = k
        value_cache_flat[slot_mapping] = v

        # ============================================================
        # Step B: Compute attention via flash_attn_varlen_func
        #
        # vLLM 对应: FlashAttentionImpl.forward() 里的核心调用
        #
        # flash_attn_varlen_func 在 flash-attn >= 2.6 支持 block_table 参数:
        #   - k, v 传的是整个 paged cache（不需要 gather）
        #   - kernel 内部根据 block_table 寻址读取正确的 K/V
        #   - prefill 和 decode 统一处理，区别仅在 query 长度
        #
        # 参数含义:
        #   q:            本步所有请求的 query tokens
        #   k, v:         整个 paged KV cache
        #   cu_seqlens_q: Q 的累积长度 (哪些 Q token 属于哪个请求)
        #   seqused_k:    每个请求 KV cache 有效长度
        #   block_table:  block_tables[i] = 请求 i 的物理 block 列表
        #   causal=True:  decoder-only 模型用 causal mask
        # ============================================================
        output = flash_attn_varlen_func(
            q=q,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=attn_metadata.query_start_loc,
            cu_seqlens_k=None,       # 用 seqused_k 替代
            max_seqlen_q=attn_metadata.max_query_len,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=True,
            block_table=attn_metadata.block_tables,
            seqused_k=attn_metadata.seq_lens,
        )

        # output: [num_tokens, num_heads, head_dim] → [num_tokens, num_heads * head_dim]
        return output.view(num_tokens, self.num_heads * self.head_dim)
