"""
GPU Model Runner — SchedulerOutput → GPU tensor → forward → sample.

Maps to: vllm/v1/worker/gpu_model_runner.py (7000 lines → ~250 here)

这是整个 Phase 1 中最复杂的文件。负责模型在 GPU 上运行的逻辑，是 CPU 世界和 GPU 世界的桥梁。

=== 职责 ===

  1. load_model():     加载 HF 权重 → trim 模型 + 分配 KV cache 显存
  2. _prepare_inputs(): SchedulerOutput (CPU, 逻辑) → GPU tensors (物理)
  3. execute_model():   prepare → model.forward → compute_logits → sample

=== vLLM 的复杂性来源 (我们跳过的) ===

  - InputBatch: 持久化 batch 状态，避免每步重新构造 (优化)
  - CUDAGraph: capture/replay 固定 shape 的 forward pass (Phase 2)
  - Spec decode: 投机解码的 draft/verify 逻辑
  - LoRA: 动态加载 adapter 权重
  - Multimodal: encoder forward + encoder cache
  - Pipeline parallelism: intermediate_tensors 传递
  - Async scheduling: 异步 GPU→CPU tensor copy
  - Persistent batch: CPU/GPU 双缓冲 + 增量更新

=== _prepare_inputs 核心翻译逻辑 ===

  SchedulerOutput (CPU)              →  GPU Tensors
  ─────────────────────                 ──────────
  req-0: token_ids=[101,2003,...]      input_ids:  [101, 2003, ..., 3681]
  req-1: token_ids=[3681]              positions:  [0, 1, ..., 20]
  req-0: block_table=[5,2]            block_tables: [[5,2,-1], [0,7,-1]]
  req-0: num_computed_tokens=0         query_start_loc: [0, 5, 6]
  req-1: num_computed_tokens=20        seq_lens: [5, 21]
                                       slot_mapping: [80,81,...,84, 36]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from transformers import AutoConfig

from trim.attention.attention import AttentionMetadata
from trim.core.schema import ModelRunnerOutput, SchedulerOutput
from trim.model.llama import LlamaForCausalLM
from trim.sample.sampler import Sampler, SamplingMetadata

if TYPE_CHECKING:
    from trim.config import CacheConfig, ModelConfig


class GPUModelRunner:
    """
    GPU-side model execution engine.

    vLLM 对应: vllm/v1/worker/gpu_model_runner.py::GPUModelRunner
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        device: torch.device,
    ) -> None:
        self.model_config = model_config
        self.cache_config = cache_config
        self.device = device
        self.dtype = model_config.get_dtype()

        self.model: LlamaForCausalLM | None = None
        self.sampler = Sampler()
        self.kv_caches: list[torch.Tensor] = []

    def load_model(self) -> None:
        """
        加载模型权重 + 填充 ModelConfig + 分配 KV cache 显存。

        vLLM 对应: GPUModelRunner.load_model()
        vLLM 使用 model_loader 框架支持多种 format (safetensors, gguf, dummy...)
        我们直接用 HuggingFace 的 from_pretrained 加载，然后手动映射权重。
        """
        # 1. 从 HuggingFace config 填充 ModelConfig 的结构参数
        hf_config = AutoConfig.from_pretrained(self.model_config.model)
        cfg = self.model_config
        cfg.vocab_size = hf_config.vocab_size
        cfg.hidden_size = hf_config.hidden_size
        cfg.num_hidden_layers = hf_config.num_hidden_layers
        cfg.num_attention_heads = hf_config.num_attention_heads
        cfg.num_key_value_heads = getattr(
            hf_config, "num_key_value_heads", hf_config.num_attention_heads
        )
        cfg.intermediate_size = hf_config.intermediate_size
        cfg.head_dim = cfg.hidden_size // cfg.num_attention_heads
        cfg.max_position_embeddings = hf_config.max_position_embeddings
        cfg.rms_norm_eps = hf_config.rms_norm_eps
        cfg.rope_theta = getattr(hf_config, "rope_theta", 500000.0)
        if cfg.max_model_len is None:
            cfg.max_model_len = cfg.max_position_embeddings

        # 2. 创建 trim 模型 (随机权重) 并移到 GPU
        self.model = LlamaForCausalLM(cfg).to(self.dtype).to(self.device)

        # 3. 流式加载 HuggingFace 权重 (逐 tensor, 不加载整个 HF 模型)
        # vLLM 对应: model_loader + AutoWeightsLoader
        from trim.model.model_loader import load_weights_from_hf
        load_weights_from_hf(self.model, cfg)

        # 4. 处理 tie_word_embeddings (lm_head 和 embed_tokens 共享权重)
        if hf_config.tie_word_embeddings:
            self.model.lm_head.weight = self.model.model.embed_tokens.weight

    def profile_and_init_kv_cache(self) -> int:
        """
        通过 dummy forward 精确测量显存占用，再分配 KV cache。

        vLLM 对应:
          Worker.determine_available_memory() → profile_run()
          Worker.initialize_cache()

        流程:
          1. 清空 CUDA 显存统计
          2. 跑一次 dummy forward (最大 batch，暴露所有隐藏分配)
          3. 测量峰值显存 → 计算剩余
          4. 剩余 × gpu_memory_utilization → KV cache 预算
          5. 分配 KV cache tensor

        为什么要 dummy forward？因为实际占用远超 model.parameters():
          model weights:     7.0 GB
          CUDA context:     ~0.5 GB  ← 运行时本身
          cuBLAS workspace: ~0.3 GB  ← matmul 临时 buffer
          activation peaks: ~1.0 GB  ← forward 中间 tensor
          fragmentation:    ~0.2 GB  ← allocator 碎片
          actual:           ~9.0 GB  ← 比参数大 2GB!

        Returns: num_gpu_blocks
        """
        cfg = self.model_config
        cache_cfg = self.cache_config

        # --- Step 1: Dummy forward profiling ---
        # vLLM 对应: GPUModelRunner.profile_run()
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.empty_cache()

        self._profile_run()

        # --- Step 2: 测量峰值，计算可用显存 ---
        peak_memory = torch.cuda.max_memory_allocated(self.device)
        total_memory = torch.cuda.get_device_properties(self.device).total_mem
        available_memory = total_memory - peak_memory

        # 清理 profiling 分配的中间 tensor
        torch.cuda.empty_cache()

        # --- Step 3: 计算 num_gpu_blocks ---
        dtype_size = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
        bytes_per_block_per_layer = (
            2 * cache_cfg.block_size * cfg.num_key_value_heads * cfg.head_dim * dtype_size
        )
        bytes_per_block = bytes_per_block_per_layer * cfg.num_hidden_layers

        kv_cache_memory = int(available_memory * cache_cfg.gpu_memory_utilization)
        num_gpu_blocks = max(kv_cache_memory // bytes_per_block, 1)

        # --- Step 4: 分配 KV cache tensor ---
        # vLLM flash_attn layout: [2, num_blocks, block_size, num_kv_heads, head_dim]
        self.kv_caches = []
        for _ in range(cfg.num_hidden_layers):
            kv_cache = torch.zeros(
                2,
                num_gpu_blocks,
                cache_cfg.block_size,
                cfg.num_key_value_heads,
                cfg.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            self.kv_caches.append(kv_cache)

        cache_cfg.num_gpu_blocks = num_gpu_blocks
        return num_gpu_blocks

    @torch.inference_mode()
    def _profile_run(self) -> None:
        """
        跑一次 dummy forward 来暴露所有隐藏的 CUDA 内存分配。

        vLLM 对应: GPUModelRunner.profile_run()
        vLLM 用 max_num_batched_tokens 构造最大 batch 来测量峰值。
        我们用一个小 batch 来近似（不如 vLLM 精确但足够用）。
        """
        cfg = self.model_config
        dummy_batch_size = min(4, cfg.max_model_len or 128)

        dummy_input_ids = torch.zeros(
            dummy_batch_size, dtype=torch.long, device=self.device
        )
        dummy_positions = torch.arange(
            dummy_batch_size, dtype=torch.long, device=self.device
        )

        # 临时构造 dummy KV cache (每层 1 block, forward 完就删)
        dummy_kv_caches = []
        for _ in range(cfg.num_hidden_layers):
            dummy_kv = torch.zeros(
                2, 1, self.cache_config.block_size,
                cfg.num_key_value_heads, cfg.head_dim,
                dtype=self.dtype, device=self.device,
            )
            dummy_kv_caches.append(dummy_kv)

        dummy_attn_metadata = AttentionMetadata(
            query_start_loc=torch.arange(
                dummy_batch_size + 1, dtype=torch.int32, device=self.device
            ),
            seq_lens=torch.ones(
                dummy_batch_size, dtype=torch.int32, device=self.device
            ),
            max_query_len=1,
            max_seq_len=1,
            block_tables=torch.zeros(
                dummy_batch_size, 1, dtype=torch.int32, device=self.device
            ),
            slot_mapping=torch.zeros(
                dummy_batch_size, dtype=torch.int64, device=self.device
            ),
            num_actual_tokens=dummy_batch_size,
        )

        # Forward pass → 触发所有 CUDA 内存分配
        hidden = self.model(
            dummy_input_ids, dummy_positions, dummy_kv_caches, dummy_attn_metadata
        )
        logits = self.model.compute_logits(hidden)

        # 清理 dummy tensors
        del logits, hidden, dummy_kv_caches, dummy_attn_metadata
        del dummy_input_ids, dummy_positions
        torch.cuda.synchronize(self.device)

    def _prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
    ) -> tuple[torch.Tensor, torch.Tensor, AttentionMetadata, list[int], SamplingMetadata]:
        """
        把 SchedulerOutput (CPU, 逻辑) 翻译成 GPU tensor (物理)。

        vLLM 对应: GPUModelRunner._prepare_inputs()
        vLLM 版本用 persistent InputBatch + numpy 预计算 + 异步 CPU→GPU copy
        做了大量优化。我们用简单直接的方式。

        Returns: (input_ids, positions, attn_metadata, logits_indices, sampling_metadata)
        """
        all_token_ids: list[int] = []
        all_positions: list[int] = []
        all_slot_mapping: list[int] = []
        query_lens: list[int] = []          # 每个请求本步的 query 长度
        seq_lens_list: list[int] = []       # 每个请求 KV cache 中的总长度
        block_tables_list: list[list[int]] = []
        sampling_params_list = []
        logits_indices: list[int] = []      # 每个请求最后一个 token 在 batch 中的位置

        token_offset = 0
        for req_data in scheduler_output.scheduled_requests:
            # Token ids for this request this step
            all_token_ids.extend(req_data.token_ids)

            # Positions: num_computed_tokens + 0, 1, 2, ...
            for i in range(req_data.num_new_tokens):
                all_positions.append(req_data.num_computed_tokens + i)

            # Slot mapping: where to write KV for each new token
            block_size = self.cache_config.block_size
            for i in range(req_data.num_new_tokens):
                token_pos = req_data.num_computed_tokens + i
                logical_block = token_pos // block_size
                offset = token_pos % block_size
                if logical_block < len(req_data.block_table):
                    physical_block = req_data.block_table[logical_block]
                    all_slot_mapping.append(physical_block * block_size + offset)

            # Query length: how many tokens this request queries with
            query_lens.append(req_data.num_new_tokens)

            # Seq length: total KV cache tokens after this step
            seq_len = req_data.num_computed_tokens + req_data.num_new_tokens
            seq_lens_list.append(seq_len)

            # Block table for this request
            block_tables_list.append(req_data.block_table)

            # Logits index: position of this request's last token in the batch
            # 只对最后一个 token 做 logits → sample
            logits_indices.append(token_offset + req_data.num_new_tokens - 1)
            token_offset += req_data.num_new_tokens

            sampling_params_list.append(req_data.sampling_params)

        # --- 构造 GPU tensors ---
        device = self.device

        input_ids = torch.tensor(all_token_ids, dtype=torch.long, device=device)
        positions = torch.tensor(all_positions, dtype=torch.long, device=device)

        # query_start_loc: cumulative query lengths
        # e.g., query_lens=[5, 1, 1] → [0, 5, 6, 7]
        cu_query = [0]
        for ql in query_lens:
            cu_query.append(cu_query[-1] + ql)
        query_start_loc = torch.tensor(cu_query, dtype=torch.int32, device=device)

        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)

        # Pad block_tables to same width
        max_blocks = max((len(bt) for bt in block_tables_list), default=0)
        padded_block_tables = [
            bt + [0] * (max_blocks - len(bt)) for bt in block_tables_list
        ]
        block_tables = torch.tensor(padded_block_tables, dtype=torch.int32, device=device)

        slot_mapping = torch.tensor(all_slot_mapping, dtype=torch.int64, device=device)

        attn_metadata = AttentionMetadata(
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            max_query_len=max(query_lens) if query_lens else 0,
            max_seq_len=max(seq_lens_list) if seq_lens_list else 0,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            num_actual_tokens=len(all_token_ids),
        )

        sampling_metadata = SamplingMetadata.from_sampling_params(
            sampling_params_list, device
        )

        return input_ids, positions, attn_metadata, logits_indices, sampling_metadata

    @torch.inference_mode()
    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """
        Full forward pass: prepare → model → logits → sample.

        vLLM 对应: GPUModelRunner.execute_model()
        vLLM 额外处理: CUDAGraph dispatch, pipeline parallelism,
          encoder forward, spec decode, async output copy, KV connector,
          micro-batching 等。

        Returns: ModelRunnerOutput with sampled token per request
        """
        if scheduler_output.is_empty:
            return ModelRunnerOutput(req_ids=[], sampled_token_ids=[])

        # 1. Prepare GPU tensors from scheduler output
        (input_ids, positions, attn_metadata,
         logits_indices, sampling_metadata) = self._prepare_inputs(scheduler_output)

        # 2. Model forward: embedding → N transformer layers → norm
        # vLLM 对应: model(input_ids, positions, intermediate_tensors)
        # vLLM 通过 forward_context 传 kv_cache 和 attn_metadata
        # 我们显式传参
        hidden_states = self.model(input_ids, positions, self.kv_caches, attn_metadata)

        # 3. Compute logits: 只取每个请求最后一个 token 的 hidden_states
        # vLLM 对应: model.compute_logits() with logits_indices
        logits_indices_tensor = torch.tensor(
            logits_indices, dtype=torch.long, device=self.device
        )
        last_hidden = hidden_states[logits_indices_tensor]  # [num_reqs, hidden_size]
        logits = self.model.compute_logits(last_hidden)     # [num_reqs, vocab_size]

        # 4. Sample next tokens
        # vLLM 对应: self.sampler(logits, sampling_metadata)
        sampled_token_ids = self.sampler(logits, sampling_metadata)

        # 5. Build output
        req_ids = [r.request_id for r in scheduler_output.scheduled_requests]
        return ModelRunnerOutput(
            req_ids=req_ids,
            sampled_token_ids=sampled_token_ids.tolist(),
        )
