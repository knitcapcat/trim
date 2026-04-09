"""
Configurations for trim project.

"""

from dataclasses import dataclass, field
from typing import Optional

import torch

@dataclass
class ModelConfig:
    """Model-level configuration."""

    # --- 用户指定 ---
    model: str                                 # HF model name or path
    dtype: str = "auto"                        # "auto", "float16", "bfloat16"
    max_model_len: Optional[int] = None        # 最大序列长度（prompt + output）
    seed: int = 0                              # 随机种子

    # --- 从 HuggingFace config 自动填充 ---
    # 在 vLLM 中这些是通过 __post_init__ 调用 get_config() 从
    # transformers.PretrainedConfig 中读取的，我们先留空，
    # 在 load_model() 时填充
    vocab_size: int = 0
    hidden_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0               # GQA: < num_attention_heads
    intermediate_size: int = 0                 # MLP 中间维度
    head_dim: int = 0                          # hidden_size // num_attention_heads
    max_position_embeddings: int = 0           # 模型支持的最大位置
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0               # RoPE base frequency

    def get_dtype(self) -> torch.dtype:
        """
        vLLM 原版: ModelConfig._get_and_verify_dtype()
        "auto" → FP16 for FP32/FP16 models, BF16 for BF16 models
        """
        if self.dtype == "auto" or self.dtype == "float16" or self.dtype == "half":
            return torch.float16
        elif self.dtype == "bfloat16":
            return torch.bfloat16
        elif self.dtype == "float32":
            return torch.float32
        return torch.float16


# ---------------------------------------------------------------------------
# CacheConfig → vllm/config/cache.py::CacheConfig
#
# vLLM 原版有: block_size, gpu_memory_utilization, cache_dtype,
#   prefix_caching, mamba_cache_*, num_gpu_blocks, num_cpu_blocks …
# 我们只保留分页 KV cache 核心字段。
# ---------------------------------------------------------------------------

@dataclass
class CacheConfig:
    """KV cache configuration (paged memory management)."""

    block_size: int = 16                       # 每个物理 block 存多少 token 的 KV
    gpu_memory_utilization: float = 0.9        # 给 KV cache 预留的 GPU 显存比例
    # 运行时通过 profiling 确定，不是用户指定的
    num_gpu_blocks: Optional[int] = None


# ---------------------------------------------------------------------------
# SchedulerConfig → vllm/config/scheduler.py::SchedulerConfig
#
# vLLM 原版有: chunked_prefill, partial_prefill, encoder_decoder,
#   async_scheduling, scheduler_cls, policy …
# 我们只保留连续批处理的核心控制参数。
# ---------------------------------------------------------------------------

@dataclass
class SchedulerConfig:
    """Scheduler configuration (continuous batching)."""

    max_num_seqs: int = 256                    # 一个 batch 最多多少个序列
    max_num_batched_tokens: int = 2048         # 一个 batch 最多多少个 token
    # Phase 1 先不实现 chunked prefill，整个 prompt 一次性 prefill
    enable_chunked_prefill: bool = False


# ---------------------------------------------------------------------------
# SamplingParams → vllm/sampling_params.py::SamplingParams
#
# vLLM 原版用 msgspec.Struct，有 30+ 字段（logprobs, penalties,
#   structured output, stop strings …）
# 我们只保留最基础的采样参数，每个 request 独立配置。
# ---------------------------------------------------------------------------

@dataclass
class SamplingParams:
    """Per-request sampling parameters."""

    temperature: float = 1.0                   # 0 = greedy
    top_p: float = 1.0                         # nucleus sampling threshold
    top_k: int = -1                            # -1 = disabled
    max_tokens: int = 256                      # 最大生成 token 数
    # vLLM 还有: stop, stop_token_ids, ignore_eos, seed,
    #   presence_penalty, frequency_penalty, repetition_penalty …

    @property
    def sampling_type(self) -> str:
        if self.temperature < 1e-5:
            return "greedy"
        return "random"


# ---------------------------------------------------------------------------
# TrimConfig → vllm/config/vllm.py::VllmConfig
#
# vLLM 原版的 VllmConfig 聚合了 15+ 个子 config（ParallelConfig,
#   LoadConfig, SpeculativeConfig, LoRAConfig …）
# 我们只聚合 Phase 1 需要的三个。
# ---------------------------------------------------------------------------

@dataclass
class TrimConfig:
    """Top-level config aggregating all sub-configs."""

    model: ModelConfig = field(default_factory=lambda: ModelConfig(model=""))
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
