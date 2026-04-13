"""
GPU Worker — 单 GPU 的设备管理者。

Maps to: vllm/v1/worker/gpu_worker.py (1000+ lines → ~60 here)

Worker 是 GPUModelRunner 的薄包装，职责分离:
  Worker:      设备管理 (set_device, 初始化环境)
  ModelRunner: 模型计算 (load, prepare, forward, sample)

为什么要分开？因为 Phase 3 分布式时:
  - 每个 GPU 有一个独立的 Worker 进程
  - Executor 通过 IPC/RPC 给每个 Worker 发指令
  - Worker 管自己的设备环境 + 进程通信
  - ModelRunner 只管模型计算，不关心自己在哪个进程

vLLM Worker 的完整职责:
  - init_device(): CUDA device + NCCL 分布式初始化
  - load_model(): 委托给 model_runner
  - determine_available_memory(): profile GPU 显存
  - compile_or_warm_up_model(): 预热 CUDAGraph
  - execute_model(): PP 通信 + 委托给 model_runner
  - get_kv_cache_spec(): KV cache 配置信息

Phase 1 只保留核心 3 个方法。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from trim.core.schema import ModelRunnerOutput, SchedulerOutput
from trim.worker.gpu_model_runner import GPUModelRunner

if TYPE_CHECKING:
    from trim.config import CacheConfig, ModelConfig


class GPUWorker:
    """
    Per-GPU worker process.

    vLLM 对应: vllm/v1/worker/gpu_worker.py::Worker
    vLLM 继承自 WorkerBase，支持 local_rank, distributed_init_method 等。
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda:0")
        self.model_config = model_config
        self.cache_config = cache_config
        self.model_runner = GPUModelRunner(model_config, cache_config, self.device)

    def init_device(self) -> int:
        """
        初始化设备 + 加载模型 + 分配 KV cache。

        vLLM 对应: Worker.init_device() + Worker.load_model() +
                   Worker.determine_available_memory() + Worker.initialize_cache()
        vLLM 把这些拆成独立方法由 Executor 按顺序调用。
        Phase 1 我们合并成一步。

        Returns: num_gpu_blocks (KV cache 能分配多少块)
        """
        torch.cuda.set_device(self.device)

        # vLLM 对应: Worker.load_model()
        self.model_runner.load_model()

        # vLLM 对应: Worker.determine_available_memory() + initialize_cache()
        num_gpu_blocks = self.model_runner.profile_and_init_kv_cache()

        return num_gpu_blocks

    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """
        执行一次 forward pass。

        vLLM 对应: Worker.execute_model()
        vLLM 额外处理: Pipeline Parallelism 的 intermediate_tensors 发送/接收,
          SP (sequence parallelism) 的 all-gather,
          DP (data parallelism) 的 batch coordination。
        Phase 1 直接委托。
        """
        return self.model_runner.execute_model(scheduler_output)
