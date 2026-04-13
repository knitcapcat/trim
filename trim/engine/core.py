"""
EngineCore — 推理引擎的主循环。

Maps to: vllm/v1/engine/core.py (2000+ lines → ~120 here)

EngineCore 是整个引擎的编排者 (orchestrator)。
每次 step() 执行一轮: schedule → execute → update。

=== vLLM 的两种 EngineCore ===

  EngineCore:      同进程版本 (用于离线推理, 测试)
  EngineCoreProc:  独立进程版本 (用于在线 serving)
    - 跑在 background process 里
    - 通过 ZMQ socket 和 LLMEngine 通信
    - 这样 scheduler 和 model execution 可以并行

我们只实现同进程版本。

=== vLLM EngineCore 的完整职责 ===

  - 持有 Executor (管理多个 Worker) 和 Scheduler
  - step(): schedule → execute → update
  - add_request(): 验证请求 → 加入 scheduler
  - abort_request(): 中止请求
  - get_grammar_bitmask(): structured output 支持
  - profile(): 性能分析
  - error handling: 记录 scheduler_output 用于 debug

=== 初始化链路 ===

  EngineCore.__init__()
    → GPUWorker.init_device()
      → GPUModelRunner.load_model()         # 加载模型权重
      → GPUModelRunner.profile_and_init_kv_cache()  # 分配 KV cache
    → KVCacheManager(num_gpu_blocks)         # 创建 block 记账系统
    → Scheduler(kv_cache_manager)            # 创建调度器
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from trim.core.kv_cache_manager import KVCacheManager
from trim.core.scheduler import Scheduler
from trim.core.schema import Request, RequestOutput
from trim.worker.gpu_worker import GPUWorker

if TYPE_CHECKING:
    from trim.config import TrimConfig


class EngineCore:
    """
    Inner loop of trim's inference engine.

    vLLM 对应: vllm/v1/engine/core.py::EngineCore
    """

    def __init__(self, config: TrimConfig) -> None:
        self.config = config

        # --- 自底向上初始化 ---

        # 1. 初始化 GPU: 加载模型 + 分配 KV cache 显存
        # vLLM 对应: Executor → Worker → ModelRunner
        self.worker = GPUWorker(config.model, config.cache)
        num_gpu_blocks = self.worker.init_device()

        # 2. 创建 KV cache 记账系统 (CPU 端)
        # vLLM 对应: EngineCore.__init__ 里创建 KVCacheManager
        self.kv_cache_manager = KVCacheManager(
            num_gpu_blocks=num_gpu_blocks,
            block_size=config.cache.block_size,
        )

        # 3. 创建调度器
        self.scheduler = Scheduler(config.scheduler, self.kv_cache_manager)

    def add_request(self, request: Request) -> None:
        """
        添加请求到调度器。

        vLLM 对应: EngineCore.add_request()
        vLLM 额外做: 请求验证, pooling_params 检查,
          kv_transfer_params 处理, 请求事件记录等。
        """
        self.scheduler.add_request(request)

    def step(self) -> list[RequestOutput]:
        """
        一轮主循环: schedule → execute → update。

        vLLM 对应: EngineCore.step()
        vLLM 的版本:
          scheduler_output = self.scheduler.schedule()
          future = self.model_executor.execute_model(scheduler_output, non_block=True)
          grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
          model_output = future.result()
          outputs = self.scheduler.update_from_output(scheduler_output, model_output)

        注意 vLLM 用 future.result() 支持异步执行:
          execute_model 提交到 GPU 后立刻返回 future,
          CPU 可以先去做 grammar_bitmask 计算,
          等真正需要结果时再 future.result() 阻塞获取。

        我们是同步版本: execute → 等结果 → update。
        """
        # 1. Schedule: 决定这一步处理什么
        scheduler_output = self.scheduler.schedule()

        if scheduler_output.is_empty:
            return []

        # 2. Execute: 在 GPU 上跑模型
        model_output = self.worker.execute_model(scheduler_output)

        # 3. Update: 更新请求状态，返回完成的请求
        finished_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        return finished_outputs

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished_requests

    def set_eos_token_id(self, eos_token_id: int) -> None:
        """设置 EOS token id (由 LLM 从 tokenizer 获取后传入)。"""
        self.scheduler.eos_token_id = eos_token_id
