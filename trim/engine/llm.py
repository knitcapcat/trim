"""
LLM — 用户面 API。

Maps to:
  vllm/entrypoints/llm.py          — LLM (用户调用的入口)
  vllm/v1/engine/llm_engine.py     — LLMEngine (内部引擎)

vLLM 分了两层:
  LLM (entrypoints/llm.py):
    最外层 API，处理 generate/chat/encode 等用户接口
    内部持有 LLMEngine

  LLMEngine (v1/engine/llm_engine.py):
    持有 InputProcessor (PromptType → EngineCoreRequest)
    持有 OutputProcessor (EngineCoreOutput → RequestOutput, detokenize)
    持有 EngineCoreClient (可以是同进程或 ZMQ 后台进程)

我们合并成一个 LLM 类，因为 Phase 1 不需要多进程和复杂的 I/O 处理。

=== 使用方式 ===

  from trim.engine.llm import LLM
  from trim.config import SamplingParams

  llm = LLM(model="meta-llama/Llama-3.2-1B")

  outputs = llm.generate(
      ["The capital of France is", "Hello, world!"],
      SamplingParams(max_tokens=128),
  )

  for output in outputs:
      print(output.output_text)
"""

from __future__ import annotations

import uuid
from typing import Optional

from transformers import AutoTokenizer

from trim.config import ModelConfig, SamplingParams, TrimConfig
from trim.core.schema import Request, RequestOutput
from trim.engine.core import EngineCore


class LLM:
    """
    Synchronous LLM inference API.

    vLLM 对应: vllm/entrypoints/llm.py::LLM
    """

    def __init__(
        self,
        model: str,
        dtype: str = "auto",
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        block_size: int = 16,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int = 2048,
    ) -> None:
        """
        vLLM 对应: LLM.__init__() → EngineArgs → VllmConfig → LLMEngine
        vLLM 有一套 EngineArgs 系统把 CLI 参数转成 VllmConfig。
        我们直接构造 TrimConfig。
        """
        self.model_name = model

        # 加载 tokenizer
        # vLLM 对应: Renderer 内部的 tokenizer 管理
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 构造配置
        config = TrimConfig(
            model=ModelConfig(model=model, dtype=dtype, max_model_len=max_model_len),
        )
        config.cache.gpu_memory_utilization = gpu_memory_utilization
        config.cache.block_size = block_size
        config.scheduler.max_num_seqs = max_num_seqs
        config.scheduler.max_num_batched_tokens = max_num_batched_tokens

        # 初始化引擎
        # vLLM 对应: LLMEngine.__init__() → EngineCoreClient.make_client()
        self.engine = EngineCore(config)
        self.engine.set_eos_token_id(self.tokenizer.eos_token_id)

    def generate(
        self,
        prompts: list[str] | str,
        sampling_params: Optional[SamplingParams] = None,
    ) -> list[RequestOutput]:
        """
        为一组 prompt 生成补全。

        vLLM 对应: LLM.generate()
        vLLM 的版本支持: PromptType (str/TokensPrompt/TextPrompt),
          多模态输入, LoRA, use_tqdm 进度条, n>1 并行采样等。

        Steps:
          1. Tokenize prompts
          2. 创建 Request 对象
          3. 添加到 EngineCore
          4. 循环 step() 直到全部完成
          5. Detokenize 输出
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        if sampling_params is None:
            sampling_params = SamplingParams()

        # 1. Tokenize + 创建 Request
        # vLLM 对应: InputProcessor.process_inputs()
        for prompt in prompts:
            request_id = str(uuid.uuid4())
            prompt_token_ids = self.tokenizer.encode(prompt)

            request = Request(
                request_id=request_id,
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
            )
            self.engine.add_request(request)

        # 2. 循环 step() 直到完成
        # vLLM 对应: LLM._run_engine() 里的 while has_unfinished
        all_outputs: list[RequestOutput] = []
        while self.engine.has_unfinished_requests():
            step_outputs = self.engine.step()
            all_outputs.extend(step_outputs)

        # 3. Detokenize
        # vLLM 对应: OutputProcessor 里的 Detokenizer
        for output in all_outputs:
            output.output_text = self.tokenizer.decode(
                output.output_token_ids, skip_special_tokens=True
            )

        return all_outputs
