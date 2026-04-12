"""
Core data structures that flow between components.

These are the "envelopes" that components use to communicate.
Every component in trim talks to others through these types:

  User prompt
    → Request                    (LLM creates, Scheduler manages)
      → SchedulerOutput          (Scheduler → Worker)
        → ModelRunnerOutput      (Worker → EngineCore)
          → RequestOutput        (EngineCore → User)

References:
  Request          → vllm/v1/request.py           (270 lines → ~60 here)
  SchedulerOutput  → vllm/v1/core/sched/output.py (250 lines → ~40 here)
  ModelRunnerOutput→ vllm/v1/outputs.py            (280 lines → ~10 here)

Key simplifications vs vLLM:
  - vLLM's SchedulerOutput splits requests into "new" (NewRequestData)
    and "cached" (CachedRequestData) to minimize IPC overhead between
    the scheduler process and worker processes. We don't need this
    because trim runs everything in one process (Phase 1).

  - vLLM's ModelRunnerOutput carries logprobs, pooler output, KV
    connector metadata, cudagraph stats, etc. We only need sampled tokens (Phase 1).

  - vLLM's Request has streaming, multi-modal, LoRA, structured output,
    speculative decoding, block hashing, priority, etc. We only keep
    the core generation lifecycle (Phase 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Optional

from trim.config import SamplingParams


# ---------------------------------------------------------------------------
# RequestStatus → vllm/v1/request.py::RequestStatus
# ---------------------------------------------------------------------------

class RequestStatus(IntEnum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED_STOPPED = auto()
    FINISHED_LENGTH_CAPPED = auto()

    @staticmethod
    def is_finished(status: RequestStatus) -> bool:
        return status >= RequestStatus.FINISHED_STOPPED


# ---------------------------------------------------------------------------
# Request → vllm/v1/request.py::Request
#
# 生命周期:
#   WAITING → (prefill) → RUNNING → (decode until done) → FINISHED_*
#
# 关键状态:
#   - prompt_token_ids: 不变，用户输入的 token
#   - output_token_ids: 逐步追加，每个 step 新增一个 sampled token
#   - num_computed_tokens: 已经计算过 KV 的 token 数量
#     第一次 prefill 后 = len(prompt_token_ids)
#     每次 decode 后 += 1
# ---------------------------------------------------------------------------

@dataclass
class Request:
    """A single inference request flowing through the engine."""

    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    status: RequestStatus = RequestStatus.WAITING

    output_token_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def num_tokens(self) -> int:
        """Total tokens = prompt + output (so far)."""
        return self.num_prompt_tokens + self.num_output_tokens

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def num_new_tokens(self) -> int:
        """Tokens that need KV computation in the NEXT step.
        - Prefill: = num_prompt_tokens (all prompt tokens are new)
        - Decode:  = 1 (only the last generated token is new)
        """
        return self.num_tokens - self.num_computed_tokens

    def append_output_token_ids(self, token_id: int) -> None:
        """
        vLLM 对应: Request.append_output_token_ids()
        注意 vLLM 同时更新 _output_token_ids 和 _all_token_ids 两个列表，
        并且触发 block hash 更新（用于 prefix caching）。
        我们只需要追加 token。
        """
        self.output_token_ids.append(token_id)

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)


# ---------------------------------------------------------------------------
# SchedulerOutput → vllm/v1/core/sched/output.py::SchedulerOutput
#
# vLLM 把调度输出分成两部分来优化 IPC:
#   - NewRequestData: 第一次调度的请求，需要发送完整的 prompt_token_ids
#   - CachedRequestData: 之前已发送过的请求，只发增量
#
# trim Phase 1 在同一个进程里，所以我们用一个简单的列表就够了。
# ---------------------------------------------------------------------------

@dataclass
class ScheduledRequestData:
    """Per-request info that the scheduler passes to the worker."""

    request_id: str
    token_ids: list[int]
    num_new_tokens: int
    block_table: list[int]
    num_computed_tokens: int
    is_prefill: bool
    sampling_params: SamplingParams


@dataclass
class SchedulerOutput:
    """Everything the scheduler decides for one step."""

    scheduled_requests: list[ScheduledRequestData]
    total_num_scheduled_tokens: int

    @property
    def num_reqs(self) -> int:
        return len(self.scheduled_requests)

    @property
    def is_empty(self) -> bool:
        return len(self.scheduled_requests) == 0


# ---------------------------------------------------------------------------
# ModelRunnerOutput → vllm/v1/outputs.py::ModelRunnerOutput
# ---------------------------------------------------------------------------

@dataclass
class ModelRunnerOutput:
    """Results of a single model forward pass."""

    req_ids: list[str]
    sampled_token_ids: list[int]


# ---------------------------------------------------------------------------
# RequestOutput — 返回给用户的最终结果
#
# vLLM 的对应物在 vllm/outputs.py::RequestOutput 和
# vllm/outputs.py::CompletionOutput
# ---------------------------------------------------------------------------

@dataclass
class RequestOutput:
    """Final output returned to the user."""

    request_id: str
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    output_text: str = ""
    finished: bool = False
