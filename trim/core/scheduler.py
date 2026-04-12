"""
Continuous Batching Scheduler.

Maps to: vllm/v1/core/sched/scheduler.py (2300 lines → ~200 here)

v1 核心思想: 没有显式的 prefill/decode 阶段。
每个请求只有 num_computed_tokens 和 num_tokens，
scheduler 让 num_computed_tokens 追上 num_tokens。
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from trim.config import SchedulerConfig
from trim.core.kv_cache_manager import KVCacheManager
from trim.core.schema import (
    ModelRunnerOutput,
    Request,
    RequestOutput,
    RequestStatus,
    ScheduledRequestData,
    SchedulerOutput,
)


class Scheduler:
    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        kv_cache_manager: KVCacheManager,
    ) -> None:
        self.config = scheduler_config
        self.kv_cache_manager = kv_cache_manager

        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.requests: dict[str, Request] = {}
        self.eos_token_id: Optional[int] = None

    def add_request(self, request: Request) -> None:
        request.status = RequestStatus.WAITING
        self.waiting.append(request)
        self.requests[request.request_id] = request

    def schedule(self) -> SchedulerOutput:
        scheduled_requests: list[ScheduledRequestData] = []
        token_budget = self.config.max_num_batched_tokens
        num_running_limit = self.config.max_num_seqs

        # 第一轮: RUNNING 请求 (decode)
        still_running: list[Request] = []
        for request in self.running:
            num_new_tokens = request.num_new_tokens
            if num_new_tokens == 0:
                still_running.append(request)
                continue

            num_new_tokens = min(num_new_tokens, token_budget)
            slot = self.kv_cache_manager.append_slot(request.request_id)
            if slot is None:
                still_running.append(request)
                continue

            block_table = self.kv_cache_manager.get_block_table(request.request_id)
            scheduled_requests.append(ScheduledRequestData(
                request_id=request.request_id,
                token_ids=request.all_token_ids[-num_new_tokens:],
                num_new_tokens=num_new_tokens,
                block_table=block_table.block_ids,
                num_computed_tokens=request.num_computed_tokens,
                is_prefill=False,
                sampling_params=request.sampling_params,
            ))
            token_budget -= num_new_tokens
            still_running.append(request)

        self.running = still_running

        # 第二轮: WAITING 请求 (prefill)
        newly_running: list[Request] = []
        while self.waiting and token_budget > 0:
            if len(self.running) + len(newly_running) >= num_running_limit:
                break

            request = self.waiting[0]
            num_new_tokens = request.num_tokens - request.num_computed_tokens

            if not self.config.enable_chunked_prefill and num_new_tokens > token_budget:
                break

            num_new_tokens = min(num_new_tokens, token_budget)

            block_table = self.kv_cache_manager.allocate_slots(
                request.request_id, request.all_token_ids,
            )
            if block_table is None:
                break

            self.waiting.popleft()
            request.status = RequestStatus.RUNNING
            is_prefill = request.num_computed_tokens == 0

            scheduled_requests.append(ScheduledRequestData(
                request_id=request.request_id,
                token_ids=request.all_token_ids[:num_new_tokens]
                    if is_prefill
                    else request.all_token_ids[-num_new_tokens:],
                num_new_tokens=num_new_tokens,
                block_table=block_table.block_ids,
                num_computed_tokens=request.num_computed_tokens,
                is_prefill=is_prefill,
                sampling_params=request.sampling_params,
            ))
            token_budget -= num_new_tokens
            newly_running.append(request)

        self.running.extend(newly_running)

        total_tokens = sum(sr.num_new_tokens for sr in scheduled_requests)
        return SchedulerOutput(
            scheduled_requests=scheduled_requests,
            total_num_scheduled_tokens=total_tokens,
        )

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_output: ModelRunnerOutput,
    ) -> list[RequestOutput]:
        finished_outputs: list[RequestOutput] = []
        finished_req_ids: list[str] = []

        for i, scheduled_req in enumerate(scheduler_output.scheduled_requests):
            req_id = scheduled_req.request_id
            request = self.requests.get(req_id)
            if request is None:
                continue

            request.num_computed_tokens += scheduled_req.num_new_tokens
            sampled_token_id = model_output.sampled_token_ids[i]
            request.append_output_token_ids(sampled_token_id)

            finished = False
            if (self.eos_token_id is not None
                    and sampled_token_id == self.eos_token_id):
                request.status = RequestStatus.FINISHED_STOPPED
                finished = True
            elif request.num_output_tokens >= request.sampling_params.max_tokens:
                request.status = RequestStatus.FINISHED_LENGTH_CAPPED
                finished = True

            if finished:
                finished_req_ids.append(req_id)
                finished_outputs.append(RequestOutput(
                    request_id=req_id,
                    prompt_token_ids=request.prompt_token_ids,
                    output_token_ids=request.output_token_ids,
                    finished=True,
                ))

        for req_id in finished_req_ids:
            self.kv_cache_manager.free(req_id)
            self.running = [r for r in self.running if r.request_id != req_id]
            self.requests.pop(req_id, None)

        return finished_outputs

    def finish_requests(
        self, request_ids: list[str],
        status: RequestStatus = RequestStatus.FINISHED_STOPPED,
    ) -> None:
        for req_id in request_ids:
            request = self.requests.pop(req_id, None)
            if request is None:
                continue
            request.status = status
            self.kv_cache_manager.free(req_id)
            self.running = [r for r in self.running if r.request_id != req_id]
            self.waiting = deque(r for r in self.waiting if r.request_id != req_id)

    @property
    def has_unfinished_requests(self) -> bool:
        return len(self.waiting) > 0 or len(self.running) > 0
