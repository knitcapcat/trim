"""
Token Sampler — 从 logits 采样下一个 token。

Maps to:
  vllm/v1/sample/sampler.py    — Sampler
  vllm/v1/sample/metadata.py   — SamplingMetadata

vLLM 原版 Sampler 的完整 pipeline:
  1. (可选) 计算 logprobs
  2. logits → float32
  3. 应用 allowed_token_ids 白名单
  4. 应用 bad_words 过滤
  5. 应用 logits processors (min_tokens, logit_bias)
  6. 应用 penalties (repetition, frequency, presence)
  7. 采样:
     a) greedy → argmax
     b) random → temperature → min_p → top_k → top_p → multinomial
  8. 收集 top-k logprobs
  9. 返回 SamplerOutput

我们只实现步骤 2 和 7（核心采样），跳过 logprobs、penalties、
bad_words、logits processors 等生产级功能。

关键设计: vLLM 支持同一 batch 内不同请求用不同采样策略
（有的 greedy，有的 random）。通过 temperature < eps 判断:
  - temperature ≈ 0 → greedy (argmax)
  - temperature > 0 → random (top-k/top-p + multinomial)
最后用 torch.where 合并两者结果。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from trim.config import SamplingParams

_SAMPLING_EPS = 1e-5


# ---------------------------------------------------------------------------
# SamplingMetadata → vllm/v1/sample/metadata.py::SamplingMetadata
# ---------------------------------------------------------------------------

@dataclass
class SamplingMetadata:
    """Pre-computed sampling tensors for the current batch."""

    temperature: torch.Tensor     # [num_reqs], float32
    top_p: torch.Tensor           # [num_reqs], float32
    top_k: torch.Tensor           # [num_reqs], int32
    all_greedy: bool
    all_random: bool

    @staticmethod
    def from_sampling_params(
        params_list: list[SamplingParams],
        device: torch.device,
    ) -> SamplingMetadata:
        temperatures = torch.tensor(
            [sp.temperature for sp in params_list],
            dtype=torch.float32, device=device,
        )
        top_ps = torch.tensor(
            [sp.top_p for sp in params_list],
            dtype=torch.float32, device=device,
        )
        top_ks = torch.tensor(
            [sp.top_k for sp in params_list],
            dtype=torch.int32, device=device,
        )
        all_greedy = bool((temperatures < _SAMPLING_EPS).all())
        all_random = bool((temperatures >= _SAMPLING_EPS).all())
        return SamplingMetadata(
            temperature=temperatures, top_p=top_ps, top_k=top_ks,
            all_greedy=all_greedy, all_random=all_random,
        )


# ---------------------------------------------------------------------------
# Sampler → vllm/v1/sample/sampler.py::Sampler
# ---------------------------------------------------------------------------

class Sampler(nn.Module):
    """Sample next tokens from model logits."""

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        logits = logits.to(torch.float32)

        greedy_tokens = logits.argmax(dim=-1)
        if sampling_metadata.all_greedy:
            return greedy_tokens

        temp = sampling_metadata.temperature
        safe_temp = torch.where(temp < _SAMPLING_EPS, 1.0, temp)
        logits = logits / safe_temp.unsqueeze(1)

        top_k = sampling_metadata.top_k
        if (top_k > 0).any():
            for i in range(logits.shape[0]):
                k = top_k[i].item()
                if k > 0:
                    topk_vals, _ = logits[i].topk(k)
                    logits[i][logits[i] < topk_vals[-1]] = float("-inf")

        top_p = sampling_metadata.top_p
        if (top_p < 1.0).any():
            sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
            probs = sorted_logits.softmax(dim=-1)
            cumulative_probs = probs.cumsum(dim=-1)
            mask = (cumulative_probs - probs) > top_p.unsqueeze(1)
            sorted_logits[mask] = float("-inf")
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        probs = logits.softmax(dim=-1)
        random_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        if sampling_metadata.all_random:
            return random_tokens

        return torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_tokens, random_tokens,
        )
