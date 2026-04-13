"""
Offline inference example — trim 的端到端使用示例。

Maps to: vllm/examples/offline_inference/basic/generate.py

Usage:
    python examples/generate.py --model meta-llama/Llama-3.2-1B
    python examples/generate.py --model meta-llama/Llama-3.2-1B --max-tokens 64
"""

import argparse
import time

from trim.config import SamplingParams
from trim.engine.llm import LLM


def main():
    parser = argparse.ArgumentParser(description="trim offline inference")
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model name or local path",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, default="auto")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    t0 = time.time()

    llm = LLM(model=args.model, dtype=args.dtype)

    print(f"Model loaded in {time.time() - t0:.1f}s")

    prompts = [
        "The capital of France is",
        "Explain quantum computing in one sentence:",
        "def fibonacci(n):",
    ]

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    print(f"\nGenerating with {len(prompts)} prompts, max_tokens={args.max_tokens}")
    print("=" * 60)

    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0

    total_output_tokens = 0
    for output in outputs:
        total_output_tokens += len(output.output_token_ids)
        print(f"\n[Request {output.request_id[:8]}...]")
        print(f"  Prompt tokens:  {len(output.prompt_token_ids)}")
        print(f"  Output tokens:  {len(output.output_token_ids)}")
        print(f"  Output: {output.output_text}")

    print("\n" + "=" * 60)
    print(f"Total time: {elapsed:.2f}s")
    print(f"Total output tokens: {total_output_tokens}")
    if elapsed > 0:
        print(f"Throughput: {total_output_tokens / elapsed:.1f} tokens/s")


if __name__ == "__main__":
    main()
