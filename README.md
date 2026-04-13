# trim

A simplified reimplementation of [vLLM v1](https://github.com/vllm-project/vllm) for learning LLM inference systems.

## Architecture

trim mirrors vLLM v1's layered architecture:

```
examples/generate.py          ← Offline inference entry point
  └─ trim.engine.LLM          ← Sync API (like vllm.entrypoints.llm.LLM)
      └─ trim.engine.EngineCore  ← Main loop: schedule → execute → update
          ├─ trim.core.Scheduler        ← Continuous batching
          ├─ trim.core.KVCacheManager   ← Paged block allocation
          └─ trim.worker.GPUWorker      ← Per-GPU execution
              └─ trim.worker.GPUModelRunner  ← Input prep + forward + sample
                  ├─ trim.model.LlamaForCausalLM  ← Model definition
                  ├─ trim.attention.Attention       ← Paged attention
                  └─ trim.sample.Sampler            ← Token sampling
```

## Phases

### Phase 1: Eager Inference (single GPU, no compilation)
- EngineCore loop: `schedule()` → `execute_model()` → `update_from_output()`
- Continuous batching scheduler with waiting/running queues
- Paged KV cache management (block allocation, slot mapping)
- Llama model with FlashAttention
- Greedy / top-k / top-p sampling

### Phase 2: Compilation Optimization
- `@support_compile` decorator → `torch.compile` integration
- `TrimBackend` (custom Dynamo backend, like VllmBackend)
- FX Graph splitting at attention ops
- `PiecewiseBackend` with multi-shape dispatch
- CUDA Graph capture and replay (FULL + PIECEWISE modes)
- Custom Inductor passes (fusion, noop elimination)

### Phase 3: Distributed Execution
- Tensor Parallelism (ColumnParallelLinear / RowParallelLinear)
- Executor → Worker multi-process communication
- Process group initialization (TP/PP groups)

## Component Mapping (trim → vLLM)

| trim                         | vLLM v1                                      |
|------------------------------|-----------------------------------------------|
| `engine.EngineCore`          | `v1.engine.core.EngineCore`                   |
| `engine.LLM`                 | `v1.engine.llm_engine.LLMEngine`              |
| `core.Scheduler`             | `v1.core.sched.scheduler.Scheduler`           |
| `core.KVCacheManager`        | `v1.core.kv_cache_manager.KVCacheManager`     |
| `worker.GPUWorker`           | `v1.worker.gpu_worker.Worker`                 |
| `worker.GPUModelRunner`      | `v1.worker.gpu_model_runner.GPUModelRunner`   |
| `model.LlamaForCausalLM`    | `model_executor.models.llama.LlamaForCausalLM`|
| `attention.Attention`        | `attention.layer.Attention`                   |
| `sample.Sampler`             | `v1.sample.sampler.Sampler`                   |
| `compilation.TrimBackend`    | `compilation.backends.VllmBackend`            |
| `compilation.PiecewiseBackend`| `compilation.piecewise_backend.PiecewiseBackend`|
| `compilation.CUDAGraphWrapper`| `compilation.cuda_graph.CUDAGraphWrapper`    |

## Quick Start

```bash
pip install -e .
python examples/generate.py --model meta-llama/Llama-3.2-1B
```
