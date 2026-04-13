"""
Model weight loader — 从 HuggingFace checkpoint 流式加载权重。

Maps to: vllm/model_executor/model_loader/ (整个目录)

vLLM 的 model_loader 支持: safetensors, pytorch bin, gguf, dummy,
  sharded checkpoints, TP weight splitting 等。
我们只实现 safetensors 流式加载。

=== 为什么不能直接 AutoModelForCausalLM.from_pretrained()？ ===

  方式 1 (我们之前的笨方法):
    trim model (随机权重)     → GPU 显存 x1
    HF model (from_pretrained) → GPU 显存 x2  ← 峰值: 2 倍显存!
    copy weights, del hf_model → GPU 显存 x1

  方式 2 (本文件的流式加载):
    trim model (随机权重)     → GPU 显存 x1
    逐个读 safetensors tensor → CPU 内存 (1 个 tensor)
    copy 到 trim model        → GPU 显存 x1  ← 峰值: 1 倍 + 1 个 tensor
    读下一个 tensor ...

  vLLM 的做法和方式 2 类似，通过 safetensors.safe_open() 逐个读取。

=== 权重映射 (stacked_params_mapping) ===

  HuggingFace checkpoint:          trim 模型:
    q_proj.weight  ─┐
    k_proj.weight  ─┼→ qkv_proj.weight  (concat along dim 0)
    v_proj.weight  ─┘
    gate_proj.weight ─┐
    up_proj.weight   ─┘→ gate_up_proj.weight  (concat along dim 0)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import safe_open

if TYPE_CHECKING:
    from trim.config import ModelConfig
    from trim.model.llama import LlamaForCausalLM


def load_weights_from_hf(
    model: LlamaForCausalLM,
    model_config: ModelConfig,
) -> None:
    """
    从 HuggingFace checkpoint 加载权重到 trim 模型。

    流程:
      1. 下载/定位 checkpoint 文件
      2. 遍历所有 safetensors 文件
      3. 对每个 tensor: 判断映射关系 → copy 到 trim 模型参数

    vLLM 对应: AutoWeightsLoader.load_weights() +
               LlamaModel.load_weights() 里的 stacked_params_mapping
    """
    cfg = model_config

    # 1. 获取模型文件路径 (可能需要下载)
    model_path = _get_model_path(cfg.model)

    # 2. 收集所有 safetensors 文件
    safetensor_files = sorted(model_path.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(
            f"No safetensors files found in {model_path}. "
            "Make sure the model is downloaded."
        )

    # 3. 准备 trim 模型参数字典
    params = dict(model.named_parameters())

    # 预计算 qkv 和 gate_up 的 size
    q_size = cfg.num_attention_heads * cfg.head_dim
    kv_size = cfg.num_key_value_heads * cfg.head_dim
    mid_size = cfg.intermediate_size

    # vLLM 的 stacked_params_mapping 格式:
    #   (fused_param_name, original_name, shard_id)
    # shard_id 可以是 "q"/"k"/"v" 或 0/1 (表示在 fused tensor 中的位置)
    stacked_params_mapping = [
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    ]

    # shard_id → (offset, size) in the fused weight
    shard_offsets = {
        "q": (0, q_size),
        "k": (q_size, kv_size),
        "v": (q_size + kv_size, kv_size),
        0: (0, mid_size),
        1: (mid_size, mid_size),
    }

    # 4. 逐文件、逐 tensor 流式加载
    # vLLM 对应: safetensors.safe_open() 逐 key 读取
    # safe_open 比 load_file 更省内存: 一次只读 1 个 tensor, 不是整个文件
    for sf_file in safetensor_files:
        with safe_open(str(sf_file), framework="pt", device="cpu") as f:
            for name in f.keys():
                if "rotary_emb.inv_freq" in name:
                    continue

                # 检查是否命中 stacked_params_mapping
                handled = False
                for fused_name, original_name, shard_id in stacked_params_mapping:
                    if original_name not in name:
                        continue
                    # 替换 original_name → fused_name 得到 trim 模型的参数名
                    target = name.replace(original_name, fused_name)
                    if target not in params:
                        break
                    weight = f.get_tensor(name)
                    offset, size = shard_offsets[shard_id]
                    params[target].data[offset:offset + size].copy_(weight)
                    handled = True
                    break

                if handled:
                    continue

                # 1:1 直接映射
                if name in params:
                    weight = f.get_tensor(name)
                    params[name].data.copy_(weight)


def _get_model_path(model_name_or_path: str) -> Path:
    """获取本地模型路径，必要时从 HuggingFace Hub 下载。"""
    local_path = Path(model_name_or_path)
    if local_path.is_dir() and any(local_path.glob("*.safetensors")):
        return local_path

    # 下载到 HuggingFace cache
    cache_dir = snapshot_download(
        model_name_or_path,
        allow_patterns=["*.safetensors", "*.json"],
    )
    return Path(cache_dir)
