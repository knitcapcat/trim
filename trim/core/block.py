"""
Block abstractions for PagedAttention.

Maps to:
  vllm/v1/core/kv_cache_utils.py  — KVCacheBlock, FreeKVCacheBlockQueue
  nano_vllm/core/block.py         — Block, BlockTable

Block: 物理内存页，存 block_size 个 token 的 K/V
BlockTable: 页表，逻辑页号 → 物理页号
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

BLOCK_SIZE = 16


def get_block_hash(
    token_ids: tuple[int, ...],
    parent_hash: Optional[int] = None,
) -> int:
    """
    计算 block 的累积 hash（包含整个 prefix chain）。

    vLLM 对应: kv_cache_utils.get_block_hash()
    """
    if parent_hash is None:
        return hash(token_ids)
    return hash((parent_hash, token_ids))


@dataclass
class Block:
    """A physical block of KV cache memory."""

    block_id: int
    ref_count: int = 0
    block_hash: Optional[int] = None
    is_full: bool = False
    is_null: bool = False

    def increment_ref(self) -> None:
        self.ref_count += 1

    def decrement_ref(self) -> int:
        self.ref_count -= 1
        return self.ref_count

    def reset(self) -> None:
        self.ref_count = 0
        self.block_hash = None
        self.is_full = False


@dataclass
class BlockTable:
    """Maps logical block positions to physical block IDs."""

    block_ids: list[int] = field(default_factory=list)
    block_size: int = BLOCK_SIZE

    def append(self, block_id: int) -> None:
        self.block_ids.append(block_id)

    @property
    def num_blocks(self) -> int:
        return len(self.block_ids)

    def get_slot_mapping(self, start: int, num_tokens: int) -> list[int]:
        slots = []
        for pos in range(start, start + num_tokens):
            logical_block = pos // self.block_size
            offset = pos % self.block_size
            physical_block = self.block_ids[logical_block]
            slots.append(physical_block * self.block_size + offset)
        return slots


def compute_num_blocks(num_tokens: int, block_size: int = BLOCK_SIZE) -> int:
    return (num_tokens + block_size - 1) // block_size
