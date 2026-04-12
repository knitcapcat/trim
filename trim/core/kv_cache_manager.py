"""
KV Cache Manager — BlockPool + 请求级管理。

Maps to:
  vllm/v1/core/block_pool.py        — BlockPool
  vllm/v1/core/kv_cache_manager.py  — KVCacheManager
  nano_vllm/core/block_manager.py   — BlockManager
"""

from __future__ import annotations

from typing import Optional

from trim.core.block import (
    Block,
    BlockTable,
    compute_num_blocks,
    get_block_hash,
)


class BlockPool:
    """Physical block pool with prefix caching support."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        enable_prefix_caching: bool = True,
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching

        self.blocks: list[Block] = [Block(block_id=i) for i in range(num_blocks)]

        self.null_block = self.blocks[0]
        self.null_block.is_null = True
        self.null_block.ref_count = 1

        self.free_blocks: list[int] = list(range(num_blocks - 1, 0, -1))

        self.prefix_cache: dict[tuple[Optional[int], tuple[int, ...]], int] = {}
        self._block_to_cache_key: dict[int, tuple[Optional[int], tuple[int, ...]]] = {}

    def allocate(self) -> Block:
        if not self.free_blocks:
            raise RuntimeError("Out of KV cache blocks")
        block_id = self.free_blocks.pop()
        block = self.blocks[block_id]
        self._maybe_evict_from_cache(block)
        block.reset()
        block.ref_count = 1
        return block

    def free(self, block: Block) -> None:
        if block.is_null:
            return
        new_ref = block.decrement_ref()
        if new_ref <= 0:
            block.ref_count = 0
            self._maybe_evict_from_cache(block)
            block.reset()
            self.free_blocks.append(block.block_id)

    def touch(self, block: Block) -> None:
        if block.ref_count == 0 and not block.is_null:
            try:
                self.free_blocks.remove(block.block_id)
            except ValueError:
                pass
        block.increment_ref()

    def try_get_cached_block(
        self, parent_hash: Optional[int], block_tokens: tuple[int, ...],
    ) -> Block | None:
        if not self.enable_prefix_caching:
            return None
        cache_key = (parent_hash, block_tokens)
        block_id = self.prefix_cache.get(cache_key)
        if block_id is not None:
            return self.blocks[block_id]
        return None

    def cache_full_block(
        self, block: Block, block_tokens: tuple[int, ...], parent_hash: Optional[int],
    ) -> int:
        if not self.enable_prefix_caching:
            return 0
        block_hash = get_block_hash(block_tokens, parent_hash)
        block.block_hash = block_hash
        block.is_full = True
        cache_key = (parent_hash, block_tokens)
        self.prefix_cache[cache_key] = block.block_id
        self._block_to_cache_key[block.block_id] = cache_key
        return block_hash

    def _maybe_evict_from_cache(self, block: Block) -> None:
        if block.block_id in self._block_to_cache_key:
            cache_key = self._block_to_cache_key.pop(block.block_id)
            self.prefix_cache.pop(cache_key, None)

    def evict_lru_block(self) -> Block | None:
        """TODO: LRU Eviction"""
        raise NotImplementedError("LRU eviction — Phase 1 后实现")

    def take_events(self) -> list:
        """TODO: KV Events"""
        return []

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)


class KVCacheManager:
    """Request-level KV cache management."""

    def __init__(
        self,
        num_gpu_blocks: int,
        block_size: int,
        enable_prefix_caching: bool = True,
    ) -> None:
        self.block_size = block_size
        self.pool = BlockPool(num_gpu_blocks, block_size, enable_prefix_caching)
        self.req_to_block_table: dict[str, BlockTable] = {}
        self.req_to_num_tokens: dict[str, int] = {}
        self.req_to_last_block_hash: dict[str, Optional[int]] = {}

    def allocate_slots(
        self, request_id: str, token_ids: list[int],
    ) -> BlockTable | None:
        num_tokens = len(token_ids)
        num_full_blocks = num_tokens // self.block_size
        has_partial = (num_tokens % self.block_size) > 0

        block_table = BlockTable(block_size=self.block_size)
        parent_hash: Optional[int] = None
        shared_prefix_len = 0

        for block_idx in range(num_full_blocks):
            start = block_idx * self.block_size
            end = start + self.block_size
            block_tokens = tuple(token_ids[start:end])

            cached_block = self.pool.try_get_cached_block(parent_hash, block_tokens)
            if cached_block is not None:
                self.pool.touch(cached_block)
                block_table.append(cached_block.block_id)
                parent_hash = cached_block.block_hash
                shared_prefix_len += self.block_size
            else:
                if self.pool.num_free_blocks == 0:
                    self._rollback_allocation(block_table)
                    return None
                block = self.pool.allocate()
                block_table.append(block.block_id)
                parent_hash = self.pool.cache_full_block(block, block_tokens, parent_hash)

        if has_partial:
            if self.pool.num_free_blocks == 0:
                self._rollback_allocation(block_table)
                return None
            block = self.pool.allocate()
            block_table.append(block.block_id)

        self.req_to_block_table[request_id] = block_table
        self.req_to_num_tokens[request_id] = num_tokens
        self.req_to_last_block_hash[request_id] = parent_hash
        return block_table

    def append_slot(self, request_id: str) -> int | None:
        block_table = self.req_to_block_table[request_id]
        num_tokens = self.req_to_num_tokens[request_id]
        offset = num_tokens % self.block_size

        if offset == 0 and num_tokens > 0:
            if self.pool.num_free_blocks == 0:
                return None
            new_block = self.pool.allocate()
            block_table.append(new_block.block_id)

        logical_block = num_tokens // self.block_size
        offset = num_tokens % self.block_size
        physical_block = block_table.block_ids[logical_block]
        slot = physical_block * self.block_size + offset

        self.req_to_num_tokens[request_id] = num_tokens + 1
        return slot

    def free(self, request_id: str) -> None:
        block_table = self.req_to_block_table.pop(request_id, None)
        self.req_to_num_tokens.pop(request_id, None)
        self.req_to_last_block_hash.pop(request_id, None)
        if block_table is None:
            return
        for block_id in reversed(block_table.block_ids):
            self.pool.free(self.pool.blocks[block_id])

    def get_block_table(self, request_id: str) -> BlockTable:
        return self.req_to_block_table.get(
            request_id, BlockTable(block_size=self.block_size)
        )

    def get_slot_mapping(self, request_id: str, start: int, num_tokens: int) -> list[int]:
        block_table = self.req_to_block_table[request_id]
        return block_table.get_slot_mapping(start, num_tokens)

    def _rollback_allocation(self, block_table: BlockTable) -> None:
        for block_id in block_table.block_ids:
            self.pool.free(self.pool.blocks[block_id])
        block_table.block_ids.clear()

    @property
    def num_free_blocks(self) -> int:
        return self.pool.num_free_blocks

    @property
    def usage(self) -> float:
        total = self.pool.num_blocks - 1
        if total == 0:
            return 0.0
        return 1.0 - (self.num_free_blocks / total)
