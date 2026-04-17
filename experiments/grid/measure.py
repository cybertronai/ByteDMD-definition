"""Generic measurement helpers for heuristic-grid experiments."""

from __future__ import annotations

from collections import defaultdict
import gc
import math
from typing import Any

from experiments.memory_management.tracer import (
    Context,
    Tracked,
    trace_to_cost_continuous,
    trace_to_cost_discrete,
    usqrt,
)


def wrap_value(ctx: Context, value: Any):
    """Recursively wrap nested Python structures in tracked scalars."""

    if isinstance(value, Tracked):
        return value
    if isinstance(value, list):
        return [wrap_value(ctx, item) for item in value]
    if isinstance(value, tuple):
        return tuple(wrap_value(ctx, item) for item in value)
    return Tracked(ctx, ctx.allocate(), value)


class FenwickTree:
    """Fenwick tree for dynamic live-rank queries in SpaceDMD."""

    def __init__(self, size: int):
        self.tree = [0] * (size + 1)
        self.size = size

    def add(self, index: int, delta: int) -> None:
        while index <= self.size:
            self.tree[index] += delta
            index += index & (-index)

    def query(self, index: int) -> int:
        total = 0
        while index > 0:
            total += self.tree[index]
            index -= index & (-index)
        return total


class SpaceDMD:
    """Density-ranked spatial liveness heuristic from the linked gist."""

    def __init__(self):
        self.counter = 0
        self.time = 0
        self.birth: dict[int, int] = {}
        self.last_use: dict[int, int] = {}
        self.accesses: defaultdict[int, int] = defaultdict(int)
        self.reads_at: list[tuple[int, ...]] = []
        self.n_reads = 0

    def allocate(self, *, is_input: bool = False) -> int:
        self.counter += 1
        key = self.counter
        born = 0 if is_input else self.time
        self.birth[key] = born
        self.last_use[key] = born
        return key

    def read(self, *keys: int) -> int:
        if not keys:
            return 0
        for key in keys:
            self.accesses[key] += 1
            self.last_use[key] = self.time
        self.reads_at.append(tuple(keys))
        self.n_reads += len(keys)
        self.time += 1
        return 0

    def free(self, _key: int) -> None:
        return

    def compute_costs(self) -> dict[str, object]:
        if not self.birth:
            return {
                "trace": [],
                "cost_discrete": 0,
                "cost_continuous": 0.0,
                "n_reads": 0,
            }

        def priority(key: int):
            lifespan = self.last_use[key] - self.birth[key] + 1
            density = self.accesses[key] / lifespan if lifespan > 0 else 0.0
            return (-density, -self.accesses[key], self.birth[key], key)

        sorted_keys = sorted(self.birth, key=priority)
        rank_map = {key: index + 1 for index, key in enumerate(sorted_keys)}

        births_at: defaultdict[int, list[int]] = defaultdict(list)
        deaths_at: defaultdict[int, list[int]] = defaultdict(list)
        for key in self.birth:
            births_at[self.birth[key]].append(key)
            deaths_at[self.last_use[key]].append(key)

        bit = FenwickTree(len(sorted_keys))
        rank_trace: list[int] = []
        discrete = 0
        continuous = 0.0

        for t in range(self.time):
            for key in births_at[t]:
                bit.add(rank_map[key], 1)

            for key in self.reads_at[t]:
                active_rank = bit.query(rank_map[key])
                rank_trace.append(active_rank)
                discrete += usqrt(active_rank)
                continuous += (2.0 / 3.0) * (active_rank ** 1.5 - (active_rank - 1) ** 1.5)

            for key in deaths_at[t]:
                bit.add(rank_map[key], -1)

        return {
            "trace": rank_trace,
            "cost_discrete": discrete,
            "cost_continuous": continuous,
            "n_reads": self.n_reads,
        }


class SpaceTracked:
    """Tracked scalar that batches operand reads for SpaceDMD events."""

    def __init__(self, ctx: SpaceDMD, key: int, val: Any):
        self._ctx = ctx
        self._key = key
        self.val = val

    def __del__(self):
        try:
            self._ctx.free(self._key)
        except Exception:
            pass

    def _do_binop(self, other, op):
        if isinstance(other, SpaceTracked):
            self._ctx.read(self._key, other._key)
            value = op(self.val, other.val)
        else:
            self._ctx.read(self._key)
            value = op(self.val, other)
        return SpaceTracked(self._ctx, self._ctx.allocate(), value)

    def __add__(self, other): return self._do_binop(other, lambda a, b: a + b)
    def __sub__(self, other): return self._do_binop(other, lambda a, b: a - b)
    def __mul__(self, other): return self._do_binop(other, lambda a, b: a * b)
    def __radd__(self, other): return self._do_binop(other, lambda a, b: b + a)
    def __rsub__(self, other): return self._do_binop(other, lambda a, b: b - a)
    def __rmul__(self, other): return self._do_binop(other, lambda a, b: b * a)


def wrap_value_space(ctx: SpaceDMD, value: Any):
    """Recursively wrap input scalars as SpaceTracked inputs."""

    if isinstance(value, SpaceTracked):
        return value
    if isinstance(value, list):
        return [wrap_value_space(ctx, item) for item in value]
    if isinstance(value, tuple):
        return tuple(wrap_value_space(ctx, item) for item in value)
    return SpaceTracked(ctx, ctx.allocate(is_input=True), value)


def measure_function(func, args: tuple[Any, ...], *, strategy: str) -> dict[str, object]:
    """Run one function under one memory-management strategy."""

    ctx = Context(strategy=strategy)
    wrapped_args = tuple(wrap_value(ctx, arg) for arg in args)
    peak_stack = [len(ctx.stack)]

    original_allocate = ctx.allocate

    def tracking_allocate():
        key = original_allocate()
        peak_stack[0] = max(peak_stack[0], len(ctx.stack))
        return key

    ctx.allocate = tracking_allocate  # type: ignore[assignment]
    result = func(*wrapped_args)

    del result
    del wrapped_args
    gc.collect()

    trace = list(ctx.trace)
    return {
        "strategy": strategy,
        "trace": trace,
        "cost_discrete": trace_to_cost_discrete(trace),
        "cost_continuous": trace_to_cost_continuous(trace),
        "n_reads": len(trace),
        "peak_stack": peak_stack[0],
    }


def measure_space_dmd(func, args: tuple[Any, ...]) -> dict[str, object]:
    """Run one function under the SpaceDMD spatial-liveness heuristic."""

    ctx = SpaceDMD()
    wrapped_args = tuple(wrap_value_space(ctx, arg) for arg in args)
    result = func(*wrapped_args)

    del result
    del wrapped_args
    gc.collect()

    measurement = ctx.compute_costs()
    measurement["strategy"] = "spacedmd"
    return measurement


def working_set_proxy(n_reads: int, peak_live: int) -> int:
    """Simple bandwidth-times-footprint proxy."""

    return n_reads * usqrt(peak_live)
