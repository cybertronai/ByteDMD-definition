"""Three-level tracing for matrix multiplication experiments.

Level 1: Python algorithms (`tiled_matmul`, `vanilla_recursive_matmul`,
`strassen_matmul`) execute over traced scalar values.

Level 2: The tracer emits an abstract load/store stream over logical values
such as ``A[0,0]`` and ``t37``.

Level 3: The abstract stream is compiled to concrete addresses under several
slot-allocation policies, producing a physical load/store trace.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import itertools
import math
from typing import Callable, Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class AbstractAccess:
    """A logical memory access without a concrete address."""

    kind: str
    value: str
    role: str


@dataclass(frozen=True)
class ConcreteAccess:
    """A compiled memory access with a concrete address."""

    kind: str
    value: str
    role: str
    address: int
    policy: str


@dataclass(frozen=True)
class TraceProgram:
    """All three levels associated with one traced algorithm run."""

    algorithm: str
    matrix_size: int
    tile_size: int | None
    result: List[List[float]]
    abstract_accesses: List[AbstractAccess]
    input_symbols: List[str]
    output_symbols: List[str]
    symbol_roles: Dict[str, str]


class TraceValue:
    """Scalar wrapper that records logical loads and stores."""

    __slots__ = ("_ctx", "name", "value", "role")

    def __init__(self, ctx: "TraceContext", name: str, value: float, role: str) -> None:
        self._ctx = ctx
        self.name = name
        self.value = value
        self.role = role

    def _coerce(self, other: "TraceValue | float | int") -> "TraceValue":
        if isinstance(other, TraceValue):
            return other
        return self._ctx.literal(other)

    def _binop(
        self,
        other: "TraceValue | float | int",
        op: Callable[[float, float], float],
        opname: str,
    ) -> "TraceValue":
        other_value = self._coerce(other)
        self._ctx.load(self)
        self._ctx.load(other_value)
        return self._ctx.new_temp(op(self.value, other_value.value), opname)

    def __add__(self, other: "TraceValue | float | int") -> "TraceValue":
        return self._binop(other, lambda a, b: a + b, "add")

    def __radd__(self, other: "TraceValue | float | int") -> "TraceValue":
        return self._coerce(other)._binop(self, lambda a, b: a + b, "add")

    def __sub__(self, other: "TraceValue | float | int") -> "TraceValue":
        return self._binop(other, lambda a, b: a - b, "sub")

    def __rsub__(self, other: "TraceValue | float | int") -> "TraceValue":
        return self._coerce(other)._binop(self, lambda a, b: a - b, "sub")

    def __mul__(self, other: "TraceValue | float | int") -> "TraceValue":
        return self._binop(other, lambda a, b: a * b, "mul")

    def __rmul__(self, other: "TraceValue | float | int") -> "TraceValue":
        return self._coerce(other)._binop(self, lambda a, b: a * b, "mul")


class TraceContext:
    """Builds the level-2 abstract load/store stream."""

    def __init__(self) -> None:
        self.abstract_accesses: List[AbstractAccess] = []
        self.symbol_roles: Dict[str, str] = {}
        self.input_symbols: List[str] = []
        self.output_symbols: List[str] = []
        self._temp_counter = 0
        self._literal_counter = 0

    def register_input_matrix(self, name: str, data: Sequence[Sequence[float]]) -> List[List[TraceValue]]:
        traced: List[List[TraceValue]] = []
        for i, row in enumerate(data):
            traced_row: List[TraceValue] = []
            for j, value in enumerate(row):
                symbol = f"{name}[{i},{j}]"
                self.symbol_roles[symbol] = "input"
                self.input_symbols.append(symbol)
                traced_row.append(TraceValue(self, symbol, value, "input"))
            traced.append(traced_row)
        return traced

    def literal(self, value: float | int) -> TraceValue:
        self._literal_counter += 1
        symbol = f"const{self._literal_counter}"
        self.symbol_roles[symbol] = "literal"
        return TraceValue(self, symbol, value, "literal")

    def load(self, value: TraceValue) -> None:
        if value.role == "literal":
            return
        self.abstract_accesses.append(AbstractAccess("load", value.name, value.role))

    def new_temp(self, value: float, opname: str) -> TraceValue:
        self._temp_counter += 1
        symbol = f"t{self._temp_counter}_{opname}"
        self.symbol_roles[symbol] = "temp"
        self.abstract_accesses.append(AbstractAccess("store", symbol, "temp"))
        return TraceValue(self, symbol, value, "temp")

    def materialize_output(self, name: str, matrix: Sequence[Sequence[TraceValue]]) -> List[List[float]]:
        numeric: List[List[float]] = []
        for i, row in enumerate(matrix):
            numeric_row: List[float] = []
            for j, cell in enumerate(row):
                symbol = f"{name}[{i},{j}]"
                self.symbol_roles[symbol] = "output"
                self.output_symbols.append(symbol)
                self.abstract_accesses.append(AbstractAccess("store", symbol, "output"))
                numeric_row.append(cell.value)
            numeric.append(numeric_row)
        return numeric


def split_matrix(matrix: Sequence[Sequence[TraceValue]]) -> tuple[list[list[TraceValue]], ...]:
    n = len(matrix)
    h = n // 2
    return (
        [[matrix[i][j] for j in range(h)] for i in range(h)],
        [[matrix[i][j] for j in range(h, n)] for i in range(h)],
        [[matrix[i][j] for j in range(h)] for i in range(h, n)],
        [[matrix[i][j] for j in range(h, n)] for i in range(h, n)],
    )


def join_matrix(
    c11: Sequence[Sequence[TraceValue]],
    c12: Sequence[Sequence[TraceValue]],
    c21: Sequence[Sequence[TraceValue]],
    c22: Sequence[Sequence[TraceValue]],
) -> List[List[TraceValue]]:
    h = len(c11)
    n = 2 * h
    out: List[List[TraceValue]] = [[c11[0][0]] * n for _ in range(n)]
    for i in range(h):
        for j in range(h):
            out[i][j] = c11[i][j]
            out[i][j + h] = c12[i][j]
            out[i + h][j] = c21[i][j]
            out[i + h][j + h] = c22[i][j]
    return out


def add_matrix(
    a: Sequence[Sequence[TraceValue]],
    b: Sequence[Sequence[TraceValue]],
) -> List[List[TraceValue]]:
    n = len(a)
    return [[a[i][j] + b[i][j] for j in range(n)] for i in range(n)]


def sub_matrix(
    a: Sequence[Sequence[TraceValue]],
    b: Sequence[Sequence[TraceValue]],
) -> List[List[TraceValue]]:
    n = len(a)
    return [[a[i][j] - b[i][j] for j in range(n)] for i in range(n)]


def tiled_matmul(
    a: Sequence[Sequence[TraceValue]],
    b: Sequence[Sequence[TraceValue]],
    *,
    tile_size: int = 4,
) -> List[List[TraceValue]]:
    """Naive matmul with one explicit level of tiling."""

    n = len(a)
    tile = max(1, min(tile_size, n))
    c: List[List[TraceValue | None]] = [[None] * n for _ in range(n)]
    for bi in range(0, n, tile):
        for bj in range(0, n, tile):
            block_h = min(tile, n - bi)
            block_w = min(tile, n - bj)
            block: List[List[TraceValue | None]] = [[None] * block_w for _ in range(block_h)]
            for bk in range(0, n, tile):
                for i_off, i in enumerate(range(bi, min(bi + tile, n))):
                    for j_off, j in enumerate(range(bj, min(bj + tile, n))):
                        acc = block[i_off][j_off]
                        for k in range(bk, min(bk + tile, n)):
                            prod = a[i][k] * b[k][j]
                            acc = prod if acc is None else acc + prod
                        block[i_off][j_off] = acc
            for i_off in range(block_h):
                for j_off in range(block_w):
                    c[bi + i_off][bj + j_off] = block[i_off][j_off]
    out: List[List[TraceValue]] = []
    for row in c:
        out_row: List[TraceValue] = []
        for cell in row:
            if cell is None:
                raise RuntimeError("tiled matmul left an output cell uninitialized")
            out_row.append(cell)
        out.append(out_row)
    return out


def vanilla_recursive_matmul(
    a: Sequence[Sequence[TraceValue]],
    b: Sequence[Sequence[TraceValue]],
) -> List[List[TraceValue]]:
    n = len(a)
    if n == 1:
        return [[a[0][0] * b[0][0]]]
    a11, a12, a21, a22 = split_matrix(a)
    b11, b12, b21, b22 = split_matrix(b)
    c11 = add_matrix(vanilla_recursive_matmul(a11, b11), vanilla_recursive_matmul(a12, b21))
    c12 = add_matrix(vanilla_recursive_matmul(a11, b12), vanilla_recursive_matmul(a12, b22))
    c21 = add_matrix(vanilla_recursive_matmul(a21, b11), vanilla_recursive_matmul(a22, b21))
    c22 = add_matrix(vanilla_recursive_matmul(a21, b12), vanilla_recursive_matmul(a22, b22))
    return join_matrix(c11, c12, c21, c22)


def strassen_matmul(
    a: Sequence[Sequence[TraceValue]],
    b: Sequence[Sequence[TraceValue]],
) -> List[List[TraceValue]]:
    n = len(a)
    if n == 1:
        return [[a[0][0] * b[0][0]]]
    a11, a12, a21, a22 = split_matrix(a)
    b11, b12, b21, b22 = split_matrix(b)

    m1 = strassen_matmul(add_matrix(a11, a22), add_matrix(b11, b22))
    m2 = strassen_matmul(add_matrix(a21, a22), b11)
    m3 = strassen_matmul(a11, sub_matrix(b12, b22))
    m4 = strassen_matmul(a22, sub_matrix(b21, b11))
    m5 = strassen_matmul(add_matrix(a11, a12), b22)
    m6 = strassen_matmul(sub_matrix(a21, a11), add_matrix(b11, b12))
    m7 = strassen_matmul(sub_matrix(a12, a22), add_matrix(b21, b22))

    c11 = add_matrix(sub_matrix(add_matrix(m1, m4), m5), m7)
    c12 = add_matrix(m3, m5)
    c21 = add_matrix(m2, m4)
    c22 = add_matrix(add_matrix(sub_matrix(m1, m2), m3), m6)
    return join_matrix(c11, c12, c21, c22)


ALGORITHMS: Dict[str, Callable[..., List[List[TraceValue]]]] = {
    "tiled": tiled_matmul,
    "recursive": vanilla_recursive_matmul,
    "strassen": strassen_matmul,
}


def trace_matmul_program(
    algorithm: str,
    a_data: Sequence[Sequence[float]],
    b_data: Sequence[Sequence[float]],
    *,
    tile_size: int = 4,
) -> TraceProgram:
    """Trace one algorithm execution and return all level-2 data."""

    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {algorithm!r}")
    ctx = TraceContext()
    a = ctx.register_input_matrix("A", a_data)
    b = ctx.register_input_matrix("B", b_data)
    if algorithm == "tiled":
        result = tiled_matmul(a, b, tile_size=tile_size)
    else:
        result = ALGORITHMS[algorithm](a, b)
    numeric = ctx.materialize_output("C", result)
    return TraceProgram(
        algorithm=algorithm,
        matrix_size=len(a_data),
        tile_size=tile_size if algorithm == "tiled" else None,
        result=numeric,
        abstract_accesses=ctx.abstract_accesses,
        input_symbols=list(ctx.input_symbols),
        output_symbols=list(ctx.output_symbols),
        symbol_roles=dict(ctx.symbol_roles),
    )


def _initial_preload_symbols(program: TraceProgram) -> List[str]:
    """Match eager init: second argument first, then first argument."""

    half = len(program.input_symbols) // 2
    a_symbols = program.input_symbols[:half]
    b_symbols = program.input_symbols[half:]
    return b_symbols + a_symbols


def _logical_last_use(program: TraceProgram) -> Dict[str, int]:
    last: Dict[str, int] = {}
    for index, access in enumerate(program.abstract_accesses):
        if access.kind == "load":
            last[access.value] = index
    return last


def abstract_reuse_depths(program: TraceProgram, *, live_only: bool) -> List[int]:
    """Compute logical reuse depths from the abstract stream."""

    stack = list(_initial_preload_symbols(program))
    last_use = _logical_last_use(program)
    depths: List[int] = []

    def compact(current_index: int) -> None:
        if not live_only:
            return
        stack[:] = [symbol for symbol in stack if last_use.get(symbol, math.inf) > current_index]

    for index, access in enumerate(program.abstract_accesses):
        symbol = access.value
        if access.kind == "store":
            stack.append(symbol)
            compact(index)
            continue

        depth = len(stack) - stack.index(symbol)
        depths.append(depth)
        stack.remove(symbol)
        stack.append(symbol)
        compact(index)
    return depths


def _rebuild_positions(stack: List[int]) -> Dict[int, int]:
    return {address: idx for idx, address in enumerate(stack)}


def _reuse_depths_from_addresses(
    accesses: Sequence[ConcreteAccess],
    preload_addresses: Sequence[int],
) -> List[int]:
    stack = list(preload_addresses)
    positions = _rebuild_positions(stack)
    depths: List[int] = []
    for access in accesses:
        address = access.address
        idx = positions.get(address)
        if access.kind == "load":
            if idx is None:
                raise RuntimeError(f"address {address} was not resident before load")
            depths.append(len(stack) - idx)

        if idx is None:
            stack.append(address)
        else:
            stack.pop(idx)
            stack.append(address)
        positions = _rebuild_positions(stack)
    return depths


def _temp_store_indices(program: TraceProgram) -> Dict[str, int]:
    indices: Dict[str, int] = {}
    for index, access in enumerate(program.abstract_accesses):
        if access.kind == "store" and access.role == "temp":
            indices.setdefault(access.value, index)
    return indices


def _temp_last_use(program: TraceProgram) -> Dict[str, int]:
    last = _logical_last_use(program)
    return {symbol: index for symbol, index in last.items() if program.symbol_roles.get(symbol) == "temp"}


def _future_load_indices(program: TraceProgram) -> Dict[str, deque[int]]:
    future: Dict[str, deque[int]] = {}
    for index, access in enumerate(program.abstract_accesses):
        if access.kind == "load":
            future.setdefault(access.value, deque()).append(index)
    return future


def _choose_belady_address(
    next_use: float,
    symbol: str,
    live_addresses: Dict[str, int],
    future_loads: Dict[str, deque[int]],
    free_addresses: List[int],
    next_address: int,
) -> tuple[int, int]:
    """Assign a stable base address using offline next-use information."""

    if not free_addresses:
        return next_address, next_address + 1

    live_order = sorted(
        (
            (loads[0], live_symbol)
            for live_symbol, loads in future_loads.items()
            if live_symbol in live_addresses and loads
        ),
        key=lambda item: (item[0], item[1]),
    )

    target_rank = 1
    for future_use, live_symbol in live_order:
        if (future_use, live_symbol) < (next_use, symbol):
            target_rank += 1
        else:
            break

    chosen = min(free_addresses, key=lambda address: (abs(address - target_rank), address))
    free_addresses.remove(chosen)
    return chosen, next_address


def _compile_belady_trace(program: TraceProgram, *, policy: str) -> List[ConcreteAccess]:
    """Two-pass Belady-inspired stable-address heuristic."""

    future_loads = _future_load_indices(program)
    initial_live = sorted(
        (
            (loads[0], symbol)
            for symbol, loads in future_loads.items()
            if program.symbol_roles.get(symbol) == "input" and loads
        ),
        key=lambda item: (item[0], item[1]),
    )

    live_addresses: Dict[str, int] = {}
    next_address = 1
    for _first_use, symbol in initial_live:
        live_addresses[symbol] = next_address
        next_address += 1

    free_addresses: List[int] = []
    compiled: List[ConcreteAccess] = []

    for _index, access in enumerate(program.abstract_accesses):
        symbol = access.value

        if access.kind == "store":
            loads = future_loads.get(symbol)
            if loads:
                address, next_address = _choose_belady_address(
                    float(loads[0]),
                    symbol,
                    live_addresses,
                    future_loads,
                    free_addresses,
                    next_address,
                )
                live_addresses[symbol] = address
            else:
                address, next_address = _choose_belady_address(
                    math.inf,
                    symbol,
                    live_addresses,
                    future_loads,
                    free_addresses,
                    next_address,
                )
        else:
            address = live_addresses[symbol]
            loads = future_loads[symbol]
            loads.popleft()
            if not loads:
                free_addresses.append(live_addresses.pop(symbol))
                free_addresses.sort()

        compiled.append(
            ConcreteAccess(
                kind=access.kind,
                value=symbol,
                role=access.role,
                address=address,
                policy=policy,
            )
        )

    return compiled


def _edf_slot_assignment(program: TraceProgram) -> Dict[str, int]:
    first_store = _temp_store_indices(program)
    last_use = _temp_last_use(program)
    intervals = [
        {"value": symbol, "start": start, "end": last_use.get(symbol, start)}
        for symbol, start in first_store.items()
    ]
    intervals.sort(key=lambda item: (item["end"], item["start"], item["value"]))
    remaining = intervals[:]
    slot_id = 0
    assignment: Dict[str, int] = {}
    while remaining:
        slot_id += 1
        packed: List[dict[str, int | str]] = []
        last_end = -1
        for interval in remaining:
            if int(interval["start"]) >= last_end:
                packed.append(interval)
                last_end = int(interval["end"]) + 1
        packed_ids = {id(interval) for interval in packed}
        remaining = [interval for interval in remaining if id(interval) not in packed_ids]
        for interval in packed:
            assignment[str(interval["value"])] = slot_id
    return assignment


def compile_concrete_trace(program: TraceProgram, *, policy: str) -> List[ConcreteAccess]:
    """Compile the level-2 abstract stream into concrete addresses."""

    normalized_policy = policy.replace("-", "_")
    if normalized_policy not in {"never_reuse", "lifo", "edf", "belady"}:
        raise ValueError(f"unknown concrete policy {policy!r}")
    if normalized_policy == "belady":
        return _compile_belady_trace(program, policy=policy)

    input_base = 1
    input_addresses = {symbol: input_base + idx for idx, symbol in enumerate(program.input_symbols)}
    output_base = input_base + len(program.input_symbols)
    output_addresses = {symbol: output_base + idx for idx, symbol in enumerate(program.output_symbols)}
    temp_base = output_base + len(program.output_symbols)

    last_use = _temp_last_use(program)
    edf_slots = _edf_slot_assignment(program) if normalized_policy == "edf" else {}
    temp_addresses: Dict[str, int] = {}
    next_temp_address = temp_base
    free_stack: List[int] = []

    compiled: List[ConcreteAccess] = []

    for index, access in enumerate(program.abstract_accesses):
        symbol = access.value
        role = access.role
        if role == "input":
            address = input_addresses[symbol]
        elif role == "output":
            address = output_addresses[symbol]
        elif normalized_policy == "edf":
            address = temp_base + edf_slots[symbol] - 1
            temp_addresses[symbol] = address
        elif normalized_policy == "never_reuse":
            if access.kind == "store":
                address = next_temp_address
                next_temp_address += 1
                temp_addresses[symbol] = address
            else:
                address = temp_addresses[symbol]
        elif normalized_policy == "lifo":
            if access.kind == "store":
                if free_stack:
                    address = free_stack.pop()
                else:
                    address = next_temp_address
                    next_temp_address += 1
                temp_addresses[symbol] = address
            else:
                address = temp_addresses[symbol]
        else:  # pragma: no cover - guarded above
            raise AssertionError(policy)

        compiled.append(
            ConcreteAccess(
                kind=access.kind,
                value=symbol,
                role=role,
                address=address,
                policy=policy,
            )
        )

        if role != "temp":
            continue
        if access.kind == "load" and last_use.get(symbol) == index:
            if normalized_policy == "lifo":
                free_stack.append(temp_addresses[symbol])
            elif normalized_policy == "edf":
                pass
            temp_addresses.pop(symbol, None)

        if access.kind == "store" and last_use.get(symbol, index) < index:
            temp_addresses.pop(symbol, None)
    return compiled


def concrete_reuse_depths(program: TraceProgram, *, policy: str) -> List[int]:
    compiled = compile_concrete_trace(program, policy=policy)
    if policy.replace("-", "_") == "belady":
        return [access.address for access in compiled if access.kind == "load"]
    preload_symbols = _initial_preload_symbols(program)
    preload_addresses = [
        1 + program.input_symbols.index(symbol)
        for symbol in preload_symbols
    ]
    return _reuse_depths_from_addresses(compiled, preload_addresses)


def bytedmd_cost(depths: Iterable[int]) -> int:
    return sum(math.isqrt(depth - 1) + 1 for depth in depths)


def memory_curve(depths: Sequence[int], memory_sizes: Sequence[int]) -> List[int]:
    return [sum(1 for depth in depths if depth > size) for size in memory_sizes]


def format_accesses(accesses: Sequence[AbstractAccess | ConcreteAccess], *, limit: int = 16) -> str:
    lines: List[str] = []
    for access in accesses[:limit]:
        if isinstance(access, ConcreteAccess):
            lines.append(f"{access.kind:>5} {access.value:<14} addr={access.address:>4} role={access.role}")
        else:
            lines.append(f"{access.kind:>5} {access.value:<14} role={access.role}")
    if len(accesses) > limit:
        lines.append(f"... ({len(accesses) - limit} more)")
    return "\n".join(lines)


def numeric_matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    n = len(a)
    out = [[0.0] * n for _ in range(n)]
    for i, j, k in itertools.product(range(n), repeat=3):
        out[i][j] += a[i][k] * b[k][j]
    return out
