"""Algorithm set for the heuristic-grid experiment."""

from __future__ import annotations

import cmath
from dataclasses import dataclass
import math
from typing import Callable

from experiments.matmul_hierarchy.hierarchy import tiled_matmul as _tiled_matmul
from experiments.memory_management.algorithms import (
    flops_naive,
    flops_rmm,
    flops_strassen,
    naive_matmul,
    rmm,
    rmm_inplace_gray,
    rmm_inplace_lex,
    strassen,
)


def make_matrix(rows: int, cols: int | None = None, offset: int = 0) -> list[list[int]]:
    """Create a deterministic dense matrix."""

    width = rows if cols is None else cols
    return [[offset + i * width + j + 1 for j in range(width)] for i in range(rows)]


def make_vector(length: int, offset: int = 0) -> list[int]:
    return [offset + i + 1 for i in range(length)]


def zeros(n: int) -> list[list[int]]:
    return [[0] * n for _ in range(n)]


def matvec(A, x):
    """Matrix-vector multiply."""

    n = len(x)
    y = [None] * n
    for i in range(n):
        acc = A[i][0] * x[0]
        for j in range(1, n):
            acc = acc + A[i][j] * x[j]
        y[i] = acc
    return y


def vecmat(A, x):
    """Vector-matrix multiply."""

    n = len(x)
    y = [None] * n
    for j in range(n):
        acc = x[0] * A[0][j]
        for i in range(1, n):
            acc = acc + x[i] * A[i][j]
        y[j] = acc
    return y


def one_level_tiled_matmul(A, B):
    """Naive matmul with one explicit tile level."""

    return _tiled_matmul(A, B, tile_size=4)


def _copy_proxy(x):
    """Tracked copy with one read and one output."""

    return x + 0


def _max2(a, b):
    """Proxy for max with the right read arity."""

    return a + b


def _exp_proxy(x):
    """Proxy for exp with one read and one output."""

    return x * x


def _inv_proxy(x):
    """Proxy for reciprocal with one read and one output."""

    return x * x


def naive_transpose(A):
    """Naive row-major transpose."""

    rows = len(A)
    cols = len(A[0])
    out = [[None] * rows for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            out[j][i] = _copy_proxy(A[i][j])
    return out


def blocked_transpose(A, *, block: int = 8):
    """Blocked transpose."""

    rows = len(A)
    cols = len(A[0])
    out = [[None] * rows for _ in range(cols)]
    for bi in range(0, rows, block):
        for bj in range(0, cols, block):
            for i in range(bi, min(bi + block, rows)):
                for j in range(bj, min(bj + block, cols)):
                    out[j][i] = _copy_proxy(A[i][j])
    return out


def recursive_transpose(A, *, leaf: int = 8):
    """Cache-oblivious recursive transpose."""

    rows = len(A)
    cols = len(A[0])
    if rows <= leaf or cols <= leaf:
        return naive_transpose(A)
    if rows >= cols:
        mid = rows // 2
        top = recursive_transpose(A[:mid], leaf=leaf)
        bottom = recursive_transpose(A[mid:], leaf=leaf)
        return [top_row + bottom_row for top_row, bottom_row in zip(top, bottom)]
    mid = cols // 2
    left = recursive_transpose([row[:mid] for row in A], leaf=leaf)
    right = recursive_transpose([row[mid:] for row in A], leaf=leaf)
    return left + right


def row_scan(A):
    """Row-major traversal sum."""

    rows = len(A)
    cols = len(A[0])
    acc = _copy_proxy(A[0][0])
    for i in range(rows):
        start = 1 if i == 0 else 0
        for j in range(start, cols):
            acc = acc + A[i][j]
    return acc


def column_scan(A):
    """Column-major traversal sum."""

    rows = len(A)
    cols = len(A[0])
    acc = _copy_proxy(A[0][0])
    for j in range(cols):
        start = 1 if j == 0 else 0
        for i in range(start, rows):
            acc = acc + A[i][j]
    return acc


def _bit_reverse(index: int, bits: int) -> int:
    out = 0
    for _ in range(bits):
        out = (out << 1) | (index & 1)
        index >>= 1
    return out


def iterative_fft(x):
    """Iterative radix-2 Cooley-Tukey FFT."""

    n = len(x)
    bits = int(math.log2(n))
    out = [_copy_proxy(x[_bit_reverse(i, bits)]) for i in range(n)]
    size = 2
    while size <= n:
        half = size // 2
        omega_m = cmath.exp(-2j * math.pi / size)
        for start in range(0, n, size):
            omega = 1.0 + 0.0j
            for j in range(half):
                t = omega * out[start + j + half]
                u = out[start + j]
                out[start + j] = u + t
                out[start + j + half] = u - t
                omega *= omega_m
        size *= 2
    return out


def recursive_fft(x):
    """Recursive radix-2 Cooley-Tukey FFT."""

    n = len(x)
    if n == 1:
        return [_copy_proxy(x[0])]
    even = recursive_fft(x[0::2])
    odd = recursive_fft(x[1::2])
    out = [None] * n
    for k in range(n // 2):
        omega = cmath.exp(-2j * math.pi * k / n)
        t = omega * odd[k]
        out[k] = even[k] + t
        out[k + n // 2] = even[k] - t
    return out


def jacobi_stencil_naive(A):
    """One Jacobi sweep in naive row-major order."""

    n = len(A)
    out = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == 0 or j == 0 or i == n - 1 or j == n - 1:
                out[i][j] = _copy_proxy(A[i][j])
                continue
            acc = A[i][j] + A[i - 1][j]
            acc = acc + A[i + 1][j]
            acc = acc + A[i][j - 1]
            acc = acc + A[i][j + 1]
            out[i][j] = 0.2 * acc
    return out


def jacobi_stencil_recursive(A, *, leaf: int = 8):
    """Tile-recursive Jacobi sweep."""

    n = len(A)
    out = [[None] * n for _ in range(n)]

    def fill(r0: int, r1: int, c0: int, c1: int) -> None:
        if (r1 - r0) <= leaf and (c1 - c0) <= leaf:
            for i in range(r0, r1):
                for j in range(c0, c1):
                    if i == 0 or j == 0 or i == n - 1 or j == n - 1:
                        out[i][j] = _copy_proxy(A[i][j])
                        continue
                    acc = A[i][j] + A[i - 1][j]
                    acc = acc + A[i + 1][j]
                    acc = acc + A[i][j - 1]
                    acc = acc + A[i][j + 1]
                    out[i][j] = 0.2 * acc
            return
        if (r1 - r0) >= (c1 - c0):
            mid = (r0 + r1) // 2
            fill(r0, mid, c0, c1)
            fill(mid, r1, c0, c1)
        else:
            mid = (c0 + c1) // 2
            fill(r0, r1, c0, mid)
            fill(r0, r1, mid, c1)

    fill(0, n, 0, n)
    return out


def regular_attention(Q, K, V):
    """Naive attention with proxy softmax ops."""

    n = len(Q)
    d = len(Q[0])

    scores = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            acc = Q[i][0] * K[j][0]
            for dd in range(1, d):
                acc = acc + Q[i][dd] * K[j][dd]
            scores[i][j] = acc

    probs = [[None] * n for _ in range(n)]
    for i in range(n):
        row_max = scores[i][0]
        for j in range(1, n):
            row_max = _max2(row_max, scores[i][j])

        row_sum = None
        for j in range(n):
            probs[i][j] = _exp_proxy(scores[i][j] - row_max)
            row_sum = probs[i][j] if row_sum is None else row_sum + probs[i][j]

        inv_sum = _inv_proxy(row_sum)
        for j in range(n):
            probs[i][j] = probs[i][j] * inv_sum

    out = [[None] * d for _ in range(n)]
    for i in range(n):
        for dd in range(d):
            acc = probs[i][0] * V[0][dd]
            for j in range(1, n):
                acc = acc + probs[i][j] * V[j][dd]
            out[i][dd] = acc
    return out


def flash_attention(Q, K, V, *, bq: int = 8, bk: int = 4):
    """Double-tiled flash attention with snake-ordered K/V blocks."""

    n = len(Q)
    d = len(Q[0])
    num_kv_blocks = (n + bk - 1) // bk
    num_q_blocks = (n + bq - 1) // bq

    out = [[None] * d for _ in range(n)]
    max_state = [None] * n
    sum_state = [None] * n
    out_state = [[None] * d for _ in range(n)]

    for qb in range(num_q_blocks):
        q_start = qb * bq
        q_end = min(q_start + bq, n)
        block_order = range(num_kv_blocks) if qb % 2 == 0 else range(num_kv_blocks - 1, -1, -1)

        for kb in block_order:
            k_start = kb * bk
            k_end = min(k_start + bk, n)
            block_size = k_end - k_start

            for i in range(q_start, q_end):
                scores = [None] * block_size
                for j in range(block_size):
                    acc = Q[i][0] * K[k_start + j][0]
                    for dd in range(1, d):
                        acc = acc + Q[i][dd] * K[k_start + j][dd]
                    scores[j] = acc

                block_max = scores[0]
                for j in range(1, block_size):
                    block_max = _max2(block_max, scores[j])

                probs = [None] * block_size
                block_sum = None
                for j in range(block_size):
                    probs[j] = _exp_proxy(scores[j] - block_max)
                    block_sum = probs[j] if block_sum is None else block_sum + probs[j]

                block_out = [None] * d
                for dd in range(d):
                    acc = probs[0] * V[k_start][dd]
                    for j in range(1, block_size):
                        acc = acc + probs[j] * V[k_start + j][dd]
                    block_out[dd] = acc

                prev_max = max_state[i]
                prev_sum = sum_state[i]
                prev_out = out_state[i]

                if prev_max is None:
                    max_state[i] = block_max
                    sum_state[i] = block_sum
                    for dd in range(d):
                        prev_out[dd] = block_out[dd]
                    continue

                merged_max = _max2(prev_max, block_max)
                alpha = _exp_proxy(prev_max - merged_max)
                beta = _exp_proxy(block_max - merged_max)
                sum_state[i] = alpha * prev_sum + beta * block_sum
                for dd in range(d):
                    prev_out[dd] = alpha * prev_out[dd] + beta * block_out[dd]
                max_state[i] = merged_max

    for i in range(n):
        inv_sum = _inv_proxy(sum_state[i])
        for dd in range(d):
            out[i][dd] = out_state[i][dd] * inv_sum
    return out


def matvec_flops(n: int) -> int:
    return n * (2 * n - 1)


def vecmat_flops(n: int) -> int:
    return matvec_flops(n)


def transpose_flops(rows: int, cols: int) -> int:
    return rows * cols


def scan_flops(rows: int, cols: int) -> int:
    return rows * cols - 1


def fft_flops(n: int) -> int:
    return 3 * (n // 2) * int(math.log2(n))


def jacobi_flops(n: int) -> int:
    interior = max(0, n - 2) ** 2
    return n * n + 5 * interior


def regular_attention_flops(n: int, d: int) -> int:
    return n * n * (2 * d - 1) + n * ((n - 1) + n + n + (n - 1) + 1 + n) + n * d * (2 * n - 1)


def flash_attention_flops(n: int, d: int, bk: int) -> int:
    total = 0
    num_blocks = (n + bk - 1) // bk
    for _ in range(n):
        for kb in range(num_blocks):
            block_size = min(bk, n - kb * bk)
            total += block_size * (2 * d - 1)
            total += max(0, block_size - 1)
            total += block_size + block_size
            total += max(0, block_size - 1)
            total += d * (2 * block_size - 1)
            if kb > 0:
                total += 1 + 2 + 2 + 3 + d * 3
        total += 1 + d
    return total


@dataclass(frozen=True)
class AlgorithmSpec:
    key: str
    label: str
    workload: str
    notes: str
    func: Callable[..., object]
    args_factory: Callable[[], tuple]
    flops: int


def build_algorithm_specs() -> list[AlgorithmSpec]:
    """Small workloads chosen so every traced metric finishes well under 10s."""

    return [
        AlgorithmSpec(
            key="matvec-32",
            label="Matvec",
            workload="32x32 by 32",
            notes="row-wise matrix-vector baseline",
            func=matvec,
            args_factory=lambda: (make_matrix(32), make_vector(32)),
            flops=matvec_flops(32),
        ),
        AlgorithmSpec(
            key="vecmat-32",
            label="Vecmat",
            workload="32 by 32x32",
            notes="column-oriented access order",
            func=vecmat,
            args_factory=lambda: (make_matrix(32), make_vector(32)),
            flops=vecmat_flops(32),
        ),
        AlgorithmSpec(
            key="transpose-naive-32",
            label="Transpose (Naive)",
            workload="32x32",
            notes="direct row-major transpose copy",
            func=naive_transpose,
            args_factory=lambda: (make_matrix(32),),
            flops=transpose_flops(32, 32),
        ),
        AlgorithmSpec(
            key="transpose-blocked-32",
            label="Transpose (Blocked)",
            workload="32x32, block=8",
            notes="blocked transpose copy",
            func=lambda A: blocked_transpose(A, block=8),
            args_factory=lambda: (make_matrix(32),),
            flops=transpose_flops(32, 32),
        ),
        AlgorithmSpec(
            key="transpose-recursive-32",
            label="Transpose (Recursive)",
            workload="32x32, leaf=8",
            notes="cache-oblivious recursive transpose",
            func=lambda A: recursive_transpose(A, leaf=8),
            args_factory=lambda: (make_matrix(32),),
            flops=transpose_flops(32, 32),
        ),
        AlgorithmSpec(
            key="scan-row-64",
            label="Row Scan",
            workload="64x64",
            notes="row-major traversal sum",
            func=row_scan,
            args_factory=lambda: (make_matrix(64),),
            flops=scan_flops(64, 64),
        ),
        AlgorithmSpec(
            key="scan-column-64",
            label="Column Scan",
            workload="64x64",
            notes="column-major traversal sum",
            func=column_scan,
            args_factory=lambda: (make_matrix(64),),
            flops=scan_flops(64, 64),
        ),
        AlgorithmSpec(
            key="naive-matmul-16",
            label="Naive Matmul",
            workload="16x16",
            notes="standard i-j-k triple loop",
            func=naive_matmul,
            args_factory=lambda: (make_matrix(16), make_matrix(16, offset=1000)),
            flops=flops_naive(16),
        ),
        AlgorithmSpec(
            key="tiled-matmul-16",
            label="Tiled Matmul",
            workload="16x16, tile=4",
            notes="one explicit blocking level",
            func=one_level_tiled_matmul,
            args_factory=lambda: (make_matrix(16), make_matrix(16, offset=1000)),
            flops=flops_naive(16),
        ),
        AlgorithmSpec(
            key="rmm-16",
            label="Recursive Matmul",
            workload="16x16",
            notes="8-way cache-oblivious recursion",
            func=rmm,
            args_factory=lambda: (make_matrix(16), make_matrix(16, offset=1000)),
            flops=flops_rmm(16),
        ),
        AlgorithmSpec(
            key="rmm-lex-16",
            label="Recursive In-Place (Lex)",
            workload="16x16",
            notes="manual in-place schedule, lexicographic order",
            func=rmm_inplace_lex,
            args_factory=lambda: (make_matrix(16), make_matrix(16, offset=1000), zeros(16)),
            flops=flops_rmm(16),
        ),
        AlgorithmSpec(
            key="rmm-gray-16",
            label="Recursive In-Place (Gray)",
            workload="16x16",
            notes="manual in-place schedule, Gray-code order",
            func=rmm_inplace_gray,
            args_factory=lambda: (make_matrix(16), make_matrix(16, offset=1000), zeros(16)),
            flops=flops_rmm(16),
        ),
        AlgorithmSpec(
            key="strassen-16",
            label="Strassen",
            workload="16x16",
            notes="leaf size 1 to expose temporary traffic",
            func=strassen,
            args_factory=lambda: (make_matrix(16), make_matrix(16, offset=1000)),
            flops=flops_strassen(16),
        ),
        AlgorithmSpec(
            key="fft-iterative-32",
            label="FFT (Iterative)",
            workload="N=32",
            notes="iterative radix-2 Cooley-Tukey",
            func=iterative_fft,
            args_factory=lambda: ([complex(v, 0.5 * v) for v in make_vector(32)],),
            flops=fft_flops(32),
        ),
        AlgorithmSpec(
            key="fft-recursive-32",
            label="FFT (Recursive)",
            workload="N=32",
            notes="recursive radix-2 Cooley-Tukey",
            func=recursive_fft,
            args_factory=lambda: ([complex(v, 0.5 * v) for v in make_vector(32)],),
            flops=fft_flops(32),
        ),
        AlgorithmSpec(
            key="jacobi-naive-32",
            label="Stencil (Naive)",
            workload="32x32, one sweep",
            notes="row-major Jacobi stencil",
            func=jacobi_stencil_naive,
            args_factory=lambda: (make_matrix(32),),
            flops=jacobi_flops(32),
        ),
        AlgorithmSpec(
            key="jacobi-recursive-32",
            label="Stencil (Recursive)",
            workload="32x32, one sweep, leaf=8",
            notes="tile-recursive Jacobi stencil",
            func=lambda A: jacobi_stencil_recursive(A, leaf=8),
            args_factory=lambda: (make_matrix(32),),
            flops=jacobi_flops(32),
        ),
        AlgorithmSpec(
            key="regular-attention-32x4",
            label="Regular Attention",
            workload="N=32, d=4",
            notes="materializes the full score matrix",
            func=regular_attention,
            args_factory=lambda: (
                make_matrix(32, 4),
                make_matrix(32, 4, offset=1000),
                make_matrix(32, 4, offset=2000),
            ),
            flops=regular_attention_flops(32, 4),
        ),
        AlgorithmSpec(
            key="flash-attention-32x4",
            label="Flash Attention",
            workload="N=32, d=4, Bq=8, Bk=4",
            notes="double-tiled Q/KV blocks with snake KV order",
            func=lambda Q, K, V: flash_attention(Q, K, V, bq=8, bk=4),
            args_factory=lambda: (
                make_matrix(32, 4),
                make_matrix(32, 4, offset=1000),
                make_matrix(32, 4, offset=2000),
            ),
            flops=flash_attention_flops(32, 4, 4),
        ),
    ]
