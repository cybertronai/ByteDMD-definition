"""Hand-scheduled fixed-address implementations for heuristic-grid kernels."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Callable

from experiments.grid.algorithms import make_pivot_matrix
from experiments.grid.fused_strassen_trace import run_fused_strassen, run_rmm
from experiments.memory_management.tracer import trace_to_cost_continuous, trace_to_cost_discrete


@dataclass(frozen=True)
class VectorView:
    base: int
    length: int
    stride: int = 1

    def addr(self, index: int) -> int:
        return self.base + index * self.stride

    def even(self) -> "VectorView":
        return VectorView(self.base, self.length // 2, self.stride * 2)

    def odd(self) -> "VectorView":
        return VectorView(self.base + self.stride, self.length // 2, self.stride * 2)

    def first_half(self) -> "VectorView":
        return VectorView(self.base, self.length // 2, self.stride)

    def second_half(self) -> "VectorView":
        return VectorView(self.base + (self.length // 2) * self.stride, self.length // 2, self.stride)


@dataclass(frozen=True)
class MatrixView:
    base: int
    rows: int
    cols: int
    stride: int

    def addr(self, row: int, col: int) -> int:
        return self.base + row * self.stride + col

    def row(self, row: int) -> VectorView:
        return VectorView(self.base + row * self.stride, self.cols, 1)

    def col(self, col: int) -> VectorView:
        return VectorView(self.base + col, self.rows, self.stride)

    def block(self, row: int, col: int, rows: int, cols: int) -> "MatrixView":
        return MatrixView(self.base + row * self.stride + col, rows, cols, self.stride)


class ManualTracer:
    """Simple fixed-address read trace."""

    def __init__(self) -> None:
        self.next_addr = 1
        self.trace: list[int] = []

    def alloc_vector(self, length: int) -> VectorView:
        view = VectorView(self.next_addr, length)
        self.next_addr += length
        return view

    def alloc_matrix(self, rows: int, cols: int, *, stride: int | None = None) -> MatrixView:
        layout_stride = cols if stride is None else stride
        view = MatrixView(self.next_addr, rows, cols, layout_stride)
        self.next_addr += rows * layout_stride
        return view

    def read(self, addr: int) -> None:
        self.trace.append(addr)

    def read_vector(self, view: VectorView) -> None:
        for index in range(view.length):
            self.read(view.addr(index))

    def result(self) -> dict[str, object]:
        return {
            "trace": list(self.trace),
            "cost_discrete": trace_to_cost_discrete(self.trace),
            "cost_continuous": trace_to_cost_continuous(self.trace),
            "n_reads": len(self.trace),
            "max_address": max(self.trace, default=0),
        }


def _make_matrix_block_views(matrix: MatrixView) -> tuple[MatrixView, MatrixView, MatrixView, MatrixView]:
    half = matrix.rows // 2
    return (
        matrix.block(0, 0, half, half),
        matrix.block(0, half, half, half),
        matrix.block(half, 0, half, half),
        matrix.block(half, half, half, half),
    )


def manual_matvec(n: int = 32) -> dict[str, object]:
    tracer = ManualTracer()
    acc = tracer.alloc_vector(1)
    x = tracer.alloc_vector(n)
    A = tracer.alloc_matrix(n, n)
    tracer.alloc_vector(n)  # y

    for i in range(n):
        tracer.read(A.addr(i, 0))
        tracer.read(x.addr(0))
        for j in range(1, n):
            tracer.read(acc.addr(0))
            tracer.read(A.addr(i, j))
            tracer.read(x.addr(j))
    return tracer.result()


def manual_vecmat(n: int = 32) -> dict[str, object]:
    tracer = ManualTracer()
    acc = tracer.alloc_vector(1)
    x = tracer.alloc_vector(n)
    A = tracer.alloc_matrix(n, n)
    tracer.alloc_vector(n)  # y

    for j in range(n):
        tracer.read(x.addr(0))
        tracer.read(A.addr(0, j))
        for i in range(1, n):
            tracer.read(acc.addr(0))
            tracer.read(x.addr(i))
            tracer.read(A.addr(i, j))
    return tracer.result()


def manual_transpose(rows: int = 32, cols: int = 32, *, block: int | None = None, recursive: bool = False) -> dict[str, object]:
    tracer = ManualTracer()
    A = tracer.alloc_matrix(rows, cols)
    tracer.alloc_matrix(cols, rows)

    def visit(r0: int, r1: int, c0: int, c1: int) -> None:
        if recursive and (r1 - r0) > 8 and (c1 - c0) > 8:
            if (r1 - r0) >= (c1 - c0):
                mid = (r0 + r1) // 2
                visit(r0, mid, c0, c1)
                visit(mid, r1, c0, c1)
            else:
                mid = (c0 + c1) // 2
                visit(r0, r1, c0, mid)
                visit(r0, r1, mid, c1)
            return
        if block is None:
            for i in range(r0, r1):
                for j in range(c0, c1):
                    tracer.read(A.addr(i, j))
            return
        for bi in range(r0, r1, block):
            for bj in range(c0, c1, block):
                for i in range(bi, min(bi + block, r1)):
                    for j in range(bj, min(bj + block, c1)):
                        tracer.read(A.addr(i, j))

    visit(0, rows, 0, cols)
    return tracer.result()


def manual_scan(rows: int = 64, cols: int = 64, *, by_column: bool) -> dict[str, object]:
    tracer = ManualTracer()
    acc = tracer.alloc_vector(1)
    A = tracer.alloc_matrix(rows, cols)

    tracer.read(A.addr(0, 0))
    if by_column:
        for j in range(cols):
            start = 1 if j == 0 else 0
            for i in range(start, rows):
                tracer.read(acc.addr(0))
                tracer.read(A.addr(i, j))
    else:
        for i in range(rows):
            start = 1 if i == 0 else 0
            for j in range(start, cols):
                tracer.read(acc.addr(0))
                tracer.read(A.addr(i, j))
    return tracer.result()


def manual_naive_matmul(n: int = 16) -> dict[str, object]:
    tracer = ManualTracer()
    acc = tracer.alloc_vector(1)
    A = tracer.alloc_matrix(n, n)
    B = tracer.alloc_matrix(n, n)
    tracer.alloc_matrix(n, n)

    for i in range(n):
        for j in range(n):
            tracer.read(A.addr(i, 0))
            tracer.read(B.addr(0, j))
            for k in range(1, n):
                tracer.read(acc.addr(0))
                tracer.read(A.addr(i, k))
                tracer.read(B.addr(k, j))
    return tracer.result()


def manual_tiled_matmul(n: int = 16, tile: int = 4) -> dict[str, object]:
    tracer = ManualTracer()
    fast_A = tracer.alloc_matrix(tile, tile)
    fast_B = tracer.alloc_matrix(tile, tile)
    fast_C = tracer.alloc_matrix(tile, tile)
    A = tracer.alloc_matrix(n, n)
    B = tracer.alloc_matrix(n, n)
    tracer.alloc_matrix(n, n)

    for i0 in range(0, n, tile):
        for j0 in range(0, n, tile):
            for k0 in range(0, n, tile):
                for i in range(tile):
                    for k in range(tile):
                        tracer.read(A.addr(i0 + i, k0 + k))
                for k in range(tile):
                    for j in range(tile):
                        tracer.read(B.addr(k0 + k, j0 + j))
                for i in range(tile):
                    for j in range(tile):
                        if k0 > 0:
                            tracer.read(fast_C.addr(i, j))
                        for k in range(tile):
                            tracer.read(fast_A.addr(i, k))
                            tracer.read(fast_B.addr(k, j))
    return tracer.result()


def manual_recursive_matmul() -> dict[str, object]:
    trace, _, _ = run_rmm(16, tile=4)
    return {
        "trace": trace,
        "cost_discrete": trace_to_cost_discrete(trace),
        "cost_continuous": trace_to_cost_continuous(trace),
        "n_reads": len(trace),
        "max_address": max(trace, default=0),
    }


def _manual_inplace_rmm(order: list[tuple[int, int, int]], n: int = 16) -> dict[str, object]:
    tracer = ManualTracer()
    A = tracer.alloc_matrix(n, n)
    B = tracer.alloc_matrix(n, n)
    C = tracer.alloc_matrix(n, n)

    def rec(ai: int, aj: int, bi: int, bj: int, ci: int, cj: int, size: int) -> None:
        if size == 1:
            tracer.read(C.addr(ci, cj))
            tracer.read(A.addr(ai, aj))
            tracer.read(B.addr(bi, bj))
            return
        half = size // 2
        for i_off, j_off, k_off in order:
            rec(
                ai + i_off * half,
                aj + k_off * half,
                bi + k_off * half,
                bj + j_off * half,
                ci + i_off * half,
                cj + j_off * half,
                half,
            )

    rec(0, 0, 0, 0, 0, 0, n)
    return tracer.result()


LEX_ORDER = [
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
    (0, 1, 1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
]

GRAY_ORDER = [
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 1),
    (0, 1, 0),
    (1, 1, 0),
    (1, 1, 1),
    (1, 0, 1),
    (1, 0, 0),
]


def _load_optimal_memory_module(module_name: str):
    path = Path(__file__).resolve().parents[1] / "optimal-memory" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def manual_strassen() -> dict[str, object]:
    module = _load_optimal_memory_module("optimal_strassen")
    ext_reads, wm_reads, _wm_writes = module.generate_strassen_traces(16)
    trace = list(ext_reads) + list(wm_reads)
    return {
        "trace": trace,
        "cost_discrete": trace_to_cost_discrete(trace),
        "cost_continuous": trace_to_cost_continuous(trace),
        "n_reads": len(trace),
        "max_address": max(trace, default=0),
    }


def manual_fused_strassen() -> dict[str, object]:
    trace, _, _ = run_fused_strassen(16, tile=4)
    return {
        "trace": trace,
        "cost_discrete": trace_to_cost_discrete(trace),
        "cost_continuous": trace_to_cost_continuous(trace),
        "n_reads": len(trace),
        "max_address": max(trace, default=0),
    }


def _bit_reverse(index: int, bits: int) -> int:
    out = 0
    for _ in range(bits):
        out = (out << 1) | (index & 1)
        index >>= 1
    return out


def manual_iterative_fft(n: int = 1024) -> dict[str, object]:
    tracer = ManualTracer()
    x = tracer.alloc_vector(n)
    out = tracer.alloc_vector(n)

    bits = n.bit_length() - 1
    for i in range(n):
        tracer.read(x.addr(_bit_reverse(i, bits)))

    size = 2
    while size <= n:
        half = size // 2
        for start in range(0, n, size):
            for j in range(half):
                tracer.read(out.addr(start + j))
                tracer.read(out.addr(start + j + half))
        size *= 2
    return tracer.result()


def _manual_recursive_fft(tracer: ManualTracer, src: VectorView, dst: VectorView) -> None:
    if src.length == 1:
        tracer.read(src.addr(0))
        return
    even_src = src.even()
    odd_src = src.odd()
    even_dst = dst.first_half()
    odd_dst = dst.second_half()
    _manual_recursive_fft(tracer, even_src, even_dst)
    _manual_recursive_fft(tracer, odd_src, odd_dst)
    for k in range(src.length // 2):
        tracer.read(even_dst.addr(k))
        tracer.read(odd_dst.addr(k))


def manual_recursive_fft(n: int = 1024) -> dict[str, object]:
    tracer = ManualTracer()
    src = tracer.alloc_vector(n)
    dst = tracer.alloc_vector(n)
    _manual_recursive_fft(tracer, src, dst)
    return tracer.result()


def manual_spatial_conv2d(rows: int = 16, cols: int = 16, k_rows: int = 5, k_cols: int = 5) -> dict[str, object]:
    tracer = ManualTracer()
    acc = tracer.alloc_vector(1)
    image = tracer.alloc_matrix(rows, cols)
    kernel = tracer.alloc_matrix(k_rows, k_cols)
    tracer.alloc_matrix(rows, cols)
    center_r = k_rows // 2
    center_c = k_cols // 2

    for i in range(rows):
        for j in range(cols):
            first = True
            for u in range(k_rows):
                ii = i + center_r - u
                if ii < 0 or ii >= rows:
                    continue
                for v in range(k_cols):
                    jj = j + center_c - v
                    if jj < 0 or jj >= cols:
                        continue
                    if not first:
                        tracer.read(acc.addr(0))
                    tracer.read(image.addr(ii, jj))
                    tracer.read(kernel.addr(u, v))
                    first = False
    return tracer.result()


def _manual_fft2d(tracer: ManualTracer, src: MatrixView, row_tmp: MatrixView, dst: MatrixView) -> None:
    for row in range(src.rows):
        _manual_recursive_fft(tracer, src.row(row), row_tmp.row(row))
    for col in range(src.cols):
        _manual_recursive_fft(tracer, row_tmp.col(col), dst.col(col))


def manual_fft_conv2d(rows: int = 16, cols: int = 16, k_rows: int = 5, k_cols: int = 5) -> dict[str, object]:
    tracer = ManualTracer()
    image = tracer.alloc_matrix(rows, cols)
    kernel = tracer.alloc_matrix(k_rows, k_cols)
    pad_rows = 32
    pad_cols = 32
    padded_image = tracer.alloc_matrix(pad_rows, pad_cols)
    padded_kernel = tracer.alloc_matrix(pad_rows, pad_cols)
    image_row_tmp = tracer.alloc_matrix(pad_rows, pad_cols)
    image_freq = tracer.alloc_matrix(pad_rows, pad_cols)
    kernel_row_tmp = tracer.alloc_matrix(pad_rows, pad_cols)
    kernel_freq = tracer.alloc_matrix(pad_rows, pad_cols)
    spectrum = tracer.alloc_matrix(pad_rows, pad_cols)
    inverse_row_tmp = tracer.alloc_matrix(pad_rows, pad_cols)
    recovered = tracer.alloc_matrix(pad_rows, pad_cols)
    tracer.alloc_matrix(rows, cols)

    for i in range(rows):
        for j in range(cols):
            tracer.read(image.addr(i, j))
    for i in range(k_rows):
        for j in range(k_cols):
            tracer.read(kernel.addr(i, j))

    _manual_fft2d(tracer, padded_image, image_row_tmp, image_freq)
    _manual_fft2d(tracer, padded_kernel, kernel_row_tmp, kernel_freq)

    for i in range(pad_rows):
        for j in range(pad_cols):
            tracer.read(image_freq.addr(i, j))
            tracer.read(kernel_freq.addr(i, j))

    _manual_fft2d(tracer, spectrum, inverse_row_tmp, recovered)
    start_r = k_rows // 2
    start_c = k_cols // 2
    for i in range(rows):
        for j in range(cols):
            tracer.read(recovered.addr(start_r + i, start_c + j))

    return tracer.result()


def manual_regular_conv(rows: int = 16, cols: int = 16, k_rows: int = 3, k_cols: int = 3, in_channels: int = 4, out_channels: int = 4) -> dict[str, object]:
    tracer = ManualTracer()
    acc = tracer.alloc_vector(1)
    image = tracer.alloc_matrix(rows * cols, in_channels)
    kernel = tracer.alloc_matrix(k_rows * k_cols * in_channels, out_channels)
    tracer.alloc_matrix(rows * cols, out_channels)
    center_r = k_rows // 2
    center_c = k_cols // 2

    for i in range(rows):
        for j in range(cols):
            pixel = i * cols + j
            for co in range(out_channels):
                first = True
                for u in range(k_rows):
                    ii = i + center_r - u
                    if ii < 0 or ii >= rows:
                        continue
                    for v in range(k_cols):
                        jj = j + center_c - v
                        if jj < 0 or jj >= cols:
                            continue
                        src_pixel = ii * cols + jj
                        for ci in range(in_channels):
                            if not first:
                                tracer.read(acc.addr(0))
                            tracer.read(image.addr(src_pixel, ci))
                            tracer.read(kernel.addr((u * k_cols + v) * in_channels + ci, co))
                            first = False
    return tracer.result()


def manual_fft_conv1d(n: int = 32) -> dict[str, object]:
    tracer = ManualTracer()
    signal = tracer.alloc_vector(n)
    kernel = tracer.alloc_vector(n)
    signal_freq = tracer.alloc_vector(n)
    kernel_freq = tracer.alloc_vector(n)
    spectrum = tracer.alloc_vector(n)
    out = tracer.alloc_vector(n)

    _manual_recursive_fft(tracer, signal, signal_freq)
    _manual_recursive_fft(tracer, kernel, kernel_freq)
    for i in range(n):
        tracer.read(signal_freq.addr(i))
        tracer.read(kernel_freq.addr(i))
    _manual_recursive_fft(tracer, spectrum, out)
    return tracer.result()


def manual_mergesort(n: int = 64) -> dict[str, object]:
    tracer = ManualTracer()
    src = tracer.alloc_vector(n)
    temp = tracer.alloc_vector(n)

    def rec(view: VectorView, scratch: VectorView) -> None:
        if view.length <= 1:
            tracer.read(view.addr(0))
            return
        left = view.first_half()
        right = view.second_half()
        left_scratch = scratch.first_half()
        right_scratch = scratch.second_half()
        rec(left, left_scratch)
        rec(right, right_scratch)

        i = 0
        j = 0
        while i < left.length and j < right.length:
            tracer.read(left.addr(i))
            tracer.read(right.addr(j))
            if i <= j:
                i += 1
            else:
                j += 1
        while i < left.length:
            tracer.read(left.addr(i))
            i += 1
        while j < right.length:
            tracer.read(right.addr(j))
            j += 1

    rec(src, temp)
    return tracer.result()


def manual_lcs_dp(m: int = 32, n: int = 32) -> dict[str, object]:
    tracer = ManualTracer()
    seq_a = tracer.alloc_vector(m)
    seq_b = tracer.alloc_vector(n)
    dp = tracer.alloc_matrix(m + 1, n + 1)

    for i in range(m):
        for j in range(n):
            tracer.read(seq_a.addr(i))
            tracer.read(seq_b.addr(j))
            tracer.read(dp.addr(i, j))
            tracer.read(dp.addr(i, j + 1))
            tracer.read(dp.addr(i + 1, j))
    return tracer.result()


def manual_gaussian_elimination(n: int = 24) -> dict[str, object]:
    tracer = ManualTracer()
    pivot_inv = tracer.alloc_vector(1)
    factor = tracer.alloc_vector(1)
    acc = tracer.alloc_vector(1)
    pivot_row = tracer.alloc_vector(n)
    rhs_pivot = tracer.alloc_vector(1)
    A = tracer.alloc_matrix(n, n)
    b = tracer.alloc_vector(n)
    x = tracer.alloc_vector(n)

    for k in range(n - 1):
        tracer.read(A.addr(k, k))
        for j in range(k + 1, n):
            tracer.read(A.addr(k, j))
        tracer.read(b.addr(k))
        for i in range(k + 1, n):
            tracer.read(A.addr(i, k))
            tracer.read(pivot_inv.addr(0))
            for j in range(k + 1, n):
                tracer.read(factor.addr(0))
                tracer.read(A.addr(i, j))
                tracer.read(pivot_row.addr(j))
            tracer.read(factor.addr(0))
            tracer.read(b.addr(i))
            tracer.read(rhs_pivot.addr(0))

    for i in range(n - 1, -1, -1):
        tracer.read(b.addr(i))
        for j in range(i + 1, n):
            tracer.read(acc.addr(0))
            tracer.read(A.addr(i, j))
            tracer.read(x.addr(j))
        tracer.read(A.addr(i, i))
        tracer.read(acc.addr(0))
        tracer.read(pivot_inv.addr(0))

    return tracer.result()


def manual_gauss_jordan_inverse(n: int = 16) -> dict[str, object]:
    tracer = ManualTracer()
    pivot_inv = tracer.alloc_vector(1)
    factor = tracer.alloc_vector(1)
    pivot_a_row = tracer.alloc_vector(n)
    pivot_inv_row = tracer.alloc_vector(n)
    A = tracer.alloc_matrix(n, n)
    inv = tracer.alloc_matrix(n, n)

    for k in range(n):
        tracer.read(A.addr(k, k))
        for j in range(n):
            tracer.read(A.addr(k, j))
            tracer.read(pivot_inv.addr(0))
            tracer.read(inv.addr(k, j))
            tracer.read(pivot_inv.addr(0))
        for i in range(n):
            if i == k:
                continue
            tracer.read(A.addr(i, k))
            for j in range(n):
                tracer.read(factor.addr(0))
                tracer.read(A.addr(i, j))
                tracer.read(pivot_a_row.addr(j))
                tracer.read(factor.addr(0))
                tracer.read(inv.addr(i, j))
                tracer.read(pivot_inv_row.addr(j))

    return tracer.result()


def _trace_lu_kernel(tracer: ManualTracer, matrix: MatrixView, pivot_inv: VectorView, factor: VectorView, row_scratch: VectorView, *, block: int | None) -> None:
    n = matrix.rows
    for k in range(n - 1):
        tracer.read(matrix.addr(k, k))
        if block is None:
            for j in range(k, n):
                tracer.read(matrix.addr(k, j))
            for i in range(k + 1, n):
                tracer.read(matrix.addr(i, k))
                tracer.read(pivot_inv.addr(0))
                for j in range(k, n):
                    tracer.read(factor.addr(0))
                    tracer.read(matrix.addr(i, j))
                    tracer.read(row_scratch.addr(j))
            continue

        for j0 in range(k, n, block):
            width = min(block, n - j0)
            for jj in range(width):
                tracer.read(matrix.addr(k, j0 + jj))
            for i in range(k + 1, n):
                tracer.read(matrix.addr(i, k))
                tracer.read(pivot_inv.addr(0))
                for jj in range(width):
                    tracer.read(factor.addr(0))
                    tracer.read(matrix.addr(i, j0 + jj))
                    tracer.read(row_scratch.addr(jj))


def _trace_recursive_lu(tracer: ManualTracer, matrix: MatrixView, pivot_inv: VectorView, factor: VectorView, row_scratch: VectorView, *, leaf: int) -> None:
    if matrix.rows <= leaf or (matrix.rows % 2) == 1:
        _trace_lu_kernel(tracer, matrix, pivot_inv, factor, row_scratch, block=None)
        return

    half = matrix.rows // 2
    q11 = matrix.block(0, 0, half, half)
    q12 = matrix.block(0, half, half, half)
    q21 = matrix.block(half, 0, half, half)
    q22 = matrix.block(half, half, half, half)

    _trace_recursive_lu(tracer, q11, pivot_inv, factor, row_scratch, leaf=leaf)

    for i in range(half):
        for j in range(half):
            tracer.read(q12.addr(i, j))
            for k in range(i):
                tracer.read(q11.addr(i, k))
                tracer.read(q12.addr(k, j))

    for i in range(half):
        for j in range(half):
            tracer.read(q21.addr(i, j))
            for k in range(j):
                tracer.read(q21.addr(i, k))
                tracer.read(q11.addr(k, j))

    for i in range(half):
        for j in range(half):
            for k in range(half):
                tracer.read(q21.addr(i, k))
                tracer.read(q12.addr(k, j))

    _trace_recursive_lu(tracer, q22, pivot_inv, factor, row_scratch, leaf=leaf)


def manual_lu_no_pivot(n: int = 24) -> dict[str, object]:
    tracer = ManualTracer()
    pivot_inv = tracer.alloc_vector(1)
    factor = tracer.alloc_vector(1)
    row_scratch = tracer.alloc_vector(n)
    A = tracer.alloc_matrix(n, n)

    _trace_lu_kernel(tracer, A, pivot_inv, factor, row_scratch, block=None)
    return tracer.result()


def manual_blocked_lu(n: int = 24, block: int = 4) -> dict[str, object]:
    tracer = ManualTracer()
    pivot_inv = tracer.alloc_vector(1)
    factor = tracer.alloc_vector(1)
    row_scratch = tracer.alloc_vector(block)
    A = tracer.alloc_matrix(n, n)

    _trace_lu_kernel(tracer, A, pivot_inv, factor, row_scratch, block=block)
    return tracer.result()


def manual_recursive_lu(n: int = 24, leaf: int = 6) -> dict[str, object]:
    tracer = ManualTracer()
    pivot_inv = tracer.alloc_vector(1)
    factor = tracer.alloc_vector(1)
    row_scratch = tracer.alloc_vector(n)
    A = tracer.alloc_matrix(n, n)

    _trace_recursive_lu(tracer, A, pivot_inv, factor, row_scratch, leaf=leaf)
    return tracer.result()


def manual_lu_partial_pivot(n: int = 24) -> dict[str, object]:
    tracer = ManualTracer()
    pivot_inv = tracer.alloc_vector(1)
    factor = tracer.alloc_vector(1)
    A = tracer.alloc_matrix(n, n)
    values = make_pivot_matrix(n)

    for k in range(n - 1):
        pivot = k
        pivot_abs = abs(values[k][k])
        for i in range(k, n):
            tracer.read(A.addr(i, k))
            candidate = abs(values[i][k])
            if candidate > pivot_abs:
                pivot = i
                pivot_abs = candidate

        if pivot != k:
            for j in range(n):
                tracer.read(A.addr(k, j))
                tracer.read(A.addr(pivot, j))
            values[k], values[pivot] = values[pivot], values[k]

        tracer.read(A.addr(k, k))
        pivot_value = values[k][k]
        for i in range(k + 1, n):
            tracer.read(A.addr(i, k))
            tracer.read(pivot_inv.addr(0))
            factor_value = values[i][k] / pivot_value
            values[i][k] = factor_value
            for j in range(k + 1, n):
                tracer.read(factor.addr(0))
                tracer.read(A.addr(i, j))
                tracer.read(A.addr(k, j))
                values[i][j] -= factor_value * values[k][j]

    return tracer.result()


def _trace_cholesky_kernel(tracer: ManualTracer, matrix: MatrixView, diag: VectorView, dot: VectorView, *, block: int | None) -> None:
    n = matrix.rows
    if block is None:
        for k in range(n):
            tracer.read(matrix.addr(k, k))
            for s in range(k):
                tracer.read(matrix.addr(k, s))
                tracer.read(matrix.addr(k, s))
            tracer.read(diag.addr(0))
            for i in range(k + 1, n):
                tracer.read(matrix.addr(i, k))
                for s in range(k):
                    tracer.read(matrix.addr(i, s))
                    tracer.read(matrix.addr(k, s))
                tracer.read(dot.addr(0))
        return

    for k0 in range(0, n, block):
        k1 = min(k0 + block, n)
        for k in range(k0, k1):
            tracer.read(matrix.addr(k, k))
            for s in range(k0, k):
                tracer.read(matrix.addr(k, s))
                tracer.read(matrix.addr(k, s))
            tracer.read(diag.addr(0))
            for i in range(k + 1, n):
                tracer.read(matrix.addr(i, k))
                for s in range(k0, k):
                    tracer.read(matrix.addr(i, s))
                    tracer.read(matrix.addr(k, s))
                tracer.read(dot.addr(0))

        for i0 in range(k1, n, block):
            for j0 in range(k1, i0 + 1, block):
                i1 = min(i0 + block, n)
                j1 = min(j0 + block, n)
                for i in range(i0, i1):
                    for j in range(j0, min(j1, i + 1)):
                        tracer.read(matrix.addr(i, j))
                        for s in range(k0, k1):
                            tracer.read(matrix.addr(i, s))
                            tracer.read(matrix.addr(j, s))


def _trace_recursive_cholesky(tracer: ManualTracer, matrix: MatrixView, diag: VectorView, dot: VectorView, *, leaf: int) -> None:
    if matrix.rows <= leaf or (matrix.rows % 2) == 1:
        _trace_cholesky_kernel(tracer, matrix, diag, dot, block=None)
        return

    half = matrix.rows // 2
    q11 = matrix.block(0, 0, half, half)
    q21 = matrix.block(half, 0, half, half)
    q22 = matrix.block(half, half, half, half)

    _trace_recursive_cholesky(tracer, q11, diag, dot, leaf=leaf)

    for i in range(half):
        for j in range(half):
            tracer.read(q21.addr(i, j))
            for k in range(j):
                tracer.read(q21.addr(i, k))
                tracer.read(q11.addr(j, k))
            tracer.read(diag.addr(0))

    for i in range(half):
        for j in range(i + 1):
            tracer.read(q22.addr(i, j))
            for k in range(half):
                tracer.read(q21.addr(i, k))
                tracer.read(q21.addr(j, k))

    _trace_recursive_cholesky(tracer, q22, diag, dot, leaf=leaf)


def manual_cholesky(n: int = 24) -> dict[str, object]:
    tracer = ManualTracer()
    diag = tracer.alloc_vector(1)
    dot = tracer.alloc_vector(1)
    A = tracer.alloc_matrix(n, n)

    _trace_cholesky_kernel(tracer, A, diag, dot, block=None)
    return tracer.result()


def manual_blocked_cholesky(n: int = 24, block: int = 4) -> dict[str, object]:
    tracer = ManualTracer()
    diag = tracer.alloc_vector(1)
    dot = tracer.alloc_vector(1)
    A = tracer.alloc_matrix(n, n)

    _trace_cholesky_kernel(tracer, A, diag, dot, block=block)
    return tracer.result()


def manual_recursive_cholesky(n: int = 24, leaf: int = 6) -> dict[str, object]:
    tracer = ManualTracer()
    diag = tracer.alloc_vector(1)
    dot = tracer.alloc_vector(1)
    A = tracer.alloc_matrix(n, n)

    _trace_recursive_cholesky(tracer, A, diag, dot, leaf=leaf)
    return tracer.result()


def _trace_householder_qr_region(tracer: ManualTracer, matrix: MatrixView, v: VectorView, tau: VectorView, *, block: int | None) -> None:
    rows = matrix.rows
    cols = matrix.cols
    active = min(rows, cols)

    for k in range(active):
        for i in range(k, rows):
            tracer.read(matrix.addr(i, k))
        for i in range(k, rows):
            tracer.read(matrix.addr(i, k))

        if block is None:
            column_blocks = [(k, cols)]
        else:
            column_blocks = [(j0, min(j0 + block, cols)) for j0 in range(k, cols, block)]

        for j0, j1 in column_blocks:
            for j in range(j0, j1):
                for i in range(k, rows):
                    tracer.read(v.addr(i - k))
                    tracer.read(matrix.addr(i, j))
                for i in range(k, rows):
                    tracer.read(tau.addr(0))
                    tracer.read(v.addr(i - k))
                    tracer.read(matrix.addr(i, j))


def manual_householder_qr(rows: int = 48, cols: int = 12) -> dict[str, object]:
    tracer = ManualTracer()
    v = tracer.alloc_vector(rows)
    tau = tracer.alloc_vector(1)
    A = tracer.alloc_matrix(rows, cols)

    _trace_householder_qr_region(tracer, A, v, tau, block=None)
    return tracer.result()


def manual_blocked_qr(rows: int = 48, cols: int = 12, block: int = 4) -> dict[str, object]:
    tracer = ManualTracer()
    v = tracer.alloc_vector(rows)
    tau = tracer.alloc_vector(1)
    A = tracer.alloc_matrix(rows, cols)

    _trace_householder_qr_region(tracer, A, v, tau, block=block)
    return tracer.result()


def manual_tsqr(rows: int = 48, cols: int = 12, leaf_rows: int = 12) -> dict[str, object]:
    tracer = ManualTracer()
    v = tracer.alloc_vector(rows)
    tau = tracer.alloc_vector(1)
    A = tracer.alloc_matrix(rows, cols)
    stacked_r = tracer.alloc_matrix(2 * cols, cols)

    def rec(view: MatrixView) -> None:
        if view.rows <= leaf_rows or view.rows <= 2 * view.cols:
            _trace_householder_qr_region(tracer, view, v, tau, block=None)
            return
        half = view.rows // 2
        top = view.block(0, 0, half, view.cols)
        bottom = view.block(half, 0, view.rows - half, view.cols)
        rec(top)
        rec(bottom)
        _trace_householder_qr_region(tracer, stacked_r.block(0, 0, 2 * view.cols, view.cols), v, tau, block=None)

    rec(A)
    return tracer.result()


def manual_jacobi(rows: int = 32, cols: int = 32, *, recursive: bool) -> dict[str, object]:
    tracer = ManualTracer()
    A = tracer.alloc_matrix(rows, cols)
    tracer.alloc_matrix(rows, cols)

    def sweep(r0: int, r1: int, c0: int, c1: int) -> None:
        if recursive and (r1 - r0) > 8 and (c1 - c0) > 8:
            if (r1 - r0) >= (c1 - c0):
                mid = (r0 + r1) // 2
                sweep(r0, mid, c0, c1)
                sweep(mid, r1, c0, c1)
            else:
                mid = (c0 + c1) // 2
                sweep(r0, r1, c0, mid)
                sweep(r0, r1, mid, c1)
            return
        for i in range(r0, r1):
            for j in range(c0, c1):
                if i == 0 or j == 0 or i == rows - 1 or j == cols - 1:
                    tracer.read(A.addr(i, j))
                    continue
                tracer.read(A.addr(i, j))
                tracer.read(A.addr(i - 1, j))
                tracer.read(A.addr(i + 1, j))
                tracer.read(A.addr(i, j - 1))
                tracer.read(A.addr(i, j + 1))

    sweep(0, rows, 0, cols)
    return tracer.result()


def manual_regular_attention(n: int = 32, d: int = 4) -> dict[str, object]:
    tracer = ManualTracer()
    row_max = tracer.alloc_vector(1)
    row_sum = tracer.alloc_vector(1)
    Q = tracer.alloc_matrix(n, d)
    K = tracer.alloc_matrix(n, d)
    V = tracer.alloc_matrix(n, d)
    scores = tracer.alloc_matrix(n, n)
    probs = tracer.alloc_matrix(n, n)
    tracer.alloc_matrix(n, d)

    for i in range(n):
        for j in range(n):
            for dd in range(d):
                tracer.read(Q.addr(i, dd))
                tracer.read(K.addr(j, dd))
        tracer.read(scores.addr(i, 0))
        for j in range(1, n):
            tracer.read(row_max.addr(0))
            tracer.read(scores.addr(i, j))

        for j in range(n):
            if j > 0:
                tracer.read(row_sum.addr(0))
            tracer.read(scores.addr(i, j))

        for dd in range(d):
            tracer.read(probs.addr(i, 0))
            tracer.read(V.addr(0, dd))
            for j in range(1, n):
                tracer.read(probs.addr(i, j))
                tracer.read(V.addr(j, dd))
    return tracer.result()


def manual_flash_attention(n: int = 32, d: int = 4, bq: int = 8, bk: int = 4) -> dict[str, object]:
    tracer = ManualTracer()
    scores = tracer.alloc_vector(bk)
    probs = tracer.alloc_vector(bk)
    block_out = tracer.alloc_vector(d)
    max_state = tracer.alloc_vector(n)
    sum_state = tracer.alloc_vector(n)
    out_state = tracer.alloc_matrix(n, d)
    Q = tracer.alloc_matrix(n, d)
    K = tracer.alloc_matrix(n, d)
    V = tracer.alloc_matrix(n, d)
    tracer.alloc_matrix(n, d)

    num_q_blocks = (n + bq - 1) // bq
    num_kv_blocks = (n + bk - 1) // bk
    for qb in range(num_q_blocks):
        q_start = qb * bq
        q_end = min(q_start + bq, n)
        block_order = range(num_kv_blocks) if qb % 2 == 0 else range(num_kv_blocks - 1, -1, -1)

        for kb in block_order:
            k_start = kb * bk
            k_end = min(k_start + bk, n)
            block_size = k_end - k_start

            for i in range(q_start, q_end):
                for j in range(block_size):
                    for dd in range(d):
                        tracer.read(Q.addr(i, dd))
                        tracer.read(K.addr(k_start + j, dd))

                tracer.read(scores.addr(0))
                for j in range(1, block_size):
                    tracer.read(scores.addr(j))

                for j in range(block_size):
                    tracer.read(scores.addr(j))
                for dd in range(d):
                    tracer.read(probs.addr(0))
                    tracer.read(V.addr(k_start, dd))
                    for j in range(1, block_size):
                        tracer.read(probs.addr(j))
                        tracer.read(V.addr(k_start + j, dd))

                if kb == block_order.start:
                    continue
                tracer.read(max_state.addr(i))
                tracer.read(sum_state.addr(i))
                for dd in range(d):
                    tracer.read(out_state.addr(i, dd))
                    tracer.read(block_out.addr(dd))

    for i in range(n):
        tracer.read(sum_state.addr(i))
        for dd in range(d):
            tracer.read(out_state.addr(i, dd))

    return tracer.result()


MANUAL_IMPLEMENTATIONS: dict[str, Callable[[], dict[str, object]]] = {
    "matvec-32": manual_matvec,
    "vecmat-32": manual_vecmat,
    "matvec-row-64": lambda: manual_matvec(64),
    "matvec-col-64": lambda: manual_vecmat(64),
    "transpose-naive-32": manual_transpose,
    "transpose-blocked-32": lambda: manual_transpose(block=8),
    "transpose-recursive-32": lambda: manual_transpose(recursive=True),
    "scan-row-64": lambda: manual_scan(by_column=False),
    "scan-column-64": lambda: manual_scan(by_column=True),
    "naive-matmul-16": manual_naive_matmul,
    "tiled-matmul-16": manual_tiled_matmul,
    "rmm-16": manual_recursive_matmul,
    "rmm-lex-16": lambda: _manual_inplace_rmm(LEX_ORDER),
    "rmm-gray-16": lambda: _manual_inplace_rmm(GRAY_ORDER),
    "strassen-16": manual_strassen,
    "fused-strassen-16": manual_fused_strassen,
    "fft-iterative-1024": manual_iterative_fft,
    "fft-recursive-1024": manual_recursive_fft,
    "conv2d-spatial-16x16-k5": manual_spatial_conv2d,
    "spatial-conv-32x32-k5": lambda: manual_spatial_conv2d(32, 32, 5, 5),
    "regular-conv-16x16-k3-c4": manual_regular_conv,
    "conv2d-fft-16x16-k5": manual_fft_conv2d,
    "fft-conv-32": manual_fft_conv1d,
    "jacobi-naive-32": lambda: manual_jacobi(recursive=False),
    "jacobi-recursive-32": lambda: manual_jacobi(recursive=True),
    "regular-attention-32x4": manual_regular_attention,
    "naive-attention-32x2": lambda: manual_regular_attention(d=2),
    "flash-attention-32x4": manual_flash_attention,
    "flash-attention-32x2-b8": lambda: manual_flash_attention(d=2, bk=8),
    "mergesort-64": manual_mergesort,
    "lcs-dp-32x32": manual_lcs_dp,
    "gaussian-elimination-24": manual_gaussian_elimination,
    "gauss-jordan-inverse-16": manual_gauss_jordan_inverse,
    "lu-no-pivot-24": manual_lu_no_pivot,
    "blocked-lu-24": lambda: manual_blocked_lu(24, 4),
    "recursive-lu-24": lambda: manual_recursive_lu(24, 6),
    "lu-partial-pivot-24": manual_lu_partial_pivot,
    "cholesky-24": manual_cholesky,
    "blocked-cholesky-24": lambda: manual_blocked_cholesky(24, 4),
    "recursive-cholesky-24": lambda: manual_recursive_cholesky(24, 6),
    "householder-qr-48x12": manual_householder_qr,
    "blocked-qr-48x12": lambda: manual_blocked_qr(48, 12, 4),
    "tsqr-48x12": lambda: manual_tsqr(48, 12, 12),
}


def measure_manual_2d(key: str) -> dict[str, object]:
    try:
        return MANUAL_IMPLEMENTATIONS[key]()
    except KeyError as exc:
        raise KeyError(f"No hand-scheduled Manual-2D implementation registered for {key}") from exc
