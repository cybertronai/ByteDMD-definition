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


def make_volume(rows: int, cols: int, channels: int, offset: int = 0) -> list[list[list[int]]]:
    return [
        [
            [offset + ((i * cols + j) * channels + c) + 1 for c in range(channels)]
            for j in range(cols)
        ]
        for i in range(rows)
    ]


def make_filter(k_rows: int, k_cols: int, in_channels: int, out_channels: int, offset: int = 0) -> list[list[list[list[int]]]]:
    return [
        [
            [
                [
                    offset
                    + ((((u * k_cols + v) * in_channels + ci) * out_channels + co) + 1)
                    for co in range(out_channels)
                ]
                for ci in range(in_channels)
            ]
            for v in range(k_cols)
        ]
        for u in range(k_rows)
    ]


def make_linear_system_matrix(n: int, offset: int = 0) -> list[list[float]]:
    """Create a deterministic strictly diagonally dominant matrix."""

    matrix = []
    for i in range(n):
        row = []
        off_diag_sum = 0.0
        for j in range(n):
            value = float(((offset + 7 * i + 11 * j) % 9) + 1)
            row.append(value)
            if i != j:
                off_diag_sum += abs(value)
        row[i] = off_diag_sum + float(n + 1)
        matrix.append(row)
    return matrix


def make_linear_system(n: int, offset: int = 0) -> tuple[list[list[float]], list[float], list[float]]:
    """Create a dense linear system with a known solution."""

    A = make_linear_system_matrix(n, offset=offset)
    x_true = [float(((3 * i) % 5) + 1) for i in range(n)]
    b = [sum(A[i][j] * x_true[j] for j in range(n)) for i in range(n)]
    return A, b, x_true


def make_pivot_matrix(n: int, offset: int = 0) -> list[list[float]]:
    """Create a dense matrix that tends to trigger early partial-pivot swaps."""

    base = make_linear_system_matrix(n, offset=offset)
    return [[float(value) for value in base[(i + 1) % n]] for i in range(n)]


def make_spd_matrix(n: int, offset: int = 0) -> list[list[float]]:
    """Create a deterministic symmetric positive-definite matrix."""

    base = make_matrix(n, n, offset=offset)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            total = 0.0
            for k in range(n):
                total += float(base[k][i]) * float(base[k][j])
            if i == j:
                total += float(n)
            out[i][j] = total
    return out


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


@dataclass(frozen=True)
class MatrixBlock:
    """A mutable square view into a matrix."""

    base: list[list[object]]
    row: int
    col: int
    size: int

    def get(self, i: int, j: int):
        return self.base[self.row + i][self.col + j]

    def set(self, i: int, j: int, value) -> None:
        self.base[self.row + i][self.col + j] = value


def _copy_proxy(x):
    """Tracked copy with one read and one output."""

    return x + 0


def _record_read(x) -> None:
    if hasattr(x, "_ctx") and hasattr(x, "_key"):
        x._ctx.read(x._key)


def _raw_value(x):
    return x.val if hasattr(x, "val") else x


def _recorded_le(a, b) -> bool:
    _record_read(a)
    _record_read(b)
    return _raw_value(a) <= _raw_value(b)


def _recorded_eq(a, b) -> bool:
    _record_read(a)
    _record_read(b)
    return _raw_value(a) == _raw_value(b)


def _max_actual(a, b):
    return _copy_proxy(b) if _recorded_le(a, b) else _copy_proxy(a)


def _max2(a, b):
    """Proxy for max with the right read arity."""

    return a + b


def _exp_proxy(x):
    """Proxy for exp with one read and one output."""

    return x * x


def _inv_proxy(x):
    """Proxy for reciprocal with one read and one output."""

    return x * x


def _sqrt_actual(x):
    if hasattr(x, "_ctx") and hasattr(x, "_key"):
        x._ctx.read(x._key)
        return type(x)(x._ctx, x._ctx.allocate(), math.sqrt(x.val))
    return math.sqrt(x)


def _tracked_constant_like(template, value):
    if hasattr(template, "_ctx") and hasattr(template, "_key"):
        return type(template)(template._ctx, template._ctx.allocate(), value)
    return value


def _reciprocal_actual(x):
    if hasattr(x, "_ctx") and hasattr(x, "_key"):
        x._ctx.read(x._key)
        return type(x)(x._ctx, x._ctx.allocate(), 1.0 / x.val)
    return 1.0 / x


def _matrix_copy(A):
    return [[_copy_proxy(value) for value in row] for row in A]


def _matrix_identity(n: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _matrix_subtract(A, B):
    rows = len(A)
    cols = len(A[0])
    out = [[None] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            out[i][j] = A[i][j] - B[i][j]
    return out


def _matrix_multiply(A, B):
    rows = len(A)
    inner = len(A[0])
    cols = len(B[0])
    out = [[None] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            acc = A[i][0] * B[0][j]
            for k in range(1, inner):
                acc = acc + A[i][k] * B[k][j]
            out[i][j] = acc
    return out


def _forward_solve_unit_lower(L, B):
    n = len(L)
    cols = len(B[0])
    out = [[None] * cols for _ in range(n)]
    for i in range(n):
        for j in range(cols):
            acc = _copy_proxy(B[i][j])
            for k in range(i):
                acc = acc - L[i][k] * out[k][j]
            out[i][j] = acc
    return out


def _solve_right_upper(B, U):
    rows = len(B)
    n = len(U)
    out = [[None] * n for _ in range(rows)]
    for i in range(rows):
        for j in range(n):
            acc = _copy_proxy(B[i][j])
            for k in range(j):
                acc = acc - out[i][k] * U[k][j]
            out[i][j] = acc * _reciprocal_actual(U[j][j])
    return out


def _solve_right_lower_transpose(B, L):
    rows = len(B)
    n = len(L)
    out = [[None] * n for _ in range(rows)]
    for i in range(rows):
        for j in range(n):
            acc = _copy_proxy(B[i][j])
            for k in range(j):
                acc = acc - out[i][k] * L[j][k]
            out[i][j] = acc * _reciprocal_actual(L[j][j])
    return out


def _join_quadrants(q11, q12, q21, q22):
    top = [row11 + row12 for row11, row12 in zip(q11, q12)]
    bottom = [row21 + row22 for row21, row22 in zip(q21, q22)]
    return top + bottom


def _slice_copy(A, r0: int, r1: int, c0: int, c1: int):
    return [[_copy_proxy(A[i][j]) for j in range(c0, c1)] for i in range(r0, r1)]


def _split_block(block: MatrixBlock) -> tuple[MatrixBlock, MatrixBlock, MatrixBlock, MatrixBlock]:
    half = block.size // 2
    return (
        MatrixBlock(block.base, block.row, block.col, half),
        MatrixBlock(block.base, block.row, block.col + half, half),
        MatrixBlock(block.base, block.row + half, block.col, half),
        MatrixBlock(block.base, block.row + half, block.col + half, half),
    )


def _quadrant_terms(
    terms: list[tuple[int, MatrixBlock]],
    quadrant: int,
) -> list[tuple[int, MatrixBlock]]:
    return [(sign, _split_block(block)[quadrant]) for sign, block in terms]


def _virtual_value(terms: list[tuple[int, MatrixBlock]], i: int, j: int):
    value = None
    for sign, block in terms:
        term = block.get(i, j)
        signed = term if sign == 1 else (-1 * term)
        value = signed if value is None else value + signed
    return value


def _accumulate_targets(targets: list[tuple[int, MatrixBlock]], i: int, j: int, value) -> None:
    for sign, block in targets:
        contribution = (1 * value) if sign == 1 else (-1 * value)
        existing = block.get(i, j)
        block.set(i, j, contribution if existing is None else existing + contribution)


def _fused_strassen_accumulate(
    a_terms: list[tuple[int, MatrixBlock]],
    b_terms: list[tuple[int, MatrixBlock]],
    c_targets: list[tuple[int, MatrixBlock]],
    *,
    leaf: int,
) -> None:
    size = a_terms[0][1].size
    if size <= leaf:
        for i in range(size):
            for j in range(size):
                total = None
                for k in range(size):
                    a_value = _virtual_value(a_terms, i, k)
                    b_value = _virtual_value(b_terms, k, j)
                    product = a_value * b_value
                    total = product if total is None else total + product
                _accumulate_targets(c_targets, i, j, total)
        return

    m1_targets: list[tuple[int, MatrixBlock]] = []
    m2_targets: list[tuple[int, MatrixBlock]] = []
    m3_targets: list[tuple[int, MatrixBlock]] = []
    m4_targets: list[tuple[int, MatrixBlock]] = []
    m5_targets: list[tuple[int, MatrixBlock]] = []
    m6_targets: list[tuple[int, MatrixBlock]] = []
    m7_targets: list[tuple[int, MatrixBlock]] = []

    for sign, block in c_targets:
        q11, q12, q21, q22 = _split_block(block)
        m1_targets.extend([(sign, q11), (sign, q22)])
        m2_targets.extend([(sign, q21), (-sign, q22)])
        m3_targets.extend([(sign, q12), (sign, q22)])
        m4_targets.extend([(sign, q11), (sign, q21)])
        m5_targets.extend([(-sign, q11), (sign, q12)])
        m6_targets.append((sign, q22))
        m7_targets.append((sign, q11))

    a11 = _quadrant_terms(a_terms, 0)
    a12 = _quadrant_terms(a_terms, 1)
    a21 = _quadrant_terms(a_terms, 2)
    a22 = _quadrant_terms(a_terms, 3)
    b11 = _quadrant_terms(b_terms, 0)
    b12 = _quadrant_terms(b_terms, 1)
    b21 = _quadrant_terms(b_terms, 2)
    b22 = _quadrant_terms(b_terms, 3)

    _fused_strassen_accumulate(a11 + a22, b11 + b22, m1_targets, leaf=leaf)
    _fused_strassen_accumulate(a21 + a22, b11, m2_targets, leaf=leaf)
    _fused_strassen_accumulate(a11, b12 + [(-sign, block) for sign, block in b22], m3_targets, leaf=leaf)
    _fused_strassen_accumulate(a22, b21 + [(-sign, block) for sign, block in b11], m4_targets, leaf=leaf)
    _fused_strassen_accumulate(a11 + a12, b22, m5_targets, leaf=leaf)
    _fused_strassen_accumulate(a21 + [(-sign, block) for sign, block in a11], b11 + b12, m6_targets, leaf=leaf)
    _fused_strassen_accumulate(a12 + [(-sign, block) for sign, block in a22], b21 + b22, m7_targets, leaf=leaf)


def fused_strassen(A, B, *, leaf: int = 8):
    """Zero-allocation fused Strassen with virtual sums and direct accumulation."""

    n = len(A)
    if n == 0 or n != len(B) or n != len(B[0]) or n != len(A[0]):
        raise ValueError("fused_strassen expects same-size square matrices")
    if n & (n - 1):
        raise ValueError("fused_strassen requires a power-of-two size")

    out = [[None] * n for _ in range(n)]
    _fused_strassen_accumulate(
        [(1, MatrixBlock(A, 0, 0, n))],
        [(1, MatrixBlock(B, 0, 0, n))],
        [(1, MatrixBlock(out, 0, 0, n))],
        leaf=leaf,
    )
    return out


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


def _recursive_fft_core(x, *, inverse: bool):
    n = len(x)
    if n == 1:
        return [_copy_proxy(x[0])]
    even = _recursive_fft_core(x[0::2], inverse=inverse)
    odd = _recursive_fft_core(x[1::2], inverse=inverse)
    out = [None] * n
    direction = 1 if inverse else -1
    for k in range(n // 2):
        omega = cmath.exp(direction * 2j * math.pi * k / n)
        t = omega * odd[k]
        out[k] = even[k] + t
        out[k + n // 2] = even[k] - t
    return out


def recursive_fft(x):
    """Recursive radix-2 Cooley-Tukey FFT."""

    return _recursive_fft_core(x, inverse=False)


def inverse_recursive_fft(x):
    """Inverse recursive radix-2 FFT."""

    scale = 1.0 / len(x)
    return [scale * value for value in _recursive_fft_core(x, inverse=True)]


def naive_circular_conv1d(x, kernel):
    """Naive circular 1D convolution."""

    n = len(x)
    out = [None] * n
    for i in range(n):
        total = None
        for j in range(n):
            product = x[j] * kernel[(i - j) % n]
            total = product if total is None else total + product
        out[i] = total
    return out


def fft_conv1d(x, kernel):
    """Circular 1D convolution via FFT."""

    x_freq = recursive_fft(x)
    kernel_freq = recursive_fft(kernel)
    spectrum = [left * right for left, right in zip(x_freq, kernel_freq)]
    return inverse_recursive_fft(spectrum)


def _transpose_plain(A):
    return [list(col) for col in zip(*A)]


def _next_power_of_two(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _pad_matrix(A, rows: int, cols: int):
    out = [[0.0] * cols for _ in range(rows)]
    for i, row in enumerate(A):
        for j, value in enumerate(row):
            out[i][j] = _copy_proxy(value)
    return out


def fft2d_recursive(A, *, inverse: bool = False):
    """2D FFT via row/column recursive 1D FFTs."""

    fft_1d = inverse_recursive_fft if inverse else recursive_fft
    rows = [fft_1d(row) for row in A]
    cols = _transpose_plain(rows)
    transformed_cols = [fft_1d(col) for col in cols]
    return _transpose_plain(transformed_cols)


def spatial_conv2d(image, kernel):
    """Same-size 2D convolution with zero padding."""

    rows = len(image)
    cols = len(image[0])
    k_rows = len(kernel)
    k_cols = len(kernel[0])
    center_r = k_rows // 2
    center_c = k_cols // 2

    out = [[None] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            total = None
            for u in range(k_rows):
                ii = i + center_r - u
                if ii < 0 or ii >= rows:
                    continue
                for v in range(k_cols):
                    jj = j + center_c - v
                    if jj < 0 or jj >= cols:
                        continue
                    product = image[ii][jj] * kernel[u][v]
                    total = product if total is None else total + product
            out[i][j] = total
    return out


def fft_conv2d(image, kernel):
    """Same-size 2D convolution via zero-padded FFT."""

    rows = len(image)
    cols = len(image[0])
    k_rows = len(kernel)
    k_cols = len(kernel[0])
    full_rows = rows + k_rows - 1
    full_cols = cols + k_cols - 1
    pad_rows = _next_power_of_two(full_rows)
    pad_cols = _next_power_of_two(full_cols)

    padded_image = _pad_matrix(image, pad_rows, pad_cols)
    padded_kernel = _pad_matrix(kernel, pad_rows, pad_cols)

    image_freq = fft2d_recursive(padded_image)
    kernel_freq = fft2d_recursive(padded_kernel)
    spectrum = [
        [image_freq[i][j] * kernel_freq[i][j] for j in range(pad_cols)]
        for i in range(pad_rows)
    ]
    full = fft2d_recursive(spectrum, inverse=True)

    start_r = k_rows // 2
    start_c = k_cols // 2
    out = [[None] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            out[i][j] = _copy_proxy(full[start_r + i][start_c + j])
    return out


def regular_conv2d(image, kernel):
    """Same-size zero-padded convolution with channels."""

    rows = len(image)
    cols = len(image[0])
    k_rows = len(kernel)
    k_cols = len(kernel[0])
    in_channels = len(kernel[0][0])
    out_channels = len(kernel[0][0][0])
    center_r = k_rows // 2
    center_c = k_cols // 2

    out = [[[None] * out_channels for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            for co in range(out_channels):
                total = None
                for u in range(k_rows):
                    ii = i + center_r - u
                    if ii < 0 or ii >= rows:
                        continue
                    for v in range(k_cols):
                        jj = j + center_c - v
                        if jj < 0 or jj >= cols:
                            continue
                        for ci in range(in_channels):
                            product = image[ii][jj][ci] * kernel[u][v][ci][co]
                            total = product if total is None else total + product
                out[i][j][co] = total
    return out


def mergesort(x):
    """Mergesort with tracked comparisons."""

    n = len(x)
    if n <= 1:
        return [_copy_proxy(x[0])] if n == 1 else []

    mid = n // 2
    left = mergesort(x[:mid])
    right = mergesort(x[mid:])
    out = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        if _recorded_le(left[i], right[j]):
            out.append(_copy_proxy(left[i]))
            i += 1
        else:
            out.append(_copy_proxy(right[j]))
            j += 1
    while i < len(left):
        out.append(_copy_proxy(left[i]))
        i += 1
    while j < len(right):
        out.append(_copy_proxy(right[j]))
        j += 1
    return out


def lcs_dp(seq_a, seq_b):
    """Standard dynamic-programming LCS with tracked comparisons."""

    m = len(seq_a)
    n = len(seq_b)
    zero = seq_a[0] - seq_a[0]
    dp = [[_copy_proxy(zero) for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            if _recorded_eq(seq_a[i], seq_b[j]):
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = _max_actual(dp[i][j + 1], dp[i + 1][j])
    return dp[m][n]


def gaussian_elimination(A, b):
    """Dense Gaussian elimination solve without pivoting."""

    n = len(A)
    work = [[_copy_proxy(A[i][j]) for j in range(n)] for i in range(n)]
    rhs = [_copy_proxy(value) for value in b]

    for k in range(n - 1):
        pivot_inv = _reciprocal_actual(work[k][k])
        for i in range(k + 1, n):
            factor = work[i][k] * pivot_inv
            for j in range(k + 1, n):
                work[i][j] = work[i][j] - factor * work[k][j]
            rhs[i] = rhs[i] - factor * rhs[k]

    x = [None] * n
    for i in range(n - 1, -1, -1):
        acc = _copy_proxy(rhs[i])
        for j in range(i + 1, n):
            acc = acc - work[i][j] * x[j]
        x[i] = acc * _reciprocal_actual(work[i][i])
    return x


def gauss_jordan_inverse(A):
    """Dense Gauss-Jordan inverse without pivoting."""

    n = len(A)
    work = [[_copy_proxy(A[i][j]) for j in range(n)] for i in range(n)]
    template = A[0][0]
    inv = [
        [_tracked_constant_like(template, 1.0 if i == j else 0.0) for j in range(n)]
        for i in range(n)
    ]

    for k in range(n):
        pivot_inv = _reciprocal_actual(work[k][k])
        for j in range(n):
            work[k][j] = work[k][j] * pivot_inv
            inv[k][j] = inv[k][j] * pivot_inv
        for i in range(n):
            if i == k:
                continue
            factor = _copy_proxy(work[i][k])
            for j in range(n):
                work[i][j] = work[i][j] - factor * work[k][j]
                inv[i][j] = inv[i][j] - factor * inv[k][j]
    return inv


def lu_no_pivot(A):
    """Doolittle LU without pivoting."""

    n = len(A)
    U = _matrix_copy(A)
    L = _matrix_identity(n)

    for k in range(n - 1):
        pivot_inv = _reciprocal_actual(U[k][k])
        for i in range(k + 1, n):
            factor = U[i][k] * pivot_inv
            L[i][k] = factor
            for j in range(k, n):
                U[i][j] = U[i][j] - factor * U[k][j]
    return L, U


def blocked_lu(A, *, block: int = 4):
    """Tile-oriented LU without pivoting."""

    n = len(A)
    U = _matrix_copy(A)
    L = _matrix_identity(n)

    for k0 in range(0, n - 1, block):
        k1 = min(k0 + block, n)
        for k in range(k0, k1):
            pivot_inv = _reciprocal_actual(U[k][k])
            for i in range(k + 1, n):
                factor = U[i][k] * pivot_inv
                L[i][k] = factor
                for j0 in range(k, n, block):
                    for j in range(j0, min(j0 + block, n)):
                        U[i][j] = U[i][j] - factor * U[k][j]
    return L, U


def recursive_lu(A, *, leaf: int = 6):
    """Recursive block LU without pivoting."""

    n = len(A)
    if n <= leaf or (n % 2) == 1:
        return lu_no_pivot(A)

    half = n // 2
    A11 = _slice_copy(A, 0, half, 0, half)
    A12 = _slice_copy(A, 0, half, half, n)
    A21 = _slice_copy(A, half, n, 0, half)
    A22 = _slice_copy(A, half, n, half, n)

    L11, U11 = recursive_lu(A11, leaf=leaf)
    U12 = _forward_solve_unit_lower(L11, A12)
    L21 = _solve_right_upper(A21, U11)
    S = _matrix_subtract(A22, _matrix_multiply(L21, U12))
    L22, U22 = recursive_lu(S, leaf=leaf)

    zero_top_right = [[0.0] * (n - half) for _ in range(half)]
    zero_bottom_left = [[0.0] * half for _ in range(n - half)]
    L = _join_quadrants(L11, zero_top_right, L21, L22)
    U = _join_quadrants(U11, U12, zero_bottom_left, U22)
    return L, U


def lu_partial_pivot(A):
    """LU with partial pivoting and explicit row-copy traffic."""

    n = len(A)
    U = _matrix_copy(A)
    L = _matrix_identity(n)
    perm = list(range(n))

    for k in range(n - 1):
        pivot = k
        pivot_abs = None
        for i in range(k, n):
            _record_read(U[i][k])
            value = abs(_raw_value(U[i][k]))
            if pivot_abs is None or value > pivot_abs:
                pivot = i
                pivot_abs = value

        if pivot != k:
            for j in range(n):
                temp = _copy_proxy(U[k][j])
                U[k][j] = _copy_proxy(U[pivot][j])
                U[pivot][j] = temp
            for j in range(k):
                temp = _copy_proxy(L[k][j])
                L[k][j] = _copy_proxy(L[pivot][j])
                L[pivot][j] = temp
            perm[k], perm[pivot] = perm[pivot], perm[k]

        pivot_inv = _reciprocal_actual(U[k][k])
        for i in range(k + 1, n):
            factor = U[i][k] * pivot_inv
            L[i][k] = factor
            for j in range(k, n):
                U[i][j] = U[i][j] - factor * U[k][j]
    return perm, L, U


def cholesky(A):
    """Lower-triangular Cholesky factorization."""

    n = len(A)
    L = [[0.0] * n for _ in range(n)]

    for k in range(n):
        diag = _copy_proxy(A[k][k])
        for s in range(k):
            diag = diag - L[k][s] * L[k][s]
        L[k][k] = _sqrt_actual(diag)
        diag_inv = _reciprocal_actual(L[k][k])
        for i in range(k + 1, n):
            acc = _copy_proxy(A[i][k])
            for s in range(k):
                acc = acc - L[i][s] * L[k][s]
            L[i][k] = acc * diag_inv
    return L


def blocked_cholesky(A, *, block: int = 4):
    """Tile-oriented Cholesky factorization."""

    n = len(A)
    work = _matrix_copy(A)
    L = [[0.0] * n for _ in range(n)]

    for k0 in range(0, n, block):
        k1 = min(k0 + block, n)
        for k in range(k0, k1):
            diag = _copy_proxy(work[k][k])
            for s in range(k0, k):
                diag = diag - L[k][s] * L[k][s]
            L[k][k] = _sqrt_actual(diag)
            diag_inv = _reciprocal_actual(L[k][k])
            for i in range(k + 1, n):
                acc = _copy_proxy(work[i][k])
                for s in range(k0, k):
                    acc = acc - L[i][s] * L[k][s]
                L[i][k] = acc * diag_inv

        for i0 in range(k1, n, block):
            for j0 in range(k1, i0 + 1, block):
                i1 = min(i0 + block, n)
                j1 = min(j0 + block, n)
                for i in range(i0, i1):
                    for j in range(j0, min(j1, i + 1)):
                        acc = _copy_proxy(work[i][j])
                        for s in range(k0, k1):
                            acc = acc - L[i][s] * L[j][s]
                        work[i][j] = acc
                        work[j][i] = acc
    return L


def recursive_cholesky(A, *, leaf: int = 6):
    """Recursive block Cholesky factorization."""

    n = len(A)
    if n <= leaf or (n % 2) == 1:
        return cholesky(A)

    half = n // 2
    A11 = _slice_copy(A, 0, half, 0, half)
    A21 = _slice_copy(A, half, n, 0, half)
    A22 = _slice_copy(A, half, n, half, n)

    L11 = recursive_cholesky(A11, leaf=leaf)
    L21 = _solve_right_lower_transpose(A21, L11)
    schur = _matrix_subtract(A22, _matrix_multiply(L21, _transpose_plain(L21)))
    L22 = recursive_cholesky(schur, leaf=leaf)

    zero_top_right = [[0.0] * (n - half) for _ in range(half)]
    return _join_quadrants(L11, zero_top_right, L21, L22)


def _householder_qr_core(A, *, block: int | None) -> list[list[object]]:
    work = _matrix_copy(A)
    rows = len(work)
    cols = len(work[0])
    active = min(rows, cols)

    for k in range(active):
        norm_sq = None
        for i in range(k, rows):
            term = work[i][k] * work[i][k]
            norm_sq = term if norm_sq is None else norm_sq + term

        norm = _sqrt_actual(norm_sq)
        _record_read(work[k][k])
        sign = 1.0 if _raw_value(work[k][k]) >= 0 else -1.0
        alpha = _tracked_constant_like(norm, sign * _raw_value(norm))

        v = [_copy_proxy(work[i][k]) for i in range(k, rows)]
        v[0] = v[0] + alpha

        vtv = None
        for value in v:
            term = value * value
            vtv = term if vtv is None else vtv + term
        beta = _tracked_constant_like(vtv, 2.0) * _reciprocal_actual(vtv)

        if block is None:
            column_blocks = [(k, cols)]
        else:
            column_blocks = [(j0, min(j0 + block, cols)) for j0 in range(k, cols, block)]

        for j0, j1 in column_blocks:
            for j in range(j0, j1):
                dot = None
                for i in range(len(v)):
                    term = v[i] * work[k + i][j]
                    dot = term if dot is None else dot + term
                tau = beta * dot
                for i in range(len(v)):
                    work[k + i][j] = work[k + i][j] - v[i] * tau

    return [row[:cols] for row in work[:active]]


def householder_qr(A):
    """Unblocked Householder QR returning the R factor."""

    return _householder_qr_core(A, block=None)


def blocked_qr(A, *, block: int = 4):
    """Column-blocked Householder QR returning the R factor."""

    return _householder_qr_core(A, block=block)


def tsqr(A, *, leaf_rows: int = 12):
    """Tall-skinny QR returning the final R factor."""

    rows = len(A)
    cols = len(A[0])
    if rows <= leaf_rows or rows <= 2 * cols:
        return householder_qr(A)

    mid = rows // 2
    top = tsqr(A[:mid], leaf_rows=leaf_rows)
    bottom = tsqr(A[mid:], leaf_rows=leaf_rows)
    return householder_qr(top + bottom)


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


def conv2d_spatial_flops(rows: int, cols: int, k_rows: int, k_cols: int) -> int:
    return rows * cols * k_rows * k_cols * 2


def conv2d_fft_flops(rows: int, cols: int, k_rows: int, k_cols: int) -> int:
    pad_rows = _next_power_of_two(rows + k_rows - 1)
    pad_cols = _next_power_of_two(cols + k_cols - 1)
    one_pass = pad_rows * fft_flops(pad_cols) + pad_cols * fft_flops(pad_rows)
    return 3 * one_pass + pad_rows * pad_cols


def regular_conv_flops(rows: int, cols: int, k_rows: int, k_cols: int, in_channels: int, out_channels: int) -> int:
    return rows * cols * k_rows * k_cols * in_channels * out_channels * 2


def fft_conv1d_flops(n: int) -> int:
    return 3 * fft_flops(n) + n


def mergesort_flops(n: int) -> int:
    return 2 * n * int(math.log2(n))


def lcs_flops(m: int, n: int) -> int:
    return 3 * m * n


def gaussian_elimination_flops(n: int) -> int:
    elimination = 2 * n * (n - 1) * (n + 1) // 3
    backsolve = n * n
    return elimination + backsolve


def gauss_jordan_flops(n: int) -> int:
    return 4 * n * n * n


def lu_flops(n: int) -> int:
    return (2 * n * n * n) // 3


def cholesky_flops(n: int) -> int:
    return (n * n * n) // 3


def householder_qr_flops(rows: int, cols: int) -> int:
    return 2 * rows * cols * cols - (2 * cols * cols * cols) // 3


def tsqr_flops(rows: int, cols: int) -> int:
    levels = max(1, rows // max(1, cols))
    return householder_qr_flops(rows, cols) + levels * householder_qr_flops(2 * cols, cols)


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
            key="matvec-row-64",
            label="Matvec Row",
            workload="64x64 by 64",
            notes="row-major matrix-vector multiply",
            func=matvec,
            args_factory=lambda: (make_matrix(64), make_vector(64)),
            flops=matvec_flops(64),
        ),
        AlgorithmSpec(
            key="matvec-col-64",
            label="Matvec Column",
            workload="64 by 64x64",
            notes="column-major vector-matrix multiply",
            func=vecmat,
            args_factory=lambda: (make_matrix(64), make_vector(64)),
            flops=vecmat_flops(64),
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
            key="fused-strassen-16",
            label="Fused Strassen",
            workload="16x16, leaf=8",
            notes="zero-allocation virtual sums with direct accumulation into C",
            func=lambda A, B: fused_strassen(A, B, leaf=8),
            args_factory=lambda: (make_matrix(16), make_matrix(16, offset=1000)),
            flops=flops_strassen(16),
        ),
        AlgorithmSpec(
            key="gaussian-elimination-24",
            label="Gaussian Elimination",
            workload="N=24",
            notes="dense solve without pivoting",
            func=gaussian_elimination,
            args_factory=lambda: make_linear_system(24)[:2],
            flops=gaussian_elimination_flops(24),
        ),
        AlgorithmSpec(
            key="gauss-jordan-inverse-16",
            label="Gauss-Jordan Inverse",
            workload="N=16",
            notes="dense matrix inverse without pivoting",
            func=gauss_jordan_inverse,
            args_factory=lambda: (make_linear_system_matrix(16, offset=200),),
            flops=gauss_jordan_flops(16),
        ),
        AlgorithmSpec(
            key="lu-no-pivot-24",
            label="LU (No Pivot)",
            workload="N=24",
            notes="Doolittle LU without row swaps",
            func=lu_no_pivot,
            args_factory=lambda: (make_linear_system_matrix(24, offset=300),),
            flops=lu_flops(24),
        ),
        AlgorithmSpec(
            key="blocked-lu-24",
            label="LU (Blocked)",
            workload="N=24, block=4",
            notes="tile-oriented LU without pivoting",
            func=lambda A: blocked_lu(A, block=4),
            args_factory=lambda: (make_linear_system_matrix(24, offset=300),),
            flops=lu_flops(24),
        ),
        AlgorithmSpec(
            key="recursive-lu-24",
            label="LU (Recursive)",
            workload="N=24, leaf=6",
            notes="recursive block LU without pivoting",
            func=lambda A: recursive_lu(A, leaf=6),
            args_factory=lambda: (make_linear_system_matrix(24, offset=300),),
            flops=lu_flops(24),
        ),
        AlgorithmSpec(
            key="lu-partial-pivot-24",
            label="LU (Partial Pivot)",
            workload="N=24",
            notes="partial pivoting with row-copy traffic",
            func=lu_partial_pivot,
            args_factory=lambda: (make_pivot_matrix(24, offset=350),),
            flops=lu_flops(24),
        ),
        AlgorithmSpec(
            key="cholesky-24",
            label="Cholesky",
            workload="N=24",
            notes="lower-triangular Cholesky factorization",
            func=cholesky,
            args_factory=lambda: (make_spd_matrix(24, offset=400),),
            flops=cholesky_flops(24),
        ),
        AlgorithmSpec(
            key="blocked-cholesky-24",
            label="Cholesky (Blocked)",
            workload="N=24, block=4",
            notes="tile-oriented Cholesky factorization",
            func=lambda A: blocked_cholesky(A, block=4),
            args_factory=lambda: (make_spd_matrix(24, offset=400),),
            flops=cholesky_flops(24),
        ),
        AlgorithmSpec(
            key="recursive-cholesky-24",
            label="Cholesky (Recursive)",
            workload="N=24, leaf=6",
            notes="recursive block Cholesky factorization",
            func=lambda A: recursive_cholesky(A, leaf=6),
            args_factory=lambda: (make_spd_matrix(24, offset=400),),
            flops=cholesky_flops(24),
        ),
        AlgorithmSpec(
            key="householder-qr-48x12",
            label="Householder QR",
            workload="48x12",
            notes="unblocked Householder QR returning R",
            func=householder_qr,
            args_factory=lambda: (make_matrix(48, 12, offset=500),),
            flops=householder_qr_flops(48, 12),
        ),
        AlgorithmSpec(
            key="blocked-qr-48x12",
            label="Blocked QR",
            workload="48x12, block=4",
            notes="column-blocked Householder QR returning R",
            func=lambda A: blocked_qr(A, block=4),
            args_factory=lambda: (make_matrix(48, 12, offset=500),),
            flops=householder_qr_flops(48, 12),
        ),
        AlgorithmSpec(
            key="tsqr-48x12",
            label="TSQR",
            workload="48x12, leaf_rows=12",
            notes="tall-skinny recursive QR returning the final R",
            func=lambda A: tsqr(A, leaf_rows=12),
            args_factory=lambda: (make_matrix(48, 12, offset=500),),
            flops=tsqr_flops(48, 12),
        ),
        AlgorithmSpec(
            key="fft-iterative-1024",
            label="FFT (Iterative)",
            workload="N=1024",
            notes="iterative radix-2 Cooley-Tukey",
            func=iterative_fft,
            args_factory=lambda: ([complex(v, 0.5 * v) for v in make_vector(1024)],),
            flops=fft_flops(1024),
        ),
        AlgorithmSpec(
            key="fft-recursive-1024",
            label="FFT (Recursive)",
            workload="N=1024",
            notes="recursive radix-2 Cooley-Tukey",
            func=recursive_fft,
            args_factory=lambda: ([complex(v, 0.5 * v) for v in make_vector(1024)],),
            flops=fft_flops(1024),
        ),
        AlgorithmSpec(
            key="conv2d-spatial-16x16-k5",
            label="2D Convolution (Spatial)",
            workload="16x16, kernel=5x5",
            notes="same-size zero-padded spatial convolution",
            func=spatial_conv2d,
            args_factory=lambda: (make_matrix(16), make_matrix(5, 5, offset=5000)),
            flops=conv2d_spatial_flops(16, 16, 5, 5),
        ),
        AlgorithmSpec(
            key="spatial-conv-32x32-k5",
            label="Spatial Conv",
            workload="32x32, kernel=5x5",
            notes="same-size zero-padded spatial convolution",
            func=spatial_conv2d,
            args_factory=lambda: (make_matrix(32), make_matrix(5, 5, offset=5000)),
            flops=conv2d_spatial_flops(32, 32, 5, 5),
        ),
        AlgorithmSpec(
            key="regular-conv-16x16-k3-c4",
            label="Regular Conv",
            workload="16x16, kernel=3x3, Cin=4, Cout=4",
            notes="direct same-size convolution over 4 input/output channels",
            func=regular_conv2d,
            args_factory=lambda: (make_volume(16, 16, 4), make_filter(3, 3, 4, 4, offset=9000)),
            flops=regular_conv_flops(16, 16, 3, 3, 4, 4),
        ),
        AlgorithmSpec(
            key="conv2d-fft-16x16-k5",
            label="2D Convolution (FFT)",
            workload="16x16, kernel=5x5, pad=32x32",
            notes="same-size convolution via zero-padded recursive 2D FFT",
            func=fft_conv2d,
            args_factory=lambda: (make_matrix(16), make_matrix(5, 5, offset=5000)),
            flops=conv2d_fft_flops(16, 16, 5, 5),
        ),
        AlgorithmSpec(
            key="fft-conv-32",
            label="FFT Conv",
            workload="N=32",
            notes="circular 1D convolution via recursive FFT",
            func=fft_conv1d,
            args_factory=lambda: (make_vector(32), make_vector(32, offset=5000)),
            flops=fft_conv1d_flops(32),
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
            key="naive-attention-32x2",
            label="Naive Attention",
            workload="N=32, d=2",
            notes="materializes the full score matrix",
            func=regular_attention,
            args_factory=lambda: (
                make_matrix(32, 2),
                make_matrix(32, 2, offset=1000),
                make_matrix(32, 2, offset=2000),
            ),
            flops=regular_attention_flops(32, 2),
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
        AlgorithmSpec(
            key="flash-attention-32x2-b8",
            label="Flash Attention (Bk=8)",
            workload="N=32, d=2, Bq=8, Bk=8",
            notes="double-tiled Q/KV blocks with wider KV tiles",
            func=lambda Q, K, V: flash_attention(Q, K, V, bq=8, bk=8),
            args_factory=lambda: (
                make_matrix(32, 2),
                make_matrix(32, 2, offset=1000),
                make_matrix(32, 2, offset=2000),
            ),
            flops=flash_attention_flops(32, 2, 8),
        ),
        AlgorithmSpec(
            key="mergesort-64",
            label="Mergesort",
            workload="N=64",
            notes="top-down mergesort with tracked comparisons",
            func=mergesort,
            args_factory=lambda: (list(reversed(make_vector(64))),),
            flops=mergesort_flops(64),
        ),
        AlgorithmSpec(
            key="lcs-dp-32x32",
            label="LCS DP",
            workload="32x32",
            notes="dynamic programming longest common subsequence",
            func=lcs_dp,
            args_factory=lambda: (make_vector(32), make_vector(32, offset=16)),
            flops=lcs_flops(32, 32),
        ),
    ]
