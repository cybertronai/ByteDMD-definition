"""Manual (no-magic) matmul on a bump-allocated 1-D physical address space.

Every access to an existing address `d >= 1` costs `ceil(sqrt(d))`; writes
are free. There is no tracer, no LRU stack, and no liveness oracle —
addresses are assigned by an explicit bump allocator, and the scratchpad
is software-managed.

Two entry points:

    matmul_naive_manual(A, B)
        Triple for-loop over the main-memory buffers.

    matmul_rmm_manual(A, B, tile_size=4)
        Cache-oblivious 8-way recursive RMM with a Hamiltonian traversal
        that keeps one block per step in the scratchpad. When the
        recursion reaches size == tile_size, the kernel DMAs A, B, C
        tiles into low-address L1 slots and performs the inner-loop
        multiply-accumulate entirely on the scratchpad.

Both return the total ByteDMD cost of the run.

Reference: user-supplied manual implementation (see REPORT.md).
"""

from __future__ import annotations

import math
from typing import List, Optional


class ManualAllocator:
    """Bump-allocated 1-D physical memory.

    - `alloc(size)` returns a base address; subsequent calls get higher addrs.
    - `write(addr, val)` is free.
    - `read(addr)` charges `ceil(sqrt(addr))`.
    """

    def __init__(self) -> None:
        self.memory: dict = {}
        self.cost: int = 0
        self.next_addr: int = 1

    def alloc(self, size: int) -> int:
        addr = self.next_addr
        self.next_addr += size
        return addr

    def write(self, addr: int, val: float) -> None:
        if addr < 1:
            raise ValueError("addresses must be >= 1")
        self.memory[addr] = val

    def read(self, addr: int) -> float:
        if addr < 1:
            raise ValueError("addresses must be >= 1")
        self.cost += math.isqrt(addr - 1) + 1
        return self.memory.get(addr, 0.0)


def _load_matrix(alloc: ManualAllocator, M: List[List[float]]) -> int:
    n = len(M)
    ptr = alloc.alloc(n * n)
    for i in range(n):
        for j in range(n):
            alloc.write(ptr + i * n + j, M[i][j])
    return ptr


class ScratchpadRMM:
    """Software-managed L1 scratchpad pinned to the lowest addresses.

    Holds three `T x T` tiles (A, B, C) at addresses 1..3*T*T. Reads from
    these addresses cost at most `ceil(sqrt(3*T*T))`, which is small —
    this is where the "cache" savings come from.
    """

    def __init__(self, alloc: ManualAllocator, n: int, tile_size: int) -> None:
        self.alloc = alloc
        self.n = n
        self.T = tile_size

        self.fast_A = alloc.alloc(self.T * self.T)
        self.fast_B = alloc.alloc(self.T * self.T)
        self.fast_C = alloc.alloc(self.T * self.T)

        self.loaded_A: Optional[tuple] = None
        self.loaded_B: Optional[tuple] = None
        self.loaded_C: Optional[tuple] = None
        self.dirty_C: bool = False

    def sync_A(self, ptrA: int, rA: int, cA: int) -> None:
        if self.loaded_A == (rA, cA):
            return
        for i in range(self.T):
            for j in range(self.T):
                val = self.alloc.read(ptrA + (rA + i) * self.n + (cA + j))
                self.alloc.write(self.fast_A + i * self.T + j, val)
        self.loaded_A = (rA, cA)

    def sync_B(self, ptrB: int, rB: int, cB: int) -> None:
        if self.loaded_B == (rB, cB):
            return
        for i in range(self.T):
            for j in range(self.T):
                val = self.alloc.read(ptrB + (rB + i) * self.n + (cB + j))
                self.alloc.write(self.fast_B + i * self.T + j, val)
        self.loaded_B = (rB, cB)

    def sync_C(self, ptrC: int, rC: int, cC: int) -> None:
        if self.loaded_C == (rC, cC):
            return
        self.flush_C(ptrC)
        for i in range(self.T):
            for j in range(self.T):
                val = self.alloc.read(ptrC + (rC + i) * self.n + (cC + j))
                self.alloc.write(self.fast_C + i * self.T + j, val)
        self.loaded_C = (rC, cC)
        self.dirty_C = False

    def flush_C(self, ptrC: int) -> None:
        if self.loaded_C is not None and self.dirty_C:
            rC, cC = self.loaded_C
            for i in range(self.T):
                for j in range(self.T):
                    val = self.alloc.read(self.fast_C + i * self.T + j)
                    self.alloc.write(ptrC + (rC + i) * self.n + (cC + j), val)
            self.dirty_C = False

    def compute_tile(self, ptrA: int, ptrB: int, ptrC: int,
                     rA: int, cA: int, rB: int, cB: int, rC: int, cC: int) -> None:
        self.sync_A(ptrA, rA, cA)
        self.sync_B(ptrB, rB, cB)
        self.sync_C(ptrC, rC, cC)
        for i in range(self.T):
            for j in range(self.T):
                c_val = self.alloc.read(self.fast_C + i * self.T + j)
                for k in range(self.T):
                    a_val = self.alloc.read(self.fast_A + i * self.T + k)
                    b_val = self.alloc.read(self.fast_B + k * self.T + j)
                    c_val += a_val * b_val
                self.alloc.write(self.fast_C + i * self.T + j, c_val)
        self.dirty_C = True


def matmul_rmm_manual(A_in: List[List[float]], B_in: List[List[float]],
                      tile_size: int = 4) -> int:
    """Cache-aware recursive matmul. Returns total ByteDMD read cost.

    The 8-way recursion uses a Hamiltonian ordering so that between every
    pair of consecutive recursive calls, exactly one of the (A, B, C)
    tiles changes. That tile survives in the scratchpad, and `sync_*`
    noops it as a cache hit.
    """
    n = len(A_in)
    alloc = ManualAllocator()
    sp = ScratchpadRMM(alloc, n, tile_size)

    ptrA = _load_matrix(alloc, A_in)
    ptrB = _load_matrix(alloc, B_in)
    C_in = [[0.0] * n for _ in range(n)]
    ptrC = _load_matrix(alloc, C_in)

    def recurse(rA: int, cA: int, rB: int, cB: int, rC: int, cC: int, size: int) -> None:
        if size == tile_size:
            sp.compute_tile(ptrA, ptrB, ptrC, rA, cA, rB, cB, rC, cC)
            return
        h = size // 2
        # Hamiltonian 8-way traversal: each transition changes exactly two
        # quadrants, so one tile always survives in the scratchpad.
        recurse(rA,     cA,     rB,     cB,     rC,     cC,     h)  # C11 += A11 * B11
        recurse(rA,     cA,     rB,     cB + h, rC,     cC + h, h)  # C12 += A11 * B12
        recurse(rA + h, cA,     rB,     cB + h, rC + h, cC + h, h)  # C22 += A21 * B12
        recurse(rA + h, cA,     rB,     cB,     rC + h, cC,     h)  # C21 += A21 * B11
        recurse(rA + h, cA + h, rB + h, cB,     rC + h, cC,     h)  # C21 += A22 * B21
        recurse(rA + h, cA + h, rB + h, cB + h, rC + h, cC + h, h)  # C22 += A22 * B22
        recurse(rA,     cA + h, rB + h, cB + h, rC,     cC + h, h)  # C12 += A12 * B22
        recurse(rA,     cA + h, rB + h, cB,     rC,     cC,     h)  # C11 += A12 * B21

    recurse(0, 0, 0, 0, 0, 0, n)
    sp.flush_C(ptrC)
    return alloc.cost


def matmul_naive_manual(A_in: List[List[float]], B_in: List[List[float]]) -> int:
    """Naive triple for-loop on the bump-allocated main memory."""
    n = len(A_in)
    alloc = ManualAllocator()
    ptrA = _load_matrix(alloc, A_in)
    ptrB = _load_matrix(alloc, B_in)
    C_in = [[0.0] * n for _ in range(n)]
    ptrC = _load_matrix(alloc, C_in)

    for i in range(n):
        for j in range(n):
            c_val = alloc.read(ptrC + i * n + j)
            for k in range(n):
                c_val += alloc.read(ptrA + i * n + k) * alloc.read(ptrB + k * n + j)
            alloc.write(ptrC + i * n + j, c_val)
    return alloc.cost


def _allocate_hierarchy(alloc: ManualAllocator, N: int) -> dict:
    """Pre-allocate an inverted pyramid of buffers: smallest at lowest addrs.

    Level K gets 3*K*K addresses for A, B, C buffers. K=1 is at the absolute
    fastest addresses; K=N is at the deepest (main RAM). Total footprint
    < 4*N*N thanks to the geometric series.
    """
    offsets = {}
    K = 1
    while K <= N:
        base = alloc.alloc(3 * K * K)
        offsets[K] = {'A': base, 'B': base + K * K, 'C': base + 2 * K * K}
        K *= 2
    return offsets


def _dma_copy(alloc: ManualAllocator,
              src_base: int, src_stride: int, src_r: int, src_c: int,
              dst_base: int, dst_stride: int, dst_r: int, dst_c: int,
              H: int) -> None:
    """Block-copy an H x H sub-matrix from src to dst. Reads are priced."""
    for i in range(H):
        for j in range(H):
            val = alloc.read(src_base + (src_r + i) * src_stride + (src_c + j))
            alloc.write(dst_base + (dst_r + i) * dst_stride + (dst_c + j), val)


def _rmm_explicit_recurse(alloc: ManualAllocator, ptrs: dict,
                          size: int, pA: int, pB: int, pC: int,
                          stride: int) -> None:
    """Hierarchical RMM: DMA data into child-level buffers before recursing."""
    if size == 1:
        a = alloc.read(pA)
        b = alloc.read(pB)
        c = alloc.read(pC)
        alloc.write(pC, c + a * b)
        return

    H = size // 2
    sA, sB, sC = ptrs[H]['A'], ptrs[H]['B'], ptrs[H]['C']

    def compute_quadrant(rC, cC, rA1, cA1, rB1, cB1, rA2, cA2, rB2, cB2):
        _dma_copy(alloc, pC, stride, rC, cC, sC, H, 0, 0, H)
        _dma_copy(alloc, pA, stride, rA1, cA1, sA, H, 0, 0, H)
        _dma_copy(alloc, pB, stride, rB1, cB1, sB, H, 0, 0, H)
        _rmm_explicit_recurse(alloc, ptrs, H, sA, sB, sC, H)
        _dma_copy(alloc, pA, stride, rA2, cA2, sA, H, 0, 0, H)
        _dma_copy(alloc, pB, stride, rB2, cB2, sB, H, 0, 0, H)
        _rmm_explicit_recurse(alloc, ptrs, H, sA, sB, sC, H)
        _dma_copy(alloc, sC, H, 0, 0, pC, stride, rC, cC, H)

    compute_quadrant(0, 0,  0, 0, 0, 0,  0, H, H, 0)
    compute_quadrant(0, H,  0, 0, 0, H,  0, H, H, H)
    compute_quadrant(H, 0,  H, 0, 0, 0,  H, H, H, 0)
    compute_quadrant(H, H,  H, 0, 0, H,  H, H, H, H)


def matmul_explicit_rmm(A_in: List[List[float]], B_in: List[List[float]]) -> int:
    """Hierarchical-scratchpad RMM. Returns total ByteDMD read cost.

    Pre-allocates an inverted pyramid: 1x1 buffers at the fastest addrs,
    NxN buffers at the deepest. Each recursion level DMAs sub-blocks from
    parent to child buffers before recursing, so base-case reads always
    hit addresses 1..3. Achieves O(N^3 log N) without any LRU magic.

    Reference: gemini/bytedmd_explicit_rmm.md.
    """
    n = len(A_in)
    alloc = ManualAllocator()
    ptrs = _allocate_hierarchy(alloc, n)
    main_A = ptrs[n]['A']
    main_B = ptrs[n]['B']
    main_C = ptrs[n]['C']
    for i in range(n):
        for j in range(n):
            alloc.write(main_A + i * n + j, A_in[i][j])
            alloc.write(main_B + i * n + j, B_in[i][j])
            alloc.write(main_C + i * n + j, 0.0)
    _rmm_explicit_recurse(alloc, ptrs, n, main_A, main_B, main_C, n)
    return alloc.cost


if __name__ == "__main__":
    for N in [8, 16, 32]:
        A = [[1 for _ in range(N)] for _ in range(N)]
        B = [[1 for _ in range(N)] for _ in range(N)]
        cost_naive = matmul_naive_manual(A, B)
        cost_manual = matmul_rmm_manual(A, B, tile_size=4)
        print(f"N={N:3d}  naive={cost_naive:>12,}  rmm+scratchpad={cost_manual:>12,}"
              f"  speedup={cost_naive / cost_manual:.2f}x")
