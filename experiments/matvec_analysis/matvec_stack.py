#!/usr/bin/env python3
"""
Track the LRU stack state at every step of the matvec algorithm.

Outputs:
  1. A step-by-step table (step, operation, key read, depth, stack)
  2. Mathematica functions for:
     - stackState[k] : the LRU stack (list of keys) after step k
     - readDepth[k]  : the depth of the k-th read
     - trace         : the full list of depths
"""

import math
import sys


class LRUStack:
    """Minimal LRU stack with step-by-step recording."""

    def __init__(self):
        self.stack = []
        self.counter = 0
        self.steps = []   # list of (step_idx, event, key, depth, stack_before)
        self.step_idx = 0
        self.trace = []

    def allocate(self, label=None):
        self.counter += 1
        self.stack.append(self.counter)
        return self.counter

    def read(self, key, label="read"):
        idx = self.stack.index(key)
        depth = len(self.stack) - idx
        self.steps.append((self.step_idx, label, key, depth, list(self.stack)))
        self.trace.append(depth)
        self.step_idx += 1
        # LRU update
        del self.stack[idx]
        self.stack.append(key)
        return depth

    def read_pair(self, k1, k2, label="binop"):
        """Read two keys (both depths from same stack state, then both move)."""
        idx1 = self.stack.index(k1)
        idx2 = self.stack.index(k2)
        d1 = len(self.stack) - idx1
        d2 = len(self.stack) - idx2
        snap = list(self.stack)
        self.steps.append((self.step_idx, f"{label}_L", k1, d1, snap))
        self.trace.append(d1)
        self.step_idx += 1
        self.steps.append((self.step_idx, f"{label}_R", k2, d2, snap))
        self.trace.append(d2)
        self.step_idx += 1
        # Move both to top in order
        self.stack.remove(k1)
        self.stack.append(k1)
        self.stack.remove(k2)
        self.stack.append(k2)


def matvec_traced(N):
    """Run N×N matvec with full stack instrumentation.

    Returns the LRUStack with all steps recorded.

    Stack layout:
        A elements: keys 1..N²  (row-major: A[i][j] = key i*N + j + 1)
        x elements: keys N²+1..N²+N
    """
    ctx = LRUStack()

    # Allocate A (row-major)
    A = {}
    for i in range(N):
        for j in range(N):
            A[i, j] = ctx.allocate(f"A[{i},{j}]")

    # Allocate x
    x = {}
    for j in range(N):
        x[j] = ctx.allocate(f"x[{j}]")

    # Run matvec: y[i] = sum_j A[i][j] * x[j]
    for i in range(N):
        # j = 0: s = A[i][0] * x[0]
        ctx.read_pair(A[i, 0], x[0], f"mul_A{i}0_x0")
        s_key = ctx.allocate(f"prod_{i}_0")

        for j in range(1, N):
            # t = A[i][j] * x[j]
            ctx.read_pair(A[i, j], x[j], f"mul_A{i}{j}_x{j}")
            t_key = ctx.allocate(f"prod_{i}_{j}")

            # s = s + t
            ctx.read_pair(s_key, t_key, f"add_s{i}_{j}")
            s_key = ctx.allocate(f"sum_{i}_{j}")

    return ctx


def print_steps(ctx, max_steps=None):
    print(f"{'step':>4}  {'event':<16} {'key':>4} {'depth':>5}  stack (bottom→top)")
    print("-" * 80)
    for step, event, key, depth, stack in ctx.steps[:max_steps]:
        print(f"{step:>4}  {event:<16} {key:>4} {depth:>5}  {stack}")


def to_mathematica(ctx, N):
    """Generate Mathematica code for the stack state and depth functions."""
    lines = []
    lines.append(f"(* Matvec N={N}: {len(ctx.trace)} reads *)")
    lines.append(f"(* Initial stack: A[i,j] = keys 1..{N*N}, x[j] = keys {N*N+1}..{N*N+N} *)")
    lines.append("")

    # readDepth: list of depths
    lines.append(f"readDepth{N} = {{{', '.join(str(d) for d in ctx.trace)}}};")
    lines.append("")

    # stackState: list of stack snapshots AFTER each step
    # We need stacks after each step. The recorded snapshots are BEFORE each read.
    # Let's also record the stack after all reads.
    # Actually the steps record stack BEFORE each read. Let's output those.
    lines.append(f"(* stackState{N}[[k]] = stack BEFORE the k-th read (1-indexed) *)")
    stack_strs = []
    for step, event, key, depth, stack in ctx.steps:
        stack_strs.append("{" + ", ".join(str(s) for s in stack) + "}")
    lines.append(f"stackState{N} = {{{', '.join(stack_strs)}}};")
    lines.append("")

    # Cost function
    lines.append(f"(* ByteDMD cost = Sum[Ceiling[Sqrt[d]], {{d, readDepth{N}}}] *)")
    cost = sum(math.isqrt(d - 1) + 1 for d in ctx.trace)
    lines.append(f"byteDMDCost{N} = {cost};")
    lines.append("")

    # Depth as function of step
    lines.append(f"(* Plot: ListPlot[readDepth{N}, PlotLabel -> \"Read depth at each step\"] *)")
    lines.append(f"(* Plot: ListPlot[Ceiling[Sqrt[#]] & /@ readDepth{N}, PlotLabel -> \"ByteDMD cost per read\"] *)")

    return "\n".join(lines)


def to_mathematica_symbolic(N):
    """Generate a symbolic Mathematica function for the depth pattern.

    For a general N×N matvec, the depth of each read follows a periodic
    pattern determined by the inner-loop structure. This function outputs
    the symbolic formula.
    """
    lines = []
    lines.append(f"(* Symbolic matvec depth formula for general N *)")
    lines.append(f"(*")
    lines.append(f"   Stack layout: A[i,j] at keys 1..N^2 (row-major), x[j] at keys N^2+1..N^2+N")
    lines.append(f"   Operations per row i (0-indexed):")
    lines.append(f"     j=0: mul(A[i,0], x[0]) -> 2 reads + 1 alloc")
    lines.append(f"     j>0: mul(A[i,j], x[j]) -> 2 reads + 1 alloc")
    lines.append(f"           add(prev_sum, product) -> 2 reads + 1 alloc")
    lines.append(f"   Reads per row: 2 + 4*(N-1) = 4N-2")
    lines.append(f"   Total reads: N*(4N-2)")
    lines.append(f"   Allocs per row: 1 + 2*(N-1) = 2N-1")
    lines.append(f"   Total allocs: N*(2N-1)")
    lines.append(f"*)")
    lines.append(f"")
    lines.append(f"(* The depth of A[i,j] when first read depends on:")
    lines.append(f"   - Its original position: i*N + j (0-indexed from bottom)")
    lines.append(f"   - How many items have been pushed above it since initialization")
    lines.append(f"   - How many items below it have been moved to top (via reads)")
    lines.append(f"   For the unmanaged strategy, the stack only grows, so the depth")
    lines.append(f"   of A[i,j] at its first read is:")
    lines.append(f"     depth = (current stack size) - (current position of A[i,j])")
    lines.append(f"   Current stack size at start of row i:")
    lines.append(f"     N^2 + N + i*(2N-1)  (initial + allocs from previous rows)")
    lines.append(f"*)")
    lines.append(f"")
    lines.append(f"matvecStackSize[n_, i_, j_] := n^2 + n + i*(2n - 1) + 2*j;")
    lines.append(f"(* Stack size just before reading A[i,j] (the j-th column in row i) *)")
    lines.append(f"(* Note: each column j in a row does 2 allocs (except j=0 which does 1) *)")
    return "\n".join(lines)


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    print(f"=== Matvec N={N} ===\n")
    ctx = matvec_traced(N)

    print_steps(ctx, max_steps=30)
    print(f"\n... ({len(ctx.steps)} total steps)\n")
    print(f"Trace: {ctx.trace}")
    cost = sum(math.isqrt(d - 1) + 1 for d in ctx.trace)
    print(f"ByteDMD cost: {cost}")
    print()

    mma = to_mathematica(ctx, N)
    print("=== Mathematica code ===")
    print(mma)
    print()

    sym = to_mathematica_symbolic(N)
    print(sym)
