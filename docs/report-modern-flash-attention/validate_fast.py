#!/usr/bin/env python3
"""Validate that bytedmd_fast produces identical results to bytedmd."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bytedmd_fast import traced_eval as traced_eval_fast, trace_to_bytedmd as ttb_fast
from bytedmd import traced_eval as traced_eval_orig, trace_to_bytedmd as ttb_orig


def my_add(a, b, c, d):
    return b + c


def my_composite_func(a, b, c, d):
    e = b + c
    f = a + d
    return e > f


def dot(a, b):
    return a[0]*b[0] + a[1]*b[1]


def my_relu(a):
    if a > 0:
        return a * 2
    return a


def test_case(name, func, args, bpe=1):
    trace_orig, res_orig = traced_eval_orig(func, args)
    trace_fast, res_fast = traced_eval_fast(func, args)
    
    cost_orig = ttb_orig(trace_orig, bpe)
    cost_fast = ttb_fast(trace_fast, bpe)
    
    ok_trace = trace_orig == trace_fast
    ok_cost = cost_orig == cost_fast
    
    status = "✓" if (ok_trace and ok_cost) else "✗"
    print(f"  {status} {name}: trace={'match' if ok_trace else 'MISMATCH'}, "
          f"cost={'match' if ok_cost else 'MISMATCH'} ({cost_orig})")
    
    if not ok_trace:
        print(f"    orig trace: {trace_orig}")
        print(f"    fast trace: {trace_fast}")
    if not ok_cost:
        print(f"    orig cost: {cost_orig}")
        print(f"    fast cost: {cost_fast}")
    
    return ok_trace and ok_cost


if __name__ == '__main__':
    print("Validating bytedmd_fast against bytedmd...")
    all_ok = True
    
    all_ok &= test_case("my_add", my_add, (1, 2, 3, 4))
    all_ok &= test_case("a+a", lambda a: a + a, (5,))
    all_ok &= test_case("composite", my_composite_func, (1, 2, 3, 4))
    all_ok &= test_case("dot product", dot, ([0, 1], [2, 3]))
    all_ok &= test_case("relu(5)", my_relu, (5,))
    all_ok &= test_case("relu(-5)", my_relu, (-5,))
    all_ok &= test_case("range", lambda n: [i for i in range(n)], (3,))
    all_ok &= test_case("divmod", lambda a, b: divmod(a, b)[0] + divmod(a, b)[1], (10, 3))
    
    # Test with bytes_per_element
    all_ok &= test_case("my_add(bpe=2)", my_add, (1, 2, 3, 4), bpe=2)
    
    # Larger test: small attention-like access pattern
    def mini_matmul(A, B):
        n = len(A)
        C = [[None] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                s = A[i][0] * B[0][j]
                for k in range(1, n):
                    s = s + A[i][k] * B[k][j]
                C[i][j] = s
        return C
    
    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[5.0, 6.0], [7.0, 8.0]]
    all_ok &= test_case("2x2 matmul", mini_matmul, (A, B))
    
    A3 = [[float(i*3+j) for j in range(3)] for i in range(3)]
    B3 = [[float(i*3+j+9) for j in range(3)] for i in range(3)]
    all_ok &= test_case("3x3 matmul", mini_matmul, (A3, B3))
    
    print()
    if all_ok:
        print("All validations passed! ✓")
    else:
        print("SOME VALIDATIONS FAILED! ✗")
        sys.exit(1)
