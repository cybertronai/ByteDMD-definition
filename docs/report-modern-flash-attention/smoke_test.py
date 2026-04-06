#!/usr/bin/env python3
"""Quick smoke test: run benchmarks at tiny sizes to validate correctness."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bytedmd_fast import traced_eval, trace_to_bytedmd
from bytedmd import traced_eval as traced_eval_orig, trace_to_bytedmd as ttb_orig
from attention_variants import (
    naive_attention, flash_attention_v1, flash_attention_v2, flash_attention_v3,
    make_matrix,
)

def measure(func, args):
    trace, _ = traced_eval(func, args)
    return trace_to_bytedmd(trace, 1)

def measure_orig(func, args):
    trace, _ = traced_eval_orig(func, args)
    return ttb_orig(trace, 1)

N, d = 4, 2
Q, K, V = make_matrix(N, d), make_matrix(N, d), make_matrix(N, d)

# Validate fast tracer matches original on naive attention
naive_fast = measure(naive_attention, (make_matrix(N,d), make_matrix(N,d), make_matrix(N,d)))
naive_orig = measure_orig(naive_attention, (make_matrix(N,d), make_matrix(N,d), make_matrix(N,d)))
print(f"Naive N=4,d=2: fast={naive_fast}, orig={naive_orig}, match={naive_fast==naive_orig}")

# Validate flash v1 matches
v1_fast = measure(lambda Q,K,V: flash_attention_v1(Q,K,V,Bk=2), 
                  (make_matrix(N,d), make_matrix(N,d), make_matrix(N,d)))
v1_orig = measure_orig(lambda Q,K,V: flash_attention_v1(Q,K,V,Bk=2),
                       (make_matrix(N,d), make_matrix(N,d), make_matrix(N,d)))
print(f"Flash v1 N=4,d=2,Bk=2: fast={v1_fast}, orig={v1_orig}, match={v1_fast==v1_orig}")

# All variants at N=4
print(f"\nAll variants at N={N}, d={d}:")
print(f"  naive:                {measure(naive_attention, (make_matrix(N,d), make_matrix(N,d), make_matrix(N,d)))}")
print(f"  flash_v1(Bk=2):      {measure(lambda Q,K,V: flash_attention_v1(Q,K,V,Bk=2), (make_matrix(N,d), make_matrix(N,d), make_matrix(N,d)))}")
print(f"  flash_v2(Bk=2):      {measure(lambda Q,K,V: flash_attention_v2(Q,K,V,Bk=2), (make_matrix(N,d), make_matrix(N,d), make_matrix(N,d)))}")
print(f"  flash_v3(Bq=2,Bk=2): {measure(lambda Q,K,V: flash_attention_v3(Q,K,V,Bq=2,Bk=2), (make_matrix(N,d), make_matrix(N,d), make_matrix(N,d)))}")

# N=8 to see trends
N = 8
print(f"\nAll variants at N={N}, d={d}:")
print(f"  naive:                {measure(naive_attention, (make_matrix(N,d), make_matrix(N,d), make_matrix(N,d)))}")
print(f"  flash_v1(Bk=2):      {measure(lambda Q,K,V: flash_attention_v1(Q,K,V,Bk=2), (make_matrix(N,d), make_matrix(N,d), make_matrix(N,d)))}")
print(f"  flash_v1(Bk=4):      {measure(lambda Q,K,V: flash_attention_v1(Q,K,V,Bk=4), (make_matrix(N,d), make_matrix(N,d), make_matrix(N,d)))}")
print(f"  flash_v2(Bk=2):      {measure(lambda Q,K,V: flash_attention_v2(Q,K,V,Bk=2), (make_matrix(N,d), make_matrix(N,d), make_matrix(N,d)))}")
print(f"  flash_v2(Bk=4):      {measure(lambda Q,K,V: flash_attention_v2(Q,K,V,Bk=4), (make_matrix(N,d), make_matrix(N,d), make_matrix(N,d)))}")
print(f"  flash_v3(Bq=2,Bk=2): {measure(lambda Q,K,V: flash_attention_v3(Q,K,V,Bq=2,Bk=2), (make_matrix(N,d), make_matrix(N,d), make_matrix(N,d)))}")
print(f"  flash_v3(Bq=2,Bk=4): {measure(lambda Q,K,V: flash_attention_v3(Q,K,V,Bq=2,Bk=4), (make_matrix(N,d), make_matrix(N,d), make_matrix(N,d)))}")

print("\nSmoke test complete!")
