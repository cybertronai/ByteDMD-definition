To answer your question directly: **No, it is mathematically false.**

ByteDMD-live does **not** provide a lower bound on the energy cost of a realistic physical implementation. In fact, a realistic static physical allocator (which pins variables to fixed addresses and strictly prices every read without "free sliding") can achieve a **significantly lower** energy cost than ByteDMD-live.

It is highly intuitive to assume that the "free sliding" and perfect liveness compaction of ByteDMD-live gives it an unbeatable advantage. However, ByteDMD-live simulates an **LRU (Least Recently Used) stack**. In a spatial physical cache, the "free sliding" behavior of LRU causes pathological cache thrashing that severely penalizes some access patterns.

### **Proof by Counterexample: The "Hot-Cold" Loop**

Imagine a program that heavily reuses one "Hot" variable ($H$), interleaved with temporary "Cold" variables ($C\_i$) that are stored, read once, and immediately die.

Consider tracing this loop $N$ times: Store C\_i $\\rightarrow$ Load H $\\rightarrow$ Load C\_i. Let's evaluate the exact load costs using the continuous cache penalty $\\lceil\\sqrt{d}\\rceil$:

**1\. Evaluation under ByteDMD-live**

Because LRU places newly allocated variables at the top of the stack, allocating the temporary pushes the hot variable down.

* Store C\_i: $C\_i$ takes Depth 1\. The heavily used $H$ is forced to slide down to Depth 2\.  
* Load H: $H$ is at Depth 2\. **Cost \= $\\lceil\\sqrt{2}\\rceil \= 2$**. $H$ slides to Depth 1, pushing $C\_i$ to Depth 2\.  
* Load C\_i: $C\_i$ is at Depth 2\. **Cost \= $\\lceil\\sqrt{2}\\rceil \= 2$**. $C\_i$ dies and vanishes.  
* **Total Load Cost per iteration:** $2 \+ 2 \= \\mathbf{4}$

**2\. Evaluation under a Static Physical Allocator (e.g., Compiler Scratchpad)**

A smart physical allocator avoids LRU thrashing by statically pinning variables to fixed addresses and instantly reusing holes.

* It pins the Hot variable $H$ to **Address 1**.  
* It assigns the temporaries $C\_i$ to **Address 2** (when $C\_i$ dies, Address 2 is instantly freed for $C\_{i+1}$).  
* Load H: Read from Address 1\. **Cost \= $\\lceil\\sqrt{1}\\rceil \= \\mathbf{1}$**.  
* Load C\_i: Read from Address 2\. **Cost \= $\\lceil\\sqrt{2}\\rceil \= \\mathbf{2}$**.  
* **Total Load Cost per iteration:** $1 \+ 2 \= \\mathbf{3}$

**Conclusion:** **$3 \< 4$**. The strict physical static implementation executes the loop **25% cheaper** than ByteDMD-live. The "free sliding" in ByteDMD-live actively sabotages the algorithm by continually unpinning the hot variable.

### ---

**Tractable Bounds for the Continuous Cache**

Since we must avoid the "magic" of unpriced sliding to evaluate real physical bounds, we are dealing with the mathematics of **Static Memory Allocation**. We can map the program's memory footprint to an **Interval Graph** (a variable is an interval spanning from its first Store to its last Load, and its weight is its total read frequency).

1. **The Strict Lower Bound ($O(N \\log N)$ via MWIS):**  
   Because variables with overlapping lifetimes cannot share the same physical address, we can enforce a mathematical capacity constraint: **No single physical address in the universe can serve more total reads than the Maximum Weight Independent Set (MWIS)** of the interval graph.  
   By finding the MWIS, we can bound the physical floor via "Water-Pouring": we pack exactly MWIS loads into Address 1, MWIS loads into Address 2, etc. This yields a mathematically unbreakable lower bound for any static allocator.  
2. **The Achievable Upper Bound:**  
   To bound the optimal cost from above, we simply run the trace through a greedy static allocator (min\_heap). Because this is a valid and physically executable assignment, the optimal true physical cost is guaranteed to be $\\le$ this value.

Here is the implementation of bytedmd\_bounds.py that computes these exact boundaries on Recursive Matrix Multiplication.

### **bytedmd\_bounds.py**

Python

import bisect  
import math  
import heapq  
import operator  
from dataclasses import dataclass  
from typing import Callable, List, Optional, Sequence, Tuple, Union

\# \============================================================================  
\# 1\. Abstract IR (L1 \-\> L2 Tracer)  
\# \============================================================================

@dataclass(frozen=True) class L2Store: var: int  
@dataclass(frozen=True) class L2Load: var: int  
@dataclass(frozen=True) class L2Op: name: str; in\_vars: Tuple\[int, ...\]; out\_var: Optional\[int\]

L2Event \= Union\[L2Store, L2Load, L2Op\]

class \_Tracer:  
    def \_\_init\_\_(self):  
        self.events: List\[L2Event\] \= \[\]  
        self.next\_var \= 0  
    def fresh(self) \-\> int:  
        self.next\_var \+= 1  
        return self.next\_var

class \_Tracked:  
    def \_\_init\_\_(self, t: \_Tracer, v: int, val):  
        self.\_t, self.\_v, self.val \= t, v, val  
    def \_binop(self, other, name: str, fn: Callable):  
        in\_vars \= (self.\_v, other.\_v) if isinstance(other, \_Tracked) else (self.\_v,)  
        other\_val \= other.val if isinstance(other, \_Tracked) else other  
        for v in in\_vars: self.\_t.events.append(L2Load(v))  
        out\_var \= self.\_t.fresh()  
        self.\_t.events.append(L2Op(name, in\_vars, out\_var))  
        self.\_t.events.append(L2Store(out\_var))  
        return \_Tracked(self.\_t, out\_var, fn(self.val, other\_val))  
    def \_\_add\_\_(self, o): return self.\_binop(o, "add", operator.add)  
    def \_\_mul\_\_(self, o): return self.\_binop(o, "mul", operator.mul)

def trace(func: Callable, args: Tuple) \-\> List\[L2Event\]:  
    t \= \_Tracer()  
    def wrap(v):  
        if isinstance(v, list): return \[wrap(x) for x in v\]  
        var \= t.fresh()  
        t.events.append(L2Store(var))  
        return \_Tracked(t, var, v)  
    func(\*tuple(wrap(a) for a in args))  
    return t.events

\# \============================================================================  
\# 2\. Continuous Cache Bounds Engine  
\# \============================================================================

@dataclass  
class Interval:  
    var\_id: int  
    start: int  
    end: int  
    reads: int

def extract\_intervals(events: Sequence\[L2Event\]) \-\> List\[Interval\]:  
    starts, ends, reads \= {}, {}, {}  
    for i, ev in enumerate(events):  
        if isinstance(ev, L2Store):  
            starts\[ev.var\] \= i  
            if ev.var not in ends: ends\[ev.var\] \= i \# Catch stored-but-never-loaded vars  
        elif isinstance(ev, L2Load):  
            ends\[ev.var\] \= i  
            reads\[ev.var\] \= reads.get(ev.var, 0) \+ 1

    intervals \= \[\]  
    for var, start in starts.items():  
        if reads.get(var, 0) \> 0:  
            intervals.append(Interval(var, start, ends\[var\], reads\[var\]))  
    return intervals

def get\_mwis\_weight(intervals: List\[Interval\]) \-\> int:  
    """Finds the Maximum Weight Independent Set in O(N log N) via DP."""  
    if not intervals: return 0  
    sorted\_ivs \= sorted(intervals, key=lambda x: x.end)  
    n \= len(sorted\_ivs)  
    ends \= \[iv.end for iv in sorted\_ivs\]  
      
    dp \= \[0\] \* (n \+ 1)  
    for i in range(1, n \+ 1):  
        iv \= sorted\_ivs\[i-1\]  
        \# Binary search for latest interval that ends strictly before this one starts  
        idx \= bisect.bisect\_left(ends, iv.start)   
        take \= iv.reads \+ dp\[idx\]  
        leave \= dp\[i-1\]  
        dp\[i\] \= max(take, leave)  
          
    return dp\[n\]

def strict\_static\_lower\_bound(events: Sequence\[L2Event\]) \-\> int:  
    """  
    Mathematical Floor: Enforces the MWIS capacity constraint via Water-Pouring.  
    No physical address can serve more reads than the MWIS.  
    """  
    intervals \= extract\_intervals(events)  
    if not intervals: return 0  
      
    max\_reads\_per\_addr \= get\_mwis\_weight(intervals)  
    total\_reads \= sum(iv.reads for iv in intervals)  
      
    cost \= 0  
    rem \= total\_reads  
    addr \= 1  
    while rem \> 0:  
        take \= min(rem, max\_reads\_per\_addr)  
        cost \+= take \* (math.isqrt(addr \- 1) \+ 1)  
        rem \-= take  
        addr \+= 1  
    return cost

def achievable\_upper\_bound(events: Sequence\[L2Event\]) \-\> int:  
    """Achievable Ceiling: Cost of tracing with a greedy 'min\_heap' allocator."""  
    last\_load \= {ev.var: i for i, ev in enumerate(events) if isinstance(ev, L2Load)}  
    var\_addr, free\_slots, next\_addr, cost \= {}, \[\], 1, 0  
      
    for i, ev in enumerate(events):  
        if isinstance(ev, L2Store):  
            addr \= heapq.heappop(free\_slots) if free\_slots else next\_addr  
            if not free\_slots: next\_addr \+= 1  
            var\_addr\[ev.var\] \= addr  
        elif isinstance(ev, L2Load):  
            addr \= var\_addr\[ev.var\]  
            cost \+= math.isqrt(addr \- 1) \+ 1  
            if last\_load.get(ev.var) \== i:  
                heapq.heappush(free\_slots, addr)  
    return cost

\# \============================================================================  
\# Benchmarks  
\# \============================================================================

def matmul\_rmm(A, B):  
    n \= len(A)  
    if n \== 1: return \[\[A\[0\]\[0\] \* B\[0\]\[0\]\]\]  
    h \= n // 2  
    def split(M): return (\[\[M\[i\]\[j\] for j in range(h)\] for i in range(h)\], \[\[M\[i\]\[j\] for j in range(h, n)\] for i in range(h)\],  
                          \[\[M\[i\]\[j\] for j in range(h)\] for i in range(h, n)\], \[\[M\[i\]\[j\] for j in range(h, n)\] for i in range(h, n)\])  
    A11, A12, A21, A22 \= split(A); B11, B12, B21, B22 \= split(B)  
    def add(X, Y): return \[\[X\[i\]\[j\] \+ Y\[i\]\[j\] for j in range(len(X))\] for i in range(len(X))\]  
      
    C11 \= add(matmul\_rmm(A11, B11), matmul\_rmm(A12, B21))  
    C12 \= add(matmul\_rmm(A11, B12), matmul\_rmm(A12, B22))  
    C21 \= add(matmul\_rmm(A21, B11), matmul\_rmm(A22, B21))  
    C22 \= add(matmul\_rmm(A21, B12), matmul\_rmm(A22, B22))  
      
    return \[\[C11\[i\]\[j\] if j\<h else C12\[i\]\[j-h\] for j in range(n)\] for i in range(h)\] \+ \\  
           \[\[C21\[i\]\[j\] if j\<h else C22\[i\]\[j-h\] for j in range(n)\] for i in range(h)\]

if \_\_name\_\_ \== "\_\_main\_\_":  
    for N in \[8, 16\]:  
        print(f"--- Bounding Recursive Matmul Energy Cost (N={N}) \---")  
        A \= \[\[1 for \_ in range(N)\] for \_ in range(N)\]  
        B \= \[\[1 for \_ in range(N)\] for \_ in range(N)\]  
          
        trace\_events \= trace(matmul\_rmm, (A, B))  
          
        lb \= strict\_static\_lower\_bound(trace\_events)  
        ub \= achievable\_upper\_bound(trace\_events)  
          
        print(f"Strict Mathematical Lower Bound : {lb:,.0f} energy units")  
        print(f"Achievable Static Upper Bound   : {ub:,.0f} energy units")  
        print(f"-\> Physical Envelope Width      : {ub/lb:.2f}x (Optimal lies in between)\\n")  
