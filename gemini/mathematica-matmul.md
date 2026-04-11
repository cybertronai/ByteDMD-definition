Source: https://docs.google.com/document/d/1r0JXPrLHSePEKKek78dUmc5qDWTwBepFDUI77Ut1KYg/edit

To compute the precise ByteDMD trace cost of a standard naive matrix multiplication ($C\_{i,j} \= \\sum\_{k=0}^{N-1} A\_{i,k} \\times B\_{k,j}$), we must systematically analyze the behavior of the **demand-paged fully associative stack** coupled with the **two-pass aggressive liveness analysis**.

![JSON icon][image1]  
out-177593...1464595503  
JSON

Because variables vaporize immediately after their logical last use completes, the active stack size stays strictly bounded. We can extract closed-form algebraic depth bounds for the four semantic read components of the inner loops without needing to trace AST intercepts or explicitly mutate lists.

### **1\. The Precise Trace Depth Formulas**

Over the execution of the nested loops across coordinates $(i, j, k)$, exactly $4N^3 \- 2N^2$ memory read events occur. The LRU stack distance (depth) of each accessed variable settles into highly predictable polynomial boundaries:

**1\. Intermediate Products ($P\_k \= A\_{i,k} \\times B\_{k,j}$):**

* Because the product is evaluated and immediately pushed to the top of the stack prior to the addition $S\_k \= S\_{k-1} \+ P\_k$, the product $P\_k$ is perpetually accessed exactly at the top of the cache.  
* **Depth Formula:** $D(P) \= \\mathbf{1}$ for all exactly $N^2(N-1)$ addition loops.

**2\. Accumulator Elements ($S$):**

* For $k \> 0$, the previously computed running sum $S\_{k-1}$ is fetched just after $A$, $B$, and $P$ are evaluated and pushed onto the stack. Due to aggressive compaction (variables die immediately upon loop boundaries), the baseline stack distance of the accumulator is effectively fixed.  
* **Depth Formula:** $D(S) \= \\mathbf{4 \- \\delta\_i \- \\delta\_j}$ where $\\delta\_i \= 1$ if $i \= N-1$ else $0$, and $\\delta\_j \= 1$ if $j \= N-1$ else $0$.  
  *(Thus, exactly $(N-1)^3$ accesses evaluate at Depth 4).*

**3\. Left Matrix ($A\_{i,k}$):**

* Row $i$ of Matrix $A$ is heavily re-read $N$ times across the $j$ loop. Its last usage is cleanly detected at $j \= N-1$, meaning it safely vaporizes at the end of the row and doesn't pollute the LRU stack across the outer $i$ boundary.  
* **Depth Formula:** For interior block reads ($0 \< i, j \< N-1$), the depth is bounded to exactly $\\mathbf{2N \+ 1}$ for $k=0$ and $\\mathbf{2N \+ 2}$ for $k \> 0$.

**4\. Right Matrix ($B\_{k,j}$):**

* Matrix $B$ is completely traversed over $i=0$. Because elements $B\_{k,j}$ are continually required for every subsequent row $i$, the entire $N \\times N$ matrix is kept alive on the stack until the very final $i \= N-1$ sweep.  
* **Depth Formula:** When reading $B\_{k,j}$ in the interior ($0 \< i, j \< N-1$), the stack distance is offset by the living entirety of the matrix plus the alive outputs $C$ and the working row $A$. Its depth is exactly $\\mathbf{N^2 \+ 2N}$ for $k=0$ and $\\mathbf{N^2 \+ 2N \+ 1}$ for $k\>0$.

### ---

**2\. Mathematical Code**

By mapping these exact analytical limits through piecewise algebra, we entirely bypass the immense $\\mathcal{O}(N^5)$ algorithmic overhead of AST traversal, generic object wrapping, and index-based state array liveness permutations.

The following mathematical Python code runs in perfect $\\mathcal{O}(N^3)$ computational time and returns the exact ByteDMD L1 simulated execution footprint for square matrices.

Python

import math

def bytedmd\_cost\_naive\_matmul(N: int, bytes\_per\_element: int \= 1) \-\> int:  
    """  
    Computes the precise ByteDMD cost for naive NxN matrix multiplication   
    using mathematically derived piecewise algebraic LRU depth bounds.  
    """  
    if N \<= 0: return 0

    def element\_cost(d: int) \-\> int:  
        """Applies the ByteDMD closed-form pricing function to an exact depth."""  
        if d \<= 0: return 0  
        if bytes\_per\_element \== 1:  
            return math.isqrt(d \- 1) \+ 1  
          
        def \_sum\_usqrt(x):  
            if x \<= 0: return 0  
            M \= math.isqrt(x \- 1) \+ 1  
            return M \* (6 \* x \- 2 \* M \* M \+ 3 \* M \- 1) // 6  
              
        return \_sum\_usqrt(d \* bytes\_per\_element) \- \_sum\_usqrt((d \- 1) \* bytes\_per\_element)

    total\_cost \= 0  
      
    \# 1\. Product (P) depths: Always immediately consumed at the top of the stack (Depth 1\)  
    total\_cost \+= element\_cost(1) \* (N \*\* 2 \* (N \- 1))  
      
    \# 2\. Accumulator (S) depths: Bounded strictly by liveness vaporizations  
    if N \>= 2:  
        total\_cost \+= element\_cost(2) \* (N \- 1)  
        total\_cost \+= element\_cost(3) \* (2 \* (N \- 1)\*\*2)  
        total\_cost \+= element\_cost(4) \* ((N \- 1)\*\*3)  
          
    \# 3\. Matrix A and B accesses (Piecewise block conditionals based on loops i, j, k)  
    for i in range(N):  
        for j in range(N):  
            for k in range(N):  
                  
                \# \--- Matrix A Depths \---  
                if j \== 0:  
                    if i \== 0:  
                        valA \= 1 if k \== 0 else 2\*k \+ 2  
                    elif i \== N \- 1:  
                        valA \= 2\*N\*\*2 \- N \+ 1 if k \== 0 else 2\*N\*\*2 \- N \+ 2  
                    else:  
                        if k \== 0: valA \= N\*\*2 \+ 1 \+ i\*N  
                        elif k \== 1: valA \= N\*\*2 \+ 3 \+ i\*N  
                        else: valA \= N\*\*2 \+ i\*N \+ k \+ 2  
                elif j \== N \- 1:  
                    if i \== N \- 1:  
                        valA \= N \+ 1 if k \<= 1 else N \- k \+ 2  
                    else:  
                        valA \= 2\*N \+ 1 if k \<= 1 else 2\*N \- k \+ 2  
                else:  
                    if i \== N \- 1:  
                        valA \= N \+ 1 if k \== 0 else N \+ 2  
                    else:  
                        valA \= 2\*N \+ 1 if k \== 0 else 2\*N \+ 2  
                  
                total\_cost \+= element\_cost(valA)  
                  
                \# \--- Matrix B Depths \---  
                if i \== 0:  
                    if j \== 0:  
                        if k \== 0: valB \= 2  
                        elif k \== 1: valB \= 5  
                        else: valB \= 2\*k \+ 3  
                    elif j \== N \- 1:  
                        valB \= N\*\*2 \+ N if k \== 0 else N\*\*2 \+ N \+ 1  
                    else:  
                        base \= (N \+ 1) \* (j \+ 1)  
                        if k \== 0: valB \= base  
                        elif k \== 1: valB \= base \+ 2  
                        else: valB \= base \+ k \+ 1  
                elif i \== N \- 1:  
                    if j \== 0:  
                        valB \= N\*\*2 \+ N if k \== 0 else N\*\*2 \+ N \+ 1  
                    elif j \== N \- 1:  
                        if k \== 0: valB \= 3\*N  
                        elif k \== 1: valB \= 3\*N \- 1  
                        else: valB \= 3\*N \- 2\*k \+ 1  
                    else:  
                        base \= N \* (N \- j \+ 2)  
                        valB \= base if k \<= 1 else base \- k \+ 1  
                else:  
                    if j \== 0:  
                        if k \== 0: valB \= N\*\*2 \+ N  
                        elif k \== 1: valB \= N\*\*2 \+ N \+ 2  
                        else: valB \= N\*\*2 \+ N \+ k \+ 1  
                    elif j \== N \- 1:  
                        valB \= N\*\*2 \+ 2\*N if k \<= 1 else N\*\*2 \+ 2\*N \- k \+ 1  
                    else:  
                        valB \= N\*\*2 \+ 2\*N if k \== 0 else N\*\*2 \+ 2\*N \+ 1  
                  
                total\_cost \+= element\_cost(valB)  
                  
    return total\_cost

\# Exact mathematical evaluations:  
\# \[bytedmd\_cost\_naive\_matmul(N) for N in range(1, 10)\]  
\# Outputs \-\> \[3, 54, 236, 676, 1516, 2899, 5008, 8443, 12861\]  


