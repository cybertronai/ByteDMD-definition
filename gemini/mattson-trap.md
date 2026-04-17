# The Mattson Stack Distance Trap

The divergence between Classic ByteDMD (based on Mattson's classic LRU stack distance) and Live-Bytes ByteDMD exposes a profound flaw in how early caching models analyzed algorithms.

The flaw is the **"Fossil Record" effect** (or Sibling Branch Pollution). Because Mattson's original stack lacks a concept of variable death, any algorithm that allocates temporary arrays acts like a massive memory leak. Dead temporaries permanently pile up in the stack, pushing long-lived active data into deeper, slower cache layers.

Live-Bytes ByteDMD, on the other hand, accurately models a smart compiler managing a Physical Scratchpad. When a temporary array dies, its physical slot is instantly recycled, keeping long-lived active data safely pinned near the processor.

Here are three famous algorithms where Mattson's stack hallucinated a severe memory bottleneck, but Live-Bytes mathematically proves the algorithm is highly efficient.

---

## 1. Karatsuba Integer Multiplication (The Master Theorem Flip)

Karatsuba's algorithm multiplies two $N$-digit numbers. By splitting the numbers in half, it requires exactly 3 recursive multiplications instead of 4, famously reducing the theoretical complexity from $O(N^2)$ to $O(N^{\log_2 3}) \approx O(N^{1.585})$.

### The Algorithm Trace

To multiply $(X_1, X_0)$ and $(Y_1, Y_0)$:

1. $Z_0 = \text{Karatsuba}(X_0, Y_0)$
2. $Z_2 = \text{Karatsuba}(X_1, Y_1)$
3. $Z_{\text{cross}} = \text{Karatsuba}(X_1 + X_0, Y_1 + Y_0)$
4. Combine: $Z = Z_0 + (Z_{\text{cross}} - Z_0 - Z_2) \cdot \text{Base} + Z_2 \cdot \text{Base}^2$

### The Classic Mattson Trap: $\Theta(N^{1.79})$

Look at Step 4. To compute the final answer, the CPU must read $Z_0$. But $Z_0$ was computed before $Z_2$ and $Z_{\text{cross}}$.

- During the recursive calculations of $Z_2$ and $Z_{\text{cross}}$, the program functionally allocated an enormous fractal tree of temporary sub-arrays. The total number of dead temporaries generated is proportional to their compute time: $\Theta(N^{1.585})$.
- When the CPU finally reads $Z_0$ (size $N$), it is buried under all $\Theta(N^{1.585})$ of those dead arrays.
- Cost to read $Z_0$: $N$ elements $\times \sqrt{N^{1.585} \text{ distance}} = \Theta(N^{1.792})$.
- **The Recurrence:** $T(N) = 3T(N/2) + \Theta(N^{1.792})$.

Because the $O(N^{1.792})$ memory overhead strictly dominates the branching factor, the Master Theorem makes the root of the tree dominate. Classic DMD claims Karatsuba is severely memory-bound at $\Theta(N^{1.792})$, destroying its theoretical advantage!

### The Live-Bytes Reality: $\Theta(N^{1.585})$

Under Live-Bytes, when $Z_2$ and $Z_{\text{cross}}$ finish, all of their internal temporaries instantly die and their L1 physical addresses are recycled.

- $Z_0$ is only pushed down by the active $O(N)$ variables surviving in the current scope.
- Cost to read $Z_0$: $N$ elements $\times \sqrt{N \text{ distance}} = \Theta(N^{1.5})$.
- **The Recurrence:** $T(N) = 3T(N/2) + \Theta(N^{1.5})$.

Because the $3\times$ branching factor overpowers the $N^{1.5}$ memory overhead, the leaves of the tree dominate. The memory penalty completely vanishes, allowing Karatsuba to run at its true CPU-bound speed of $\Theta(N^{1.585})$.

---

## 2. Backpropagation Through Time (The Spurious Time Penalty)

Consider training a Recurrent Neural Network (RNN) via Backpropagation Through Time (BPTT) for $T$ timesteps. The weight matrix $W$ is shared, but the hidden states $H_t$ (size $N$) are unique to each timestep.

### The Algorithm Trace

1. **Forward Pass:** Compute $H_1, H_2, \ldots, H_T$.
2. **Backward Pass:** Traverse backward from $T$ down to 1. At step $t$, compute the gradient $\nabla H_t$ by reading the old hidden state $H_t$.

### The Classic Mattson Trap: $\Theta(T^{1.5} N^{1.5})$

When the backward pass reaches timestep $t=1$, it must read $H_1$.

- $H_1$ was written at the very beginning of the forward pass.
- Between writing $H_1$ and reading $H_1$, the network allocated every future hidden state ($H_2 \ldots H_T$) and every future gradient ($\nabla H_T \ldots \nabla H_2$).
- Mattson assumes all $2T$ of these unique arrays permanently push $H_1$ deeper into the cache. The depth of $H_t$ is $\Theta(t \cdot N)$.
- The cost to read $H_t$ is $N \times \sqrt{t \cdot N}$.
- Summing this penalty across all $T$ timesteps yields $\sum_{t=1}^{T} N \sqrt{tN} = \Theta(T^{1.5} N^{1.5})$. Mattson falsely claims that running an RNN for longer makes the data-movement per step exponentially worse!

### The Live-Bytes Reality: $\Theta(T \cdot N^{1.5})$

A real AI compiler aggressively reuses scratchpad buffers.

- As the backward pass proceeds, the future hidden states and gradients ($H_T, \nabla H_T$, etc.) die and their physical SRAM addresses are instantly recycled.
- When the backward pass reaches step $t$, the massive pile of tensors from steps $t+1 \ldots T$ have vanished. The only live variable more recently accessed than $H_t$ is the current gradient $\nabla H_t$.
- The depth of $H_t$ is bounded strictly to $O(N)$.
- The cost to read $H_t$ is $N \times \sqrt{N}$, completely independent of time. Summing over $T$ steps yields the true linear physical cost: $\Theta(T \cdot N^{1.5})$.

---

## 3. Out-of-Place FFT (Sibling Branch Pollution)

Divide-and-Conquer algorithms that use functional, out-of-place memory allocation (like Cooley-Tukey FFT or Top-Down Merge Sort) contain a trap called the **"Combine Step Penalty."**

### The Algorithm Trace

1. $E = \text{FFT}(\text{Even\_Elements})$
2. $O = \text{FFT}(\text{Odd\_Elements})$
3. **Combine:** Loop over $N$, evaluating $\text{Result}[k] = E[k] + W \cdot O[k]$

### The Classic Mattson Trap: $\Theta(N^{1.5} \sqrt{\log N})$

When the Combine step reads the $E$ array, what is its Mattson stack distance?

- After $E$ was generated, the $O$ recursive branch spawned a massive fractal tree of temporary sub-arrays, totaling $\Theta(N \log N)$ unique variable allocations.
- Classic Mattson forces the $E$ array to look past all $\Theta(N \log N)$ pieces of garbage.
- The cost to read $E$ during the combine step is $N \times \sqrt{N \log N}$.
- The data movement recurrence is $T(N) = 2T(N/2) + N^{1.5} \sqrt{\log N}$, artificially penalizing the algorithm with a spurious $\sqrt{\log N}$ factor.

### The Live-Bytes Reality: $\Theta(N^{1.5})$

- A physical memory allocator overwrites the exact same scratchpad slots over and over during the $O$ recursive call. By the time it finishes, all internal sub-arrays are dead.
- The only thing surviving is the final output array $O$. The $E$ array is only pushed down by the active size of $O$ (which is $O(N)$).
- The recurrence becomes $T(N) = 2T(N/2) + N^{1.5}$, which resolves mathematically to exactly $\Theta(N^{1.5})$.

---

## Summary

Whenever an algorithm features temporal gaps between the creation of a variable and its later use, and functionally allocates temporary arrays during that gap, Mattson's classic stack distance mathematically fails. Live-Bytes metrics correctly protect these algorithms, reflecting how well-written code utilizes spatial scratchpads.
