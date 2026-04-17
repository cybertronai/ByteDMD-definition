# Strassen vs Systolic Array: A Physical Energy Analysis

To understand how a **Systolic Array** (like a Google TPU or NVIDIA Tensor Core) compares to the **Single-Processor** architecture modeled in your scripts, we must evaluate them through the exact physical hardware model you built: `cost += ceil(sqrt(addr))`.

In VLSI silicon hardware, memory requires 2D physical space. A cache holding $S$ elements has a physical area of $S$, making the average wire length to access it $O(\sqrt{S})$. Because driving electrical signals across wires charges parasitic capacitance, **energy is strictly proportional to physical distance**.

Assuming the ALUs (the actual math) are free, the entire energy footprint of the system is determined by **data movement**. Here is the mathematical breakdown of how the architectures diverge.

---

## 1. The Single Processor (Von Neumann Bottleneck)

In a single-processor CPU, all operations funnel through one central point in space. To prevent the ALU from constantly waiting on main memory, you introduced an **L1 Scratchpad** of capacity $S$. Let's call the scratchpad's physical grid dimension $T \times T$ (so $S = T^2$).

To multiply $N \times N$ matrices, you tile the problem into $(N/T)^3$ block multiplications.

**Global Fetches** (RAM → Scratchpad): You fetch $T^2$ elements from Main Memory (which has a physical radius of $N$).

$$\text{Global Energy} = (N/T)^3 \times O(T^2) \times O(N) = \mathbf{O(N^4 / T)}$$

**Local Fetches** (Scratchpad → ALU): The central ALU must read/write to the Scratchpad $T^3$ times. The physical radius of the scratchpad is $O(T)$.

$$\text{Local Energy} = (N/T)^3 \times O(T^3) \times O(T) = \mathbf{O(N^3 \cdot T)}$$

**Total CPU Energy:**

$$O(N^4/T + N^3 \cdot T)$$

Notice the physical trap here: If you increase the cache size $T$ to save off-chip memory trips, your cache physically expands, causing the local wire energy $O(N^3 \cdot T)$ to explode! If you perfectly balance this equation ($T = N^{0.5}$), the best a single-processor can achieve is $\mathbf{O(N^{3.5})}$.

> *Note: Even with an infinitely perfect, log-deep multi-level cache hierarchy, this only drops to $O(N^3 \log N)$.*

---

## 2. The Systolic Array (Spatial Compute)

A Systolic Array breaks this bottleneck by **destroying the central cache**. Instead of an empty $T \times T$ SRAM grid, it interleaves a grid of $T \times T$ ALUs directly into the memory layout.

**Global Fetches** (RAM → Edge): The matrix tiles are still fetched from RAM.

$$\text{Global Energy} = \mathbf{O(N^4 / T)} \quad \text{(a tie with the CPU)}$$

**Local Fetches** (ALU → ALU): Here is the magic. Once data is loaded onto the edges of the grid, it does not sit in a cache waiting to be queried. It rhythmically steps from one ALU directly to its **physically adjacent neighbor**. The distance between neighbors is always $O(1)$, completely independent of how large the array grows.

$$\text{Local Energy} = (N/T)^3 \times O(T^3) \times O(1) = \mathbf{O(N^3)}$$

**Total Systolic Energy:**

$$O(N^4/T + N^3)$$

By turning centralized $O(T)$ cache lookups into local $O(1)$ wire shifts, the local compute energy drops from $O(N^3 \cdot T)$ to a flat $O(N^3)$. You can now scale the array up to hold the entire matrix ($T=N$), resulting in an absolute total energy bound of exactly $\mathbf{O(N^3)}$.

---

## 3. The Grand Strassen Paradox

But wait, you just proved in your script that Strassen's Algorithm drops the logical operations on a Single-Processor down to $O(N^{2.807})$. Does a single-processor running Strassen beat a Systolic Array running standard matmul?

**No. They tie at exactly $O(N^3)$ data movement energy.**

We can mathematically prove this using your 2D physical cost model. Let's calculate the energy for Strassen on an optimal multi-level CPU cache hierarchy:

- For any cache level of capacity $M$, the physical wire distance is $O(M^{0.5})$.
- The number of cache misses Strassen incurs at capacity $M$ is known to be $O(N^{2.807} / M^{0.403})$.

$$\text{Energy at cache level } M = \text{Misses} \times \text{Distance} = N^{2.807} \times M^{0.097}$$

Because the exponent of $M$ is positive ($0.097 > 0$), the energy across the cache hierarchy is a **geometric series dominated entirely by the highest level** (Main Memory, where $M = N^2$).

If we evaluate the energy at $M = N^2$:

$$\text{Total CPU Strassen Energy} = N^{2.807} \times (N^2)^{0.097} = N^{2.807} \times N^{0.194} = \mathbf{O(N^3)}$$

---

## Conclusion

This reveals a profound intersection between algorithmic complexity and the physics of our universe:

> Just to **read** an $N \times N$ matrix from a 2D memory layout into a central processor requires fetching $N^2$ elements across an average physical radius of $N$. Therefore, the **absolute physical lower-bound energy** for any matrix algorithm — even if operations are free — is $N^2 \times N = O(N^3)$.

- A **Single Processor running Strassen** achieves $O(N^3)$ energy because it reduces operations enough to stop thrashing the inner caches.
- A **Systolic Array** achieves $O(N^3)$ energy by physically decentralizing the inner cache.

The industry chose Systolic Arrays because Strassen requires heavy recursive memory allocations and non-local matrix additions that demand chaotic, long-distance wiring. A Systolic Array achieves the theoretical $O(N^3)$ limits of physical space using predictable, purely local $O(1)$ wires, proving that **optimizing hardware geometry is often infinitely more powerful than optimizing mathematical algorithms**.
