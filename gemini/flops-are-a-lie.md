# FLOPs are a lie

The realization that "FLOPs are a lie" is the cornerstone of modern
High-Performance Computing (HPC) and AI optimization.

The standard theoretical model (the RAM model) assumes that calculating
$1 + 1$ takes the exact same amount of time as fetching a number from
Main Memory ($\mathcal{O}(1)$). But in physical reality, processors have
scaled exponentially faster than memory bandwidth (a phenomenon known
as the **Memory Wall**). Today, FLOPs are practically free; moving data
across a silicon chip is what you actually pay for.

Beyond Strassen's algorithm, here are some of the most famous examples
across computer science where theoretical FLOPs are deeply misleading,
categorized by *why* the standard model fails.

---

## Category 1: Doing fewer FLOPs is actually slower

These are algorithms that look theoretically optimal on a whiteboard,
but fall apart physically because they trade math for chaotic,
unpredictable memory access.

### 1. Fibonacci Heaps vs. Binary Heaps (Dijkstra's algorithm)

- **The FLOPs illusion.** To find the shortest path in a graph,
  Dijkstra's algorithm using a standard **Binary Heap** takes
  $\mathcal{O}((V + E) \log V)$ operations. In 1984, the Fibonacci
  Heap was invented, mathematically lowering the bound to
  $\mathcal{O}(E + V \log V)$.
- **The reality.** Fibonacci Heaps are almost never used in production.
  Standard Binary Heaps crush them.
- **The memory trap.** A Binary Heap is a flat, contiguous 1D array. A
  Fibonacci Heap is a sprawling forest of dynamically allocated,
  doubly-linked tree nodes. Doing math in the L1 cache is practically
  free; stalling the CPU to chase a randomized pointer across RAM costs
  hundreds of cycles. The cache misses completely eclipse the
  mathematical savings.

### 2. Sparse vs. Dense matrix math

- **The FLOPs illusion.** If a matrix is 90% zeros, you should use a
  sparse algorithm to skip them, theoretically saving 90% of the FLOPs.
- **The reality.** On modern GPUs, it is frequently much faster to
  pretend the matrix is 100% full, multiply all the zeros (explicitly
  wasting billions of FLOPs), and add them up.
- **The memory trap.** Dense matrices are contiguous blocks that stream
  perfectly into specialized GPU Tensor Cores. Sparse algorithms
  require indirect addressing (e.g., fetching `A[row_ptr[i]]`). This
  chaotic memory scattering breaks hardware prefetchers and starves the
  math units. Sparsity usually doesn't beat "dumb" dense math until the
  matrix is > 95% empty.

### 3. Winograd convolutions vs. Im2Col (computer vision)

- **The FLOPs illusion.** For a standard $3 \times 3$ image
  convolution, the **Winograd** algorithm uses a mathematical
  transformation to reduce the number of multiplications by 2.25× vs
  standard sliding windows.
- **The reality.** Hardware frameworks often favor **Im2Col** (Image to
  Column), which literally duplicates the image data in memory to
  flatten sliding windows into one giant matrix.
- **The memory trap.** Winograd requires complex memory read/write
  strides and massive temporary register usage. Im2Col intentionally
  bloats the memory footprint *and* the FLOP count so the data can be
  blasted through a highly-optimized, linear Dense Matrix Multiplication
  (GEMM) block. The brute-force streaming momentum obliterates the
  "smart" FLOP reduction.

---

## Category 2: Doing more FLOPs is actually faster

Because memory bandwidth is the ultimate bottleneck, it is frequently
much faster to redundantly compute the exact same math twice rather
than write the answer to RAM and read it back later.

### 4. Gradient checkpointing (LLM training)

- **The FLOPs illusion.** When training a neural network, the standard
  model says you must save the "forward pass" activations to memory so
  you don't waste FLOPs recalculating them during the "backward pass"
  (gradient updates).
- **The reality.** Modern AI frameworks intentionally delete the
  activations and re-calculate the entire forward pass from scratch
  during the backward pass.
- **The memory trap.** Writing gigabytes of intermediate tensors to GPU
  High Bandwidth Memory (HBM) and reading them back is incredibly slow.
  We deliberately increase the total FLOPs of the training run by ~30%,
  and the model trains faster in wall-clock time because it stays off
  the memory bus.

### 5. The death of the Lookup Table (LUT)

- **The FLOPs illusion.** In the 1990s, to calculate $\sin x$ or
  $e^x$, programmers precomputed the values into an array. Reading
  the array requires $\mathcal{O}(1)$ FLOPs. Recomputing the value
  using a Taylor series approximation requires ~20 FLOPs.
- **The reality.** Today, we perform dozens of "unnecessary" FLOPs
  instead of looking up the answer.
- **The memory trap.** A modern ALU can execute 20 math operations
  natively in the registers in 3 to 4 clock cycles. If the lookup table
  has fallen out of the L1 cache, fetching that single precomputed
  value from Main RAM will take 200 to 300 clock cycles. Recomputing
  math from scratch is vastly faster than looking it up.

---

## Category 3: Same math, different data geometry

These algorithms execute the exact same number of operations, but one
fundamentally alters the dataflow to cheat the memory hierarchy.

### 6. Heapsort vs. Quicksort

- **The FLOPs illusion.** Heapsort is theoretically superior. It
  guarantees $\mathcal{O}(N \log N)$ worst-case time. Quicksort has an
  average case of $\mathcal{O}(N \log N)$ but a catastrophic worst-case
  of $\mathcal{O}(N^2)$. Both algorithms do roughly the same number of
  comparisons.
- **The reality.** Quicksort routinely demolishes Heapsort in
  real-world benchmarks and is the default in almost all standard
  libraries.
- **The memory trap.** Heapsort navigates an array as a binary tree
  (comparing index $i$ with $2i$). It violently bounces back and forth
  across RAM. Quicksort puts two pointers at opposite ends of the array
  and streams sequentially inward. The CPU's hardware prefetcher
  recognizes this linear pattern and silently loads the data into the
  L1 cache before the CPU even asks for it.

---

## Extending the Flash Attention example

You mentioned FlashAttention vs. Naive Attention, but looking at
**FlashAttention vs. Sparse Attention** perfectly encapsulates this
entire phenomenon:

- Before FlashAttention, researchers tried to solve the
  $\mathcal{O}(N^2)$ memory bottleneck of transformers by building
  "Sparse Attention" (e.g., Longformer, BigBird). By only looking at
  local windows of tokens, they dropped the FLOP count to
  $\mathcal{O}(N \log N)$.
- However, dynamically gathering these scattered tokens caused massive
  indirect memory reads.
- When Tri Dao invented FlashAttention, he didn't *lower* the FLOP
  count — FlashAttention is still fully dense $\mathcal{O}(N^2)$ math.
  But by tiling the sequence, he trapped the dense math entirely
  inside the GPU's ultra-fast SRAM. The dense $\mathcal{O}(N^2)$
  FlashAttention algorithm finishes executing before the sparse
  $\mathcal{O}(N \log N)$ algorithm even finishes fetching its data
  from Main RAM.
