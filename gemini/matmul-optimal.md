# VLSI Matrix-Vector Multiplication Analysis

Created with Gemini 3 Deep Think April 9, 2026 at 09:20 PM
Published April 10, 2026 at 03:54 PM

Source: https://gemini.google.com/share/30594364a3ce

---

## User

I'm interseted in using THompson 2D vlsi model or more precisely the spatial computer attached to analyze the complexity N*N matrix vector multiplication. Tell me what is the energy lower bound, is it possible to achieve it, what is a low energy way to implement matrix vector multiplication, give me detailed specification of a spatial computer that implements such thing

## Gemini

Based on the Spatial Computer model introduced by Gianinazzi et al.—which abstracts modern 2D spatial architectures by strictly pricing communication energy based on the physical Manhattan distance messages travel—here is the formal complexity analysis and detailed architectural specification for an N×N matrix-vector multiplication (y=Ax).

1. What is the Energy Lower Bound?

For an N×N matrix-vector multiplication, the absolute energy lower bound on the spatial computer is Ω(N
2
).

Proof of the bound:
In the Spatial Computer model, processors are restricted to a constant O(1) memory. To simply store the N
2
 elements of the dense matrix A, you are mathematically forced to distribute them across a 2D physical subgrid of at least N×N processors. This creates a grid with a physical diameter of Θ(N).

To compute the output vector y, every element x
j
	​

 of the N-dimensional input vector must be multiplied by every element in the corresponding j-th column of matrix A. Because the N matrix elements of that column are spatially spread across the grid, routing a single scalar x
j
	​

 to all of them requires messages to traverse a minimum physical distance (Steiner tree weight) of Ω(N). Doing this for all N elements of vector x takes N×Ω(N)=Ω(N
2
) energy. A symmetric Ω(N
2
) energy cost applies to physically gathering and summing the partial products across the rows to form the output vector y.

2. Is it Possible to Achieve This Bound?

Yes. It is perfectly possible to achieve the exact Θ(N
2
) energy bound.

The paper points out a heavy Ω(n
1.5
) energy "permutation bottleneck" for sorting and square matrix-matrix multiplication (which corresponds to Ω(N
3
) for an N×N matrix). Matrix-vector multiplication entirely avoids this bottleneck. Because the matrix data requires O(N
2
) area but the vectors only represent O(N) data, we can keep the massive matrix completely stationary and only route the vectors, keeping communication strictly localized to hit the Θ(N
2
) bound.

(Note: There is an energy-depth tradeoff here. If you use the paper's 1D Broadcast tree to speed up the operation to O(logN) depth, the long wires would increase the energy to a sub-optimal O(N
2
logN). To get optimal energy, we must accept an O(N) depth).

3. A Low-Energy Way to Implement It

To hit the absolute minimum Θ(N
2
) energy, we completely avoid long-distance tree routing. Instead, we implement a 2D Systolic Wave-front Pipeline (conceptually similar to Cannon's algorithm).

The Strategy:
The matrix A remains entirely stationary. The vector x is streamed vertically downward column-by-column. As elements of x pass over elements of A, they are locally multiplied. The accumulating sums for the output vector y are then streamed horizontally to the left row-by-row. Because messages only ever travel to immediate physical neighbors, the energy cost per message is precisely 1 unit, strictly minimizing the total energy and wire-depth.

4. Detailed Specification of the Spatial Computer

Below is the strict node-level and lifecycle specification mapping this optimal algorithm to the Spatial Computer rules defined in the paper.

A. Hardware Architecture per Node

Grid Allocation: An unbounded 2D Cartesian grid is used, bounding active execution to an N×N subgrid where processor p
i,j
	​

 sits at coordinates (i,j) for 0≤i,j<N.

Local Memory (O(1) constraint): Each processor utilizes exactly four constant-sized registers:

A_val: Permanently stores the matrix element A
i,j
	​

.

x_val: Temporarily stores the streaming vector element x
j
	​

.

prod: Stores the locally computed scalar product.

sum_val: Stores the accumulating horizontal sum.

Queues: A constant-sized receive queue. To prevent undefined behavior, the execution protocol below guarantees a processor will never receive more than 2 messages in a single time-step.

ALU: Capable of O(1) arithmetic operations per cycle (a fused multiply-add).

B. Initial State

Processor p
i,j
	​

 holds its respective A
i,j
	​

 in A_val.

The top row of processors (p
0,j
	​

) holds the input vector x
j
	​

 in their x_val registers.

Output will naturally materialize at the left-most column (p
i,0
	​

).

C. Execution Protocol (Synchronous Time-Steps)

All processors share a global clock t (starting at t=0). To respect data dependencies, execution flows as a diagonal wavefront. Every clock cycle, all active processors simultaneously run the following logic based on their local physical (i,j) coordinates:

Python
# Phase 1: Vector Shift & Local Multiply
if t == i:
    if i > 0:
        x_val = dequeue_from(North)     # Receive x_j from above
        
    prod = A_val * x_val                # O(1) Local ALU computation
    
    if i < N - 1:
        send_to(South, x_val)           # Route x_j downwards (Cost: 1 Energy)

# Phase 2: Row Reduction (Accumulating y_i)
if t == i + (N - 1 - j):
    if j < N - 1:
        sum_val = dequeue_from(East)    # Receive partial sum from the right
    else:
        sum_val = 0                     # Right-most column starts at 0
        
    sum_val = sum_val + prod            # O(1) Local ALU computation
    
    if j > 0:
        send_to(West, sum_val)          # Route partial sum leftwards (Cost: 1 Energy)
    else:
        # Final output y_i is complete and stored in sum_val at j=0
        terminate()

D. Final Complexity Output

Total Energy (E):

The vector x takes N−1 distance-1 hops down each of the N columns = N(N−1) energy.

The sums take N−1 distance-1 hops left across each of the N rows = N(N−1) energy.

Total Energy = 2N
2
−2N=Θ(N
2
). This perfectly matches the lower bound.

Depth (D): The longest critical dependency chain is the time it takes x
N−1
	​

 to reach the bottom right (N−1 steps), plus the time for y
N−1
	​

's sum to reach the bottom left (N−1 steps). Depth = 2N−2=Θ(N).

Wire-Depth (D
w
	​

): Because no individual message traverses a physical distance greater than 1, the maximum energy expended over any dependent chain is completely proportional to the depth. Wire-depth = Θ(N).
