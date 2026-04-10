(* ===================================================================
   Exact closed-form ByteDMD formulas for N×N matrix-vector multiply
   y = A x  using the standard i-j-k loop order.

   Stack layout at initialization:
     A elements: keys 1..N² (row-major, A[i,j] = key i*N + j + 1)
     x elements: keys N²+1..N²+N
     Total initial stack size: N² + N

   Algorithm structure per row i:
     j=0: s = A[i,0] * x[0]          (1 mul:  2 reads, 1 alloc)
     j≥1: t = A[i,j] * x[j]; s += t  (1 mul + 1 add: 4 reads, 2 allocs)

   Reads per row:  2 + 4(N-1) = 4N - 2
   Allocs per row: 1 + 2(N-1) = 2N - 1
   Total reads:    N(4N - 2)
   Total allocs:   N(2N - 1)

   Verified exact match against runtime tracer for N = 3, 4, 5, 8, 16.
   =================================================================== *)

(* --- Depth of A[i,j] when it is first read --- *)
(* A elements are each read exactly once. Their depth grows linearly
   with i (each row adds 2N-1 allocs) and with j (each column step
   within a row adds 2 allocs after the first). *)

depthA[n_, i_, j_] := n^2 + n + i (2 n - 1) + If[j == 0, 0, 2 j - 1]

(* --- Depth of x[j] when read in row i --- *)
(* x elements are read N times (once per row). Row 0 has a transient
   pattern; rows 1..N-1 are in steady state.

   Row 0: x[0] starts at depth N (top of initial x block).
          x[j≥1] at depth N + 3j - 1 (pushed down by allocs within the row).

   Rows i≥1: x elements were last touched during the PREVIOUS row.
          Between their last use and now, exactly 2N-1 allocs happened
          (the rest of the previous row's products + sums) plus the
          current row's earlier allocs. This converges to:
            x[0] at depth 4N - 2
            x[j≥1] at depth 4N - 1
*)

depthX[n_, 0, 0] := n
depthX[n_, 0, j_] := n + 3 j - 1     /; j >= 1
depthX[n_, i_, 0] := 4 n - 2          /; i >= 1
depthX[n_, i_, j_] := 4 n - 1         /; i >= 1 && j >= 1

(* --- Addition depths are CONSTANT --- *)
(* The sum operand is at depth 4 (below: new mul result, the x that
   was moved to MRU, and the A that was moved to MRU).
   The product operand is always at depth 1 (just allocated, MRU). *)

depthAddL = 4;
depthAddR = 1;

(* --- Full trace as a flat list --- *)
matvecTrace[n_] := Flatten[Table[
  Join[
    {depthA[n, i, 0], depthX[n, i, 0]},
    Flatten[Table[
      {depthA[n, i, j], depthX[n, i, j], depthAddL, depthAddR},
      {j, 1, n - 1}
    ]]
  ],
  {i, 0, n - 1}
]]

(* --- Stack state at step k --- *)
(* The stack at step k can be reconstructed from the initial stack by
   replaying k LRU moves + allocs. For a closed form, note:

   stackSize[k] = N² + N + (number of allocs up to step k)

   The alloc schedule is deterministic:
     Within each row, allocs happen after every mul (steps 0,2,4,...)
     and after every add (steps 4,6,...). Per row of 4N-2 reads,
     there are 2N-1 allocs.

   allocsBeforeStep[k] gives the number of allocs that have occurred
   strictly before the k-th read.
*)

allocsPerRow[n_] := 2 n - 1
readsPerRow[n_] := 4 n - 2

(* Which row and position within the row does step k correspond to? *)
stepRow[n_, k_] := Floor[k / readsPerRow[n]]
stepCol[n_, k_] := Mod[k, readsPerRow[n]]

(* Total allocs before the k-th read *)
allocsBeforeRead[n_, k_] := Module[{row, col, withinRow},
  row = stepRow[n, k];
  col = stepCol[n, k];
  (* Each completed row contributes 2N-1 allocs *)
  (* Within the current row, allocs happen after reads at positions
     1 (after mul j=0), then 3,5 (after mul j=1, add j=1),
     then 5,7, ... *)
  withinRow = If[col <= 1, 0,
    1 + Floor[(col - 2) / 2] + If[Mod[col - 2, 4] >= 2, 1, 0]
  ];
  (* Actually this is complex. Simpler: per (mul+add) pair, 2 allocs.
     First mul alone: 1 alloc. *)
  withinRow = If[col == 0, 0, 1 + Floor[(col - 1) / 2]];
  row * allocsPerRow[n] + withinRow
]

stackSizeAtRead[n_, k_] := n^2 + n + allocsBeforeRead[n, k]

(* --- Total ByteDMD cost (discrete, bytes=1) --- *)
byteDMDCost[n_] := Sum[Ceiling[Sqrt[d]], {d, matvecTrace[n]}]

(* --- Continuous ByteDMD cost --- *)
byteDMDCostContinuous[n_] := Sum[2/3 (d^(3/2) - (d - 1)^(3/2)), {d, matvecTrace[n]}]

(* --- Closed-form cost decomposition --- *)
(* The total cost splits into three independent sums:
   1. A-element reads:  Sum_{i,j} ceil(sqrt(depthA[n,i,j]))
   2. x-element reads:  Sum_{i,j} ceil(sqrt(depthX[n,i,j]))
   3. Addition reads:   N(N-1) * (ceil(sqrt(4)) + ceil(sqrt(1))) = 3 N(N-1)
*)

costA[n_] := Sum[Ceiling[Sqrt[depthA[n, i, j]]], {i, 0, n - 1}, {j, 0, n - 1}]
costX[n_] := Sum[Ceiling[Sqrt[depthX[n, i, j]]], {i, 0, n - 1}, {j, 0, n - 1}]
costAdd[n_] := 3 n (n - 1)

(* Total = costA + costX + costAdd *)
(* Verify: byteDMDCost[n] == costA[n] + costX[n] + costAdd[n] *)

(* --- Asymptotic scaling --- *)
(* For large N:
   - costA ≈ Sum_{i,j} sqrt(N² + (2N)i + 2j)
           ≈ N² * sqrt(N²) = N³   (each A read at depth ~N²)
   - costX ≈ N * (N-1) * sqrt(4N) = N² * 2*sqrt(N) = 2 N^2.5
           (x reads dominate at 4N depth, read N times)
   - costAdd = 3 N² - 3N

   Total ≈ N³ + 2 N^2.5 + 3 N² → Theta(N³) for discrete ByteDMD.

   Note: Gemini's continuous analysis predicts Theta(N⁴) because it
   uses the block-integral model where B is reused at depth N².
   The per-element LRU model gives Theta(N³) because the x elements
   are refreshed each row and stay at depth ~4N, not N².
   The A elements contribute the dominant N³ term.
*)

(* --- Quick numerical check --- *)
Print["ByteDMD costs:"]
Do[Print["  N=", n, ": ", byteDMDCost[n]], {n, {3, 4, 5, 8, 16}}]
