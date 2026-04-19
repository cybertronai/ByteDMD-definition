Based on the provided article by Bill Dally ("On the Model of Computation") and his Stanford AHA Retreat slides, here is the mathematical re-derivation of the heuristic, the core assumptions it relies on, and the broader context of the scenario.
1. Re-Deriving the Heuristic
The heuristic "An add is worth 10µm of movement" equates the energy required to perform a basic addition to the energy required to physically move the necessary operands across the chip. The provided documents offer two ways to derive this:
Derivation A: The Per-Bit Estimate (From Slide 8)
* Cost of Arithmetic: The energy cost of an addition is approximately $1\text{ fJ/bit}$ (femtojoule per bit).
* Cost of Communication: On-chip wire movement costs approximately $100\text{ fJ/bit-mm}$.
* The Math: To find the distance ($d$) at which the communication energy equals the operation energy:
$$1\text{ fJ/bit} = 100\text{ fJ/bit-mm} \times d$$
$$d = 0.01\text{ mm} = \mathbf{10\text{ µm}}$$
Derivation B: The 32-bit Operands Estimate (From the CACM Article, Page 1)
   * Cost of Arithmetic: A single 32-bit addition operation takes $20\text{ fJ}$.
   * Cost of Communication: To feed this 2-input operation, you must move two 32-bit words (64 bits total). The article states that moving these 64 bits a distance of 1 mm takes $1.9\text{ pJ}$ ($1,900\text{ fJ}$).
   * The Math: The cost to move the operands per micrometer is $1,900\text{ fJ} / 1,000\text{ µm} = 1.9\text{ fJ/µm}$. To find the distance equivalent to the $20\text{ fJ}$ operation cost:
$$d = \frac{20\text{ fJ}}{1.9\text{ fJ/µm}} \approx 10.52\text{ µm} \approx \mathbf{10\text{ µm}}$$
2. Common Assumptions Made
To arrive at this specific heuristic, several architectural and physical assumptions are baked in:
      1. Energy is the Primary Metric: The heuristic equates "worth" with dynamic energy expenditure ($E = \frac{1}{2}CV^2$), rather than time/latency. While latency is mentioned, minimizing energy is the primary bottleneck for scaling modern compute.
      2. Domain-Specific Architectures (No CPU Overhead): The baseline cost (20 fJ) measures the raw physics of activating the logic gates. It assumes execution on bare-metal hardware or a Domain-Specific Architecture (DSA). A general-purpose CPU introduces immense overhead (instruction fetch, decode, pipeline management). As Dally notes, CPU overhead turns a 20 fJ add into an 80 pJ instruction (or a 250 pJ instruction per Slide 11), which would completely mask the physical gap between math and data movement.
      3. On-Chip Wire Capacitance: The 10µm distance applies strictly to movement across standard on-chip interconnects. Going off-chip is catastrophically more expensive. As shown on Slide 28, accessing off-chip LPDDR DRAM costs $\sim640\text{ pJ/word}$, which is tens of thousands of times more expensive than an add.
      4. Moving the Full Payload: The communication cost accounts for routing both operands required to feed the ALU.
3. More Details of the Scenario
Bill Dally uses this heuristic to highlight a fundamental crisis in computer science: our traditional models of computation evaluate algorithms incorrectly.
      * The Flaw in RAM/PRAM Models: For decades, software engineers have used models like RAM (Random-Access Memory) and PRAM (Parallel RAM) to measure asymptotic complexity (Big-O notation). These models assume that an arithmetic operation and a global memory fetch both have a uniform "unit cost" of $O(1)$. Historically, this was a fair approximation. However, as semiconductor technology advanced, the cost of logic shrank by orders of magnitude while the capacitance of wires did not. Today, fetching two 32-bit words from main memory ($1.3\text{ nJ}$) takes 64,000x more energy than the addition itself.
      * Constant Factors Matter (Store vs. Recompute): Because data movement completely dominates power consumption, constant factors now routinely override asymptotic complexity. Dally highlights neural network back-propagation:
      * Standard Big-O analysis prefers storing activation layers to off-chip memory ($O(n)$ complexity) to avoid recalculating them ($O(n^2)$ complexity).
      * However, storing values off-chip costs $640\text{ pJ}$, while recomputing them on-chip costs only $160\text{ fJ/MAC}$. For typical batch sizes (e.g., $n=256$), the $O(n^2)$ recomputation is actually 16x cheaper in energy.
      * The Proposed Solution (PECM): Dally proposes shifting to the Parallel Explicit Communication Model (PECM). Under PECM, arithmetic operations remain a unit cost, but memory operations are assigned a higher cost proportional to a physical distance function (like Manhattan distance on a 2D chip).
      * A Prescription for Engineers: To accommodate this reality, hardware and software should be co-designed to minimize movement. As Dally outlines in his slides, this means computing locally (using distributed SRAMs), utilizing smaller data types (quantizing weights to INT8, INT4, or Log numbers so they cost less to transport), and designing networks sparsely.