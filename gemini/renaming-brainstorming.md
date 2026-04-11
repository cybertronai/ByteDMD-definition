# Renaming Data Movement Distance Metric

Created with Gemini 3 Deep Think April 8, 2026 at 03:57 PM
Published April 8, 2026 at 04:03 PM

Source: https://gemini.google.com/share/af75298aa427

---

## User

I want to reuse the idea of data movement distance but maybe relax byte and use something else instead of byteDMD. Brainstorm other potential names I can use in place of ByteDMD. I want to reuse the existing data movement distance, but perhaps emphasize that I'm doing a very concrete count. I'm essentially tightening up their metric, which is meant to be complexity-based, but I want something that has concrete numbers connected to it.

## Gemini

Here are several brainstorming directions for renaming ByteDMD. Since you want to contrast your exact, execution-based tally against Ding and Smith’s theoretical, asymptotic complexity metric, the new name should bridge the "Data Movement" lineage with the idea of a hard numerical score.

Relaxing the word "Byte" also makes sense, as your metric is highly flexible and operates on logical objects, elements, or multi-byte scalars.

1. Emphasizing the "Exact Tally" (Contrasting with Big-O)

These names highlight that you are providing a concrete integer sum rather than an abstract asymptotic bound.

ExactDMD: Directly contrasts with the asymptotic bounds of the original paper. It tells the academic/engineering reader immediately: "This is DMD, but calculated exactly."

TraceDMD: Highlights how you get the concrete number—by dynamically tracing the execution path. It sets your empirical tool apart from static, on-paper analysis.

CountDMD: Emphasizes the "counting" of discrete operations.

TallyDMD: "Tally" implies a discrete, running sum, which fits perfectly with how your Python tracer accumulates costs instruction by instruction.

DiscreteDMD: Highlights the jump from continuous mathematical bounds (like O(N
3.5
)) to discrete integer steps and ceiling functions (⌈
d
	​

⌉).

2. Generalizing the Data Unit (Replacing "Byte")

If you want to keep the <Word>DMD naming convention but make it less inherently tied to 8-bit increments, you can use a broader term for the data being tracked.

ElementDMD (or ElemDMD): Highly accurate to your computation model. Your README explicitly states: "An idealized processor operates directly on an element-level LRU stack." It scales naturally whether the unit is an 8-bit char or a 64-bit float.

WordDMD: A classic architecture term that implies a concrete machine fetch without locking you into exactly 8 bits.

ItemDMD: Friendly, Pythonic, and abstract enough to apply to any data type passing through your _Tracked proxy wrapper.

UnitDMD: Very flexible. A "unit" could be a byte, a token, or a struct, determined by the bytes_per_element multiplier in your code.

3. Evolving the Acronym (Replacing "Distance")

"Distance" naturally sounds like a continuous, spatial measurement or a theoretical property. If you want to emphasize a final concrete tally, you might replace "Distance" with "Count", "Cost", or "Score", while keeping "Data Movement."

DMC (Data Movement Cost): A very natural pivot. While DMD represents the spatial distance (
d
	​

), the DMC is the actual sum total (∑⌈
d
	​

⌉) you pay at the end of the execution.

DMC (Data Movement Count): Similar to the above, but directly incorporates your desire to emphasize "counting."

DMS (Data Movement Score): Gives it a benchmarking/leaderboard feel (e.g., "Strassen gets a worse DMS than Naive Matmul").

4. Leaning into the Physical Metaphor (VLSI & 2D Routing)

Since your ceiling and square-root logic is explicitly motivated by "routing wire length on a 2D grid", you could draw naming inspiration from the physical reality it simulates to make the metric feel grounded.

WireDMD (or WireCost): Highly evocative. "Wire" inherently implies a physical, measurable connection with a finite length, rather than an abstract mathematical distance.

GridDMD: Evokes the 2D Manhattan distance grid visual you have in your documentation.

RouteCost: Highlights that you are paying a concrete toll for the routing length required to fetch data to the processor.

Summary Recommendations

To emphasize the shift to a concrete number: Go with ExactDMD or TraceDMD. This instantly tells readers of your reports that you are doing concrete execution tracking, not asymptotic math.

To keep the original rhythm but fix the "byte" issue: Go with ElementDMD. It reads exactly the same as ByteDMD, but perfectly matches the "element-level" phrasing already in your README.

To make it sound like a true benchmarking metric: Data Movement Cost (DMC) is incredibly clean. You could introduce it as: "While DMD gives us the theoretical spatial penalty, we calculate the concrete Data Movement Cost (DMC) of an algorithm."
