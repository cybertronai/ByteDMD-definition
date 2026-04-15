# Experiments

Stand-alone experiments that go beyond the canonical benchmarks in `benchmarks/`. Each experiment lives in its own subdirectory with its own runner, raw results, and a written report.

## Index

- [`memory_management/`](memory_management/report.md) — Three memory management strategies (unmanaged, tombstone GC, aggressive compaction) tested on five matmul algorithms (naive, recursive with temporaries, in-place RMM lex, in-place RMM Gray-code, Strassen) at sizes N=2..64. Both discrete (`ceil(sqrt(d))`) and continuous (integral) ByteDMD costs are computed. Headline finding: at N=64 under aggressive memory management, **Gray-code swizzled in-place RMM beats Strassen by 2.4×**. The polynomial shapes match Gemini's analytical formulas with extraordinary fit quality (max error <1.2% for in-place RMM), but the leading constants are 0.4×–0.85× of the analytical upper bounds because the LRU dynamics give shallower depths than the static-watermark model.
- [`matmul_hierarchy/`](matmul_hierarchy/report.md) — Final report for an explicit three-level tracing pipeline for tiled, recursive, and Strassen matmul: Python source, abstract logical load/store trace, and compiled concrete-address trace. Includes `N=16` cache-size sweep plots, a glossary for `ByteDMD-classic`, `ByteDMD-live`, `never-reuse`, `belady`, `lifo`, and `edf`, plus a note that `belady` is an offline oracle lower bound rather than a stable-address compiler.
