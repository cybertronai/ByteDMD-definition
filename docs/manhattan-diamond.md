# Manhattan Diamond

(aka, simplified explicit communication model)

Bill Dally ([*On the Model of Computation*, CACM
2022](https://cacm.acm.org/opinion/on-the-model-of-computation-point/)) proposed modeling algorithm data movement explicitly on the Manhattan grid.

This is a simplified implementation of this model for a single processor, designed to price a single function call.

docs/manhattan_function_figure.svg

- Processor is in the center and the memory is arranged on a 2D grid around it.
- Space is linearly indexed , with layout chosen to make it easy to compute Manhattan distance, ceiling(sqrt(idx)) gives the cost

[TODO(claude): insert a figure like https://github.com/cybertronai/ByteDMD/blob/dev/docs/manhattan_figure.svg, but drop the example with memory access at 12, I basically want the same layout and colors as docs/manhattan_function_figure.svg, but I want to illustrate the memory access pattern

- We only price reads. Since every write and instruction call involves a read, we absorb those costs into the read cost.

- Each cell in a 2D grid, except y=0 line is indexed using a single integer
- Arguments are pre-loaded on the read-only part of the memory (negative indices)
- Outputs and temporaries are in the writeable part of the memory (positive indices)
- At the end of the function call, every output value is read (This is to make sure that there are no free reads when chaining function calls)
