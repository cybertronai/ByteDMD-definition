#!/usr/bin/env python3
"""
Run the 4x4 strategy catalog and emit ranked output.
"""
import json
import os

from algorithms_4x4 import STRATEGIES


HERE = os.path.dirname(__file__)


def main():
    rows = [fn() for fn in STRATEGIES]
    rows.sort(key=lambda r: r['cost'])

    with open(os.path.join(HERE, 'results_4x4.json'), 'w') as f:
        json.dump(rows, f, indent=2)

    lines = [
        '| rank | strategy | cost | peak_scratch | peak_arg | n_reads |',
        '|-----:|----------|-----:|-------------:|---------:|--------:|',
    ]
    for rank, r in enumerate(rows, start=1):
        lines.append(
            f"| {rank} | `{r['name']}` | {r['cost']} | "
            f"{r['peak_scratch']} | {r['peak_arg']} | {r['reads']} |"
        )
    table = '\n'.join(lines)
    with open(os.path.join(HERE, 'ranked_4x4.md'), 'w') as f:
        f.write(table + '\n')

    print(f"Ran {len(rows)} strategies:\n")
    best = rows[0]['cost']
    for r in rows:
        delta = r['cost'] - best
        marker = '*' if r['cost'] == best else ' '
        print(f"  {marker} {r['name']:28s}  cost={r['cost']:>5}  "
              f"(+{delta:>4})  scratch_peak={r['peak_scratch']:>3}  "
              f"reads={r['reads']}")
    print(f"\nBest = {best}. Wrote results_4x4.json, ranked_4x4.md.")


if __name__ == '__main__':
    main()
