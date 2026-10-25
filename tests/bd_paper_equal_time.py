"""Paper-style fixed-target ULA: BD per-iteration gain vs wall-clock cost.

Runs the four-mode, 2D Gaussian-mixture diagnostic from
``bd_paper_toy_diagnostic.py`` and compares BD against ULA at *equal sampler
steps* and at *equal wall-clock* (by running ULA for many more steps).  This
separates "BD converges in fewer iterations" from "BD is a faster sampler".

Example (full run used in FINDINGS.md):
    python3 tests/bd_paper_equal_time.py --counts 16,64,128,1000 \
        --seeds 7,17,29,41 --bd-steps 5000 --ula-steps 20000,50000,100000 \
        --output samples/bd-paper-equal-time/metrics.json
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

import bd_paper_toy_diagnostic as diagnostic  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", default="16,64,128,1000")
    parser.add_argument("--seeds", default="7,17,29")
    parser.add_argument("--bd-steps", type=int, default=5000)
    parser.add_argument("--bd-bandwidths", default="0.3,0.7,1.2")
    parser.add_argument("--ula-steps", default="20000,50000,100000")
    parser.add_argument("--step-size", type=float, default=1e-3)
    parser.add_argument("--output", default="samples/bd-paper-equal-time/metrics.json")
    args = parser.parse_args()

    counts = [int(value) for value in args.counts.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    bandwidths = [float(value) for value in args.bd_bandwidths.split(",")]
    ula_steps = [int(value) for value in args.ula_steps.split(",")]

    rows = []
    for count in counts:
        for seed in seeds:
            methods = [("ula", None, steps) for steps in ula_steps]
            methods += [(f"bd{h:g}", h, args.bd_steps) for h in bandwidths]
            for name, bandwidth, steps in methods:
                target = diagnostic.PaperGaussianMixture()
                start = time.perf_counter()
                records = diagnostic._run(
                    target, count, steps, args.step_size, bandwidth,
                    every_k_steps=1, seed=seed, checkpoints={steps})
                elapsed = time.perf_counter() - start
                record = records[-1]
                rows.append({
                    "count": count, "seed": seed, "method": name,
                    "bandwidth": bandwidth, "steps": steps,
                    "elapsed": elapsed,
                    "mean_nll": record["mean_nll"],
                    "component_tv": record["component_responsibility_tv"],
                    "mse": record["mean_squared_error"],
                })
                print(f"N={count:>4} seed={seed:>2} {name:>6} steps={steps:>6} "
                      f"t={elapsed:7.2f}s TV={record['component_responsibility_tv']:.3f} "
                      f"NLL={record['mean_nll']:.3f} "
                      f"MSE={record['mean_squared_error']:.3f}", flush=True)

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Saved {output.resolve()}")

    groups = {}
    for row in rows:
        groups.setdefault((row["count"], row["method"], row["steps"]), []).append(row)
    print(f"\n{'N':>5} {'method':>6} {'steps':>7} {'TV':>6} {'NLL':>6} "
          f"{'MSE':>6} {'time(s)':>8}")
    for key in sorted(groups):
        values = groups[key]
        print(f"{key[0]:>5} {key[1]:>6} {key[2]:>7} "
              f"{statistics.fmean(v['component_tv'] for v in values):>6.3f} "
              f"{statistics.fmean(v['mean_nll'] for v in values):>6.3f} "
              f"{statistics.fmean(v['mse'] for v in values):>6.2f} "
              f"{statistics.fmean(v['elapsed'] for v in values):>8.2f}")


if __name__ == "__main__":
    main()
