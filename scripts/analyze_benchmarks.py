#!/usr/bin/env python3

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
INPUT = RESULTS / "benchmark.csv"
SUMMARY = RESULTS / "benchmark_summary.csv"

groups = defaultdict(list)

with INPUT.open(newline="") as f:
    for row in csv.DictReader(f):
        key = (
            int(row["size"]),
            row["implementation"],
            int(row["processes"]),
        )
        groups[key].append(float(row["elapsed_seconds"]))

summary = []
sequential_means = {}

for (size, implementation, processes), times in sorted(groups.items()):
    mean = statistics.mean(times)
    stddev = statistics.stdev(times) if len(times) > 1 else 0.0

    if implementation == "sequential":
        sequential_means[size] = mean

    summary.append({
        "size": size,
        "implementation": implementation,
        "processes": processes,
        "runs": len(times),
        "mean_seconds": mean,
        "stddev_seconds": stddev,
    })

for row in summary:
    baseline = sequential_means[row["size"]]
    row["speedup_vs_sequential"] = (
        baseline / row["mean_seconds"]
    )

with SUMMARY.open("w", newline="") as f:
    fields = [
        "size", "implementation", "processes", "runs",
        "mean_seconds", "stddev_seconds", "speedup_vs_sequential"
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for row in summary:
        output = row.copy()
        for key in ("mean_seconds", "stddev_seconds",
                    "speedup_vs_sequential"):
            output[key] = f"{output[key]:.9f}"
        writer.writerow(output)

sizes = sorted(sequential_means)
configurations = [
    ("sequential", 1, "Sequential C"),
    ("mpi", 1, "MPI (1 process)"),
    ("mpi", 2, "MPI (2 processes)"),
    ("mpi", 4, "MPI (4 processes)"),
]

# Plot 1: end-to-end runtime. Log scale makes small sequential times visible.
fig, ax = plt.subplots(figsize=(9, 5.5))

for implementation, processes, label in configurations:
    values = [
        next(r["mean_seconds"] for r in summary
             if r["size"] == size
             and r["implementation"] == implementation
             and r["processes"] == processes)
        for size in sizes
    ]
    ax.plot(sizes, values, marker="o", linewidth=2, label=label)

ax.set_yscale("log")
ax.set_xlabel("Length of each input sequence (characters)")
ax.set_ylabel("Mean end-to-end runtime (seconds, log scale)")
ax.set_title("LCS Runtime: Sequential vs MPI")
ax.grid(True, which="both", linestyle="--", alpha=0.4)
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / "runtime_comparison.png", dpi=200)
plt.close(fig)

# Plot 2: speedup relative to sequential execution.
fig, ax = plt.subplots(figsize=(9, 5.5))

for implementation, processes, label in configurations:
    values = [
        next(r["speedup_vs_sequential"] for r in summary
             if r["size"] == size
             and r["implementation"] == implementation
             and r["processes"] == processes)
        for size in sizes
    ]
    ax.plot(sizes, values, marker="o", linewidth=2, label=label)

ax.axhline(1.0, linestyle="--", linewidth=1, label="No speedup (1×)")
ax.set_xlabel("Length of each input sequence (characters)")
ax.set_ylabel("Speedup (sequential runtime / runtime)")
ax.set_title("LCS Speedup Relative to Sequential C")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / "speedup_comparison.png", dpi=200)
plt.close(fig)

print(f"Summary saved to {SUMMARY}")
print("Graphs saved to:")
print(f"  {RESULTS / 'runtime_comparison.png'}")
print(f"  {RESULTS / 'speedup_comparison.png'}")
print()
print(f"{'Size':>6} {'Implementation':>16} {'Procs':>5} "
      f"{'Mean (s)':>12} {'Speedup':>10}")
print("-" * 56)

for row in summary:
    print(
        f"{row['size']:>6} {row['implementation']:>16} "
        f"{row['processes']:>5} {row['mean_seconds']:>12.6f} "
        f"{row['speedup_vs_sequential']:>10.4f}x"
    )
