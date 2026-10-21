
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
kernel_groups = defaultdict(list)

with INPUT.open(newline="") as f:
    for row in csv.DictReader(f):
        key = (
            int(row["size"]),
            row["implementation"],
            int(row["processes"]),
        )

        groups[key].append(float(row["elapsed_seconds"]))

        if row.get("gpu_kernel_ms", "").strip():
            kernel_groups[key].append(float(row["gpu_kernel_ms"]))

summary = []
sequential_means = {}

for (size, implementation, processes), times in sorted(groups.items()):
    mean = statistics.mean(times)
    stddev = statistics.stdev(times) if len(times) > 1 else 0.0

    if implementation == "sequential":
        sequential_means[size] = mean

    kernel_times = kernel_groups.get(
        (size, implementation, processes), []
    )

    summary.append({
        "size": size,
        "implementation": implementation,
        "processes": processes,
        "runs": len(times),
        "mean_seconds": mean,
        "stddev_seconds": stddev,
        "mean_gpu_kernel_ms": (
            statistics.mean(kernel_times) if kernel_times else ""
        ),
        "stddev_gpu_kernel_ms": (
            statistics.stdev(kernel_times)
            if len(kernel_times) > 1 else
            (0.0 if kernel_times else "")
        ),
    })

for row in summary:
    baseline = sequential_means[row["size"]]
    row["speedup_vs_sequential"] = (
        baseline / row["mean_seconds"]
        if row["mean_seconds"] > 0 else float("nan")
    )

with SUMMARY.open("w", newline="") as f:
    fields = [
        "size",
        "implementation",
        "processes",
        "runs",
        "mean_seconds",
        "stddev_seconds",
        "speedup_vs_sequential",
        "mean_gpu_kernel_ms",
        "stddev_gpu_kernel_ms",
    ]

    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for row in summary:
        output = row.copy()

        for key in (
            "mean_seconds",
            "stddev_seconds",
            "speedup_vs_sequential",
        ):
            output[key] = f"{output[key]:.9f}"

        for key in ("mean_gpu_kernel_ms", "stddev_gpu_kernel_ms"):
            if output[key] != "":
                output[key] = f"{output[key]:.6f}"

        writer.writerow(output)

sizes = sorted(sequential_means)

configurations = [
    ("sequential", 1, "Sequential C"),
    ("mpi", 1, "MPI (1 process)"),
    ("mpi", 2, "MPI (2 processes)"),
    ("mpi", 4, "MPI (4 processes)"),
    ("cuda", 1, "CUDA GPU"),
]


def get_metric(size, implementation, processes, metric):
    return next(
        row[metric]
        for row in summary
        if row["size"] == size
        and row["implementation"] == implementation
        and row["processes"] == processes
    )


# Plot 1: end-to-end runtime.
fig, ax = plt.subplots(figsize=(10, 6))

for implementation, processes, label in configurations:
    values = [
        get_metric(
            size, implementation, processes, "mean_seconds"
        )
        for size in sizes
    ]
    ax.plot(sizes, values, marker="o", linewidth=2, label=label)

ax.set_yscale("log")
ax.set_xlabel("Length of each input sequence (characters)")
ax.set_ylabel("Mean end-to-end runtime (seconds, log scale)")
ax.set_title("LCS End-to-End Runtime: Sequential, MPI and CUDA")
ax.grid(True, which="both", linestyle="--", alpha=0.4)
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / "runtime_comparison.png", dpi=200)
plt.close(fig)

# Plot 2: end-to-end speedup relative to Sequential C.
fig, ax = plt.subplots(figsize=(10, 6))

for implementation, processes, label in configurations:
    values = [
        get_metric(
            size, implementation, processes, "speedup_vs_sequential"
        )
        for size in sizes
    ]
    ax.plot(sizes, values, marker="o", linewidth=2, label=label)

ax.axhline(
    1.0, linestyle="--", linewidth=1, label="No speedup (1x)"
)
ax.set_xlabel("Length of each input sequence (characters)")
ax.set_ylabel("Speedup (sequential runtime / runtime)")
ax.set_title("LCS End-to-End Speedup Relative to Sequential C")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / "speedup_comparison.png", dpi=200)
plt.close(fig)

# Plot 3: CUDA kernel time only.
cuda_rows = [
    row for row in summary
    if row["implementation"] == "cuda"
    and row["mean_gpu_kernel_ms"] != ""
]

if cuda_rows:
    cuda_rows.sort(key=lambda row: row["size"])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        [row["size"] for row in cuda_rows],
        [row["mean_gpu_kernel_ms"] for row in cuda_rows],
        marker="o",
        linewidth=2,
        label="CUDA diagonal kernels",
    )
    ax.set_xlabel("Length of each input sequence (characters)")
    ax.set_ylabel("Mean GPU diagonal computation time (ms)")
    ax.set_title("CUDA Kernel Time")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "cuda_kernel_comparison.png", dpi=200)
    plt.close(fig)

print(f"Summary saved to {SUMMARY}")
print("Graphs saved to:")
print(f"  {RESULTS / 'runtime_comparison.png'}")
print(f"  {RESULTS / 'speedup_comparison.png'}")
if cuda_rows:
    print(f"  {RESULTS / 'cuda_kernel_comparison.png'}")

print()
print(
    f"{'Size':>6} {'Implementation':>16} {'Procs':>5} "
    f"{'Mean (s)':>12} {'Speedup':>10} {'GPU kernel (ms)':>16}"
)
print("-" * 82)

for row in summary:
    kernel = row["mean_gpu_kernel_ms"]
    kernel_text = f"{kernel:.4f}" if kernel != "" else "-"

    print(
        f"{row['size']:>6} {row['implementation']:>16} "
        f"{row['processes']:>5} {row['mean_seconds']:>12.6f} "
        f"{row['speedup_vs_sequential']:>9.4f}x "
        f"{kernel_text:>16}"
    )
