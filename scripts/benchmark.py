#!/usr/bin/env python3

import csv
import random
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "benchmark.csv"
SIZES = [100, 250, 500, 1000]
PROCESS_COUNTS = [1, 2, 4]
REPEATS = 3


def make_sequence(seed, size):
    rng = random.Random(seed)
    return "".join(rng.choices("ACGT", k=size))


def run_program(command):
    start = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}\n{result.stderr}"
        )

    match = re.search(r"LCS length:\s*(\d+)", result.stdout)
    if not match:
        raise RuntimeError(
            f"Could not parse LCS length from output:\n{result.stdout}"
        )

    return elapsed, int(match.group(1))


def main():
    subprocess.run(["make", "sequential", "mpi"], cwd=ROOT, check=True)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for size in SIZES:
        seq1 = make_sequence(size, size)
        seq2 = make_sequence(size + 1, size)

        print(f"Benchmarking size {size}...", flush=True)

        for repeat in range(1, REPEATS + 1):
            seq_time, seq_length = run_program(
                [str(ROOT / "lcs_sequential"), seq1, seq2]
            )
            rows.append({
                "size": size,
                "implementation": "sequential",
                "processes": 1,
                "repeat": repeat,
                "elapsed_seconds": f"{seq_time:.9f}",
                "lcs_length": seq_length,
            })

            for processes in PROCESS_COUNTS:
                mpi_time, mpi_length = run_program([
                    "mpirun", "--oversubscribe", "-n", str(processes),
                    str(ROOT / "lcs_mpi"), seq1, seq2
                ])

                if mpi_length != seq_length:
                    raise RuntimeError(
                        f"Incorrect result for size {size}, "
                        f"MPI processes {processes}: "
                        f"sequential={seq_length}, MPI={mpi_length}"
                    )

                rows.append({
                    "size": size,
                    "implementation": "mpi",
                    "processes": processes,
                    "repeat": repeat,
                    "elapsed_seconds": f"{mpi_time:.9f}",
                    "lcs_length": mpi_length,
                })

    with RESULTS.open("w", newline="") as f:
        fields = [
            "size", "implementation", "processes", "repeat",
            "elapsed_seconds", "lcs_length"
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} measurements to {RESULTS}")
    print("All sequential/MPI LCS lengths matched.")


if __name__ == "__main__":
    main()
