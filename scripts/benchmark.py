
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


def run_program(command, expect_gpu_time=False):
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
            f"Command failed: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    length_match = re.search(
        r"LCS length:\s*(\d+)", result.stdout
    )
    if not length_match:
        raise RuntimeError(
            f"Could not parse LCS length:\n{result.stdout}"
        )

    kernel_match = re.search(
        r"GPU diagonal computation time:\s*"
        r"([0-9.eE+-]+)\s*milliseconds",
        result.stdout,
    )

    if expect_gpu_time and not kernel_match:
        raise RuntimeError(
            "Could not parse CUDA kernel timing:\n"
            f"{result.stdout}"
        )

    kernel_ms = (
        float(kernel_match.group(1)) if kernel_match else ""
    )

    return elapsed, int(length_match.group(1)), kernel_ms


def add_row(rows, size, implementation, processes, repeat,
            elapsed, length, kernel_ms=""):
    rows.append({
        "size": size,
        "implementation": implementation,
        "processes": processes,
        "repeat": repeat,
        "elapsed_seconds": f"{elapsed:.9f}",
        "gpu_kernel_ms": (
            f"{kernel_ms:.6f}" if kernel_ms != "" else ""
        ),
        "lcs_length": length,
    })


def main():
    subprocess.run(
        ["make", "sequential", "mpi", "cuda"],
        cwd=ROOT,
        check=True,
    )
    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for size in SIZES:
        seq1 = make_sequence(size, size)
        seq2 = make_sequence(size + 1, size)

        print(f"\nBenchmarking sequence length {size}...",
              flush=True)

        # Warm up CUDA before collecting measured runs.
        _, warmup_length, _ = run_program(
            [str(ROOT / "lcs_cuda"), seq1, seq2],
            expect_gpu_time=True,
        )

        # Establish the reference answer.
        _, reference_length, _ = run_program(
            [str(ROOT / "lcs_sequential"), seq1, seq2]
        )

        if warmup_length != reference_length:
            raise RuntimeError(
                f"CUDA warm-up correctness failure at size {size}: "
                f"sequential={reference_length}, CUDA={warmup_length}"
            )

        for repeat in range(1, REPEATS + 1):
            print(f"  Repetition {repeat}/{REPEATS}", flush=True)

            seq_time, seq_length, _ = run_program(
                [str(ROOT / "lcs_sequential"), seq1, seq2]
            )

            add_row(
                rows, size, "sequential", 1, repeat,
                seq_time, seq_length,
            )

            if seq_length != reference_length:
                raise RuntimeError(
                    f"Sequential result changed at size {size}"
                )

            for processes in PROCESS_COUNTS:
                mpi_time, mpi_length, _ = run_program([
                    "mpirun", "--oversubscribe", "-n",
                    str(processes),
                    str(ROOT / "lcs_mpi"), seq1, seq2,
                ])

                if mpi_length != seq_length:
                    raise RuntimeError(
                        f"MPI correctness failure at size {size}, "
                        f"{processes} processes: "
                        f"sequential={seq_length}, MPI={mpi_length}"
                    )

                add_row(
                    rows, size, "mpi", processes, repeat,
                    mpi_time, mpi_length,
                )

            cuda_time, cuda_length, kernel_ms = run_program(
                [str(ROOT / "lcs_cuda"), seq1, seq2],
                expect_gpu_time=True,
            )

            if cuda_length != seq_length:
                raise RuntimeError(
                    f"CUDA correctness failure at size {size}: "
                    f"sequential={seq_length}, CUDA={cuda_length}"
                )

            add_row(
                rows, size, "cuda", 1, repeat,
                cuda_time, cuda_length, kernel_ms,
            )

    fields = [
        "size",
        "implementation",
        "processes",
        "repeat",
        "elapsed_seconds",
        "gpu_kernel_ms",
        "lcs_length",
    ]

    with RESULTS.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} measurements to {RESULTS}")
    print("All Sequential C, MPI, and CUDA LCS lengths matched.")
    print("CUDA kernel time is recorded separately from end-to-end time.")


if __name__ == "__main__":
    main()
