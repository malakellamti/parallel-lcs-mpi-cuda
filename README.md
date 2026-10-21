# Parallel LCS: Sequential C, MPI, and CUDA

Implementations and performance evaluation of the **Longest Common Subsequence (LCS)** problem using sequential C, MPI-based distributed computation, and CUDA-based GPU computation.

This project compares three approaches to the same dynamic programming problem, with automated correctness tests, reproducible benchmarks, and performance visualizations.

## Overview

The Longest Common Subsequence problem finds the longest sequence of characters that appears in two input strings in the same relative order, without requiring the characters to be contiguous.

For example, given `ABCBDAB` and `BDCABA`, one valid longest common subsequence is `BCBA`, with length 4.

This repository contains three implementations:

- **Sequential C:** Dynamic programming baseline.
- **MPI:** Anti-diagonal wavefront computation distributed across MPI processes.
- **CUDA:** Anti-diagonal wavefront computation using GPU threads.

The project explores the benefits and limitations of parallel computation by comparing execution times and validating results across implementations.

## Project Objectives

- Implement LCS using sequential dynamic programming.
- Distribute computations across MPI processes.
- Implement GPU-based computation using CUDA.
- Validate implementations through automated tests.
- Compare end-to-end execution times across implementations.
- Measure CUDA kernel execution time separately.
- Generate benchmark summaries and performance plots.
- Document experimental results and implementation limitations.

## Repository Structure

```text
parallel-lcs-mpi-cuda/
├── src/
│   ├── lcs_sequential.c
│   ├── lcs_mpi.c
│   └── lcs_cuda.cu
├── tests/
│   ├── test_sequential.sh
│   ├── test_mpi.sh
│   └── test_cuda.sh
├── scripts/
│   ├── benchmark.py
│   └── analyze_benchmarks.py
├── results/
│   ├── benchmark.csv
│   ├── benchmark_summary.csv
│   ├── runtime_comparison.png
│   ├── speedup_comparison.png
│   └── cuda_kernel_comparison.png
├── Makefile
├── README.md
├── LICENSE
└── .gitignore
```

## Technologies

- C
- MPI (Message Passing Interface)
- CUDA C++
- GNU Make
- Bash
- Python 3
- Matplotlib
- Git and GitHub

## Requirements

### Sequential and MPI implementations

- GCC or a compatible C compiler
- Open MPI or another compatible MPI implementation
- GNU Make
- Bash

### CUDA implementation

- NVIDIA CUDA Toolkit, including `nvcc`
- A compatible NVIDIA GPU and driver
- A C++ host compiler supported by the installed CUDA Toolkit

### Benchmark analysis

Python 3 and Matplotlib are required to generate the benchmark analysis and plots.

Install Matplotlib if necessary:

```bash
python3 -m pip install matplotlib
```

The development environment used for this project includes WSL2 with Ubuntu, CUDA Toolkit 12.9, Open MPI 5.0.10, and `g++-14` as the CUDA host compiler.

## Build

Clone the repository and enter the project directory:

```bash
git clone https://github.com/malakellamti/parallel-lcs-mpi-cuda.git
cd parallel-lcs-mpi-cuda
```

Build the sequential and MPI implementations:

```bash
make all
```

Build the CUDA implementation:

```bash
make cuda
```

To rebuild all executables from a clean state:

```bash
make clean
make all
make cuda
```

The generated executables are `lcs_sequential`, `lcs_mpi`, and `lcs_cuda`.

## Usage

Each executable accepts two input strings. Each input sequence is limited to 10,000 characters. Actual performance and memory requirements depend on the input sizes and the implementation used.

### Sequential C

```bash
./lcs_sequential ABCBDAB BDCABA
```

### MPI

Run the program using four MPI processes:

```bash
mpirun -n 4 ./lcs_mpi ABCBDAB BDCABA
```

### CUDA

```bash
./lcs_cuda ABCBDAB BDCABA
```

For the example above, each implementation should report an LCS length of 4. The actual subsequence may differ when multiple longest subsequences are valid.

Run an executable without arguments to display its usage instructions.

## Automated Testing

Run all test suites:

```bash
make test
```

Run individual suites:

```bash
make test-sequential
make test-mpi
make test-cuda
```

The current test suites contain 22 test cases:

| Implementation | Test cases |
|---|---:|
| Sequential C | 7 |
| MPI | 8 |
| CUDA | 7 |
| **Total** | **22** |

The tests cover representative cases such as identical strings, no common characters, empty inputs, single-character matches, subsequence relationships, and inputs of different lengths.

All 22 tests passed in the latest recorded clean build and test run. The benchmark script also verifies that the LCS lengths returned by the three implementations agree on generated inputs.

These tests provide practical correctness checks but do not constitute an exhaustive proof of correctness.

## Algorithm Design

### Sequential dynamic programming

The sequential implementation constructs a two-dimensional dynamic programming table.

For input strings `X` and `Y`, `DP[i][j]` represents the LCS length of the first `i` characters of `X` and the first `j` characters of `Y`.

The recurrence is:

1. If `i = 0` or `j = 0`:

   `DP[i][j] = 0`

2. If `X[i-1] == Y[j-1]`:

   `DP[i][j] = DP[i-1][j-1] + 1`

3. Otherwise:

   `DP[i][j] = max(DP[i-1][j], DP[i][j-1])`

The final table entry contains the LCS length. The implementation then traverses the table backward to reconstruct one valid subsequence.

- **Time complexity:** `O(mn)`
- **Space complexity:** `O(mn)`

Here, `m` and `n` are the lengths of the input strings.

### MPI distributed computation

The MPI implementation uses anti-diagonal wavefront computation. Cells on an anti-diagonal depend on cells from earlier diagonals, allowing the current diagonal's work to be distributed across processes.

The implementation uses `MPI_All
`MPI_Allreduce` to share diagonal values between processes before advancing to the next diagonal.

After the distributed computation determines the LCS length, rank 0 reconstructs a valid subsequence using a sequential dynamic programming table.

The MPI timing reported by the program covers the distributed length calculation, not the subsequent sequential reconstruction. The benchmark's end-to-end timing includes the complete program execution.

Collective communication at every diagonal can create substantial overhead, especially for small inputs.

### CUDA GPU computation

The CUDA implementation assigns GPU threads to cells on an anti-diagonal. It launches successive kernels to process diagonals in dependency order and maintains the previous diagonal values needed for the recurrence.

The GPU calculates the LCS length, while the CPU reconstructs a valid subsequence using sequential dynamic programming.

CUDA event timing measures the diagonal kernel phase separately. It excludes costs such as program startup, memory allocation, data transfers, and CPU reconstruction.

The need to launch a kernel for each diagonal can limit performance for relatively small workloads.

## Benchmarking Methodology

Generate benchmark measurements with:

```bash
python3 scripts/benchmark.py
```

The benchmark uses reproducible, randomly generated DNA-like strings containing `A`, `C`, `G`, and `T`.

### Experimental configuration

| Parameter | Configuration |
|---|---|
| Input lengths | 100, 250, 500, 1,000 |
| MPI process counts | 1, 2, 4 |
| Repetitions | 3 per configuration |
| Total measured runs | 60 |
| Correctness validation | LCS length compared against sequential C |

CUDA is warmed up before the measured runs for each input size.

### Timing measurements

**End-to-end runtime:** Measured externally using Python's `time.perf_counter()`. This includes program invocation and startup overhead, computation, and reconstruction.

**CUDA kernel runtime:** Measured using CUDA events around the diagonal kernel phase. It represents only part of the CUDA program's total execution time.

The raw measurements are written to `results/benchmark.csv`. Summary statistics are written to `results/benchmark_summary.csv`.

## Results and Performance Analysis

The following table presents the mean end-to-end runtimes from the recorded benchmark run.

| Input length | Sequential C | MPI (1 process) | MPI (2 processes) | MPI (4 processes) | CUDA |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.000576 s | 0.244971 s | 0.252494 s | 0.264593 s | 0.185573 s |
| 250 | 0.000713 s | 0.245649 s | 0.272988 s | 0.276573 s | 0.216950 s |
| 500 | 0.001012 s | 0.242054 s | 0.252475 s | 0.255267 s | 0.201196 s |
| 1,000 | 0.002297 s | 0.258155 s | 0.259593 s | 0.264009 s | 0.224144 s |

### Key observations

- **Sequential C was fastest** for every tested input size.
- **CUDA was faster end-to-end than the tested MPI configurations**, but did not outperform the sequential implementation.
- **Increasing the MPI process count did not improve end-to-end runtime** in these experiments. Communication overhead outweighed any benefit from distributing the work.
- **CUDA kernel time was only part of total execution time.** At input length 1,000, mean kernel time was approximately 23.81 ms, compared with approximately 224.14 ms end-to-end.

These findings reflect the current algorithms, workload sizes, and tested environment. They should not be interpreted as universal evidence that MPI or CUDA is inherently slower than sequential execution.

### Performance visualizations

**End-to-end runtime**

![Runtime comparison](results/runtime_comparison.png)

**Speedup relative to sequential C**

![Speedup comparison](results/speedup_comparison.png)

**CUDA kernel execution time**

![CUDA kernel comparison](results/cuda_kernel_comparison.png)

Speedup is calculated by dividing the sequential mean runtime by the mean runtime of the implementation being evaluated:

`Speedup = Sequential mean runtime / Implementation mean runtime`

- **Speedup greater than 1:** The implementation was faster than the sequential baseline.
- **Speedup equal to 1:** Both had the same runtime.
- **Speedup below 1:** The implementation was slower than the sequential baseline.

To regenerate the analysis and plots, run:

```bash
python3 scripts/analyze_benchmarks.py
```

## Limitations

- The current benchmark covers input lengths up to 1,000 characters.
- MPI experiments cover 1, 2, and 4 processes.
- MPI communication occurs at every anti-diagonal.
- CUDA launches a kernel for each anti-diagonal.
- LCS reconstruction remains sequential in both parallel implementations.
- Each configuration has three measured repetitions.
- Performance results depend on the hardware, software environment, workload, and system load.

## Future Work

- Expand automated testing with randomized inputs and additional edge cases.
- Evaluate larger input sizes and additional MPI process counts.
- Investigate more efficient MPI communication and work-distribution strategies.
- Explore CUDA approaches that reduce kernel-launch overhead.
- Investigate parallel LCS reconstruction.
- Compare alternative algorithms and optimizations.
- Prepare a technical report documenting implementation decisions, validation, experimental methodology, and results.

## Project Status

- [x] Sequential C implementation
- [x] MPI implementation using anti-diagonal wavefront computation
- [x] CUDA implementation using anti-diagonal wavefront computation
- [x] Automated tests for all three implementations
- [x] Cross-implementation LCS length validation in the benchmark suite
- [x] Reproducible benchmark generation
- [x] Runtime and speedup analysis
- [x] CUDA kernel timing analysis
- [x] Benchmark results and visualization plots
- [x] Build, usage, and testing documentation
- [x] Clean build and test workflow
- [ ] Extended scalability evaluation
- [ ] Further optimization of MPI communication and CUDA kernel launches
- [x] Technical report

## Reproducibility

To rebuild the executables and run all automated tests:

```bash
make clean
make all
make cuda
make test
```

To regenerate the measurements and analysis:

```bash
python3 scripts/benchmark.py
python3 scripts/analyze_benchmarks.py
```

The recorded benchmark files are included so that the current results can be inspected without rerunning the experiments.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
