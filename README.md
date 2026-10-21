# Parallel Longest Common Subsequence (LCS): C, MPI and CUDA

A comparative implementation of the Longest Common Subsequence (LCS) problem using sequential C, distributed-memory parallelism with MPI, and GPU parallelism with CUDA.

The project explores algorithm design, parallel computation, correctness testing, benchmarking, and performance analysis.

## Project status

- [x] Sequential C implementation
- [x] Automated sequential tests
- [x] MPI implementation using anti-diagonal wavefront computation
- [x] Automated MPI tests
- [x] CUDA implementation using anti-diagonal wavefront computation
- [x] Automated CUDA tests
- [x] Reproducible benchmark scripts
- [x] Performance analysis and plots
- [ ] Extended scalability evaluation
- [ ] Technical report

## 1. Problem description

The Longest Common Subsequence problem consists of finding the longest sequence of characters that appears in two input sequences in the same relative order. Characters do not need to be contiguous.

For example, given:

- Sequence 1: `ABCBDAB`
- Sequence 2: `BDCABA`

One valid longest common subsequence is `BCBA`, with length 4.

The project implements the same problem using three approaches:

1. **Sequential C:** a conventional dynamic programming algorithm.
2. **MPI:** distributes dynamic programming cells across processes using an anti-diagonal wavefront.
3. **CUDA:** computes dynamic programming cells in parallel on an NVIDIA GPU, one thread per cell on each anti-diagonal.

All implementations calculate the LCS length and reconstruct one valid LCS.

## 2. Repository structure

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
│   ├── benchmark.sh
│   ├── benchmark.py
│   └── analyze_benchmarks.py
├── results/
│   ├── benchmark.csv
│   ├── benchmark_summary.csv
│   ├── runtime_comparison.png
│   └── speedup_comparison.png
├── .gitignore
├── Makefile
└── README.md
```

## 3. Requirements

### Sequential and MPI implementations

- GCC
- Open MPI
- GNU Make
- Bash

### CUDA implementation

- NVIDIA CUDA Toolkit with `nvcc`
- A compatible NVIDIA GPU and driver
- A host C++ compiler supported by the installed CUDA Toolkit

The current development environment uses GCC for the sequential implementation, Open MPI for MPI, and `g++-14` as the CUDA host compiler.

Python 3 and Matplotlib are required to regenerate the benchmark analysis and plots.

## 4. Build

Build the sequential and MPI implementations:

```bash
make all
```

Build the CUDA implementation:

```bash
make cuda
```

Build all three executables from a clean state:

```bash
make clean
make all
make cuda
```

The generated executables are:

- `lcs_sequential`
- `lcs_mpi`
- `lcs_cuda`

Compiled binaries are excluded from version control.

## 5. Usage

### Sequential C

```bash
./lcs_sequential ABCBDAB BDCABA
```

### MPI

Run with two MPI processes:

```bash
mpirun -n 2 ./lcs_mpi ABCBDAB BDCABA
```

Run with four MPI processes:

```bash
mpirun -n 4 ./lcs_mpi ABCBDAB BDCABA
```

### CUDA

```bash
./lcs_cuda ABCBDAB BDCABA
```

Each implementation should report an LCS length of 4 and a valid LCS, such as `BCBA`.

The implementations may reconstruct different valid subsequences when multiple optimal solutions exist.

Each input sequence is limited to 10,000 characters by the current command-line implementations. Actual feasible input sizes also depend on available memory and execution time.

## 6. Automated testing

Run all three test suites:

```bash
make test
```

Or run them individually:

```bash
make test-sequential
make test-mpi
make test-cuda
```

The current test suites cover:

- The classic LCS example
- Identical sequences
- Sequences with no common characters
- An empty sequence
- A single matching character
- A sequence that is a subsequence of another
- Inputs of different lengths
- MPI execution with different process counts

**Current validation:** all 7 sequential tests, 8 MPI tests, and 7 CUDA tests pass in the development environment.

The tests verify the expected LCS lengths. They do not constitute exhaustive proof of correctness for every possible input.

## 7. Algorithm design

### 7.1 Sequential dynamic programming

Let \(X\) and \(Y\) be the two input sequences. Define \(L(i,j)\) as the LCS length of their first \(i\) and \(j\) characters.

The recurrence is:

\[
L(i,j)=
\begin{cases}
0 & i=0 \text{ or } j=0\\
L(i-1,j-1)+1 & X_i=Y_j\\
\max(L(i-1,j),L(i,j-1)) & X_i\ne Y_j
\end{cases}
\]

The full dynamic programming table is filled row by row, then traversed backwards to reconstruct one LCS.

- Time complexity: \(O(mn)\)
- Space complexity: \(O(mn)\)

Here, \(m\) and \(n\) are the sequence lengths.

### 7.2 MPI distributed computation

The MPI implementation processes the dynamic programming table by anti-diagonals. Cells on the same anti-diagonal can be computed concurrently once the required previous diagonals are available.

Each process computes a subset of the cells on the current anti-diagonal. `MPI_Allreduce` combines the local results so that all processes can access the values needed for the next step.

This approach demonstrates distributed computation but incurs communication overhead on every anti-diagonal.

The distributed phase calculates the LCS length. Rank 0 then performs a sequential dynamic programming pass to reconstruct the actual subsequence.

Consequently, the MPI implementation does not parallelize the complete end-to-end workflow. Its measured total runtime includes both distributed computation and sequential reconstruction.

### 7.3 CUDA GPU computation

The CUDA implementation also uses anti-diagonal wavefront computation.

For each anti-diagonal:

1. The host determines the valid range of cells.
2. CUDA threads are launched to calculate those cells in parallel.
3. The device rotates the buffers holding the previous two diagonals and the current diagonal.
4. The process continues until the final diagonal has been calculated.

The LCS length is copied back from the GPU. A CPU dynamic programming pass then reconstructs one valid LCS for display.

The reported GPU computation time measures the diagonal-kernel phase using CUDA events. It excludes device allocation, host-to-device input copies, result transfer, and CPU reconstruction.

This timing is useful for examining GPU computation, but it is not directly equivalent to end-to-end program runtime.

## 8. Benchmarking and performance analysis

The repository includes scripts to run experiments and analyze the results.

### Reproduce the benchmarks

```bash
python3 scripts/benchmark.py
```

The benchmark script generates deterministic DNA-like sequences over the alphabet `ACGT`, tests the sequential and MPI implementations, checks the resulting LCS lengths, and records elapsed wall-clock times.

The current recorded experiment uses sequence lengths of 100, 250, 500, and 1,000 characters, with three repetitions per configuration. MPI runs use 1, 2, and 4 processes.

### Regenerate summary statistics and plots

```bash
python3 scripts/analyze_benchmarks.py
```

The analysis script generates summary statistics and comparison plots from `results/benchmark.csv`.

### Recorded results

In the recorded experiment, the sequential implementation outperformed the MPI implementation at every tested sequence length.

For sequences of length 1,000, the mean end-to-end runtimes were approximately:

| Implementation | Processes | Mean runtime |
|---|---:|---:|
| Sequential C | 1 | 0.00375 s |
| MPI | 1 | 0.25491 s |
| MPI | 2 | 0.25787 s |
| MPI | 4 | 0.26414 s |

These measurements show no MPI speedup for the tested workload. Communication overhead, process startup, and sequential reconstruction outweigh the benefits of distributing this relatively small computation.

The results are specific to the tested implementation, environment, input sizes, and repetition count. They should not be interpreted as a general comparison of MPI and sequential computation.

**Important:** the recorded benchmark dataset currently covers sequential C and MPI, not CUDA. CUDA performance should be evaluated separately until the benchmark harness records CUDA end-to-end runtime and GPU kernel time consistently.

### Benchmark artifacts

The repository includes:

- `results/benchmark.csv` — recorded benchmark measurements
- `results/benchmark_summary.csv` — summary statistics
- `results/runtime_comparison.png` — runtime comparison plot
- `results/speedup_comparison.png` — speedup comparison plot

#### Runtime comparison

![Runtime comparison](results/runtime_comparison.png)

#### Speedup comparison

![Speedup comparison](results/speedup_comparison.png)

## 9. Limitations and future work

Potential improvements include:

- Extend automated tests with randomized inputs and cross-implementation comparisons.
- Improve MPI communication efficiency and evaluate alternative work-distribution strategies.
- Investigate CUDA approaches that reduce kernel-launch overhead.
- Add CUDA to the benchmark harness and distinguish end-to-end runtime from GPU kernel time.
- Evaluate larger input sizes and additional process counts.
- Document the hardware and software environment for every performance experiment.
- Complete a technical report describing implementation choices, correctness validation, experimental methodology, and results.

## 10. Reproducibility

To rebuild the executables and run the automated tests:

```bash
make clean
make all
make cuda
make test
```

To regenerate the sequential and MPI benchmark measurements and their analysis:

```bash
python3 scripts/benchmark.py
python3 scripts/analyze_benchmarks.py
```

Recorded benchmark files are retained in the repository so that the published results can be inspected without rerunning the experiments.

## License

No license has been specified yet. Add an appropriate license before explicitly permitting reuse or redistribution of this project.
