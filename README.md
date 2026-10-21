# Parallel LCS: MPI and CUDA

An implementation and performance study of the Longest Common Subsequence (LCS) problem using sequential C, MPI, and CUDA.

## Project status

- [x] Sequential C baseline
- [x] Automated tests for sequential and MPI implementations
- [x] MPI distributed LCS length calculation
- [ ] CUDA implementation
- [ ] Benchmarking and performance analysis
- [ ] Technical report

## Problem

Given two sequences, find their longest common subsequence (LCS). A subsequence preserves character order but does not need to be contiguous.

For example, the LCS of `ABCBDAB` and `BDCABA` has length `4`. One valid result is `BCBA`.

## Sequential algorithm

The baseline uses dynamic programming.

For sequences of lengths `m` and `n`:

- Time complexity: `O(m × n)`
- Space complexity: `O(m × n)`

The program reports the LCS length and one valid subsequence.

## MPI implementation

The MPI implementation calculates the LCS length using an anti-diagonal wavefront approach.

- Each anti-diagonal contains cells that can be computed using values from the previous two anti-diagonals.
- MPI processes divide the cells on each anti-diagonal among themselves.
- `MPI_Allreduce` combines the partial results so that every process has the values needed for the next anti-diagonal.
- Rank 0 reconstructs one valid LCS sequence using a sequential dynamic-programming table after the distributed length calculation.

The distributed length calculation uses `O(m)` auxiliary memory per process, where `m` is the length of the first sequence. Its total dynamic-programming work remains `O(m × n)`. Communication overhead must be considered when evaluating performance.

## Requirements

- GCC
- GNU Make
- Open MPI for the MPI implementation
- NVIDIA CUDA Toolkit for the planned CUDA implementation

## Build

Build the sequential and MPI programs:

```bash
make
```

Build only the sequential program:

```bash
make sequential
```

Build only the MPI program:

```bash
make mpi
```

## Run

Sequential version:

```bash
./lcs_sequential ABCBDAB BDCABA
```

MPI version with two processes:

```bash
mpirun -n 2 ./lcs_mpi ABCBDAB BDCABA
```

Both should report an LCS length of `4` and a valid LCS sequence.

## Automated tests

Run all available tests:

```bash
make test
```

The sequential test suite covers seven cases, and the MPI test suite covers eight cases, including empty sequences, identical sequences, and different MPI process counts.

## Planned evaluation

The project will compare correctness, execution time, speedup, and scalability across sequential, MPI, and CUDA implementations. Benchmarking will use larger inputs and multiple process counts.

## Learning objectives

- Dynamic programming
- Distributed-memory parallelism with MPI
- GPU parallelism with CUDA
- Benchmarking and performance analysis
- Reproducible software development

## Performance evaluation

### Methodology

The sequential C and MPI implementations were tested on reproducibly generated
DNA-like sequences over the alphabet `ACGT`. Each input pair was tested three
times at lengths of 100, 250, 500, and 1,000 characters.

End-to-end runtime was measured using Python's high-resolution
`time.perf_counter()`. MPI measurements include process startup and the
program's sequential LCS reconstruction on rank 0. Consequently, these
measurements represent total program runtime, not only the distributed
calculation.

### Results

On the tested input sizes, sequential C was faster than the MPI implementation.
For example, at a sequence length of 1,000, the mean measured runtimes were
approximately:

| Implementation | Mean runtime |
|---|---:|
| Sequential C | 0.00375 s |
| MPI, 1 process | 0.25491 s |
| MPI, 2 processes | 0.25787 s |
| MPI, 4 processes | 0.26414 s |

The MPI implementation did not demonstrate a speedup on these inputs.
Communication on every anti-diagonal and MPI startup overhead outweigh the
benefit of distributing the computation at these sizes. Increasing the process
count also increased runtime slightly in this experiment.

These results are limited to the tested implementation, environment, input
sizes, and three repetitions per configuration. Larger inputs and further
optimization are needed to evaluate scalability more fully.

### Benchmark artifacts

The `scripts/benchmark.py` script regenerates the measurements.
The `scripts/analyze_benchmarks.py` script produces summary statistics and
plots from the recorded CSV data.

![Runtime comparison](results/runtime_comparison.png)

![Speedup comparison](results/speedup_comparison.png)
