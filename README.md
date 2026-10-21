
# Parallel LCS: MPI and CUDA

An implementation and performance study of the Longest Common Subsequence (LCS) problem using sequential C, MPI, and CUDA.

## Project status

- [x] Sequential C baseline
- [ ] Automated tests
- [ ] MPI implementation
- [ ] CUDA implementation
- [ ] Benchmarking and performance analysis
- [ ] Technical report

## Problem

Given two sequences, find their longest common subsequence. A subsequence preserves character order but does not need to be contiguous.

## Sequential algorithm

The baseline uses dynamic programming.

For sequences of lengths `m` and `n`:

- Time complexity: `O(m × n)`
- Space complexity: `O(m × n)`

The program reports the LCS length and one valid subsequence.

## Requirements

- GCC
- GNU Make
- Open MPI (for the MPI version)
- NVIDIA CUDA Toolkit (for the CUDA version)

## Build

```bash
make
```

## Run

```bash
./lcs_sequential ABCBDAB BDCABA
```

Expected LCS length: `4`.

## Planned evaluation

The project will compare correctness, execution time, speedup, and scalability across sequential, MPI, and CUDA implementations.

## Learning objectives

- Dynamic programming
- Distributed-memory parallelism with MPI
- GPU parallelism with CUDA
- Benchmarking and performance analysis
- Reproducible software development
