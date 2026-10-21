# Parallel Longest Common Subsequence: Sequential C, MPI, and CUDA

**Technical Report**

**Author:** Malak Ellamti  
**Project:** Parallel LCS Computation  
**Date:** October 2026

---

## Abstract

The Longest Common Subsequence (LCS) problem consists of finding the longest sequence of elements shared by two input sequences while preserving their relative order. This project implements the LCS problem using three approaches: a sequential implementation in C, a distributed-memory implementation using the Message Passing Interface (MPI), and a GPU implementation using CUDA.

The implementations use dynamic programming. The sequential version evaluates the dynamic programming table in row-major order, whereas the MPI and CUDA versions process its cells along anti-diagonals to expose parallelism. Experiments compare execution times for input sequences of 100, 250, 500, and 1,000 characters. Correctness is checked by comparing the LCS lengths returned by the implementations against the sequential baseline.

The results show that sequential C is the fastest approach for all tested input sizes. The parallel implementations incur substantial overhead relative to the small computational workloads. These findings demonstrate that parallelization does not automatically improve performance: workload size, communication, kernel-launch overhead, and process startup must also be considered.

## 1. Introduction

Parallel computing can accelerate computationally intensive problems by dividing work across multiple processing units. However, the benefit depends on the structure of the algorithm and the overhead required to coordinate parallel execution.

The Longest Common Subsequence problem is a useful example because its dynamic programming formulation contains dependencies between neighboring cells. These dependencies limit which computations can execute simultaneously.

The purpose of this project is to implement and compare three approaches:

- Sequential execution using C.
- Distributed computation using MPI.
- GPU computation using CUDA.

The project evaluates correctness, execution time, and speedup relative to the sequential implementation.

### 1.1 Objectives

The main objectives are:

1. Implement a correct sequential LCS algorithm.
2. Parallelize the computation using MPI.
3. Implement a CUDA-based GPU version.
4. Validate results across all implementations.
5. Automate benchmarking and performance analysis.
6. Identify the overheads and limitations affecting performance.

## 2. Theoretical Background

### 2.1 Definition of the LCS problem

Given two sequences, X and Y, a common subsequence is a sequence that appears in both inputs in the same relative order. Its elements do not need to be contiguous.

For example, consider:

- X = `ABCBDAB`
- Y = `BDCABA`

One possible longest common subsequence is `BCBA`, with length 4.

The objective is to compute the length of the longest common subsequence. The implementations also produce a subsequence, although the parallel versions reconstruct it on the CPU after computing the length.

### 2.2 Dynamic programming recurrence

Let DP[i][j] represent the LCS length of the first i characters of X and the first j characters of Y.

The recurrence is:

1. If i = 0 or j = 0, then DP[i][j] = 0.
2. If X[i-1] equals Y[j-1], then DP[i][j] = DP[i-1][j-1] + 1.
3. Otherwise, DP[i][j] = max(DP[i-1][j], DP[i][j-1]).

The final LCS length is DP[m][n], where m and n are the lengths of the input sequences.

For two sequences of lengths m and n, the conventional dynamic programming algorithm requires O(mn) time and O(mn) space when the full table is stored for subsequence reconstruction.

### 2.3 Anti-diagonal parallelism

The recurrence depends on cells above, to the left, and diagonally above-left. Therefore, cells cannot all be computed simultaneously.

Cells on the same anti-diagonal can be computed in parallel because their required predecessor values belong to earlier anti-diagonals. Processing the table this way creates a sequence of computation stages, with synchronization between stages.

This strategy is used by the MPI and CUDA implementations in this project.

## 3. Implementation Design

### 3.1 Sequential C implementation

The sequential implementation computes the dynamic programming table in row-major order. Each cell is evaluated after the cells on which it depends have been calculated.

After filling the table, the implementation reconstructs an LCS by tracing backward through the table. This produces a reference result for validating the parallel implementations.

The sequential implementation serves as the performance baseline.

### 3.2 MPI implementation

The MPI implementation distributes anti-diagonal computation across multiple MPI processes. Processes calculate assigned cells and use MPI communication to make diagonal values available for subsequent computation.

At each anti-diagonal, every process initializes its local contribution array to zero and computes only the cells assigned to its rank. The implementation uses `MPI_Allreduce` with `MPI_SUM` to combine the local arrays and distribute the resulting values to all processes. This collective operation makes the completed diagonal available for the next wavefront stage and synchronizes the computation across processes.

After the distributed computation calculates the LCS length, rank 0 reconstructs a subsequence sequentially.

MPI introduces communication and synchronization overhead. In particular, the repeated collective operation can be expensive when each diagonal contains relatively little work. The measured runtime also includes the cost of launching the MPI program and its processes.

### 3.3 CUDA implementation

The CUDA implementation assigns anti-diagonal computations to GPU threads. A kernel is launched for each diagonal, and successive launches respect the dependencies between stages.

Once the GPU has computed the LCS length, the CPU reconstructs a subsequence.

The implementation records GPU computation time using CUDA events placed immediately before and after the anti-diagonal computation loop. The measured interval covers the sequence of diagonal kernel launches and their GPU execution. It excludes memory allocation, host-to-device input transfers, memory initialization, the final device-to-host transfer of the LCS length, and CPU-side subsequence reconstruction. Consequently, this kernel-phase measurement is distinct from the end-to-end runtime reported in the benchmark.

Launching a kernel for each diagonal introduces overhead. For relatively short input sequences, this overhead can outweigh the benefits of GPU parallelism.

## 4. Experimental Methodology

### 4.1 Test environment

The project was developed and tested in a Linux environment using WSL2 on a Windows laptop equipped with an NVIDIA RTX 5060 GPU. The software environment uses GCC 15.2.0, Open MPI 5.0.10, CUDA Toolkit 12.9, Python 3.14.4, and Matplotlib 3.10.7.

### 4.2 Input generation

The benchmark script generates two DNA-like sequences over the alphabet A, C, G, and T. A deterministic random generator is used so that inputs are reproducible for a given size.

The tested sequence lengths are:

| Length of each sequence | Number of DP cells |
|---:|---:|
| 100 | 10,000 |
| 250 | 62,500 |
| 500 | 250,000 |
| 1,000 | 1,000,000 |

The number of DP cells is proportional to the product of the two sequence lengths.

### 4.3 Benchmark configurations

Each input size is tested using the sequential implementation, MPI with 1, 2, and 4 processes, and CUDA with one GPU.

Three measured repetitions are collected for each configuration. CUDA is warmed up before the measurements, and the benchmark checks its result against the sequential reference. MPI results are also checked against the sequential LCS length.

The benchmark collects 60 measured executions in total:

4 input sizes × 5 configurations × 3 repetitions = 60 executions.

### 4.4 Timing and analysis

End-to-end runtime is measured externally using Python's high-resolution performance counter around each program execution. This includes program startup and process-launch overhead, as well as computation and other work performed before termination.

For each configuration, the analysis calculates the mean and sample standard deviation of the measured runtimes.

Speedup is defined as:

Speedup = Mean sequential runtime / Mean runtime of the compared implementation

A speedup greater than 1 indicates that the compared implementation is faster than the sequential baseline. A value below 1 indicates that it is slower.

CUDA kernel timing is analyzed separately and must not be interpreted as end-to-end runtime.

## 5. Results

### 5.1 Mean execution times

The following table summarizes the measured mean runtimes from the benchmark dataset.

| Input length | Sequential C (s) | MPI, 1 process (s) | MPI, 2 processes (s) | MPI, 4 processes (s) | CUDA end-to-end (s) |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.000576 | 0.244971 | 0.252494 | 0.264593 | 0.185573 |
| 250 | 0.000713 | 0.245649 | 0.272988 | 0.276573 | 0.216950 |
| 500 | 0.001012 | 0.242054 | 0.252475 | 0.255267 | 0.201196 |
| 1,000 | 0.002297 | 0.258155 | 0.259593 | 0.264009 | 0.224144 |

These measurements are specific to the test environment and benchmark procedure. They should not be generalized to all input sizes or hardware.

### 5.2 Runtime comparison

![End-to-end runtime comparison](../results/runtime_comparison.png)

The sequential implementation has the lowest measured runtime at every tested input size. Its runtime increases as the input sequences grow, which is consistent with the quadratic time complexity of the dynamic programming algorithm.

The MPI and CUDA implementations have substantially higher end-to-end runtimes for these small workloads. Their fixed execution overhead is large compared with the useful computation performed on relatively small DP tables.

Among the tested parallel configurations, CUDA has lower end-to-end runtime than the MPI configurations. Nevertheless, it remains much slower than sequential C in these experiments.

### 5.3 Speedup comparison

![Speedup comparison](../results/speedup_comparison.png)

The sequential implementation defines the baseline, so its speedup is 1 by definition. The parallel implementations have speedup values below 1 for every tested size, indicating a slowdown rather than an acceleration.

Increasing the MPI process count from one to four does not improve performance in the tested configurations. The additional communication and synchronization costs exceed any benefit from distributing the computation.

The results emphasize that parallel performance depends not only on available processing units but also on the amount of work available to amortize coordination overhead.

### 5.4 CUDA kernel timing

![CUDA kernel computation time](../results/cuda_kernel_comparison.png)

The CUDA kernel measurements isolate the diagonal-computation phase. For input length 1,000, the mean kernel time is approximately 23.81 milliseconds, while the mean end-to-end CUDA runtime is approximately 224.14 milliseconds.

This difference illustrates why kernel time and total application runtime must be reported separately. Kernel timing alone does not account for the complete cost of executing the program.

## 6. Discussion

The experiments show that the sequential implementation is the most efficient approach for the tested input sizes.

First, the sequential algorithm performs a relatively small amount of work on one machine process. Its low overhead allows it to complete the test cases in milliseconds or less.

Second, the MPI implementation synchronizes after every anti-diagonal using a collective communication operation. For small inputs, communication and process-management overhead dominate the useful computation. Using more MPI processes does not automatically reduce total runtime.

Third, the CUDA implementation launches a kernel for each anti-diagonal. The dependency structure therefore requires many successive kernel launches, each with associated overhead. Memory management, transfers, and CPU-side reconstruction further contribute to the total execution time.

The CUDA implementation nevertheless outperforms the MPI configurations in end-to-end time in these measurements. This does not establish that CUDA will outperform MPI for larger inputs; larger workloads and additional experiments are needed to draw that conclusion.

## 7. Limitations and Future Work

Several limitations should be considered when interpreting these results.

- The benchmark tests input lengths only up to 1,000 characters, despite the implementations supporting longer sequences.
- Only three measured repetitions are collected per configuration.
- End-to-end timing includes process startup and runtime overhead, which can obscure computational performance.
- MPI timing is not a pure measurement of arithmetic operations; it includes communication and synchronization.
- CUDA kernel timing covers only the GPU diagonal-computation phase and is not directly comparable to end-to-end runtime.
- The current MPI and CUDA implementations use CPU-side subsequence reconstruction.
- The benchmark does not yet provide an extended scalability study across larger inputs, more process counts, or alternative GPU strategies.

Future improvements could include:

1. Benchmarking larger sequences within the implementation limits.
2. Increasing the number of repetitions and reporting variability.
3. Separating initialization, communication, transfer, kernel, and reconstruction costs.
4. Reducing MPI synchronization overhead.
5. Reducing CUDA kernel-launch overhead through more efficient kernels.
6. Comparing the current approach with alternative parallel LCS algorithms.
7. Documenting exact hardware and software versions for reproducibility.

## 8. Conclusion

This project implements the Longest Common Subsequence problem using sequential C, MPI, and CUDA. All three approaches are tested against a common correctness reference, and an automated benchmark suite measures runtime and analyzes performance.

For the tested sequence lengths of 100 to 1,000 characters, sequential C is the fastest implementation. MPI and CUDA introduce overhead that exceeds the benefit of parallel execution at these problem sizes. CUDA is faster end-to-end than the tested MPI configurations, but neither parallel implementation achieves a speedup over sequential C.

The results demonstrate the importance of evaluating complete application performance rather than assuming that parallelization guarantees acceleration. Larger-scale experiments and further optimization are needed to determine how these implementations behave on more computationally demanding inputs.

## References

1. MPI Forum. *MPI: A Message-Passing Interface Standard*. https://www.mpi-forum.org/
2. NVIDIA. *CUDA C++ Programming Guide*. https://docs.nvidia.com/cuda/cuda-c-programming-guide/
3. Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. *Introduction to Algorithms*. MIT Press.
