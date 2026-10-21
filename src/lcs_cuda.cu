
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#define MAX_LENGTH 10000
#define THREADS_PER_BLOCK 256

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

/*
 * One CUDA thread computes one cell on an anti-diagonal.
 * prev1 contains diagonal k-1; prev2 contains diagonal k-2.
 * Arrays are indexed by the row coordinate i.
 */
__global__ void compute_diagonal(
    const char *x,
    const char *y,
    int m,
    int n,
    int k,
    int i_min,
    int count,
    const int *prev2,
    const int *prev1,
    int *current)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= count) {
        return;
    }

    int i = i_min + tid;
    int j = k - i;

    if (i == 0 || j == 0) {
        current[i] = 0;
    } else if (x[i - 1] == y[j - 1]) {
        current[i] = prev2[i - 1] + 1;
    } else {
        int above = prev1[i - 1];
        int left = prev1[i];
        current[i] = (above > left) ? above : left;
    }
}

static int compute_lcs_length_cuda(
    const char *x,
    const char *y,
    int m,
    int n,
    float *elapsed_ms)
{
    size_t x_bytes = static_cast<size_t>(m) + 1;
    size_t y_bytes = static_cast<size_t>(n) + 1;
    size_t dp_bytes = (static_cast<size_t>(m) + 1) * sizeof(int);

    char *d_x = nullptr;
    char *d_y = nullptr;
    int *d_prev2 = nullptr;
    int *d_prev1 = nullptr;
    int *d_current = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x, x_bytes));
    CUDA_CHECK(cudaMalloc(&d_y, y_bytes));
    CUDA_CHECK(cudaMalloc(&d_prev2, dp_bytes));
    CUDA_CHECK(cudaMalloc(&d_prev1, dp_bytes));
    CUDA_CHECK(cudaMalloc(&d_current, dp_bytes));

    CUDA_CHECK(cudaMemcpy(d_x, x, x_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y, y_bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_prev2, 0, dp_bytes));
    CUDA_CHECK(cudaMemset(d_prev1, 0, dp_bytes));
    CUDA_CHECK(cudaMemset(d_current, 0, dp_bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int k = 0; k <= m + n; ++k) {
        int i_min = std::max(0, k - n);
        int i_max = std::min(m, k);
        int count = i_max - i_min + 1;

        int blocks = (count + THREADS_PER_BLOCK - 1)
                   / THREADS_PER_BLOCK;

        compute_diagonal<<<blocks, THREADS_PER_BLOCK>>>(
            d_x, d_y, m, n, k, i_min, count,
            d_prev2, d_prev1, d_current);

        CUDA_CHECK(cudaGetLastError());

        int *temporary = d_prev2;
        d_prev2 = d_prev1;
        d_prev1 = d_current;
        d_current = temporary;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(elapsed_ms, start, stop));

    int result = 0;
    CUDA_CHECK(cudaMemcpy(
        &result, d_prev1 + m, sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_prev2));
    CUDA_CHECK(cudaFree(d_prev1));
    CUDA_CHECK(cudaFree(d_current));

    return result;
}

/* Reconstruct one LCS on the CPU for display. */
static std::string reconstruct_lcs(
    const char *x, const char *y, int m, int n)
{
    size_t rows = static_cast<size_t>(m) + 1;
    size_t cols = static_cast<size_t>(n) + 1;

    if (rows > std::numeric_limits<size_t>::max() / cols ||
        rows * cols > std::numeric_limits<size_t>::max() / sizeof(int)) {
        fprintf(stderr, "DP table size overflow.\n");
        exit(EXIT_FAILURE);
    }

    std::vector<int> dp(rows * cols, 0);

    auto cell = [&dp, cols](int i, int j) -> int& {
        return dp[static_cast<size_t>(i) * cols
                + static_cast<size_t>(j)];
    };

    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (x[i - 1] == y[j - 1]) {
                cell(i, j) = cell(i - 1, j - 1) + 1;
            } else {
                cell(i, j) = std::max(cell(i - 1, j), cell(i, j - 1));
            }
        }
    }

    std::string result;
    int i = m;
    int j = n;

    while (i > 0 && j > 0) {
        if (x[i - 1] == y[j - 1]) {
            result.push_back(x[i - 1]);
            --i;
            --j;
        } else if (cell(i - 1, j) >= cell(i, j - 1)) {
            --i;
        } else {
            --j;
        }
    }

    std::reverse(result.begin(), result.end());
    return result;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s SEQUENCE1 SEQUENCE2\n", argv[0]);
        fprintf(stderr, "Example: %s ABCBDAB BDCABA\n", argv[0]);
        return EXIT_FAILURE;
    }

    size_t m_size = std::strlen(argv[1]);
    size_t n_size = std::strlen(argv[2]);

    if (m_size > MAX_LENGTH || n_size > MAX_LENGTH) {
        fprintf(stderr, "Error: each sequence must be at most %d characters.\n",
                MAX_LENGTH);
        return EXIT_FAILURE;
    }

    int m = static_cast<int>(m_size);
    int n = static_cast<int>(n_size);

    int length = 0;
    float gpu_time_ms = 0.0f;

    length = compute_lcs_length_cuda(
        argv[1], argv[2], m, n, &gpu_time_ms);

    std::string sequence = reconstruct_lcs(argv[1], argv[2], m, n);

    if (static_cast<int>(sequence.size()) != length) {
        fprintf(stderr, "Error: GPU length and CPU reconstruction disagree.\n");
        return EXIT_FAILURE;
    }

    printf("LCS length: %d\n", length);
    printf("LCS sequence: %s\n", sequence.c_str());
    printf("GPU diagonal computation time: %.6f milliseconds\n",
           gpu_time_ms);

    return EXIT_SUCCESS;
}
