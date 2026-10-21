
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LENGTH 10000

/*
 * Longest Common Subsequence (LCS)
 * Sequential C implementation.
 *
 * Time complexity: O(m * n)
 * Space complexity: O(m * n)
 */

static int compute_lcs(
    const char *x,
    const char *y,
    int m,
    int n,
    char *result
) {
    int cols = n + 1;

    int *dp = calloc((size_t)(m + 1) * (size_t)cols, sizeof(int));
    if (dp == NULL) {
        fprintf(stderr, "Error: unable to allocate DP table.\n");
        return -1;
    }

    #define DP(i, j) dp[(size_t)(i) * (size_t)cols + (size_t)(j)]

    /* Fill the dynamic programming table. */
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (x[i - 1] == y[j - 1]) {
                DP(i, j) = DP(i - 1, j - 1) + 1;
            } else {
                int above = DP(i - 1, j);
                int left = DP(i, j - 1);
                DP(i, j) = (above > left) ? above : left;
            }
        }
    }

    int length = DP(m, n);

    /* Reconstruct one LCS by walking backwards through the table. */
    result[length] = '\0';

    int i = m;
    int j = n;
    int k = length;

    while (i > 0 && j > 0) {
        if (x[i - 1] == y[j - 1]) {
            result[--k] = x[i - 1];
            i--;
            j--;
        } else if (DP(i - 1, j) >= DP(i, j - 1)) {
            i--;
        } else {
            j--;
        }
    }

    free(dp);
    #undef DP

    return length;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s SEQUENCE1 SEQUENCE2\n", argv[0]);
        fprintf(stderr, "Example: %s ABCBDAB BDCABA\n", argv[0]);
        return EXIT_FAILURE;
    }

    size_t m_size = strlen(argv[1]);
    size_t n_size = strlen(argv[2]);

    if (m_size > MAX_LENGTH || n_size > MAX_LENGTH) {
        fprintf(stderr, "Error: each sequence must be at most %d characters.\n",
                MAX_LENGTH);
        return EXIT_FAILURE;
    }

    int m = (int)m_size;
    int n = (int)n_size;

    char *lcs = malloc((size_t)(m < n ? m : n) + 1);
    if (lcs == NULL) {
        fprintf(stderr, "Error: unable to allocate result buffer.\n");
        return EXIT_FAILURE;
    }

    int length = compute_lcs(argv[1], argv[2], m, n, lcs);
    if (length < 0) {
        free(lcs);
        return EXIT_FAILURE;
    }

    printf("Sequence 1: %s\n", argv[1]);
    printf("Sequence 2: %s\n", argv[2]);
    printf("LCS length: %d\n", length);
    printf("LCS: %s\n", lcs);

    free(lcs);
    return EXIT_SUCCESS;
}
