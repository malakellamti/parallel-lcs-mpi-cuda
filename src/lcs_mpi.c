
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LENGTH 10000

static int max_int(int a, int b)
{
    return (a > b) ? a : b;
}

/*
 * Calculate LCS length using distributed anti-diagonal wavefronts.
 * Each DP cell is calculated by exactly one MPI process.
 */
static int distributed_lcs_length(
    const char *x, const char *y,
    int m, int n, int rank, int size)
{
    int *prev2 = calloc((size_t)m + 1, sizeof(int));
    int *prev1 = calloc((size_t)m + 1, sizeof(int));
    int *local = calloc((size_t)m + 1, sizeof(int));
    int *current = calloc((size_t)m + 1, sizeof(int));

    if (!prev2 || !prev1 || !local || !current) {
        fprintf(stderr, "Rank %d: memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int k = 2; k <= m + n; k++) {
        int i_min = (k - n > 1) ? k - n : 1;
        int i_max = (k - 1 < m) ? k - 1 : m;

        memset(local, 0, ((size_t)m + 1) * sizeof(int));

        for (int i = i_min; i <= i_max; i++) {
            if ((i - i_min) % size != rank)
                continue;

            int j = k - i;

            if (x[i - 1] == y[j - 1]) {
                int diagonal = (i > 1 && j > 1)
                    ? prev2[i - 1] : 0;
                local[i] = diagonal + 1;
            } else {
                int above = (i > 1) ? prev1[i - 1] : 0;
                int left  = (j > 1) ? prev1[i] : 0;
                local[i] = max_int(above, left);
            }
        }

        MPI_Allreduce(
            local, current, m + 1,
            MPI_INT, MPI_SUM, MPI_COMM_WORLD
        );

        int *temporary = prev2;
        prev2 = prev1;
        prev1 = current;
        current = temporary;
    }

    int result = prev1[m];

    free(prev2);
    free(prev1);
    free(local);
    free(current);

    return result;
}

/* Reconstruct an actual LCS on rank 0 after the distributed length pass. */
static char *reconstruct_lcs(
    const char *x, const char *y, int m, int n, int length)
{
    size_t rows = (size_t)m + 1;
    size_t cols = (size_t)n + 1;

    if (rows > (size_t)-1 / cols ||
        rows * cols > (size_t)-1 / sizeof(int)) {
        return NULL;
    }

    int *dp = calloc(rows * cols, sizeof(int));
    char *sequence = malloc((size_t)length + 1);

    if (!dp || !sequence) {
        free(dp);
        free(sequence);
        return NULL;
    }

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            size_t cell = (size_t)i * cols + (size_t)j;

            if (x[i - 1] == y[j - 1]) {
                dp[cell] = dp[(size_t)(i - 1) * cols
                              + (size_t)(j - 1)] + 1;
            } else {
                int above = dp[(size_t)(i - 1) * cols + (size_t)j];
                int left = dp[(size_t)i * cols + (size_t)(j - 1)];
                dp[cell] = max_int(above, left);
            }
        }
    }

    int i = m;
    int j = n;
    int pos = length;

    sequence[length] = '\0';

    while (i > 0 && j > 0) {
        if (x[i - 1] == y[j - 1]) {
            sequence[--pos] = x[i - 1];
            i--;
            j--;
        } else {
            int above = dp[(size_t)(i - 1) * cols + (size_t)j];
            int left = dp[(size_t)i * cols + (size_t)(j - 1)];

            if (above >= left)
                i--;
            else
                j--;
        }
    }

    free(dp);
    return sequence;
}

int main(int argc, char **argv)
{
    int rank, size;
    int valid = 1;
    int m = 0, n = 0;
    char *x = NULL, *y = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        if (argc != 3) {
            fprintf(stderr, "Usage: mpirun -n <processes> "
                    "./lcs_mpi SEQUENCE1 SEQUENCE2\n");
            valid = 0;
        } else {
            size_t len_x = strlen(argv[1]);
            size_t len_y = strlen(argv[2]);

            if (len_x > MAX_LENGTH || len_y > MAX_LENGTH) {
                fprintf(stderr, "Sequences must be at most %d "
                        "characters long.\n", MAX_LENGTH);
                valid = 0;
            } else {
                m = (int)len_x;
                n = (int)len_y;
            }
        }
    }

    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!valid) {
        MPI_Finalize();
        return 1;
    }

    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    x = malloc((size_t)m + 1);
    y = malloc((size_t)n + 1);

    if (!x || !y) {
        fprintf(stderr, "Rank %d: memory allocation failed\n", rank);
        free(x);
        free(y);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        memcpy(x, argv[1], (size_t)m + 1);
        memcpy(y, argv[2], (size_t)n + 1);
    }

    MPI_Bcast(x, m + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(y, n + 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    double start = MPI_Wtime();

    int length = distributed_lcs_length(x, y, m, n, rank, size);

    double local_elapsed = MPI_Wtime() - start;
    double elapsed = 0.0;

    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        char *sequence = reconstruct_lcs(x, y, m, n, length);

        if (!sequence) {
            fprintf(stderr, "Could not reconstruct LCS: "
                    "memory allocation failed\n");
            free(x);
            free(y);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        printf("LCS length: %d\n", length);
        printf("LCS sequence: %s\n", sequence);
        printf("MPI processes: %d\n", size);
        printf("Distributed length computation time: %.6f seconds\n",
               elapsed);

        free(sequence);
    }

    free(x);
    free(y);

    MPI_Finalize();
    return 0;
}
