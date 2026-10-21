#!/usr/bin/env bash

set -euo pipefail

SIZES=(100 250 500 1000)
PROCESS_COUNTS=(1 2 4)
REPEATS=3

mkdir -p results
CSV="results/benchmark.csv"
TIMEFILE=$(mktemp)
trap 'rm -f "$TIMEFILE"' EXIT

echo "size,implementation,processes,repeat,elapsed_seconds" > "$CSV"

# Build both implementations before measuring.
make sequential mpi

for size in "${SIZES[@]}"; do
    # Generate reproducible DNA-like sequences using fixed seeds.
    seq1=$(python3 -c \
        'import random,sys; random.seed(int(sys.argv[1])); print("".join(random.choices("ACGT", k=int(sys.argv[2]))))' \
        "$size" "$size")

    seq2=$(python3 -c \
        'import random,sys; random.seed(int(sys.argv[1])); print("".join(random.choices("ACGT", k=int(sys.argv[2]))))' \
        "$((size + 1))" "$size")

    echo "Benchmarking sequence length: $size"

    for repeat in $(seq 1 "$REPEATS"); do
        /usr/bin/time -f '%e' -o "$TIMEFILE" \
            ./lcs_sequential "$seq1" "$seq2" >/dev/null

        elapsed=$(cat "$TIMEFILE")
        echo "$size,sequential,1,$repeat,$elapsed" >> "$CSV"

        for processes in "${PROCESS_COUNTS[@]}"; do
            /usr/bin/time -f '%e' -o "$TIMEFILE" \
                mpirun --oversubscribe -n "$processes" \
                ./lcs_mpi "$seq1" "$seq2" >/dev/null

            elapsed=$(cat "$TIMEFILE")
            echo "$size,mpi,$processes,$repeat,$elapsed" >> "$CSV"
        done
    done
done

echo
echo "Benchmark complete: $CSV"
echo "Rows recorded: $(($(wc -l < "$CSV") - 1))"
