
#!/usr/bin/env bash

set -u

passed=0
failed=0

run_test() {
    local description="$1"
    local processes="$2"
    local sequence1="$3"
    local sequence2="$4"
    local expected="$5"

    local output
    if ! output=$(mpirun --oversubscribe -n "$processes" \
        ./lcs_mpi "$sequence1" "$sequence2" 2>&1); then
        echo "FAIL: $description (MPI execution failed)"
        echo "$output"
        failed=$((failed + 1))
        return
    fi

    local actual
    actual=$(printf '%s\n' "$output" |
        awk -F': ' '/^LCS length:/ {print $2}')

    if [[ "$actual" == "$expected" ]]; then
        echo "PASS: $description"
        passed=$((passed + 1))
    else
        echo "FAIL: $description (expected $expected, got ${actual:-missing})"
        failed=$((failed + 1))
    fi
}

run_test "Classic example, 2 processes" 2 ABCBDAB BDCABA 4
run_test "Identical sequences, 2 processes" 2 ABC ABC 3
run_test "No common characters, 2 processes" 2 ABC DEF 0
run_test "Empty first sequence, 2 processes" 2 "" ABC 0
run_test "Single matching character, 2 processes" 2 A A 1
run_test "One sequence is a subsequence, 2 processes" 2 ABCDE ACE 3
run_test "Different lengths, 4 processes" 4 XMJYAUZ MZJAWXU 4
run_test "Classic example, 1 process" 1 ABCBDAB BDCABA 4

echo
echo "Results: $passed passed, $failed failed"

if [[ "$failed" -ne 0 ]]; then
    exit 1
fi
