
#!/usr/bin/env bash
set -euo pipefail

passed=0

run_test() {
    local name="$1"
    local seq1="$2"
    local seq2="$3"
    local expected="$4"

    local output
    output=$(./lcs_cuda "$seq1" "$seq2")

    if ! grep -q "LCS length: $expected" <<< "$output"; then
        echo "FAIL: $name"
        echo "$output"
        exit 1
    fi

    echo "PASS: $name"
    passed=$((passed + 1))
}

run_test "Classic example" "ABCBDAB" "BDCABA" 4
run_test "Identical sequences" "ABC" "ABC" 3
run_test "No common characters" "ABC" "DEF" 0
run_test "Empty first sequence" "" "ABC" 0
run_test "Single matching character" "A" "A" 1
run_test "Subsequence" "ABCDE" "ACE" 3
run_test "Different lengths" "XMJYAUZ" "MZJAWXU" 4

echo "$passed CUDA tests passed."
