
#!/usr/bin/env bash
set -euo pipefail

PROGRAM="./lcs_sequential"
passed=0
failed=0

run_test() {
    local name="$1"
    local seq1="$2"
    local seq2="$3"
    local expected_length="$4"

    local output
    output=$("$PROGRAM" "$seq1" "$seq2")
    local actual_length
    actual_length=$(printf '%s\n' "$output" | awk -F': ' '/^LCS length:/ {print $2}')

    if [[ "$actual_length" == "$expected_length" ]]; then
        printf 'PASS: %s\n' "$name"
        ((passed += 1))
    else
        printf 'FAIL: %s (expected length %s, got %s)\n' \
            "$name" "$expected_length" "${actual_length:-missing}"
        ((failed += 1))
    fi
}

if [[ ! -x "$PROGRAM" ]]; then
    echo "Program not found. Run 'make' first."
    exit 1
fi

run_test "Classic example" "ABCBDAB" "BDCABA" 4
run_test "Identical sequences" "ABC" "ABC" 3
run_test "No common characters" "ABC" "DEF" 0
run_test "Empty first sequence" "" "ABC" 0
run_test "Single matching character" "A" "A" 1
run_test "One sequence is a subsequence" "ABCDE" "ACE" 3
run_test "Different lengths" "XMJYAUZ" "MZJAWXU" 4

echo
echo "Results: $passed passed, $failed failed"

if (( failed > 0 )); then
    exit 1
fi
