#!/bin/bash
set -euo pipefail

# event_rateCU — near machine-precision tolerance on integrated rates
./build/src/app/event_rateCU 2>/dev/null | grep -E '^\||Result|Flavor' > /tmp/event_rate_check.txt
echo "baseline event rates:"
cat baselines/event_rate.golden
echo "---"
echo "current event rates:"
cat /tmp/event_rate_check.txt

# Extract the Result column (second number field) from both, compare
grep -oP '(?<=\|)[\s]*[\d.]+(?=[\s]*\|[\s]*[\d.]+[\s]*\|[\s]*[\d.]+\s*%)' baselines/event_rate.golden | tr -d ' ' > /tmp/golden_vals.txt
grep -oP '(?<=\|)[\s]*[\d.]+(?=[\s]*\|[\s]*[\d.]+[\s]*\|[\s]*[\d.]+\s*%)' /tmp/event_rate_check.txt  | tr -d ' ' > /tmp/check_vals.txt
paste /tmp/golden_vals.txt /tmp/check_vals.txt | while read golden check; do
  rel_diff=$(python3 -c "print(abs($check - $golden) / max(abs($golden), 1.0))")
  if python3 -c "exit(0 if $rel_diff < 0.001 else 1)"; then
    echo "  OK: $golden vs $check (rel diff = $rel_diff)"
  else
    echo "FAIL: event rate $golden diverged to $check (rel diff = $rel_diff)"
    exit 1
  fi
done || { echo "FAIL: event_rateCU diverged from baseline"; exit 1; }
echo "PASS: event_rateCU matches baseline (within 0.1%)"

# chi2fittestCU — Fval within 0.5%
./build/src/app/chi2fittestCU 2>/dev/null | tail -9 > /tmp/chi2fittest_check.txt
GOLDEN_FVAL=$(grep '^Fval:' baselines/chi2fittest.golden | grep -oP '[\d.e+\-]+' | head -1)
CHECK_FVAL=$( grep '^Fval:' /tmp/chi2fittest_check.txt   | grep -oP '[\d.e+\-]+' | head -1)
if [ -z "$GOLDEN_FVAL" ] || [ -z "$CHECK_FVAL" ]; then
  echo "FAIL: could not parse Fval from output"
  exit 1
fi
REL_DIFF=$(python3 -c "print(abs($CHECK_FVAL - $GOLDEN_FVAL) / abs($GOLDEN_FVAL))")
if python3 -c "exit(0 if $REL_DIFF < 0.005 else 1)"; then
  echo "PASS: chi2fittestCU Fval within 0.5% (relative diff = $REL_DIFF)"
else
  echo "FAIL: chi2fittestCU Fval diverged (relative diff = $REL_DIFF > 0.5%)"
  echo "  golden: $GOLDEN_FVAL"
  echo "  check:  $CHECK_FVAL"
  exit 1
fi

echo "ALL BASELINES PASSED"
