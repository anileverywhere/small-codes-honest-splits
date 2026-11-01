#!/bin/bash
# Paired bootstrap tests for the in-domain comparisons (Table II text). usage: bash scripts/sig_e1.sh
cd "$(dirname "$0")/.." || exit 1; export HF_HUB_DISABLE_PROGRESS_BARS=1
M=distilbert-base-uncased
for ds in sst5 tweet amz_all; do
  for pair in "ae|64|0.5 ft_linear|0|0.0" "bn|64|0.0 ft_linear|0|0.0" "ae|64|0.5 bn|64|0.0" "lora_linear|0|0.0 ft_linear|0|0.0" "frozen_ae|64|0.5 frozen_linear|0|0.0"; do
    set -- $pair
    echo "$ds $1 vs $2: $(python src/aggregate.py results "E1|$ds|$M|$1" "E1|$ds|$M|$2" $ds 2>/dev/null | tail -1)"
  done
done
echo SIG_E1_DONE
