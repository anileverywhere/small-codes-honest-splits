#!/bin/bash
# Run a job file (one exp.py argument line per job) with N parallel workers; finished jobs are skipped on re-run.
# usage (from the repo root): bash scripts/run_queue.sh jobs/jobs_q1.txt 3
# Note: never run a base-size encoder job (jobs_q2) alongside other jobs on a 16 GB GPU.
cd "$(dirname "$0")/.." || exit 1
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p logs results/preds
run_one() { key=$(echo "$1" | md5sum | cut -c1-12); if grep -q '^DONE' logs/$key.log 2>/dev/null; then return; fi
  echo "$1" > logs/$key.log; python src/exp.py $1 >> logs/$key.log 2>&1; }
export -f run_one
grep -v '^\s*#' "$1" | grep -v '^\s*$' | xargs -P "${2:-2}" -I{} bash -c 'run_one "{}"'
echo QUEUE_FINISHED "$1"
