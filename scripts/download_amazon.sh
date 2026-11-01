#!/bin/bash
# Download the raw Amazon Reviews 2023 files used to build Amazon-4D: the FIRST 300,000 lines of each category file
# (McAuley-Lab/Amazon-Reviews-2023 on Hugging Face). Then build the splits:
#   bash scripts/download_amazon.sh && python src/prep_data.py data/amazon data/processed
#   python src/prep_leak.py data/amazon data/processed && python src/prep_inject.py data/processed
# Check your splits against results/summary/split_checksums.json with scripts/check_splits.py.
set -e
cd "$(dirname "$0")/.."
mkdir -p data/amazon
for d in Books Electronics Home_and_Kitchen Movies_and_TV; do
  echo "downloading $d (first 300k reviews)"
  curl -sL "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/$d.jsonl" \
    | head -n 300000 > "data/amazon/$d.jsonl" || true
  test "$(wc -l < data/amazon/$d.jsonl)" -eq 300000
done
