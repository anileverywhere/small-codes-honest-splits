# Compact Bottleneck Classifiers on DistilBERT

Code, data-split recipes and full results for the paper

> **Compact Bottleneck Classifiers on DistilBERT for Cross-Domain Sentiment Analysis**
> Jyothi K. S., G. Anil Kumar, Shantala C. P.
> Dept. of CSE, Channabasaveshwara Institute of Technology, Gubbi, Tumkur, Karnataka, India

## What the paper finds

- **Compression is free in-domain.** The [CLS] representation of fine-tuned DistilBERT can be compressed to d=16 (64 bytes) with no loss of SST-5 macro-F1.
- **The bottleneck helps out of domain; reconstruction does not.**
  - In single-source transfer across four Amazon domains, a d=64 bottleneck gains +0.3 to +0.5 out-of-domain macro-F1 over a linear head (paired bootstrap, p < 0.02).
  - The reconstruction term (decoder to the frozen pretrained embedding) adds nothing measurable on top.
- **Random splits inflate scores.**
  - Random splits of the same reviews report about 4 points more macro-F1, without any significant gain on a common unseen test subset.
  - When 50% of test texts are duplicated in training, the inflation reaches 10.7 points.

## Repository layout

| Path | Contents |
|---|---|
| `src/exp.py` | One training/evaluation run: TF-IDF, frozen and fine-tuned linear heads, bottleneck (BN), reconstruction-regularised bottleneck (RRB, `--method ae`), LoRA; DistilBERT, RoBERTa, DeBERTaV3, ModernBERT |
| `src/prep_data.py` | Builds **Amazon-4D**: deduplicated, temporal 75% cut-off, product-disjoint test set, class-balanced 2000/300/600 reviews per class |
| `src/prep_leak.py`, `src/prep_inject.py` | Leakage-experiment splits: random, random after deduplication, and 10% / 50% injected test duplicates |
| `src/aggregate.py` | Aggregates runs (mean ± SD over seeds) and runs paired bootstrap tests |
| `src/sig_e3.py`, `scripts/sig_e1.sh` | Significance tests for Table IV and Section V-B |
| `src/leak_eval.py` | Leakage analysis on the common uncontaminated test subset (Table V) |
| `src/bench.py`, `src/llm_zeroshot.py` | Inference cost and zero-shot Qwen2.5 baselines (Table II) |
| `src/make_figs.py`, `src/fig_arch.py` | Figures |
| `jobs/*.txt` | The exact argument lines of every run in the paper (one run per line) |
| `results/{A,E1,E3,L}.jsonl`, `results/llm.jsonl` | One JSON record per run: config, validation and test metrics, time, memory |
| `results/preds/` | Test-set predictions of every run (`.npy`), used by the significance tests |
| `results/summary/` | Aggregated tables, bootstrap results, benchmark, leakage analysis, dataset statistics, split checksums |

## Setup

Tested with Python 3.11, CUDA GPU (RTX A4000, 16 GB), on Ubuntu 22.04.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce

All commands run from the repository root.

**1. Data.** SST-5 (`SetFit/sst5`) and TweetEval (`cardiffnlp/tweet_eval`) download automatically. Amazon-4D is rebuilt from Amazon Reviews 2023 (McAuley-Lab). We redistribute no review text, only the recipe and checksums:

```bash
bash scripts/download_amazon.sh                   # first 300k reviews per category
python src/prep_data.py data/amazon data/processed  # Amazon-4D
python src/prep_leak.py data/amazon data/processed  # random / random-dedup splits
python src/prep_inject.py data/processed            # 10% / 50% injected duplicates
python scripts/check_splits.py                      # compare with results/summary/split_checksums.json
```

**2. Runs.**

| Paper item | Jobs | Command |
|---|---|---|
| Fig. 2 sweep (tag A), Table II DistilBERT rows (E1) | `jobs/jobs_q1.txt`, `jobs/jobs_add1.txt` | `bash scripts/run_queue.sh jobs/jobs_q1.txt 3` |
| Table II base-size encoders (E1) | `jobs/jobs_q2.txt` | `bash scripts/run_queue.sh jobs/jobs_q2.txt 1` |
| Table IV cross-domain transfer (E3) | `jobs/jobs_add1.txt` | `bash scripts/run_queue.sh jobs/jobs_add1.txt 3` |
| Table V leakage (L) | `jobs/jobs_add2.txt` | `bash scripts/run_queue.sh jobs/jobs_add2.txt 2` |
| Table II cost and LLM rows | | `python src/bench.py results/bench.json`; `python src/llm_zeroshot.py Qwen/Qwen2.5-3B-Instruct sst5,tweet,amz_all results/llm.jsonl` |

Run a base-size encoder job (`jobs_q2`) alone on a 16 GB GPU. Running it next to other jobs can exhaust GPU memory.

**3. Analysis.**

```bash
python src/aggregate.py results > results/agg.txt   # tables, mean ± SD
bash scripts/sig_e1.sh                               # in-domain paired bootstrap
python src/sig_e3.py                                 # cross-domain paired bootstrap
python src/leak_eval.py results                      # leakage analysis
python src/make_figs.py results figures
```

The committed `results/` already contain every run of the paper. Step 3 works without re-training once the data of step 1 exists, because the tests need the test labels.

## Protocol notes

- Hyper-parameters (d, λ) were selected on the **SST-5 validation set only** and then fixed for every other corpus. Test sets were never used for selection.
- Every configuration was run with 3 seeds. Paired bootstrap tests use 2,000 test-set resamples.
- DeBERTaV3 was trained and benchmarked in fp32, because it diverges in bf16/fp16.

## License

Code: MIT (see `LICENSE`). Amazon Reviews 2023, SST-5 and TweetEval keep their original licenses and terms of use.

## Citation

```bibtex
@inproceedings{jyothi2027compact,
  author    = {Jyothi K. S. and G. Anil Kumar and Shantala C. P.},
  title     = {Compact Bottleneck Classifiers on {DistilBERT} for Cross-Domain Sentiment Analysis},
  booktitle = {Proc. ICRCICN},
  year      = {2027}
}
```
