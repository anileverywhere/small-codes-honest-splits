"""Build the evaluation corpora used in the paper.

Amazon-4D: four product domains from McAuley-Lab/Amazon-Reviews-2023
(Books, Electronics, Home_and_Kitchen, Movies_and_TV), labelled by star rating
(1-5 -> 5 ordinal classes). Leakage controls, which the original draft lacked:
  * exact + normalised-text deduplication inside each domain
  * temporal split: test reviews are strictly later than every train/val review
  * product-disjoint test: test parent_asins never appear in train/val
  * class-balanced sampling so macro-F1 and accuracy are comparable across domains
SST-5 and TweetEval-sentiment use their official splits.
"""
import json, re, hashlib, random, os, sys
import pandas as pd

RAW = sys.argv[1] if len(sys.argv) > 1 else "data/amazon"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/processed"
DOMAINS = ["Books", "Electronics", "Home_and_Kitchen", "Movies_and_TV"]
N_TRAIN, N_VAL, N_TEST = 2000, 300, 600  # per star class -> 10k / 1.5k / 3k per domain
SEED = 13
os.makedirs(OUT, exist_ok=True)


def norm(t):
    t = t.lower()
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


stats = {}
for dom in DOMAINS:
    rows = []
    with open(f"{RAW}/{dom}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            text = (r.get("title", "") + ". " + r.get("text", "")).strip()
            text = re.sub(r"<br\s*/?>", " ", text)
            if not (30 <= len(text) <= 3000):
                continue
            rows.append((text, int(round(r["rating"])) - 1, r["timestamp"], r["parent_asin"]))
    df = pd.DataFrame(rows, columns=["text", "label", "ts", "asin"])
    n_raw = len(df)
    df["h"] = df.text.map(lambda t: hashlib.md5(norm(t).encode()).hexdigest())
    df = df.drop_duplicates("h")
    n_dedup = len(df)
    df = df.sort_values("ts").reset_index(drop=True)
    cut = df.ts.quantile(0.75)
    pool_tr, pool_te = df[df.ts <= cut], df[df.ts > cut]
    pool_te = pool_te[~pool_te.asin.isin(set(pool_tr.asin))]

    def take(pool, n, seed):
        parts = []
        for c in range(5):
            p = pool[pool.label == c]
            parts.append(p.sample(n=min(n, len(p)), random_state=seed))
        return pd.concat(parts).sample(frac=1, random_state=seed)

    te = take(pool_te, N_TEST, SEED)
    trv = take(pool_tr, N_TRAIN + N_VAL, SEED)
    va = pd.concat([trv[trv.label == c].iloc[:N_VAL] for c in range(5)])
    tr = trv.drop(va.index)
    assert set(tr.h).isdisjoint(te.h) and set(va.h).isdisjoint(te.h)
    for name, part in [("train", tr), ("val", va), ("test", te)]:
        part[["text", "label"]].assign(domain=dom).to_json(f"{OUT}/amz_{dom}_{name}.jsonl", orient="records", lines=True)
    stats[dom] = dict(raw=n_raw, dedup=n_dedup, dup_removed=n_raw - n_dedup, train=len(tr), val=len(va), test=len(te),
                      train_counts=tr.label.value_counts().sort_index().tolist(),
                      test_counts=te.label.value_counts().sort_index().tolist(),
                      cutoff_ts=int(cut), test_products_overlap=int(len(set(te.asin) & set(tr.asin))))
    print(dom, stats[dom], flush=True)

json.dump(stats, open(f"{OUT}/amazon_stats.json", "w"), indent=1)
