"""Leakage experiment data: naive random splits vs the leakage-controlled Amazon-4D split.

For each domain, build splits with exactly the clean split's sizes and class balance
(train 2000/class, val 300/class, test 600/class):
  rand   : random split of the RAW reviews (no dedup, any time, products may overlap)
  randdd : random split after exact/normalised dedup (products and time still overlap)
Also reports, for each split: train-test exact-duplicate rate and product (parent_asin) overlap.
"""
import json, re, hashlib, os, sys
import pandas as pd

RAW, OUT = sys.argv[1], sys.argv[2]
DOMAINS = ["Books", "Electronics", "Home_and_Kitchen", "Movies_and_TV"]
NTR, NVA, NTE, SEED = 2000, 300, 600, 13


def norm(t):
    t = re.sub(r"<[^>]+>|https?://\S+", " ", t.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", t)).strip()


stats = {}
for dom in DOMAINS:
    rows = []
    for line in open(f"{RAW}/{dom}.jsonl"):
        r = json.loads(line)
        text = re.sub(r"<br\s*/?>", " ", (r.get("title", "") + ". " + r.get("text", "")).strip())
        if 30 <= len(text) <= 3000:
            rows.append((text, int(round(r["rating"])) - 1, r["parent_asin"]))
    df = pd.DataFrame(rows, columns=["text", "label", "asin"])
    df["h"] = df.text.map(lambda t: hashlib.md5(norm(t).encode()).hexdigest())
    for name, pool in [("rand", df), ("randdd", df.drop_duplicates("h"))]:
        parts = [pool[pool.label == c].sample(n=NTR + NVA + NTE, random_state=SEED) for c in range(5)]
        tr = pd.concat([p.iloc[:NTR] for p in parts]); va = pd.concat([p.iloc[NTR:NTR + NVA] for p in parts])
        te = pd.concat([p.iloc[NTR + NVA:] for p in parts])
        for split, part in [("train", tr), ("val", va), ("test", te)]:
            part.sample(frac=1, random_state=SEED)[["text", "label"]].assign(domain=dom).to_json(
                f"{OUT}/amz_{name}_{dom}_{split}.jsonl", orient="records", lines=True)
        trh, tra = set(tr.h), set(tr.asin)
        stats[f"{name}_{dom}"] = dict(test_dup_in_train=float(te.h.isin(trh).mean()),
                                      test_product_in_train=float(te.asin.isin(tra).mean()))
        print(name, dom, stats[f"{name}_{dom}"], flush=True)
json.dump(stats, open(f"{OUT}/leak_stats.json", "w"), indent=1)
