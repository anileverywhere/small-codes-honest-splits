"""Controlled duplicate injection: copy a fraction r of each class's TEST reviews into TRAIN
(replacing the same number of train reviews of that class), keeping sizes and class balance.
Simulates a retweet/near-duplicate-heavy corpus split at random. Test and val are unchanged.
"""
import sys, pandas as pd
D, DOMS, SEED = sys.argv[1], ["Books", "Electronics", "Home_and_Kitchen", "Movies_and_TV"], 13
for r in [0.1, 0.5]:
    for dom in DOMS:
        tr = pd.read_json(f"{D}/amz_{dom}_train.jsonl", lines=True)
        te = pd.read_json(f"{D}/amz_{dom}_test.jsonl", lines=True)
        va = pd.read_json(f"{D}/amz_{dom}_val.jsonl", lines=True)
        parts = []
        for c in range(5):
            tc, ec = tr[tr.label == c], te[te.label == c]
            k = int(round(r * len(ec)))
            parts += [tc.sample(n=len(tc) - k, random_state=SEED), ec.sample(n=k, random_state=SEED)]
        new = pd.concat(parts).sample(frac=1, random_state=SEED)
        tag = f"inj{int(r * 100)}"
        new.to_json(f"{D}/amz_{tag}_{dom}_train.jsonl", orient="records", lines=True)
        va.to_json(f"{D}/amz_{tag}_{dom}_val.jsonl", orient="records", lines=True)
        te.to_json(f"{D}/amz_{tag}_{dom}_test.jsonl", orient="records", lines=True)
        print(tag, dom, len(new), "test-in-train:", round(te.text.isin(set(new.text)).mean(), 3))
