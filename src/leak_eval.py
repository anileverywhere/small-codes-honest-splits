"""Leakage analysis on a common, uncontaminated test subset.
Random-split training sets (rand/randdd) contain 8-12% of the clean test reviews, so scoring them on the full
clean test is itself leaky. For each domain we keep only the clean-test reviews that occur in neither the rand
nor the randdd training set ("unseen"), and score clean/rand/randdd there (3 seeds; paired bootstrap vs clean).
inj conditions: the clean test is their own test by design; we report F1 on the leaked and the non-leaked items.
Also reports rand/randdd own-test F1.
usage: python leak_eval.py results > results/leak.txt   (writes results/leak.json)
"""
import sys, json, numpy as np, pandas as pd
from sklearn.metrics import f1_score

R = sys.argv[1] if len(sys.argv) > 1 else "results"
D = "data/processed"
DOMS = ["Books", "Electronics", "Home_and_Kitchen", "Movies_and_TV"]
COND = ["rand", "randdd", "inj10", "inj50"]
runs = pd.read_csv(f"{R}/all_runs.csv")
runs = runs[runs.method == "ft_linear"]
rd = lambda f: pd.read_json(f"{D}/{f}.jsonl", lines=True)


def ids(tag, train, test):
    return sorted(set(runs[(runs.tag == tag) & (runs.train == train) & (runs.test == test)].run_id))


def f1(y, p): return f1_score(y, p, average="macro")


out, B, rng = {}, 2000, np.random.default_rng(0)
for dom in DOMS:
    ct = rd(f"amz_{dom}_test"); y = ct.label.values
    # exclude clean-test items that occur in the rand/randdd train sets (inj conditions contain test items by design)
    seen = set().union(*[set(rd(f"amz_{c}_{dom}_train").text) for c in ["rand", "randdd"]])
    unseen = ~ct.text.isin(seen).values
    res = {"n_test": len(ct), "n_unseen": int(unseen.sum())}
    P = {"clean": [np.load(f"{R}/preds/{r}__amz_{dom}.npy") for r in ids("E3", f"amz_{dom}", f"amz_{dom}")]}
    for c in COND:
        P[c] = [np.load(f"{R}/preds/{r}__amz_{dom}.npy") for r in ids("L", f"amz_{c}_{dom}", f"amz_{dom}")]
    for c in ["rand", "randdd"]:  # own (random) test split
        te = rd(f"amz_{c}_{dom}_test")
        own = [np.load(f"{R}/preds/{r}__amz_{c}_{dom}.npy") for r in ids("L", f"amz_{c}_{dom}", f"amz_{c}_{dom}")]
        res[f"{c}_own"] = float(np.mean([f1(te.label.values, p) for p in own]))
    for c, ps in P.items():
        assert len(ps) == 3, (dom, c, len(ps))
        res[f"{c}_full"] = float(np.mean([f1(y, p) for p in ps]))
        res[f"{c}_unseen"] = float(np.mean([f1(y[unseen], p[unseen]) for p in ps]))
    for c in ["inj10", "inj50"]:
        leaked = ct.text.isin(set(rd(f"amz_{c}_{dom}_train").text)).values
        res[f"{c}_leaked_items_f1"] = float(np.mean([f1(y[leaked], p[leaked]) for p in P[c]]))
        res[f"{c}_nonleaked_items_f1"] = float(np.mean([f1(y[~leaked], p[~leaked]) for p in P[c]]))
        res[f"clean_on_{c}_leaked_items_f1"] = float(np.mean([f1(y[leaked], p[leaked]) for p in P["clean"]]))
        res[f"{c}_leaked_frac"] = float(leaked.mean())
    # paired bootstrap on the unseen subset: condition vs clean
    yu = y[unseen]; n = len(yu)
    for c in ["rand", "randdd"]:
        diffs = []
        for _ in range(B):
            i = rng.integers(0, n, n)
            diffs.append(np.mean([f1(yu[i], p[unseen][i]) for p in P[c]]) - np.mean([f1(yu[i], p[unseen][i]) for p in P["clean"]]))
        diffs = np.array(diffs)
        res[f"{c}_vs_clean_unseen"] = dict(d=res[f"{c}_unseen"] - res["clean_unseen"], lo=float(np.percentile(diffs, 2.5)),
                                           hi=float(np.percentile(diffs, 97.5)), p=float(max(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1 / B)))
    out[dom] = res
    print(dom, json.dumps(res), flush=True)

keys = [k for k in out[DOMS[0]] if isinstance(out[DOMS[0]][k], float)]
out["mean"] = {k: float(np.mean([out[d][k] for d in DOMS])) for k in keys}
print("MEAN", json.dumps(out["mean"]))
json.dump(out, open(f"{R}/leak.json", "w"), indent=1)
