"""Aggregate run results into paper tables (mean +/- SD over seeds) and paired significance tests.
usage: python aggregate.py results/  -> writes results/summary_*.csv and prints markdown tables
"""
import json, glob, sys, os, itertools, numpy as np, pandas as pd

R = sys.argv[1] if len(sys.argv) > 1 else "results"
rows = []
for f in glob.glob(f"{R}/*.jsonl*"):
    if "llm" in f or "bench" in f: continue
    for line in open(f):
        r = json.loads(line)
        for t, m in r["test"].items():
            rows.append(dict(tag=r["tag"], train=r["train"], test=t, model=r["model"].split("/")[-1], method=r["method"],
                             d=r["d"] if r["method"] in ("bn", "ae", "frozen_ae") else 0,
                             lam=r.get("lam_effective", 0.0), seed=r["seed"], run_id=r["run_id"],
                             val_f1=r.get("val_macro_f1"), time_s=r.get("time_s"), params_trainable=r.get("params_trainable"),
                             peak_mem_gb=r.get("peak_mem_gb"), **{k: m[k] for k in ["acc", "macro_f1", "mcc", "qwk", "ece", "mae", "adjacent_err_share"]}))
df = pd.DataFrame(rows)
df.to_csv(f"{R}/all_runs.csv", index=False)
keys = ["tag", "train", "test", "model", "method", "d", "lam"]
agg = df.groupby(keys).agg(n=("seed", "count"), **{f"{k}_{s}": (k, s) for k in ["macro_f1", "acc", "qwk", "mcc", "ece", "mae", "val_f1"] for s in ["mean", "std"]},
                           adj=("adjacent_err_share", "mean"), time_s=("time_s", "mean"), params=("params_trainable", "mean"), mem=("peak_mem_gb", "mean")).reset_index()
agg.to_csv(f"{R}/summary.csv", index=False)
pd.set_option("display.width", 250, "display.max_rows", 500)
fmt = lambda a, b: f"{100 * a:.1f}±{100 * (0 if pd.isna(b) else b):.1f}"
agg["F1"] = [fmt(a, b) for a, b in zip(agg.macro_f1_mean, agg.macro_f1_std)]
agg["Acc"] = [fmt(a, b) for a, b in zip(agg.acc_mean, agg.acc_std)]
agg["QWK"] = [fmt(a, b) for a, b in zip(agg.qwk_mean, agg.qwk_std)]
agg["valF1"] = [fmt(a, b) for a, b in zip(agg.val_f1_mean, agg.val_f1_std)]
print(agg[keys + ["n", "valF1", "F1", "Acc", "QWK", "ece_mean", "adj"]].to_string(index=False))


# ---- paired bootstrap: proposed vs reference, seed-matched predictions, macro-F1 of the seed-averaged decision quality
def labels(test):
    sys.path.insert(0, os.path.dirname(__file__))
    from datasets import load_dataset
    if test == "sst5": return np.array(load_dataset("SetFit/sst5")["test"]["label"])
    if test == "tweet": return np.array(load_dataset("cardiffnlp/tweet_eval", "sentiment")["test"]["label"])
    doms = ["Books", "Electronics", "Home_and_Kitchen", "Movies_and_TV"] if test == "amz_all" else [test[4:]]
    return pd.concat([pd.read_json(f"data/processed/amz_{x}_test.jsonl", lines=True) for x in doms]).label.values


def macro_f1(y, p, C):
    f = []
    for c in range(C):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        f.append(0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return np.mean(f)


def paired(run_ids_a, run_ids_b, test, B=2000, seed=0):
    y = labels(test); C = y.max() + 1; rng = np.random.default_rng(seed)
    PA = [np.load(f"{R}/preds/{r}__{test}.npy") for r in run_ids_a]
    PB = [np.load(f"{R}/preds/{r}__{test}.npy") for r in run_ids_b]
    obs = np.mean([macro_f1(y, p, C) for p in PA]) - np.mean([macro_f1(y, p, C) for p in PB])
    diffs = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        diffs.append(np.mean([macro_f1(y[i], p[i], C) for p in PA]) - np.mean([macro_f1(y[i], p[i], C) for p in PB]))
    diffs = np.array(diffs)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return obs, np.percentile(diffs, 2.5), np.percentile(diffs, 97.5), max(p, 1 / B)


if __name__ == "__main__" and len(sys.argv) > 2:
    # e.g. python aggregate.py results "E1|sst5|distilbert-base-uncased|ae|128|0.5" "E1|sst5|distilbert-base-uncased|ft_linear|0|0.0" sst5
    def ids(spec):
        tag, tr, mo, me, d, lam = spec.split("|")
        q = df[(df.tag == tag) & (df.train == tr) & (df.model == mo) & (df.method == me) & (df.d == int(d)) & (np.isclose(df.lam, float(lam)))]
        return sorted(set(q.run_id))
    a, b = ids(sys.argv[2]), ids(sys.argv[3])
    for t in sys.argv[4].split(","):
        print(t, "dF1=%.4f CI[%.4f, %.4f] p=%.4f" % paired(a, b, t), len(a), len(b))
