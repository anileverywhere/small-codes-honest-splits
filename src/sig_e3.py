"""Paired bootstrap for E3: system A vs B on pooled out-of-domain predictions (and per pair).
Seed-matched predictions; statistic = mean over seeds of macro-F1. 2000 resamples.
"""
import json, glob, numpy as np, pandas as pd
R, B = "results", 2000
D = ["Books", "Electronics", "Home_and_Kitchen", "Movies_and_TV"]
Y = {d: pd.read_json(f"data/processed/amz_{d}_test.jsonl", lines=True).label.values for d in D}


def f1(y, p):
    s = 0.0
    for c in range(5):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        s += 0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
    return s / 5


def rid(src, spec, s):
    m, d, l = spec
    return f"E3_amz_{src}_distilbert-base-uncased_{m}_d{d}_l{l}_s{s}"


SYS = {"ft_linear": ("ft_linear", 128, 0.5), "bn": ("bn", 64, 0.5), "ae0.5": ("ae", 64, 0.5), "ae2": ("ae", 64, 2.0),
       "frozen_lin": ("frozen_linear", 128, 0.5), "frozen_ae": ("frozen_ae", 64, 0.5)}


def pooled(src, name, tgts):
    ys = np.concatenate([Y[t] for t in tgts])
    ps = [np.concatenate([np.load(f"{R}/preds/{rid(src, SYS[name], s)}__amz_{t}.npy") for t in tgts]) for s in [1, 2, 3]]
    return ys, ps


def boot(a, b, srcs, ood=True, seed=0):
    Ys, PA, PB = [], [], []
    for src in srcs:
        tg = [t for t in D if t != src] if ood else [src]
        y, pa = pooled(src, a, tg); _, pb = pooled(src, b, tg)
        Ys.append(y); PA.append(pa); PB.append(pb)
    y = np.concatenate(Ys); pa = [np.concatenate([x[s] for x in PA]) for s in range(3)]; pb = [np.concatenate([x[s] for x in PB]) for s in range(3)]
    obs = np.mean([f1(y, p) for p in pa]) - np.mean([f1(y, p) for p in pb])
    rng = np.random.default_rng(seed); diffs = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        diffs.append(np.mean([f1(y[i], p[i]) for p in pa]) - np.mean([f1(y[i], p[i]) for p in pb]))
    diffs = np.array(diffs); p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return dict(diff=100 * obs, lo=100 * np.percentile(diffs, 2.5), hi=100 * np.percentile(diffs, 97.5), p=max(p, 1 / B))


out = {}
for a, b in [("ae2", "ft_linear"), ("ae0.5", "ft_linear"), ("bn", "ft_linear"), ("ae2", "bn"), ("ae0.5", "bn"), ("frozen_ae", "frozen_lin")]:
    out[f"{a}_vs_{b}_OOD_all"] = boot(a, b, D, True)
    out[f"{a}_vs_{b}_IND_all"] = boot(a, b, D, False)
    print(a, b, "OOD", {k: round(v, 3) for k, v in out[f"{a}_vs_{b}_OOD_all"].items()}, "IND", {k: round(v, 3) for k, v in out[f"{a}_vs_{b}_IND_all"].items()}, flush=True)
for src in D:
    out[f"ae2_vs_ft_linear_OOD_{src}"] = r = boot("ae2", "ft_linear", [src], True)
    print("ae2 vs ft_linear OOD from", src, {k: round(v, 3) for k, v in r.items()}, flush=True)
json.dump(out, open(f"{R}/sig_e3.json", "w"), indent=1)
