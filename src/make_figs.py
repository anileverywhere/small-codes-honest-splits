"""Build paper figures/tables from results pulled from the server.
usage: python make_figs.py RESULTS_DIR OUT_FIG_DIR
"""
import sys, json, os, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(f"{R}/all_runs.csv")
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})  # TrueType: IEEE PDF eXpress rejects Type 3
PAL = ["#1b6ca8", "#e07b39", "#3a9d5d", "#b8446b", "#7d5ba6", "#8c8c8c"]

# ---- Fig: d x lambda sweep on SST-5 (val and test macro-F1)
A = df[(df.tag == "A")].copy(); A.loc[A.method == "bn", "lam"] = 0.0
if len(A):
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.95), sharey=True)
    for ax, col, title in [(axs[0], "val_f1", "SST-5 validation"), (axs[1], "macro_f1", "SST-5 test")]:
        for i, lam in enumerate(sorted(A.lam.unique())):
            g = A[A.lam == lam].groupby("d")[col].agg(["mean", "std"]).reset_index()
            ax.errorbar(g.d, 100 * g["mean"], yerr=100 * g["std"], marker="o", ms=3, lw=1, capsize=2, color=PAL[i], label=f"$\\lambda$={lam:g}")
        ax.set_xscale("log", base=2); ax.set_xticks([16, 32, 64, 128, 256]); ax.set_xticklabels([16, 32, 64, 128, 256])
        ax.set_xlabel("latent width $d$"); ax.set_ylabel("macro-F1 (%)" if ax is axs[0] else ""); ax.set_title(title, fontsize=7)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6, ncol=5, frameon=False, loc="upper center", handlelength=1.2, columnspacing=0.9)
    plt.tight_layout(rect=(0, 0, 1, 0.9)); plt.savefig(f"{OUT}/sweep.pdf"); plt.close()

# ---- Table: E3 cross-domain transfer (macro-F1), mean over seeds, averaged over target domains
E3 = df[df.tag == "E3"].copy()
if len(E3):
    E3["name"] = E3.method + E3.apply(lambda r: f"-d{r.d}-l{r.lam:g}" if r.method in ("ae", "frozen_ae", "bn") else "", axis=1)
    E3["indomain"] = E3.train == E3.test
    piv = E3.groupby(["name", "train", "test"]).macro_f1.mean().unstack("test")
    piv.to_csv(f"{R}/e3_matrix.csv")
    s = E3.groupby(["name", "indomain", "seed"]).macro_f1.mean().unstack("indomain")
    s.columns = ["ood", "ind"]; s["gap"] = s.ind - s.ood
    summ = s.groupby("name").agg(["mean", "std"]); summ.to_csv(f"{R}/e3_summary.csv"); print(summ.round(4))
    q = E3.groupby(["name", "indomain", "seed"]).qwk.mean().unstack("indomain"); q.columns = ["ood", "ind"]
    print(q.groupby("name").agg(["mean", "std"]).round(4))

# ---- Fig: accuracy-cost frontier (CPU latency vs SST-5 / Amazon macro-F1)
bpath = f"{R}/bench.json"
if os.path.exists(bpath):
    B = pd.DataFrame(json.load(open(bpath))["rows"]); B.to_csv(f"{R}/bench.csv", index=False); print(B.round(3))
