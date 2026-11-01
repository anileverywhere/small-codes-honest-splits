"""Fig. 1 drawn at native IEEE column width (3.45 in) so text prints at ~6.2 pt."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.size": 6.2, "mathtext.fontset": "dejavusans", "pdf.fonttype": 42, "ps.fonttype": 42})  # TrueType: IEEE PDF eXpress rejects Type 3
fig, ax = plt.subplots(figsize=(3.45, 2.15)); ax.set_xlim(0, 100); ax.set_ylim(0, 66); ax.axis("off")


def box(x, y, w, h, t, fc, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.5", fc=fc, ec="#333", lw=0.7, ls=ls))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center")


def arr(a, b, dashed=False):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=6, lw=0.7,
                                 ls="--" if dashed else "-", color="#777" if dashed else "#222"))


ax.text(50, 63, "solid: inference path      dashed: training only", fontsize=6, style="italic", color="#444", ha="center", va="center")
# row 1: deployed path
box(0, 44, 16, 11, "Input text\n(128 tok.)", "#f2f2f2")
box(20, 44, 22, 11, r"DistilBERT $f_\theta$" + "\n(fine-tuned)", "#dbe9f6")
box(47, 44, 24, 11, r"Bottleneck $\mathbf{h}$" + "\n" + r"($d$-dim, GELU)", "#fde5c8")
box(76, 44, 23, 11, r"Classifier $V$" + "\n" + r"$\to\mathcal{L}_{cls}$ (CE)", "#d9f0d3")
arr((16.5, 49.5), (19.5, 49.5)); arr((42.5, 49.5), (46.5, 49.5)); arr((71.5, 49.5), (75.5, 49.5))
# row 2: training-only decoder
box(47, 24, 24, 11, r"Decoder $W'$" + "\n" + r"$\hat{\mathbf{x}}\in\mathbb{R}^{768}$", "#eeeeee", "--")
box(76, 24, 23, 11, r"$\lambda\,\mathcal{L}_{rec}$ (MSE)", "#ffffff", "--")
arr((59, 43.5), (59, 35.5), True); arr((71.5, 29.5), (75.5, 29.5), True)
# row 3: frozen teacher
box(16, 4, 26, 11, "Frozen pretrained\n" + r"DistilBERT $f_{\theta_0}$", "#eeeeee", "--")
box(47, 4, 24, 11, r"target $\mathrm{LN}(\bar{\mathbf{x}})$", "#eeeeee", "--")
arr((6, 43.5), (15.5, 10), True); arr((42.5, 9.5), (46.5, 9.5), True); arr((71.5, 9.5), (87, 23.5), True)
plt.savefig("figures/architecture.pdf", bbox_inches="tight", pad_inches=0.02)
plt.savefig("figures/architecture.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
