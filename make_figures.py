#!/usr/bin/env python3
"""make_figures.py -- regenerates the five figures of the paper from the data
shipped in data/ (no hard-coded numbers).  Output goes to ../paper/figs/.

  fig_4_1_displacement_scatter.png   Figure 4.1  (reference_8p4e9.csv)
  fig_5_1_euler_loading.png          Figure 5.1  (floors_all_*, logP_pairs_*)
  fig_5_2_decomposition.png          Figure 5.2  (floors_all_8p4e9, logP_pairs_8p4e9, zsum_pairs_8p4e9)
  fig_6_1_euler_deficit.png          Figure 6.1  (logP_pairs_8p4e9, logP_ctrl_8p4e9)
  fig_7_1_far_field_law.png          Figure 7.1  (lehmer_rows_*)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("RH_DATA", os.path.join(HERE, "data"))
OUT = os.environ.get("FIG_DIR", os.path.join(HERE, "..", "paper", "figs"))
os.makedirs(OUT, exist_ok=True)
XS = [10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7]
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def ols(Xm, y):
    A = np.column_stack([Xm, np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    return coef, 1.0 - float(res @ res / np.sum((y - y.mean()) ** 2))


# ------------------------------------------------------------ Figure 4.1
d = np.genfromtxt(os.path.join(DATA, "reference_8p4e9.csv"), delimiter=",", names=True)
fig, ax = plt.subplots(figsize=(6.2, 4.6))
lim = 1.1 * max(np.abs(d["delta"]).max(), np.abs(d["prediction"]).max())
ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="$y=x$")
ax.scatter(d["prediction"], d["delta"], s=28, zorder=3)
ax.set_xlabel(r"prediction $(g^2/8)\,\lambda_{\mathrm{far}}$")
ax.set_ylabel(r"measured displacement $\delta$")
ax.set_title(r"Displacement law, 15 tightest pairs at $t\approx8.436\times10^{9}$")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_4_1_displacement_scatter.png"), dpi=180)
plt.close(fig)

# ------------------------------------------------------------ Figure 5.1
f8 = np.load(os.path.join(DATA, "floors_all_8p4e9.npy"))
lp8 = np.load(os.path.join(DATA, "logP_pairs_8p4e9.npy"))
f2 = np.load(os.path.join(DATA, "floors_all_2M.npy"))
lp2 = np.load(os.path.join(DATA, "logP_pairs_2M.npy"))
rows = {}
for name, f, lp in (("8.4e9", f8, lp8), ("2M", f2, lp2)):
    lg, lf = np.log(f[:, 2]), np.log(f[:, 5])
    rows[name] = [ols(np.column_stack([lg, lp[:, j]]), lf) for j in range(6)]
fig, ax = plt.subplots(figsize=(6.2, 4.4))
ax2 = ax.twinx()
lx = np.log10(XS)
for name, mk, col in (("8.4e9", "o", "C0"), ("2M", "s", "C1")):
    b = [r[0][1] for r in rows[name]]
    r2 = [r[1] for r in rows[name]]
    ax.plot(lx, b, mk + "-", color=col, label=r"$b$, $t\approx%s$" % ("8.4\\times10^{9}" if name == "8.4e9" else "1.2\\times10^{6}"))
    ax2.plot(lx, r2, mk + ":", color=col, mfc="none", label=r"$R^2$, %s" % name)
ax.axhline(1.0, color="k", ls="--", lw=0.8)
ax.set_xlabel(r"$\log_{10} X$")
ax.set_ylabel(r"Euler loading $b$")
ax2.set_ylabel(r"$R^2$")
ax.set_ylim(0.65, 1.02)
ax2.set_ylim(0.92, 0.99)
ax2.grid(False)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)
ax.set_title(r"Euler regression $\log F = a\log g + b\log|P_X| + c$")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_5_1_euler_loading.png"), dpi=180)
plt.close(fig)

# ------------------------------------------------------------ Figure 5.2
zs8 = np.load(os.path.join(DATA, "zsum_pairs_8p4e9.npy"))
lf8 = np.log(f8[:, 5])
j = 5
pred = lp8[:, j] + zs8[1, :, j]
R = lf8 - pred
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.2))
a1.plot([pred.min(), pred.max()], [pred.min(), pred.max()], "k--", lw=0.8)
a1.scatter(pred, lf8, s=5, alpha=0.5)
a1.set_xlabel(r"$\log|P_X(t_0)| + S_{1000}(t_0;X)$,  $X=10^{7}$")
a1.set_ylabel(r"$\log F$ (measured floor)")
a1.set_title("Decomposition of the floor, 2,736 pairs")
a2.hist(R, bins=60, color="C0")
a2.set_xlabel(r"residual $R_{1000}(t_0;10^{7})$")
a2.set_ylabel("pairs")
a2.set_title("median %+.4f, s.d. %.4f" % (np.median(R), R.std()))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_5_2_decomposition.png"), dpi=180)
plt.close(fig)

# ------------------------------------------------------------ Figure 6.1
lpc = np.load(os.path.join(DATA, "logP_ctrl_8p4e9.npy"))
j = 4  # X = 1e6
fig, ax = plt.subplots(figsize=(6.4, 4.2))
bins = np.linspace(min(lp8[:, j].min(), lpc[:, j].min()), max(lp8[:, j].max(), lpc[:, j].max()), 50)
ax.hist(lp8[:, j], bins=bins, density=True, alpha=0.75, color="C0", label="tight-pair midpoints (2,736)")
ax.hist(lpc[:, j], bins=bins, density=True, alpha=0.55, color="C1", label="control mid-gaps (300)")
ax.axvline(lp8[:, j].mean(), color="C0", lw=1.5)
ax.axvline(lpc[:, j].mean(), color="C1", lw=1.5)
ax.set_xlabel(r"$\log|P_X(\frac{1}{2}+it)|$,  $X=10^{6}$")
ax.set_ylabel("density")
ax.set_title(r"Euler deficit at Lehmer midpoints, $t\approx8.436\times10^{9}$")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_6_1_euler_deficit.png"), dpi=180)
plt.close(fig)

# ------------------------------------------------------------ Figure 7.1
dens = {"100k": 1.494, "2M": 1.935, "8p4e9": 3.3451, "1e12": 3.896, "1e21": 7.095}
pts = []
for k, dv in dens.items():
    a = np.load(os.path.join(DATA, "lehmer_rows_%s.npy" % k))
    pts.append((dv ** 2, float(a[:, 5].mean()), k))
pts.sort()
fig, ax = plt.subplots(figsize=(6.0, 4.4))
xx = np.array([p[0] for p in pts])
yy = np.array([p[1] for p in pts])
xs = np.logspace(np.log10(xx.min() / 1.5), np.log10(xx.max() * 1.5), 50)
ax.plot(xs, 4 * np.pi ** 2 / 15 * xs, "k--", lw=0.9, label=r"$(4\pi^2/15)\,\mathrm{dens}^2$")
ax.scatter(xx, yy, s=36, zorder=3)
for x0, y0, k in pts:
    ax.annotate(k, (x0, y0), textcoords="offset points", xytext=(6, -10), fontsize=8)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\mathrm{dens}^2$")
ax.set_ylabel(r"mean far field of tight pairs")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_7_1_far_field_law.png"), dpi=180)
plt.close(fig)
print("figures written to", OUT)
