"""V4 -- the detector on certified pairs (Section 3.6, Proposition 3.12)."""
import numpy as np

from tightpairs import geometry as G
from tightpairs.constants import W
from tightpairs.report import close, equal, below, report, section


def run(ctx):
    section("V4: the sweep of the transposed detector (Section 3.6)")
    det = ctx.csv("detector_sweep_8p4e9.csv")
    equal("V4 sweep rows", int(det.size), 2736)
    equal("V4 sweep sign changes", int(det["n_sign_changes"].max()), 0)
    close("V4 largest Ghat over all pairs (-3.93e3)", det["max_Ghat"].max(), -3.93e3, 10.0, "%.2f", "%.2f")
    k = int(np.argmax(det["max_Ghat"]))
    close("V4 ... attained at a pair with g = 0.0448", det["g"][k], 0.0448, 1e-4, "%.5f", "%.4f")
    equal("V4 ... among the ten widest gaps", bool(k in np.argsort(det["g"])[-10:]), True)
    hr = det["H"] / (det["g"] / 2.0)
    close("V4 min H/(g/2) (3.49)", hr.min(), 3.49, 0.01, "%.3f", "%.2f")
    close("V4 median H/(g/2) (16.0)", np.median(hr), 16.0, 0.05, "%.2f", "%.1f")
    # condition (3.23)
    r8 = ctx.rows["8p4e9"]
    zl, idx8 = ctx.zl, ctx.idx8
    g8 = r8[:, 2]
    dprev = np.asarray(ctx.t0_8 - zl[idx8 - 1], float)
    dnext = np.asarray(zl[idx8 + 2] - ctx.t0_8, float)
    sk = g8 / (2 * np.minimum(dprev, dnext))
    rhs = 1.0 / (8 * (1 - sk) ** 2)
    close("V4 max RHS of (3.23) (0.346)", rhs.max(), 0.346, 0.002, "%.4f", "%.3f")
    equal("V4 pairs violating (3.23)", int(np.sum(r8[:, 7] <= rhs)), 0)
    # fresh Ghat on random pairs
    n = 200 if ctx.full else 20
    gmax = max(G.Ghat_on_gap(zl, idx8[j], 257, W) for j in ctx.rng.choice(len(idx8), n, replace=False))
    report("V4 max Ghat on %d random pairs (<0)" % n, gmax, "<0", "-", gmax < 0, "%.1f")
    # the sweep file's geometry agrees with the rows table
    below("V4 sweep far vs rows far, max rel dev", float(np.max(np.abs(det["far"] / r8[:, 5] - 1))), 1e-10)
    below("V4 sweep r vs rows r, max rel dev", float(np.max(np.abs(det["r"] / r8[:, -1] - 1))), 1e-10)
