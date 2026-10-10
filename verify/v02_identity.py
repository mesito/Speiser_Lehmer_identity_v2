"""V2 -- the Speiser-Lehmer identity (Theorem A(a)) on five ensembles; self-calibration (Section 3.5)."""
import numpy as np

from tightpairs import geometry as G
from tightpairs.constants import W, dens
from tightpairs.report import close, rel, equal, below, info, section


def run(ctx):
    section("V2: the identity, Table 3.1; self-calibration, Observation 3.4")
    ti = ctx.table("tab_identity")
    names = {"100k": "100k", "2M": "2M", "8p4e9": "LMFDB", "1e12": "1e12", "1e21": "1e21"}
    for k, a in ctx.rows.items():
        dev = float(np.max(np.abs(G.identity_deviation(a[:, -1], a[:, 7]))))
        below("V2 identity max dev (double) %s <= 1e-15" % k, dev, 1e-15)
        rel("V2 identity max dev %s equals Table 3.1" % k, dev, ti[names[k]]["max_dev_double"], 1e-6)
    r8 = ctx.rows["8p4e9"]
    g_ld, far_ld = r8[:, 2].astype(np.longdouble), r8[:, 5].astype(np.longdouble)
    r_ld = np.sqrt(2.0 / (far_ld + 8 / g_ld ** 2)) / (g_ld / 2)
    Cm_ld = (1 / g_ld ** 2) / far_ld
    dev80 = float(np.max(np.abs(G.identity_deviation(r_ld, Cm_ld))))
    below("V2 identity max dev (80-bit) <= 1e-18", dev80, 1e-18)
    rel("V2 identity 80-bit dev equals Table 3.1", dev80, ti["LMFDB"]["max_dev_80bit"], 1e-6)
    # the rows themselves: fresh geometry on a subset (all on --full)
    sub = np.arange(len(ctx.idx8)) if ctx.full else np.sort(ctx.rng.choice(len(ctx.idx8), 60, replace=False))
    fresh = G.rows_table(ctx.zl, ctx.idx8[sub], W)
    below("V2 lehmer_rows_8p4e9 recomputed on %d pairs, max rel dev" % len(sub), float(np.max(np.abs(fresh[:, 1:] / r8[sub, 1:] - 1))), 1e-12)
    # the window rows recomputed from the raw tables (pipeline/compute_windows.py)
    import importlib.util, os
    spec = importlib.util.spec_from_file_location("compute_windows", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline", "compute_windows.py"))
    cw = importlib.util.module_from_spec(spec); spec.loader.exec_module(cw)
    for ens, name in (("1e12", "zeros3"), ("1e21", "zeros4")):
        _, _, _, fresh_w = cw.window_rows(name)
        below("V2 lehmer_rows_%s recomputed from %s, max rel dev" % (ens, name), float(np.max(np.abs(fresh_w / ctx.rows[ens] - 1))), 1e-14)
    for ens, tg in (("2M", None), ("100k", None)):
        info("V2 median r %s" % ens, float(np.median(ctx.rows[ens][:, -1])), "%.4f")
    rr, Cmid = r8[:, -1], r8[:, 7]
    close("V2 min r (0.9612)", rr.min(), 0.9612, 5e-4, "%.6f")
    close("V2 median r (0.9981)", np.median(rr), 0.9981, 5e-4, "%.6f")
    close("V2 max r (0.99998)", rr.max(), 0.99998, 1e-5, "%.6f", "%.5f")
    close("V2 min C_mid (1.52)", Cmid.min(), 1.52, 0.01, "%.3f", "%.2f")
    close("V2 median C_mid (32.0)", np.median(Cmid), 32.0, 0.1, "%.2f", "%.1f")
    close("V2 max C_mid (2520)", Cmid.max(), 2520.0, 1.0, "%.1f", "%.0f")
    zl, idx8 = ctx.zl, ctx.idx8
    g8 = r8[:, 2]
    dprev = np.asarray(ctx.t0_8 - zl[idx8 - 1], float)
    dnext = np.asarray(zl[idx8 + 2] - ctx.t0_8, float)
    sk = g8 / (2 * np.minimum(dprev, dnext))
    close("V2 median sqrt(kappa) (0.050)", np.median(sk), 0.050, 0.002, "%.4f", "%.3f")
    close("V2 max sqrt(kappa) (0.399)", sk.max(), 0.399, 0.002, "%.4f", "%.3f")
    # self-calibration constant (Corollary A.3)
    s8 = r8[:, 3]
    x = (1 - rr) / s8 ** 2
    close("V2 mean (1-r)/s^2 (0.156)", x.mean(), 0.156, 1e-3, "%.4f", "%.3f")
    phi = r8[:, 5] / dens(r8[:, 1]) ** 2
    close("V2 mean(far/dens^2)/16 (0.156)", phi.mean() / 16, 0.156, 1e-3, "%.4f", "%.3f")
    close("V2 the far-field constant 2.501 equals Table 7.1", phi.mean(), ctx.table("tab_farlaw")["LMFDB"]["ratio"], 1e-12, "%.4f", "%.4f")
    close("V2 far-field constant (2.501)", phi.mean(), 2.501, 5e-4, "%.4f", "%.3f")
    close("V2 pi^2/60", np.pi ** 2 / 60, 0.1645, 1e-4, "%.4f", "%.4f")
    # the flip depth of Proposition A.4: (x-1)/(1+x)^2 = 1/(16 C_mid)
    y = 1 / (16 * Cmid)
    a_, b_, c_ = y, 2 * y - 1, y + 1
    xf = (-b_ - np.sqrt(b_ * b_ - 4 * a_ * c_)) / (2 * a_)
    hf = np.sqrt(xf)
    below("V2 flip-depth equation residual", float(np.max(np.abs((xf - 1) / (1 + xf) ** 2 - y))), 1e-11)
    close("V2 median h_flip/eps (1.0039)", np.median(hf), 1.0039, 2e-4, "%.5f", "%.4f")
    below("V2 max |h_flip/eps - 1 - 1/(8 C_mid)| / C_mid^-2", float(np.max(np.abs(hf - 1 - 1 / (8 * Cmid)) * Cmid ** 2)), 0.1)
    below("V2 max |h_thr/eps - 1 + 1/(16 C_mid)| / C_mid^-2", float(np.max(np.abs(rr - 1 + 1 / (16 * Cmid)) * Cmid ** 2)), 0.1)
    # Corollary A.2: Lehmer condition C > 5/4 <=> r > sqrt(20/21) up to eta
    C_ = r8[:, 6]
    lehmer = C_ > 1.25
    equal("V2 Lehmer pairs (C > 5/4) have r > sqrt(20/21)", bool(np.all(rr[lehmer] > np.sqrt(20 / 21))), True)
    equal("V2 tight pairs with C > 5/4 (2735 of 2736)", int(lehmer.sum()), 2735)
    equal("V2 the exception is the pair with the smallest C_mid", int(np.argmin(Cmid)), int(np.flatnonzero(~lehmer)[0]))
    close("V2 sqrt(20/21)", np.sqrt(20 / 21), 0.9759, 1e-4, "%.5f", "%.4f")
