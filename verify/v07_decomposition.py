"""V7 -- the Euler-Hadamard decomposition of the floor (Observation 5.7, Table 5.3)
and the loading accounting (Observation 5.8)."""
import numpy as np
from scipy.special import sici

from tightpairs import kernel as K, stats as S
from tightpairs.constants import XS, GAMMA_E
from tightpairs.report import close, report, below, equal, section


def run(ctx):
    section("V7: decomposition of the floor (Section 5.4, Table 5.3)")
    td = ctx.table("tab_decomp")
    lf8, lf2 = np.log(ctx.f8[:, 5]), np.log(ctx.f2[:, 5])
    for d, Dl in enumerate((100, 1000)):
        for j, X in enumerate(XS):
            R = lf8 - ctx.lp8[:, j] - ctx.zs8[d, :, j]
            tg = td["8p4e9"][str(Dl)][str(int(X))]
            ok = abs(np.median(R) - tg["median"]) <= 1e-9 and abs(R.std() - tg["sd"]) <= 1e-9 and abs(np.abs(R).max() - tg["max"]) <= 1e-9
            report("V7 R Delta=%d X=1e%d med/sd/max (Table 5.3)" % (Dl, int(np.log10(X))), "%+.4f/%.4f/%.3f" % (np.median(R), R.std(), np.abs(R).max()),
                   "%+.4f/%.4f/%.3f" % (tg["median"], tg["sd"], tg["max"]), 1e-9, ok)
    m1000 = [td["8p4e9"]["1000"][str(int(X))] for X in XS]
    report("V7 Delta=1000 medians in [0.0001, 0.0013]", "%.4f..%.4f" % (min(v["median"] for v in m1000), max(v["median"] for v in m1000)), "[0.0001,0.0013]", "-",
           min(v["median"] for v in m1000) >= 0.0001 - 5e-5 and max(v["median"] for v in m1000) <= 0.0013 + 5e-5, "%s", "%s")
    report("V7 Delta=1000 s.d. in [0.0025, 0.0038]", "%.4f..%.4f" % (min(v["sd"] for v in m1000), max(v["sd"] for v in m1000)), "[0.0025,0.0038]", "-",
           min(v["sd"] for v in m1000) >= 0.0025 - 5e-5 and max(v["sd"] for v in m1000) <= 0.0038 + 5e-5, "%s", "%s")
    m100 = [td["8p4e9"]["100"][str(int(X))]["sd"] for X in XS]
    report("V7 Delta=100 s.d. in [0.010, 0.017]", "%.4f..%.4f" % (min(m100), max(m100)), "[0.010,0.017]", "-", min(m100) >= 0.0095 and max(m100) <= 0.0175, "%s", "%s")
    for j, X in enumerate(XS):
        R = lf2 - ctx.lp2[:, j] - ctx.zs2[1, :, j]
        tg = td["2M"]["1000"][str(int(X))]
        report("V7 2M R Delta=1000 X=1e%d med/sd" % int(np.log10(X)), "%+.4f/%.4f" % (np.median(R), R.std()), "%+.4f/%.4f" % (tg["median"], tg["sd"]), 1e-9,
               abs(np.median(R) - tg["median"]) <= 1e-9 and abs(R.std() - tg["sd"]) <= 1e-9)
    below("V7 2M max |median| (0.0014)", td["2M_max_abs_median"], 0.0014 + 5e-5)
    below("V7 2M max s.d. (0.0037)", td["2M_max_sd"], 0.0037 + 5e-5)
    # kernel sums: fresh recomputation on the V6 subset
    sub = ctx._cache.get("sub8", np.sort(ctx.rng.choice(len(ctx.idx8), 40, replace=False)))
    zs_chk = np.array([K.zsum(ctx.zd, ctx.zl, ctx.t0_8[sub], Dl) for Dl in (100.0, 1000.0)])
    below("V7 S_Delta recomputed on %d pairs, max dev" % len(sub), float(np.max(np.abs(zs_chk - ctx.zs8[:, sub, :]))), 1e-9)
    # the loading explained (Observation 5.8)
    g8, lg8 = ctx.f8[:, 2], np.log(ctx.f8[:, 2])
    far8 = ctx.f8[:, 4]
    acc = td["loading"]
    tf = ctx.table("tab_fits")
    for j, X in enumerate(XS):
        Sp = K.pair_term(g8, X)
        Sf = ctx.zs8[1, :, j] - Sp
        Rr = lf8 - ctx.lp8[:, j] - ctx.zs8[1, :, j]
        (ap, bp, _), _, _, _ = S.ols(np.column_stack([lg8, ctx.lp8[:, j]]), Sp - 2 * lg8)
        (a2, b2, _), _, _, _ = S.ols(np.column_stack([lg8, ctx.lp8[:, j]]), Sf)
        (ar, br, _), _, _, _ = S.ols(np.column_stack([lg8, ctx.lp8[:, j]]), Rr)
        (a, b, c), _, _, _ = S.ols(np.column_stack([lg8, ctx.lp8[:, j]]), lf8)
        tg = acc[str(int(X))]
        close("V7 pair-term curvature in a, X=1e%d" % int(np.log10(X)), ap, tg["curv_a"], 1e-9, "%+.4f", "%+.4f")
        close("V7 slope of S_far on logP, X=1e%d" % int(np.log10(X)), b2, tg["Sfar_b"], 1e-9, "%+.4f", "%+.4f")
        close("V7 slope of S_far on log g, X=1e%d" % int(np.log10(X)), a2, tg["Sfar_a"], 1e-9, "%+.4f", "%+.4f")
        below("V7 residual slopes < 0.001, X=1e%d" % int(np.log10(X)), max(abs(ar), abs(br)), 0.001)
        close("V7 exact accounting of a, X=1e%d" % int(np.log10(X)), 2 + ap + a2 + ar, a, 1e-9, "%.6f", "%.6f")
        close("V7 exact accounting of b, X=1e%d" % int(np.log10(X)), 1 + bp + b2 + br, b, 1e-9, "%.6f", "%.6f")
        close("V7 corr(S_far, logP) X=1e%d" % int(np.log10(X)), S.corr(Sf, ctx.lp8[:, j]), tg["corr_Sfar_logP"], 1e-9, "%.3f", "%.3f")
        close("V7 corr(S_far, log far) X=1e%d" % int(np.log10(X)), S.corr(Sf, np.log(far8)), tg["corr_Sfar_logfar"], 1e-9, "%.3f", "%.3f")
        close("V7 sd(S_far) X=1e%d" % int(np.log10(X)), Sf.std(), tg["sd_Sfar"], 1e-9, "%.3f", "%.3f")
    # text numbers
    cv = [acc[str(int(X))]["curv_a"] for X in XS]
    report("V7 curvature corrections (-0.004..-0.048)", "/".join("%+.3f" % v for v in cv), "-0.004/-0.009/-0.016/-0.024/-0.035/-0.048", 1.5e-3,
           all(abs(v - t) <= 1.5e-3 for v, t in zip(cv, (-0.004, -0.009, -0.016, -0.024, -0.035, -0.048))))
    sb = [acc[str(int(X))]["Sfar_b"] for X in XS]
    report("V7 S_far slopes on logP (-0.254..-0.064)", "/".join("%+.3f" % v for v in sb), "-0.254/-0.260/-0.218/-0.167/-0.097/-0.064", 1.5e-3,
           all(abs(v - t) <= 1.5e-3 for v, t in zip(sb, (-0.254, -0.260, -0.218, -0.167, -0.097, -0.064))))
    sa = [acc[str(int(X))]["Sfar_a"] for X in XS]
    report("V7 S_far slopes on log g in [-0.016, 0.008]", "%+.4f..%+.4f" % (min(sa), max(sa)), "[-0.016,0.008]", "-", min(sa) >= -0.0165 and max(sa) <= 0.0085, "%s", "%s")
    close("V7 corr(S_far, logP) X=1e2 (-0.68)", acc["100"]["corr_Sfar_logP"], -0.68, 5e-3, "%.3f", "%.2f")
    close("V7 corr(S_far, logP) X=1e7 (-0.22)", acc["10000000"]["corr_Sfar_logP"], -0.22, 5e-3, "%.3f", "%.2f")
    close("V7 corr(S_far, log far) X=1e4 (0.56)", acc["10000"]["corr_Sfar_logfar"], 0.56, 5e-3, "%.3f", "%.2f")
    close("V7 corr(S_far, log far) X=1e7 (0.03)", acc["10000000"]["corr_Sfar_logfar"], 0.03, 5e-3, "%.3f", "%.2f")
    close("V7 sd(S_far) X=1e2 (0.32)", acc["100"]["sd_Sfar"], 0.32, 5e-3, "%.3f", "%.2f")
    close("V7 sd(S_far) X=1e7 (0.21)", acc["10000000"]["sd_Sfar"], 0.21, 5e-3, "%.3f", "%.2f")
    y0 = (g8 / 2) * np.log(1e7)
    below("V7 pair-term expansion |2Ci(y)-2(gamma+log y)| <= y^2/2", float(np.max(np.abs(2 * sici(y0)[1] - 2 * (GAMMA_E + np.log(y0))) - y0 ** 2 / 2)), 1e-12)
    close("V7 unexplained variance at X=1e7: sd(S_far)^2 (0.04)", acc["10000000"]["sd_Sfar"] ** 2, 0.044, 5e-3, "%.4f", "%.3f")
