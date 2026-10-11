"""V10 -- Theorem E (the GUE far-field constant), Observation 7.2, the reference row of Table 7.3."""
import numpy as np

from tightpairs import gue as GU, data as D
from tightpairs.report import close, rel, below, report, section


def run(ctx):
    section("V10: the GUE far-field constant (Theorem E, Observation 7.2)")
    I1, I2, CG = GU.integrals()
    close("V10 int (1-S^2)/x^2 = 2pi^2/3", I1, 2 * np.pi ** 2 / 3, 1e-5, "%.7f", "%.7f")
    close("V10 int S'^2/x^2 = 2pi^4/15", I2, 2 * np.pi ** 4 / 15, 1e-5, "%.7f", "%.7f")
    close("V10 C_GUE = 4pi^2/15", CG, GU.C_GUE, 1e-5, "%.7f", "%.7f")
    close("V10 4pi^2/15 = 2.6318945069", GU.C_GUE, 2.6318945069, 1e-9, "%.10f", "%.10f")
    close("V10 rho(x)/x^4 -> pi^4/45 at x=1e-2", GU.rho(1e-2) / 1e-8, np.pi ** 4 / 45, 1e-2, "%.5f", "%.5f")
    close("V10 conditional density ratio vs rho at x=0.8, a=1e-4", GU.Kdet(0.8, 1e-4), GU.rho(0.8), 1e-6, "%.8f", "%.8f")
    close("V10 C(a) at s=0.02 (2.6314358)", GU.C_of_a(0.01, 1024.0), 2.6314358, 2e-7, "%.7f", "%.7f")
    close("V10 deficit at s=0.02 (4.6e-4)", GU.C_GUE - GU.C_of_a(0.01, 1024.0), 4.6e-4, 2e-5, "%.2e", "%.1e")
    sg, Cg, lin, cub = GU.small_s_law()
    close("V10 (C0-C(s))/s^2 = 1.17 - 1.26 s: intercept", lin[1], 1.17, 0.005, "%.4f", "%.2f")
    close("V10 (C0-C(s))/s^2 = 1.17 - 1.26 s: slope", lin[0], -1.26, 0.01, "%.4f", "%.2f")
    below("V10 linear fit max residual <= 0.0035", float(np.max(np.abs((GU.C_GUE - Cg) / sg ** 2 - np.polyval(lin, sg)))), 0.0035)
    below("V10 cubic fit constant term <= 1.4e-5", abs(cub[3]), 1.4e-5)
    r8 = ctx.rows["8p4e9"]
    close("V10 population prediction (2.610)", float(np.mean(np.interp(r8[:, 3], sg, Cg))) - 2 / 250, 2.610, 2e-3, "%.4f", "%.3f")
    close("V10 window law int_{|x|<=100} = C - 2/100", GU.window_integral(100.0), GU.C_GUE - 0.02, 2e-4, "%.5f", "%.5f")
    close("V10 pi^2/4 differs from C_GUE by 6.7%", 100 * (GU.C_GUE / (np.pi ** 2 / 4) - 1), 6.7, 0.05, "%.2f", "%.1f")
    # the exact small-gap tail of the sine process (Table 7.3, reference row)
    tgu = ctx.table("tab_gue")["sine_kernel"]
    for s0, tg, k in ((0.05, 0.998, 0), (0.10, 0.992, 1), (0.15, 0.982, 2)):
        ex = GU.tail_ratio_exact(s0)
        close("V10 sine-kernel tail ratio s0=%.2f, Fredholm determinant (%.3f)" % (s0, tg), ex, tg, 5e-4, "%.5f", "%.3f")
        below("V10 ... Fredholm determinant vs series 1 - 18pi^2 s^2/225 + 9pi^4 s^4/2205 (<= 4e-6)", abs(ex - GU.tail_ratio_series(s0)), 4e-6)
        close("V10 ... equals Table 7.3", ex, tgu["tail_ratios"][k], 1e-12, "%.6f", "%.6f")
    rel("V10 E_2(0;s) stable under doubling the nodes, s=0.15", GU.gap_probability(0.15, 80), GU.gap_probability(0.15, 40), 1e-14, "%.15f", "%.15f")
    # Monte Carlo results
    try:
        mc = ctx.arr("gue_mc_summary.npz")
        close("V10 MC N=500 window 100 (2.5781)", float(mc["N500_w100_mean"]), 2.5781, 3 * float(mc["N500_w100_se"]) + 1e-3, "%.4f", "%.4f")
        res = ctx.arr("gue_mc_results.npz")
        gaps = res["N500_gaps"]
        sg2 = np.linspace(0.01, 0.25, 25)
        Cg2 = np.array([GU.C_of_a(s_ / 2, 1024.0) for s_ in sg2])
        EC = float(np.mean(np.interp(gaps, sg2, Cg2)))
        for w_, tg in ((25, 2.5177), (50, 2.5577), (100, 2.5777)):
            pred = EC - 2.0 / w_
            close("V10 window law E[C(s)] - 2/%d for the MC population (%.4f)" % (w_, tg), pred, tg, 2e-4, "%.4f", "%.4f")
            close("V10 MC N=500 window %d vs prediction" % w_, float(mc["N500_w%d_mean" % w_]), pred, 3 * float(mc["N500_w%d_se" % w_]) + 1e-3, "%.4f", "%.4f")
        close("V10 MC N=500 pairs (73,446)", float(mc["N500_npairs"]), 73446.0, 0.0, "%.0f", "%.0f")
        close("V10 MC increment 25->50 (+0.0400)", float(mc["N500_w50_mean"] - mc["N500_w25_mean"]), 0.04, 0.01, "%.4f", "%.4f")
        close("V10 MC increment 50->100 (+0.0200)", float(mc["N500_w100_mean"] - mc["N500_w50_mean"]), 0.02, 0.01, "%.4f", "%.4f")
        close("V10 MC total conditioned pairs (138,619)", float(mc["total_pairs"]), 138619.0, 0.0, "%.0f", "%.0f")
        # bulk spacings of the N=500 matrices against the exact sine-kernel tail (Section 7.4)
        for s0, key, tg in ((0.05, "tail_005", (1.08, 0.04)), (0.10, "tail_010", (1.03, 0.01)), (0.15, "tail_015", (0.995, 0.008))):
            v, se_ = float(mc[key]), float(mc[key + "_se"])
            report("V10 MC tail ratio s0=%.2f +- Poisson s.e. (%.3g +- %.2g)" % (s0, tg[0], tg[1]), "%.4f +- %.4f" % (v, se_), "%.3g +- %.2g" % tg, 5e-3,
                   abs(v - tg[0]) <= 5e-3 and abs(se_ - tg[1]) <= 5e-3, "%s", "%s")
            below("V10 ... MC within 2.4 s.e. of the exact value, s0=%.2f" % s0, abs(v - GU.tail_ratio_exact(s0)) / se_, 2.45, "%.2f")
        close("V10 MC bulk spacings (4.5e6)", float(mc["n_bulk_spacings"]), 4.5e6, 1e4, "%.0f", "%.2g")
        close("V10 MC lag-1, consecutive bulk spacings of the same matrix (-0.308)", float(mc["lag1"]), -0.308, 5e-4, "%.4f", "%.3f")
        close("V10 MC lag-1 equals Table 7.3", float(mc["lag1"]), ctx.table("tab_gue")["GUE_MC"]["lag1"], 1e-12, "%.4f", "%.4f")
    except FileNotFoundError:
        report("V10 MC summary file missing (run pipeline/gue_mc.py)", "missing", "present", "-", False, "%s", "%s")
    if ctx.full:
        res = GU.sample_pairs(200, 20000, seed=2024)
        for w, v in res["sums"].items():
            rel("V10 MC quick tier C_GUE (N=200, window %d, +2/w)" % int(w), float(v.mean()) + 2.0 / w, GU.C_GUE, 0.03, "%.4f", "%.4f")
        L = res["lag"]
        close("V10 MC quick tier lag-1 (-0.308)", np.corrcoef(L[:, 0], L[:, 1])[0, 1], -0.308, 0.01, "%.3f", "%.3f")
        sb = res["tails"]
        C9 = np.pi ** 2 / 9
        for s0 in (0.10, 0.15):
            close("V10 MC quick tier tail ratio s0=%.2f vs exact" % s0, float(np.mean(sb < s0)) / (C9 * s0 ** 3), GU.tail_ratio_exact(s0), 0.08, "%.3f", "%.3f")
    else:
        print("    (GUE Monte Carlo quick tier skipped; use --full)")
