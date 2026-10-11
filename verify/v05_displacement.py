"""V5 -- the displacement law (Section 4, Appendix B)."""
import numpy as np
import mpmath as mp

from tightpairs import geometry as G, instrument as I, euler as E
from tightpairs.constants import W, dens
from tightpairs.report import close, rel, equal, below, info, report, section


def run(ctx):
    section("V5: displacement law (Theorem B, Tables B.1-B.2)")
    td = ctx.table("tab_disp")
    ref = ctx.csv("reference_8p4e9.csv")
    close("V5 corr(delta, prediction), 15 pairs (0.999998)", np.corrcoef(ref["delta"], ref["prediction"])[0, 1], 0.999998, 2e-6, "%.7f", "%.6f")
    close("V5 median rel. err. (%) (0.36)", np.median(ref["abs_rel_err_pct"]), 0.36, 0.01, "%.3f", "%.2f")
    close("V5 max rel. err. (%) (7.5)", ref["abs_rel_err_pct"].max(), 7.52, 0.01, "%.2f", "%.2f")
    equal("V5 signs agree on all 15 pairs", bool(np.all(np.sign(ref["delta"]) == np.sign(ref["prediction"]))), True)
    ok = len(td["rows"]) == int(ref.size) and all(
        abs(r_["delta"] - float(ref["delta"][k])) <= 1e-12 * abs(r_["delta"]) + 1e-20 and abs(r_["prediction"] - float(ref["prediction"][k])) <= 1e-12 * abs(r_["prediction"]) + 1e-20
        and abs(r_["lambda_far"] - float(ref["lambda_far"][k])) <= 1e-11 for k, r_ in enumerate(td["rows"]))
    report("V5 Table B.2 rows equal the reference file", "%d rows" % len(td["rows"]), "%d rows" % int(ref.size), 1e-12, ok, "%s", "%s")
    close("V5 Table B.2 correlation", float(np.corrcoef(ref["delta"], ref["prediction"])[0, 1]), td["corr"], 1e-12, "%.8f", "%.8f")
    # the inequality of the proof of Theorem B: ((1 - sqrt(1-2h))/h - 1) = h/(1 - h + sqrt(1-2h)) <= 2h on [0, 1/2], with equality at 1/2
    hh = np.linspace(1e-6, 0.5, 200001)
    lhs = (1 - np.sqrt(1 - 2 * hh)) / hh - 1
    below("V5 Theorem B: |(1-sqrt(1-2h))/h - 1 - h/(1-h+sqrt(1-2h))| on [0,1/2]", float(np.max(np.abs(lhs - hh / (1 - hh + np.sqrt(1 - 2 * hh))))), 1e-9)
    below("V5 Theorem B: (1-sqrt(1-2h))/h - 1 <= 2h on [0,1/2] (max of ratio)", float(np.max(lhs / (2 * hh))), 1.0 + 1e-12)
    equal("V5 Theorem B: the ratio to h exceeds 1 beyond sqrt(2)-1", bool(np.all(lhs[hh > np.sqrt(2) - 1 + 1e-6] > hh[hh > np.sqrt(2) - 1 + 1e-6])), True)
    k = int(np.argmax(ref["abs_rel_err_pct"]))
    equal("V5 max error at the pair with the smallest |delta|", int(np.argmin(np.abs(ref["delta"]))), k)
    equal("V5 ... which has the smallest |lambda_far|", int(np.argmin(np.abs(ref["lambda_far"]))), k)
    close("V5 ... |delta| there (1.5e-7)", abs(ref["delta"][k]), 1.5e-7, 5e-9, "%.3e", "%.1e")
    close("V5 ... |lambda_far| there (0.034)", abs(ref["lambda_far"][k]), 0.034, 5e-4, "%.4f", "%.3f")
    close("V5 ... absolute error there (1.1e-8)", abs(ref["delta"][k] - ref["prediction"][k]), 1.1e-8, 1e-9, "%.2e", "%.1e")
    others = np.delete(ref["abs_rel_err_pct"], k)
    below("V5 error on the other fourteen pairs <= 1.0%", others.max(), 1.0)
    report("V5 |delta| range (1.5e-7 .. 4.1e-5)", "%.2e..%.2e" % (np.abs(ref["delta"]).min(), np.abs(ref["delta"]).max()), "1.5e-7..4.1e-5", "-",
           abs(np.abs(ref["delta"]).min() / 1.5e-7 - 1) < 0.05 and abs(np.abs(ref["delta"]).max() / 4.1e-5 - 1) < 0.05, "%s", "%s")
    sref = ref["g"] * dens(8.436e9)
    report("V5 normalised spacings of the 15 pairs (0.0139..0.0257)", "%.4f..%.4f" % (sref.min(), sref.max()), "0.0139..0.0257", "-",
           abs(sref.min() - 0.0139) < 5e-4 and abs(sref.max() - 0.0257) < 5e-4, "%s", "%s")
    info("V5 range of |delta|", "%.1e..%.1e" % (np.abs(ref["delta"]).min(), np.abs(ref["delta"]).max()), "%s")
    for ens, tg in (("100k", (30, 0.999994, 0.185, 2.859)), ("2M", (15, 0.999997, 0.459, 1.577))):
        b = ctx.csv("reference_%s.csv" % ens)
        equal("V5 %s reference pairs" % ens, int(len(b)), tg[0])
        close("V5 corr(delta, prediction) %s" % ens, np.corrcoef(b["delta"], b["prediction"])[0, 1], tg[1], 2e-6, "%.6f", "%.6f")
        close("V5 median rel. err. (%%) %s" % ens, np.median(b["abs_rel_err_pct"]), tg[2], 0.005, "%.3f", "%.3f")
        close("V5 max rel. err. (%%) %s" % ens, b["abs_rel_err_pct"].max(), tg[3], 0.005, "%.3f", "%.3f")
    # fresh recomputation on reference pairs (3 fast / 15 full), exact midpoints
    hl, zl = ctx.hl, ctx.zl
    mp.mp.dps = 30
    for k in range(15 if ctx.full else 3):
        i = int(ref["idx1"][k]) - 1
        ga = mp.mpf(float(hl[i, 0])) + mp.mpf(float(hl[i, 1]))
        gb = mp.mpf(float(hl[i + 1, 0])) + mp.mpf(float(hl[i + 1, 1]))
        t0 = (ga + gb) / 2
        g = gb - ga
        far, lam = G.far_lambda(zl, i, np.longdouble(mp.nstr(t0, 25)), W)
        delta = float(I.extremum_mp(mp, t0) - t0)
        close("V5 pair %02d lambda_far vs table" % (k + 1), lam, float(ref["lambda_far"][k]), 1e-6, "%.6f", "%.6f")
        rel("V5 pair %02d delta recomputed" % (k + 1), delta, float(ref["delta"][k]), 1e-6)
        rel("V5 pair %02d prediction" % (k + 1), float(g ** 2 / 8) * lam, float(ref["prediction"][k]), 1e-9)
    # with double-precision midpoints the error grows (Section 2.3)
    errs = []
    for k in range(15 if ctx.full else 3):
        i = int(ref["idx1"][k]) - 1
        t0d = (mp.mpf(float(hl[i, 0])) + mp.mpf(float(hl[i + 1, 0]))) / 2
        delta_d = float(I.extremum_mp(mp, t0d) - t0d)
        errs.append(100 * abs(delta_d - float(ref["prediction"][k])) / abs(delta_d))
    info("V5 errors (%%) with double-precision midpoints, %d pairs: max" % len(errs), max(errs), "%.2f")
    if ctx.full:
        close("V5 ... median (2.45)", np.median(errs), 2.45, 0.15, "%.2f", "%.2f")
        close("V5 ... max (332)", max(errs), 332.0, 5.0, "%.1f", "%.0f")
    # Lehmer's pair, out of sample
    z1 = ctx.z1
    n = 6708
    ga, gb = z1[n], z1[n + 1]
    gL, t0L = gb - ga, (ga + gb) / 2
    zl1 = z1.astype(np.longdouble)
    farL, lamL = G.far_lambda(zl1, n, np.longdouble(t0L), W)
    close("V5 Lehmer pair gamma_6709", ga, 7005.062866, 1e-6, "%.6f", "%.6f")
    close("V5 Lehmer pair g", gL, 0.037698, 1e-6, "%.6f", "%.6f")
    close("V5 Lehmer pair lambda_far", lamL, 0.7784, 1e-4, "%.4f", "%.4f")
    close("V5 Lehmer pair far", farL, 2.428, 1e-3, "%.3f", "%.3f")
    tl = ctx.table("tab_lehmer")
    predL = gL ** 2 / 8 * lamL
    tsL = I.extremum_mp(mp, mp.mpf(t0L))
    deltaL = float(tsL - mp.mpf(t0L))
    rel("V5 Lehmer delta predicted (1.383e-4)", predL, 1.383e-4, 1e-3)
    rel("V5 Lehmer delta measured (1.397e-4)", deltaL, 1.397e-4, 1e-3)
    close("V5 Lehmer rel. err. (%) (1.0)", 100 * abs(deltaL - predL) / deltaL, 1.0, 0.05, "%.2f", "%.1f")
    flL = float(abs(mp.siegelz(tsL)))
    ZppL = float(abs(mp.diff(mp.siegelz, mp.mpf(t0L), 2)))
    rel("V5 Lehmer floor measured", flL, tl["floor_meas"], 1e-9)
    rel("V5 Lehmer (g^2/8)|Z''(t0)|", gL ** 2 / 8 * ZppL, tl["floor_pred"], 1e-9)
    close("V5 Lehmer floor formula error (%) (0.03)", 100 * abs(gL ** 2 / 8 * ZppL / flL - 1), 0.03, 0.005, "%.3f", "%.2f")
    leh = ctx.arr("lehmer_pair_7005.npz")
    R_L = np.log(flL) - leh["logP"] - leh["zsum"][2]
    for j, X in ((0, 1e2), (2, 1e4), (5, 1e7)):
        close("V5 Lehmer R_3000 at X=1e%d" % int(np.log10(X)), R_L[j], tl["R3000"][str(int(X))], 1e-9, "%+.4f", "%+.4f")
    below("V5 Lehmer max |R_3000| over X (<= 0.009)", float(np.abs(R_L).max()), 0.009)
    lpL4 = E.logP_exact(np.array([np.longdouble(t0L)]), xs=[10 ** 4])[0, 0]
    close("V5 Lehmer log|P_1e4| recomputed", lpL4, leh["logP"][2], 1e-9, "%.6f", "%.6f")
