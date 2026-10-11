"""V8 -- mediation (Section 6): far-field correlation, partial correlations, the Euler deficit,
the control regression, block averaging."""
import numpy as np
import mpmath as mp

from tightpairs import data as D, euler as E, stats as S, instrument as I
from tightpairs.constants import XS
from tightpairs.report import close, rel, report, below, equal, section


def run(ctx):
    section("V8: mediation (Section 6, Tables 6.1-6.3)")
    tk = ctx.table("tab_k5")
    f100 = ctx.arr("floors_all_100k.npy")
    f12 = ctx.arr("floors_1e12.npy")
    for name, ff in (("100k", f100), ("2M", ctx.f2), ("8p4e9", ctx.f8), ("1e12", f12)):
        (_, _), _, _, rg = S.ols(np.log(ff[:, 2])[:, None], np.log(ff[:, 5]))
        close("V8 K5 = corr(resid_g, log far) %s (Table 6.1)" % name, S.corr(rg, np.log(ff[:, 4])), tk[name]["K5"], 1e-12, "%.4f", "%.4f")
    close("V8 K5 8.4e9 (-0.725)", tk["8p4e9"]["K5"], -0.725, 5e-4, "%.4f", "%.3f")
    equal("V8 100k tight pairs", int(len(f100)), 261)
    equal("V8 1e12 tight pairs with floors", int(len(f12)), 38)
    below("V8 100k floors file: geometry equals lehmer_rows_100k", float(np.max(np.abs(f100[:, [0, 1, 2, 3, 5 - 1]] - ctx.rows["100k"][:, [0, 1, 2, 3, 5]]))), 1e-9)
    below("V8 1e12 floors file: geometry equals lehmer_rows_1e12", float(np.max(np.abs(f12[:, [0, 1, 2, 3, 4]] - ctx.rows["1e12"][:, [0, 1, 2, 3, 5]]))), 1e-9)
    mp.mp.dps = 30
    sel100 = range(len(f100)) if ctx.full else ctx.rng.choice(len(f100), 5, replace=False)
    worst = 0.0
    for k in sel100:
        i = int(f100[k, 0]) - 1
        f, _ = I.floor_max_dd(mp.mpf(float(ctx.z1[i])), mp.mpf(float(ctx.z1[i + 1])), mp)
        worst = max(worst, abs(f / f100[k, 5] - 1))
    below("V8 100k floors recomputed (%d), max rel dev" % len(list(sel100)), worst, 1e-10)
    if ctx.full:
        mp.mp.dps = 40
        b3, off3 = D.load_window("zeros3")
        k = int(np.argmin(f12[:, 5]))
        i = int(f12[k, 0]) - 1
        f, _ = I.floor_max_dd(mp.mpf(b3) + mp.mpf(repr(float(off3[i]))), mp.mpf(b3) + mp.mpf(repr(float(off3[i + 1]))), mp)
        rel("V8 smallest 1e12 floor recomputed", f, f12[k, 5], 1e-10)
    lg8, lf8, far8 = np.log(ctx.f8[:, 2]), np.log(ctx.f8[:, 5]), ctx.f8[:, 4]
    (_, _), _, _, resg8 = S.ols(lg8[:, None], lf8)
    for j, X, tg in ((2, 1e4, (0.921, 0.048, -0.799)), (5, 1e7, (0.939, 0.053, -0.720))):
        close("V8 partial corr(resid, logP | far) X=1e%d" % int(np.log10(X)), S.partial_corr(resg8, ctx.lp8[:, j], far8), tg[0], 1e-3, "%.3f", "%.3f")
        close("V8 partial corr(resid, far | logP) X=1e%d" % int(np.log10(X)), S.partial_corr(resg8, far8, ctx.lp8[:, j]), tg[1], 1e-3, "%.3f", "%.3f")
        close("V8 corr(logP, log far) X=1e%d" % int(np.log10(X)), S.corr(ctx.lp8[:, j], np.log(far8)), tg[2], 1e-3, "%.3f", "%.3f")
    sub = D.rank_subsample(ctx.f8[:, 2], 300)
    (_, _), _, _, resg_s = S.ols(lg8[sub][:, None], lf8[sub])
    ts_ = ctx.table("tab_sub")
    close("V8 subsample partial corr(resid, logP | far) X=1e4", S.partial_corr(resg_s, ctx.lp8[sub, 2], far8[sub]), ts_["pcorr_sub_P_given_far"], 1e-12, "%.3f", "%.3f")
    close("V8 subsample partial corr(resid, far | logP) X=1e4", S.partial_corr(resg_s, far8[sub], ctx.lp8[sub, 2]), ts_["pcorr_sub_far_given_P"], 1e-12, "%.3f", "%.3f")
    close("V8 2M corr(logP, log far) X=1e4 (-0.426)", S.corr(ctx.lp2[:, 2], np.log(ctx.f2[:, 4])), -0.426, 1e-3, "%.3f", "%.3f")
    tf = ctx.table("tab_fits")["ladder_1e4"]
    report("V8 adding far to (g, logP) at X=1e4: R2 0.9407 -> 0.9419", "%.4f -> %.4f" % (tf["g_logP"], tf["g_logP_far"]), "0.9407 -> 0.9419", 5e-4,
           abs(tf["g_logP"] - 0.9407) <= 5e-4 and abs(tf["g_logP_far"] - 0.9419) <= 5e-4, "%s", "%s")
    # controls
    sel = D.control_indices(ctx.zd)
    equal("V8 control indices reproduced (seed 11)", bool(np.array_equal(sel, ctx.ctrl_idx)), True)
    tdf = ctx.table("tab_deficit")
    for j, X in enumerate(XS):
        mp_, mc_, sd_ = ctx.lp8[:, j].mean(), ctx.lpc[:, j].mean(), ctx.lpc[:, j].std(ddof=1)
        sh = (mp_ - mc_) / sd_
        tg = tdf[str(int(X))]
        ok = abs(mp_ - tg["mean_pairs"]) <= 1e-9 and abs(mc_ - tg["mean_ctrl"]) <= 1e-9 and abs(sd_ - tg["sd_ctrl"]) <= 1e-9 and abs(sh - tg["shift"]) <= 1e-9
        report("V8 deficit X=1e%d means/sd/shift (Table 6.2)" % int(np.log10(X)), "%.3f/%.3f/%.3f/%.2f" % (mp_, mc_, sd_, sh),
               "%.3f/%.3f/%.3f/%.2f" % (tg["mean_pairs"], tg["mean_ctrl"], tg["sd_ctrl"], tg["shift"]), 1e-9, ok)
    close("V8 shift at X=1e2 (-0.48)", tdf["100"]["shift"], -0.48, 0.01, "%.3f", "%.2f")
    close("V8 shift at X=1e7 (-3.88)", tdf["10000000"]["shift"], -3.88, 0.01, "%.3f", "%.2f")
    close("V8 factor at X=1e7 (11.4)", tdf["10000000"]["factor"], 11.4, 0.1, "%.2f", "%.1f")
    equal("V8 shift decreases monotonically with X", bool(np.all(np.diff([tdf[str(int(X))]["shift"] for X in XS]) < 0)), True)
    lpc_chk = E.logP_exact((ctx.zl[sel[:10]] + ctx.zl[sel[:10] + 1]) / 2)
    below("V8 control log|P_X| recomputed (10), max dev", float(np.max(np.abs(lpc_chk - ctx.lpc[:10]))), 1e-9)
    yc = np.log(ctx.fc[:, 2])
    for j, X, tg in ((2, 1e4, (0.810, 0.802, 0.950)), (5, 1e7, (0.979, 0.607, 0.944))):
        (m_, c_), _, _, _ = S.ols(ctx.lpc[:, j][:, None], yc)
        cc = S.corr(ctx.lpc[:, j], yc)
        tj = tdf["ctrl_regression_1e%d" % int(np.log10(X))]
        report("V8 control regression X=1e%d slope/intercept/corr" % int(np.log10(X)), "%.3f/%.3f/%.3f" % (m_, c_, cc), "%.3f/%.3f/%.3f" % tg, 2e-3,
               abs(m_ - tg[0]) <= 2e-3 and abs(c_ - tg[1]) <= 2e-3 and abs(cc - tg[2]) <= 2e-3)
        below("V8 ... equals the table", abs(m_ - tj["slope"]) + abs(cc - tj["corr"]), 1e-9)
    mp.mp.dps = 30
    hl = ctx.hl
    for i in sel[:3]:
        ga = mp.mpf(float(hl[i, 0])) + mp.mpf(float(hl[i, 1])); gb = mp.mpf(float(hl[i + 1, 0])) + mp.mpf(float(hl[i + 1, 1]))
        f, _ = I.floor_max_dd(ga, gb, mp)
        j = int(np.flatnonzero(ctx.ctrl_idx == i)[0])
        close("V8 control floor recomputed, gap %d" % (i + 1), f, ctx.fc[j, 2], 1e-10, "%.8f", "%.8f")
    # blocks
    am = ctx.arr("moment_allgaps.npy")
    lpa = ctx.arr("logP_allgaps_1e6.npy")
    tb = ctx.table("tab_blocks")
    equal("V8 all-gap rows", int(len(am)), 193179)
    lfA, lgA = np.log(am[:, 3]), np.log(am[:, 2])
    for B in (1, 16, 256, 4096):
        lfb, lgb, Lb = S.block_means(lfA, B), S.block_means(lgA, B), S.block_means(lpa, B)
        (aa, bb, cc), se, r2, _ = S.ols(np.column_stack([lgb, Lb]), lfb)
        pc = S.partial_corr(lfb, Lb, lgb)
        tg = tb[str(B)]
        ok = all(abs(x - y) <= 1e-9 for x, y in ((aa, tg["a"]), (bb, tg["b"]), (se[1], tg["se_b"]), (r2, tg["R2"]), (pc, tg["pcorr"]), (Lb.std(), tg["sd_logP"])))
        report("V8 blocks B=%d a/b/se/R2/pcorr/sd (Table 6.3)" % B, "%.3f/%.3f/%.3f/%.4f/%.3f/%.3f" % (aa, bb, se[1], r2, pc, Lb.std()),
               "%.3f/%.3f/%.3f/%.4f/%.3f/%.3f" % (tg["a"], tg["b"], tg["se_b"], tg["R2"], tg["pcorr"], tg["sd_logP"]), 1e-9, ok)
        report("V8 blocks B=%d corr(logF,logP), corr(logP,logg)" % B, "%.3f/%.3f" % (tg["corr_logF_logP"], tg["corr_logP_logg"]), "[0.92,0.95]/-", "-",
               0.915 <= tg["corr_logF_logP"] <= 0.955, "%s", "%s")
    bl = [tb[str(B)]["b"] for B in (1, 16, 256, 4096)]
    report("V8 b decays 0.693 -> 0.215 -> 0.149 -> 0.075", "/".join("%.3f" % v for v in bl), "0.693/0.215/0.149/0.075", 1.5e-3, all(abs(v - t) <= 1.5e-3 for v, t in zip(bl, (0.693, 0.215, 0.149, 0.075))))
    close("V8 B=4096: b in s.e. (1.5)", tb["4096"]["b"] / tb["4096"]["se_b"], 1.5, 0.1, "%.2f", "%.1f")
    close("V8 B=256: b in s.e. (11)", tb["256"]["b"] / tb["256"]["se_b"], 11.0, 1.0, "%.1f", "%.0f")
    ts = tb["tight_subset"]
    report("V8 blocks tight subset n/a/b/R2 (680/1.914/0.886/0.945)", "%d/%.3f/%.3f/%.3f" % (ts["n"], ts["a"], ts["b"], ts["R2"]), "680/1.914/0.886/0.945", 1.5e-3,
           ts["n"] == 680 and abs(ts["a"] - 1.914) <= 1.5e-3 and abs(ts["b"] - 0.886) <= 1.5e-3 and abs(ts["R2"] - 0.945) <= 1.5e-3)
    close("V8 tight subset s.e.(a) (0.026)", ts["se_a"], 0.026, 5e-4, "%.4f", "%.3f")
    close("V8 tight subset s.e.(b) (0.011)", ts["se_b"], 0.011, 5e-4, "%.4f", "%.3f")
    t6 = ctx.table("tab_fits")["8p4e9"]["1000000"]
    close("V8 tight subset vs Table 5.1: a, in s.e. of the difference (1.6)", abs(ts["a"] - t6["a"]) / np.hypot(ts["se_a"], t6["se_a"]), 1.6, 0.05, "%.3f", "%.1f")
    close("V8 tight subset vs Table 5.1: b, in s.e. of the difference (1.3)", abs(ts["b"] - t6["b"]) / np.hypot(ts["se_b"], t6["se_b"]), 1.3, 0.05, "%.3f", "%.1f")
    sa = ctx.rng.choice(len(am), 20, replace=False)
    lpa_chk = E.logP_exact(am[sa, 1].astype(np.longdouble), xs=[10 ** 6])[:, 0]
    below("V8 all-gap log|P_1e6| recomputed (20), max dev", float(np.max(np.abs(lpa_chk - lpa[sa]))), 1e-8)
