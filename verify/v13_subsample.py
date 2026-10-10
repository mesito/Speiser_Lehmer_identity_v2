"""V13 -- robustness of the Euler regression (Section 5.4): the rank subsample, the gap-only regression,
the variance ladder, errors in variables with the three-harmonic proxy L_X^(3); Table B.1 (rank subsample)."""
import numpy as np
import mpmath as mp

from tightpairs import data as D, euler as E, stats as S
from tightpairs.constants import XS
from tightpairs.report import close, rel, report, below, equal, section

# numbers quoted in the text of the paper (Sections 4.3, 5.1, 5.4, 6.2, 6.3)
TEXT = {
    "gslope_full": 1.953, "gcorr_full": 0.684, "gsd_full": 0.682,
    "gslope_sub": 1.865, "gcorr_sub": 0.683, "gsd_sub": 0.676,
    "shared_double": 280,
}


def _fit(lg, L, lf):
    (a, b, c), se, r2, _ = S.ols(np.column_stack([lg, L]), lf)
    return a, se[0], b, se[1], r2


def run(ctx):
    section("V13: rank subsample, gap-only regression, variance ladder, errors in variables (Section 5.4)")
    f8, lp8 = ctx.f8, ctx.lp8
    g8 = f8[:, 2]
    sub = D.rank_subsample(g8, 300)
    order = np.argsort(g8, kind="stable")
    equal("V13 subsample size", int(len(sub)), 300)
    equal("V13 subsample starts with the tightest pair", int(sub[0]), int(order[0]))
    equal("V13 subsample ends with the widest pair", int(sub[-1]), int(order[-1]))
    ranks = np.round(np.linspace(0, len(g8) - 1, 300)).astype(int)
    equal("V13 subsample = pairs at ranks round(k(N-1)/299)", bool(np.array_equal(sub, order[ranks])), True)
    equal("V13 first fifteen ranks 0,9,...,128", [int(x) for x in ranks[:15]], [0, 9, 18, 27, 37, 46, 55, 64, 73, 82, 91, 101, 110, 119, 128])
    ts = ctx.table("tab_sub")
    equal("V13 subsample indices equal Table B.1", [int(f8[i, 0]) for i in sub[:15]], ts["sub_idx1"])
    # membership with ordinates rounded to doubles
    zd = ctx.zd
    gd = np.diff(zd)[ctx.idx8]
    subd = D.rank_subsample(gd, 300)
    shared = len(set(sub.tolist()) & set(subd.tolist()))
    report("V13 subsample from double-precision gaps shares pairs", shared, TEXT["shared_double"] if TEXT["shared_double"] else "info", 0,
           TEXT["shared_double"] is None or shared == TEXT["shared_double"], "%d", "%s")
    lg, lf = np.log(g8), np.log(f8[:, 5])
    # full vs subsample fits (Table)
    for j, X in enumerate(XS):
        A = _fit(lg, lp8[:, j], lf)
        B = _fit(lg[sub], lp8[sub, j], lf[sub])
        tf, tsb = ts["full"][str(int(X))], ts["sub"][str(int(X))]
        ok = all(abs(x - y) <= 1e-10 for x, y in zip(A, (tf["a"], tf["se_a"], tf["b"], tf["se_b"], tf["R2"]))) and \
            all(abs(x - y) <= 1e-10 for x, y in zip(B, (tsb["a"], tsb["se_a"], tsb["b"], tsb["se_b"], tsb["R2"])))
        report("V13 fits X=1e%d full a/b/R2 | sub a/b/R2" % int(np.log10(X)), "%.4f/%.4f/%.4f | %.4f/%.4f/%.4f" % (A[0], A[2], A[4], B[0], B[2], B[4]),
               "table", 1e-10, ok, "%s", "%s")
        tfits = ctx.table("tab_fits")["8p4e9"][str(int(X))]
        below("V13 ... full fit equals Table 5.1", abs(A[0] - tfits["a"]) + abs(A[2] - tfits["b"]) + abs(A[4] - tfits["R2"]), 1e-10)
    # gap-only regression (Section 5.1)
    for key, ii in (("full", np.arange(len(f8))), ("sub", sub)):
        (sl, c0), se, r2, res = S.ols(lg[ii][:, None], lf[ii])
        tg = ts["g_only_" + key]
        ok = abs(sl - tg["slope"]) <= 1e-10 and abs(S.corr(lg[ii], lf[ii]) - tg["corr"]) <= 1e-10 and abs(res.std() - tg["resid_sd_nats"]) <= 1e-10
        report("V13 gap-only regression %s slope/corr/sd(nats)" % key, "%.4f/%.4f/%.4f" % (sl, S.corr(lg[ii], lf[ii]), res.std()),
               "%.4f/%.4f/%.4f" % (tg["slope"], tg["corr"], tg["resid_sd_nats"]), 1e-10, ok)
        close("V13 ... sd in dex = sd(nats)/log 10", tg["resid_sd_dex"], tg["resid_sd_nats"] / np.log(10), 1e-12, "%.4f", "%.4f")
        close("V13 text: gap-only sd in log10 units %s" % key, res.std() / np.log(10), {"full": 0.296, "sub": 0.293}[key], 5e-4, "%.4f", "%.3f")
        for nm in ("slope", "corr", "sd"):
            tv = TEXT.get("g%s_%s" % (nm, key))
            if tv is not None:
                val = {"slope": sl, "corr": S.corr(lg[ii], lf[ii]), "sd": res.std()}[nm]
                close("V13 text: gap-only %s %s" % (nm, key), val, tv, 5e-4, "%.4f", "%.3f")
    # variance ladder (Table)
    tl = ctx.table("tab_ladder")
    for ens, fa, lp in (("8p4e9", ctx.f8, ctx.lp8), ("2M", ctx.f2, ctx.lp2)):
        lgx, lfx, lfar = np.log(fa[:, 2]), np.log(fa[:, 5]), np.log(fa[:, 4])
        for j, X in ((2, 1e4), (5, 1e7)):
            L = lp[:, j]
            vals = (S.ols(lgx[:, None], lfx)[2], S.ols(np.column_stack([lgx, lfar]), lfx)[2], S.ols(np.column_stack([lgx, L]), lfx)[2],
                    S.ols(np.column_stack([lgx, L, lfar]), lfx)[2])
            tg = tl["%s_%d" % (ens, int(np.log10(X)))]
            ok = all(abs(v - tg[k]) <= 1e-10 for v, k in zip(vals, ("g", "g_far", "g_P", "g_P_far")))
            report("V13 ladder %s X=1e%d R2 g/g+far/g+P/g+P+far" % (ens, int(np.log10(X))), "/".join("%.4f" % v for v in vals),
                   "/".join("%.4f" % tg[k] for k in ("g", "g_far", "g_P", "g_P_far")), 1e-10, ok, "%s", "%s")
    t4 = tl["8p4e9_4"]
    report("V13 ladder 8.4e9 X=1e4 (0.467/0.747/0.941/0.942)", "%.3f/%.3f/%.3f/%.3f" % (t4["g"], t4["g_far"], t4["g_P"], t4["g_P_far"]), "0.467/0.747/0.941/0.942", 5e-4,
           all(abs(a - b) <= 5e-4 for a, b in zip((t4["g"], t4["g_far"], t4["g_P"], t4["g_P_far"]), (0.467, 0.747, 0.941, 0.942))), "%s", "%s")
    # errors in variables (Table)
    te = ctx.table("tab_eiv")
    pr = E.Primes(10 ** 7)
    for ens, fa, lp in (("8p4e9", ctx.f8, ctx.lp8), ("2M", ctx.f2, ctx.lp2)):
        l3 = ctx.arr("LX3_pairs_%s.npy" % ens)
        lgx, lfx = np.log(fa[:, 2]), np.log(fa[:, 5])
        for j, X in enumerate(XS):
            b = _fit(lgx, lp[:, j], lfx)[2]
            b3 = _fit(lgx, l3[:, j], lfx)[2]
            rx, rw = S.resid_on(lp[:, j], lgx[:, None]), S.resid_on(l3[:, j], lgx[:, None])
            lam = float(np.dot(rx, rw) / np.dot(rw, rw))
            tg = te[ens][str(int(X))]
            ok = abs(b - tg["b"]) <= 1e-10 and abs(b3 - tg["b3"]) <= 1e-10 and abs(lam - tg["lambda_meas"]) <= 1e-10
            report("V13 EIV %s X=1e%d b/b3/lambda_meas" % (ens, int(np.log10(X))), "%.4f/%.4f/%.4f" % (b, b3, lam), "%.4f/%.4f/%.4f" % (tg["b"], tg["b3"], tg["lambda_meas"]),
                   1e-10, ok)
            below("V13 ... |b3/b - lambda_meas|", abs(b3 / b - lam), 0.01)
            if ens == "8p4e9":
                close("V13 ... delta_X", E.delta_sup(X, pr), tg["delta_sup"], 1e-12, "%.4f", "%.4f")
                close("V13 ... lambda_rand", E.lambda_rand(X, pr), tg["lambda_rand"], 1e-12, "%.4f", "%.4f")
                below("V13 ... max |L3 - logP| <= delta_X", float(np.abs(l3[:, j] - lp[:, j]).max()), tg["delta_sup"])
    # fresh L_X^(3) at random midpoints
    hl, zl = ctx.hl, ctx.zl
    n = 40 if ctx.full else 6
    sel = ctx.rng.choice(len(ctx.idx8), n, replace=False)
    l3 = ctx.arr("LX3_pairs_8p4e9.npy")
    worst = 0.0
    for k in sel:
        i = ctx.idx8[k]
        ph = E.phases((zl[i] + zl[i + 1]) / 2, pr)
        worst = max(worst, max(abs(E.LX3(ph, pr, X) - l3[k, jx]) for jx, X in enumerate(XS)))
    below("V13 L_X^(3) recomputed on %d pairs, max dev" % n, worst, 1e-9)
    # subsample analyses quoted in Section 6
    td = ctx.table("tab_deficit")
    for j, X in enumerate(XS):
        ms = ctx.lp8[sub, j].mean()
        close("V13 deficit on the subsample X=1e%d (mean)" % int(np.log10(X)), ms, td[str(int(X))]["mean_sub"], 1e-12, "%.4f", "%.4f")
    # Table B.1: the first fifteen pairs of the subsample, 30-digit references
    ref = ctx.csv("reference_8p4e9_sub.csv")
    equal("V13 Table B.1 rows", int(ref.size), 15)
    equal("V13 Table B.1 pairs are the first 15 of the subsample", [int(x) for x in ref["idx1"]], ts["sub_idx1"])
    tds = ctx.table("tab_dispsub")
    close("V13 Table B.1 corr(delta, pred)", S.corr(ref["delta"], ref["prediction"]), tds["corr"], 1e-12, "%.7f", "%.7f")
    from tightpairs.geometry import far_lambda
    from tightpairs.constants import W
    mp.mp.dps = 30
    for k in range(15):
        i = int(ref["idx1"][k]) - 1
        # the midpoint as in pipeline/reference_pairs.py: exact (double-double) ordinates, rounded once to 80 bits;
        # the 80-bit average (zl[i] + zl[i+1])/2 can differ from it by one unit in the last place, 9e-10 at this height,
        # which moves lambda_far by far * 9e-10
        t0e = (mp.mpf(float(hl[i, 0])) + mp.mpf(float(hl[i, 1])) + mp.mpf(float(hl[i + 1, 0])) + mp.mpf(float(hl[i + 1, 1]))) / 2
        t0 = np.longdouble(mp.nstr(t0e, 25))
        _, lam = far_lambda(zl, i, t0, W)
        if k < 3 or ctx.full:
            rel("V13 Table B.1 row %d lambda_far recomputed" % (k + 1), lam, ref["lambda_far"][k], 1e-9, "%.10f", "%.10f")
    mp.mp.dps = 30
    from tightpairs import instrument as I
    rows = range(15) if ctx.full else range(1)
    for k in rows:
        i = int(ref["idx1"][k]) - 1
        ga = mp.mpf(float(hl[i, 0])) + mp.mpf(float(hl[i, 1]))
        gb = mp.mpf(float(hl[i + 1, 0])) + mp.mpf(float(hl[i + 1, 1]))
        t0 = (ga + gb) / 2
        delta = float(I.extremum_mp(mp, t0) - t0)
        rel("V13 Table B.1 row %d delta recomputed (30 digits)" % (k + 1), delta, ref["delta"][k], 1e-9, "%.10e", "%.10e")
