"""V3 -- Theorem A(b),(c): the floor law and the midpoint curvature on the reference pairs
(Table 3.2); Corollary 3.5 (endpoint-midpoint identity)."""
import numpy as np
import mpmath as mp

from tightpairs import geometry as G, instrument as I
from tightpairs.constants import W
from tightpairs.report import close, rel, below, equal, info, section


def run(ctx):
    section("V3: floor law and curvature (Theorem A(b),(c), Table 3.2); endpoint identity")
    tf = ctx.table("tab_floorid")
    ref = ctx.csv("reference_8p4e9.csv")
    below("V3 max |R_250| (3.0e-6)", float(np.abs(ref["R250"]).max()), 3.0e-6)
    below("V3 min |R_250| >= 1.9e-7", -float(np.abs(ref["R250"]).min()), -1.9e-7)
    below("V3 R_250 explained by the omitted tail (5e-11)", float(np.abs(ref["R250"] + (ref["farall"] - ref["far250"]) / ref["Sonall"]).max()), 5e-11)
    below("V3 max |R_all| (7.3e-13)", float(np.abs(ref["Rall"]).max()), 7.3e-13)
    below("V3 max |curvature deviation| (2.9e-13)", float(np.abs(ref["curv_dev"]).max()), 2.9e-13)
    equal("V3 Table 3.2 rows equal reference file", all(abs(r["Rall"] - float(ref["Rall"][k])) < 1e-20 and abs(r["F"] / float(ref["F"][k]) - 1) < 1e-12 for k, r in enumerate(tf["rows"])), True)
    for ens, bound in (("100k", 2.3e-7), ("2M", 4.1e-7), ("lehmer", 7.1e-8)):
        b = ctx.csv("reference_%s.csv" % ens)
        below("V3 max |R_all| %s (%.1e)" % (ens, bound), float(np.abs(b["Rall"]).max()), bound)
        below("V3 max |curvature deviation| %s (%.1e)" % (ens, bound), float(np.abs(b["curv_dev"]).max()), bound)
    # fresh recomputation of the floor law on reference pairs (3 fast / 15 full)
    hl, zl = ctx.hl, ctx.zl
    mp.mp.dps = 30
    for k in range(15 if ctx.full else 3):
        i = int(ref["idx1"][k]) - 1
        ga = mp.mpf(float(hl[i, 0])) + mp.mpf(float(hl[i, 1]))
        gb = mp.mpf(float(hl[i + 1, 0])) + mp.mpf(float(hl[i + 1, 1]))
        t0 = (ga + gb) / 2
        tstar = I.extremum_mp(mp, t0)
        F, Zpp = I.floor_identity_mp(mp, tstar)
        pair = 1 / (tstar - ga) ** 2 + 1 / (tstar - gb) ** 2
        ts_ld = np.longdouble(mp.nstr(tstar, 25))
        fs, _ = G.far_lambda(zl, i, ts_ld, None)
        fs += G.mirror_sum(zl, ts_ld) + G.tail_second_moment(zl, ts_ld, False)
        Rall = float(F * (float(pair) + fs + I.archimedean_curvature(float(tstar))) / Zpp - 1)
        below("V3 pair %02d |R_all| recomputed (<= 1e-12)" % (k + 1), abs(Rall), 1e-12)
        rel("V3 pair %02d floor F equals reference" % (k + 1), float(F), float(ref["F"][k]), 1e-12)
        rel("V3 pair %02d |Z''(t*)| equals reference" % (k + 1), float(Zpp), float(ref["Zpp"][k]), 1e-10)
        # the floor from the shipped array (grid + parabola, Z_dd) against the 30-digit floor
        j = int(np.flatnonzero(ctx.idx8 == i)[0])
        rel("V3 pair %02d shipped floor vs 30-digit floor (<= 1e-5)" % (k + 1), ctx.f8[j, 5], float(F), 1e-5)
        with mp.workdps(50):
            below("V3 pair %02d |Z_dd - mpmath| at the extremum (<= 1e-13)" % (k + 1), abs(I.Z_dd(tstar, mp) - float(mp.siegelz(tstar))), 1e-13)
    # Theorem A(c) for the archimedean term: (log omega)'' = -7/(4t^2)(1+O(t^-2))
    mp.mp.dps = 30
    for t in (10.0, 7005.08, 8.4e9):
        tm = mp.mpf(t)
        om2 = -2 * (tm ** 2 - mp.mpf(1) / 4) / (tm ** 2 + mp.mpf(1) / 4) ** 2 - mp.re(mp.psi(1, mp.mpf(1) / 4 + 1j * tm / 2)) / 4
        f = lambda x: mp.log(mp.mpf(1) / 2 * (x ** 2 + mp.mpf(1) / 4) * mp.pi ** (-mp.mpf(1) / 4) * abs(mp.gamma(mp.mpf(1) / 4 + 1j * x / 2)))
        rel("V3 (log omega)'' closed form vs numerical, t=%g" % t, float(om2), float(mp.diff(f, tm, 2)), 1e-8)
        below("V3 |t^2 (log omega)''| <= 7/4, t=%g" % t, abs(float(om2 * tm * tm)), 1.75 + 1e-12)
    close("V3 t^2 (log omega)'' at t=8.4e9 -> -1.75", float(-7.0 / 4), -1.75, 1e-12, "%.4f", "%.2f")
    # Corollary 3.5: S_end = 2 far + positive series, and the bounds (3.17)
    r8 = ctx.rows["8p4e9"]
    worst = 0.0
    viol = 0
    sub = np.arange(len(ctx.idx8)) if ctx.full else ctx.rng.choice(len(ctx.idx8), 200, replace=False)
    for j in sub:
        i = ctx.idx8[j]
        q = G.pair_quantities(zl, i, W)
        series = G.endpoint_correction_series(zl, i, W)
        worst = max(worst, abs(q["Send"] - 2 * q["far"] - series) / q["Send"])
        lo, hi = G.window(i, len(zl), W)
        a = np.concatenate([q["t0"] - zl[lo], q["t0"] - zl[hi]]).astype(float)
        s4 = float(np.sum(a ** -4.0))
        g2 = q["g"] ** 2
        kap = q["kappa"]
        lower = g2 / 2 * (3 - kap) * s4
        upper = 3 * g2 / 2 * (1 - kap) ** -2 * s4
        if not (lower - 1e-12 <= q["Send"] - 2 * q["far"] <= upper + 1e-12 <= 3 * g2 / 2 * (1 - kap) ** -2 * q["dmin"] ** -2 * q["far"] + 1e-9):
            viol += 1
    below("V3 endpoint identity (3.16) in 80-bit arithmetic, %d pairs (ordinate rounding 1e-8)" % len(sub), worst, 1e-8)
    equal("V3 bounds (3.17) violated", viol, 0)
    # the identity with exact ordinates, in arbitrary precision, on three pairs
    mp.mp.dps = 40
    zf = lambda m: mp.mpf(float(hl[m, 0])) + mp.mpf(float(hl[m, 1]))
    for j in sub[:3]:
        i = int(ctx.idx8[j])
        ga, gb = zf(i), zf(i + 1)
        t0, e2 = (ga + gb) / 2, ((gb - ga) / 2) ** 2
        lo, hi = G.window(i, len(zl), W)
        ms = list(lo) + list(hi)
        Send = sum(1 / (zf(m) - ga) ** 2 + 1 / (zf(m) - gb) ** 2 for m in ms)
        far = sum(1 / (t0 - zf(m)) ** 2 for m in ms)
        ser = 2 * e2 * sum((3 * (t0 - zf(m)) ** 2 - e2) / ((t0 - zf(m)) ** 2 * ((t0 - zf(m)) ** 2 - e2) ** 2) for m in ms)
        below("V3 endpoint identity exact, pair %d (<= 1e-30)" % (i + 1), float(abs(Send - 2 * far - ser) / Send), 1e-30)
    eta = (r8[:, 7] / (2 * r8[:, 6]) - 1)                   # C_mid / (2 C_pair) - 1
    info("V3 max eta = S_end/(2 far) - 1 over the ensemble", float(eta.max()), "%.4f")
    equal("V3 eta >= 0 on every pair", bool(np.all(eta >= -1e-12)), True)
