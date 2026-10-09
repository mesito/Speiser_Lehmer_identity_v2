"""V1 -- ensembles, decoding, exact ordinates, instruments (Section 2)."""
import time
from fractions import Fraction

import numpy as np
import mpmath as mp

from tightpairs import data as D, instrument as I
from tightpairs.constants import PI_LD, N_PRIMARY, N_PAIRS_PRIMARY, N_PAIRS_2M, dens
from tightpairs.report import report, close, rel, equal, below, info, section


def _Z_rs2_double_2pi(t):
    """the fast instrument with the double-precision 2pi in the phase reduction (for the error budget)."""
    tld = np.longdouble(t)
    two_pi = np.longdouble(2 * np.pi)
    tau = np.sqrt(tld / two_pi)
    N = int(np.floor(tau))
    p = float(tau - N)
    nn = np.arange(1, N + 1).astype(np.longdouble)
    thcorr = 1 / (48 * tld) + 7 / (5760 * tld ** 3) + 31 / (80640 * tld ** 5)
    ph = (tld * np.log(tau / nn) - tld / 2 - PI_LD / 8 + thcorr) % two_pi
    main = 2.0 * float(np.sum(np.cos(np.asarray(ph, float)) / np.sqrt(np.asarray(nn, float))))
    psi = np.cos(2 * np.pi * (p * p - p - 1.0 / 16.0)) / np.cos(2 * np.pi * p)
    return main + (-1) ** (N + 1) * float(tau) ** -0.5 * psi


def run(ctx):
    section("V1: ensembles, decoding and instruments (Section 2)")
    hl, zl, zd, ex = ctx.hl, ctx.zl, ctx.zd, ctx.exact
    equal("V1 LMFDB size", int(zl.size), N_PRIMARY)
    equal("V1 complete blocks decoded", int(len(ex["block_t0"])), 110)
    equal("V1 block index continuity", bool(np.all(ex["block_n1"][:-1] == ex["block_n0"][1:])), True)
    equal("V1 block t continuity", bool(np.all(ex["block_t1"][:-1] == ex["block_t0"][1:])), True)
    equal("V1 80-bit ordinates strictly increasing", bool(np.all(np.diff(zl) > 0)), True)
    equal("V1 hi column equals the double array", bool(np.array_equal(hl[:, 0], zd)), True)
    below("V1 max |lo| <= half an ulp (4.8e-7)", float(np.abs(hl[:, 1]).max()), 4.8e-7)
    # exact ordinates vs the double-double representation
    t = time.time()
    if ctx.full:
        worst = D.check_hilo(ex, hl)
        n = len(hl)
    else:
        sel = ctx.rng.choice(len(hl), 3000, replace=False)
        scale = Fraction(1, 2 ** 101)
        worst = 0.0
        for i in sel:
            f = Fraction(float(ex["block_t0"][ex["block_of"][i]])) + Fraction((int(ex["cum_hi"][i]) << 64) | int(ex["cum_lo"][i])) * scale
            worst = max(worst, float(abs(f - Fraction(float(hl[i, 0])) - Fraction(float(hl[i, 1])))))
        n = len(sel)
    below("V1 max |exact - (hi+lo)| on %d ordinates (3e-23)" % n, worst, 3e-23)
    print("    (exact check: %.1fs)" % (time.time() - t))
    if ctx.full:
        raw = open(D.path("lmfdb_top.bin"), "rb").read()
        t = time.time()
        ex2 = D.decode_platt(raw)              # applies checks (i)-(iv)
        equal("V1 fresh decode: cumulative integers identical", bool(np.array_equal(ex2["cum_lo"], ex["cum_lo"]) and np.array_equal(ex2["cum_hi"], ex["cum_hi"])), True)
        print("    (fresh decode with checks (i)-(iv): %.1fs)" % (time.time() - t))
    close("V1 LMFDB first ordinate", zd[0], 8436146000.02, 0.01, "%.2f", "%.2f")
    close("V1 LMFDB last ordinate", zd[-1], 8436376999.73, 0.01, "%.2f", "%.2f")
    equal("V1 2M size", int(ctx.z6.size), 2001052)
    equal("V1 100k size", int(ctx.z1.size), 100000)
    close("V1 dens at 8.4e9", dens(zd.mean()), 3.3451, 5e-4, "%.5f", "%.4f")
    equal("V1 tight pairs 8.4e9", int(len(D.tight_pairs(zd))), N_PAIRS_PRIMARY)
    equal("V1 tight pairs 2M", int(len(D.tight_pairs(ctx.z6))), N_PAIRS_2M)
    equal("V1 tight pairs 100k", int(len(D.tight_pairs(ctx.z1))), 261)
    equal("V1 pair index arrays consistent", bool(np.array_equal(D.tight_pairs(zd), ctx.idx8) and np.array_equal(D.tight_pairs(ctx.z6), ctx.idx2)), True)
    # instruments
    mp.mp.dps = 50
    rng = np.random.default_rng(2026)
    sel = rng.choice(772000, 20 if ctx.full else 4, replace=False)
    e_dd, e_rs = [], []
    for i in sel:
        tm = (mp.mpf(float(hl[i, 0])) + mp.mpf(float(hl[i, 1])) + mp.mpf(float(hl[i + 1, 0])) + mp.mpf(float(hl[i + 1, 1]))) / 2
        e_dd.append(abs(I.Z_dd(tm, mp) - float(mp.siegelz(tm))))
        tl = (zl[i] + zl[i + 1]) / 2
        e_rs.append(abs(I.Z_rs2(tl) - float(mp.siegelz(mp.mpf(np.format_float_positional(tl, precision=25))))))
    below("V1 |Z_dd - mpmath| on %d mid-gaps (<= 8.5e-14)" % len(sel), max(e_dd), 8.5e-14)
    below("V1 |Z_rs2 - mpmath| on %d mid-gaps (<= 3.6e-8)" % len(sel), max(e_rs), 3.6e-8)
    e2pi = []
    for i in sel:
        td = float((zl[i] + zl[i + 1]) / 2)
        e2pi.append(abs(_Z_rs2_double_2pi(td) - float(mp.siegelz(mp.mpf(td)))))
    report("V1 Z_rs2 with double-precision 2pi: max error on %d mid-gaps (<= 4e-6, > 1e-7)" % len(sel), max(e2pi), 3.9e-6, "[1e-7,4e-6]", 1e-7 <= max(e2pi) <= 4e-6, "%.2e", "%.1e")
    tlow = mp.mpf("1200000.37")
    below("V1 |Z_dd - mpmath| at t=1.2e6 (<= 1e-14)", abs(I.Z_dd(tlow, mp) - float(mp.siegelz(tlow))), 1e-14)
    # the Riemann-Siegel coefficients: the residual after C_0..C_3 is the C_4 term
    tt = mp.mpf("7005.08")
    below("V1 Z_dd at t=7005.08 with C_0..C_3 (<= 1e-10)", abs(I.Z_dd(tt, mp, 4) - float(mp.siegelz(tt))), 1e-10)
    tf = ctx.table("tab_fits")
    rel("V1 smallest floor 8.4e9 (table)", ctx.f8[:, 5].min(), tf["smallest_floor_8p4e9"], 1e-12)
    rel("V1 smallest floor 2M (table)", ctx.f2[:, 5].min(), tf["smallest_floor_2M"], 1e-12)
    info("V1 smallest floor 8.4e9", ctx.f8[:, 5].min(), "%.4e")
    # a few floors against the reference instrument
    mp.mp.dps = 40
    for k in range(3):
        i = ctx.idx8[rng.integers(0, len(ctx.idx8))]
        ga = mp.mpf(float(hl[i, 0])) + mp.mpf(float(hl[i, 1]))
        gb = mp.mpf(float(hl[i + 1, 0])) + mp.mpf(float(hl[i + 1, 1]))
        f, _ = I.floor_max_dd(ga, gb, mp)
        j = int(np.flatnonzero(ctx.idx8 == i)[0])
        rel("V1 shipped floor recomputed, pair %d" % (i + 1), f, ctx.f8[j, 5], 1e-12)
