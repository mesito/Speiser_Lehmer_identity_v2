#!/usr/bin/env python3
"""reference_pairs.py -- arbitrary-precision references on selected tight pairs
(Section 3.5, Section 4.3, Appendix B of the paper).

For each selected pair, with exact (double-double) ordinates and mpmath at 30
digits:
  t*        critical point of Z in the gap (findroot on d/dt siegelz, seeded at t0)
  delta     t* - t0 and the prediction (g^2/8) lambda_far at W = 250 (Theorem B)
  F         |Z(t*)|, the floor;  Zpp = |Z''(t*)|
  S_on(t*)  (t*-gamma_n)^-2 + (t*-gamma_{n+1})^-2 + far(t*), the pair terms in
            mpmath, the far field in 80-bit precision, with W = 250 ("250"),
            with every ordinate of the file plus the mean-density tail and the
            mirror terms ("all"), and the archimedean curvature (log omega)''
  R         F * S_on(t*) / |Z''(t*)| - 1  (Theorem A(c))
  curv      -(log|Z|)''(t0) / S_on(t0) - 1 with S_on(t0) = 8/g^2 + far(t0) ("all")
            (Theorem A(b))
  r2        8 / (g^2 S_on(t0)) at W = 250 (the Speiser ratio squared)

Ensembles: the fifteen tightest pairs (smallest normalised spacing s) of the
primary ensemble ("8p4e9"); the first fifteen pairs of the rank subsample of the
primary ensemble (tightpairs.data.rank_subsample: ranks 0, 9, 18, ... of the gap,
"8p4e9_sub", Table B.1); the thirty tightest pairs of the first 1e5 zeros; the
fifteen tightest of the 2M ensemble; and Lehmer's pair (gamma_6709, gamma_6710).
Except for 8p4e9_sub, which is in rank order, each set is ordered by increasing s.
Writes data/reference_<ens>.csv.  Runtime: about 4 minutes (primary), 1 minute
(the rest).
"""
import os
import sys
import time

import numpy as np
import mpmath as mp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tightpairs import data as D, geometry as G, instrument as I  # noqa: E402
from tightpairs.constants import W, dens  # noqa: E402

HEADER = ("rank,idx1,g,lambda_far,delta,prediction,abs_rel_err_pct,F,Zpp,Son250,Sonall,"
          "R250,Rall,curv_dev,r2,far250,farall")
FMT = ["%d", "%d", "%.12e", "%.12f", "%.12e", "%.12e", "%.5f", "%.12e", "%.12e", "%.12e", "%.12e",
       "%.4e", "%.4e", "%.4e", "%.10f", "%.10f", "%.10f"]


def run_pair(zl, i, ga_mp, gb_mp, lower_complete, label):
    t0 = (ga_mp + gb_mp) / 2
    g = gb_mp - ga_mp
    t0_ld = np.longdouble(mp.nstr(t0, 25))
    far250, lam250 = G.far_lambda(zl, i, t0_ld, W)
    farall, _ = G.far_lambda(zl, i, t0_ld, None)
    farall += G.mirror_sum(zl, t0_ld) + G.tail_second_moment(zl, t0_ld, lower_complete)
    tstar = I.extremum_mp(mp, t0)
    delta = tstar - t0
    pred = g ** 2 / 8 * lam250
    err = 100 * abs(float(delta - pred) / float(delta))
    F, Zpp = I.floor_identity_mp(mp, tstar)
    pair = 1 / (tstar - ga_mp) ** 2 + 1 / (tstar - gb_mp) ** 2
    ts_ld = np.longdouble(mp.nstr(tstar, 25))
    fs250, _ = G.far_lambda(zl, i, ts_ld, W)
    fsall, _ = G.far_lambda(zl, i, ts_ld, None)
    fsall += G.mirror_sum(zl, ts_ld) + G.tail_second_moment(zl, ts_ld, lower_complete)
    om2 = I.archimedean_curvature(float(tstar))
    Son250 = float(pair) + fs250
    Sonall = float(pair) + fsall + om2
    R250 = float(F * (float(pair) + fs250) / Zpp - 1)
    Rall = float(F * Sonall / Zpp - 1)
    curv = I.log_curvature_mp(mp, t0)
    Son0 = 8 / g ** 2 + farall + I.archimedean_curvature(float(t0))
    curv_dev = float(curv / Son0 - 1)
    r2 = float(8 / (g ** 2 * (far250 + 8 / g ** 2)))
    row = (i + 1, float(g), lam250, float(delta), float(pred), err, float(F), float(Zpp), Son250, Sonall,
           R250, Rall, curv_dev, r2, far250, farall)
    print("%s idx %d g=%.4e delta=%+.4e pred=%+.4e err=%.3f%% R250=%+.2e Rall=%+.2e curv=%+.2e r2=%.6f"
          % (label, i + 1, float(g), float(delta), float(pred), err, R250, Rall, curv_dev, r2), flush=True)
    return row


def write(name, rows):
    r = np.array([(k + 1,) + tuple(row) for k, row in enumerate(rows)])
    np.savetxt(D.path(name), r, delimiter=",", header=HEADER, comments="", fmt=FMT)
    print("wrote", D.path(name), "corr(delta,pred)=%.7f median err %.3f%% max %.3f%% max|Rall| %.2e max|curv| %.2e"
          % (np.corrcoef(r[:, 4], r[:, 5])[0, 1], np.median(r[:, 6]), r[:, 6].max(), np.abs(r[:, 12]).max(), np.abs(r[:, 13]).max()), flush=True)


def main(which=("8p4e9", "8p4e9_sub", "100k", "2M", "lehmer")):
    mp.mp.dps = 30
    t = time.time()
    if "8p4e9" in which or "8p4e9_sub" in which:
        hl = D.load_hilo()
        zl = D.load_longdouble(hl=hl)
        idx = D.tight_pairs(hl[:, 0])
        s = (hl[idx + 1, 0] - hl[idx, 0]) * dens((hl[idx, 0] + hl[idx + 1, 0]) / 2)
        g = np.asarray(zl[idx + 1] - zl[idx], dtype=float)
        sel = {"8p4e9": idx[np.argsort(s)][:15],              # the fifteen tightest pairs
               "8p4e9_sub": idx[D.rank_subsample(g)[:15]]}   # the first fifteen of the rank subsample
        for ens in ("8p4e9", "8p4e9_sub"):
            if ens not in which:
                continue
            rows = []
            for i in sel[ens]:
                i = int(i)
                ga = mp.mpf(float(hl[i, 0])) + mp.mpf(float(hl[i, 1]))
                gb = mp.mpf(float(hl[i + 1, 0])) + mp.mpf(float(hl[i + 1, 1]))
                rows.append(run_pair(zl, i, ga, gb, False, ens))
            write("reference_%s.csv" % ens, rows)
            print("(%.0fs)" % (time.time() - t))
    for ens, fname, ntop in (("100k", "zeros1", 30), ("2M", "zeros6", 15)):
        if ens not in which:
            continue
        z = D.load_odlyzko(fname)
        zl = z.astype(np.longdouble)
        g = np.diff(z)
        tt = (z[:-1] + z[1:]) / 2
        s = g * np.log(tt / (2 * np.pi)) / (2 * np.pi)
        ii = np.flatnonzero(s < 0.15)
        ii = ii[np.argsort(s[ii])][:ntop]
        rows = [run_pair(zl, int(i), mp.mpf(float(z[i])), mp.mpf(float(z[i + 1])), True, ens) for i in ii]
        write("reference_%s.csv" % ens, rows)
    if "lehmer" in which:
        z = D.load_odlyzko("zeros1")
        zl = z.astype(np.longdouble)
        n = 6708
        rows = [run_pair(zl, n, mp.mpf(float(z[n])), mp.mpf(float(z[n + 1])), True, "lehmer")]
        write("reference_lehmer.csv", rows)
    print("total %.0fs" % (time.time() - t))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("8p4e9", "8p4e9_sub", "100k", "2M", "lehmer"))
