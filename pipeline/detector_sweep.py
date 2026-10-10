#!/usr/bin/env python3
"""detector_sweep.py -- the transposed detector of Section 3.6 on the gap interior
of every tight pair of the primary ensemble:

    Ghat(t) = sum_{m outside the pair, |m-n|<=W} 1/(t-gamma_m)^2 + 2/((t-gamma_n)(t-gamma_{n+1}))

sampled at 257 interior points per pair, W = 250 ordinates on each side
(clipped at the ensemble edges), all ordinate arithmetic in 80-bit precision.
Also records far, S_on, h_thr, r, H = sqrt(2/far) and H/(g/2).  Writes
data/detector_sweep_8p4e9.csv.  Proposition 3.5 shows that no pair with
C_mid > (8(1-sqrt(kappa))^2)^{-1} can show a non-negative value; the sweep is a
test of the implementation.  Runtime about two minutes.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tightpairs import data as D, geometry as G  # noqa: E402
from tightpairs.constants import W  # noqa: E402

NPT = 257


def main():
    hl = D.load_hilo()
    zl = D.load_longdouble(hl=hl)
    idx = D.tight_pairs(hl[:, 0])
    us = np.linspace(0.0, 1.0, NPT + 2)[1:-1].astype(np.longdouble)
    rows = []
    for i in idx:
        q = G.pair_quantities(zl, i, W)
        ga, gb = zl[i], zl[i + 1]
        lo, hi = G.window(i, len(zl), W)
        nsc = 0
        gmax = -np.inf
        for u in us:
            t = ga + (gb - ga) * u
            Gv = float(np.sum(1 / (t - zl[lo]) ** 2) + np.sum(1 / (t - zl[hi]) ** 2) + 2 / ((t - ga) * (t - gb)))
            gmax = max(gmax, Gv)
            if Gv >= 0:
                nsc += 1
        H = np.sqrt(2.0 / q["far"])
        rows.append((i + 1, float(ga), float(gb), q["g"], q["t0"], q["far"], q["Son"], q["hthr"], q["r"], H, H / (q["g"] / 2), nsc, gmax))
    hdr = "pair_idx,gamma_n,gamma_n1,g,t0,far,S_on,h_thr,r,H,H_over_halfgap,n_sign_changes,max_Ghat"
    np.savetxt(D.path("detector_sweep_8p4e9.csv"), np.array(rows), delimiter=",", header=hdr, comments="",
               fmt=["%d", "%.18f", "%.18f", "%.12e", "%.12f", "%.10f", "%.10f", "%.10e", "%.10f", "%.10f", "%.10f", "%d", "%.6e"])
    a = np.array(rows)
    print("pairs %d, sign changes %d, min r %.6f, median H/(g/2) %.3f, largest Ghat %.4e" % (len(a), int(a[:, 11].sum()), a[:, 8].min(), np.median(a[:, 10]), a[:, 12].max()))


if __name__ == "__main__":
    main()
