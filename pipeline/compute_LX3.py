#!/usr/bin/env python3
"""compute_LX3.py -- the three-harmonic proxy L_X^(3)(t0) of (5.3) at the midpoints of every
tight pair of the primary and the 2M ensembles and at the 300 control mid-gaps (Section 5.4.2).

    L_X^(3)(t) = sum_{p <= X} [ p^{-1/2} cos(t log p) + p^{-1} cos(2 t log p)/2
                               + p^{-3/2} cos(3 t log p)/3 ]

i.e. the harmonics k <= 3 of every prime p <= X, against the exact log|P_X| of
compute_logP.py, which keeps exactly the prime powers p^k <= X.  The two observables
differ by delta_X(t) = sum_{sqrt X < p <= X} p^{-1} cos(2 t log p)/2
+ sum_{X^{1/3} < p <= X} p^{-3/2} cos(3 t log p)/3 - (harmonics k >= 4 with p^k <= X);
Section 5.4.2 measures the effect of this difference on the Euler regression.
Phases (t log p) mod 2pi in 80-bit precision, as in compute_logP.py.

Writes data/LX3_pairs_8p4e9.npy (2736 x 6), data/LX3_ctrl_8p4e9.npy (300 x 6),
data/LX3_pairs_2M.npy (6088 x 6), X = 1e2..1e7.  Runtime about 20 minutes on one core.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tightpairs import data as D, euler as E  # noqa: E402
from tightpairs.constants import XS  # noqa: E402


def LX3_at(t0, pr):
    out = np.empty((len(t0), len(XS)))
    for j, t in enumerate(t0):
        ph = E.phases(t, pr)
        for jx, X in enumerate(XS):
            out[j, jx] = E.LX3(ph, pr, X)
    return out


def main(which=("8p4e9", "ctrl", "2M")):
    t = time.time()
    pr = E.Primes(10 ** 7)
    if "8p4e9" in which or "ctrl" in which:
        hl = D.load_hilo()
        zl = D.load_longdouble(hl=hl)
        zd = hl[:, 0]
        if "8p4e9" in which:
            idx = D.tight_pairs(zd)
            np.save(D.path("LX3_pairs_8p4e9.npy"), LX3_at((zl[idx] + zl[idx + 1]) / 2, pr))
            print("8p4e9: %d pairs (%.0fs)" % (len(idx), time.time() - t), flush=True)
        if "ctrl" in which:
            sel = D.control_indices(zd)
            np.save(D.path("LX3_ctrl_8p4e9.npy"), LX3_at((zl[sel] + zl[sel + 1]) / 2, pr))
            print("controls (%.0fs)" % (time.time() - t), flush=True)
    if "2M" in which:
        z = D.load_odlyzko("zeros6")
        zl = z.astype(np.longdouble)
        idx = D.tight_pairs(z)
        np.save(D.path("LX3_pairs_2M.npy"), LX3_at((zl[idx] + zl[idx + 1]) / 2, pr))
        print("2M: %d pairs (%.0fs)" % (len(idx), time.time() - t), flush=True)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("8p4e9", "ctrl", "2M"))
