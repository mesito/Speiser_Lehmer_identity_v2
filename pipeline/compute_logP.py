#!/usr/bin/env python3
"""compute_logP.py -- exact log|P_X(1/2+it0)| (GHK convention, all prime powers
p^k <= X, X = 1e2..1e7) and the kernel sums S_Delta(t0; X) (Delta = 100, 1000)
at the midpoints of every tight pair of the primary and the 2M ensembles and at
the 300 control mid-gaps of the primary ensemble (Sections 5.2-5.4, 6.3).

Writes data/logP_pairs_8p4e9.npy (2736 x 6), data/zsum_pairs_8p4e9.npy (2 x 2736 x 6),
data/logP_ctrl_8p4e9.npy (300 x 6), data/ctrl_idx_8p4e9.npy, data/logP_pairs_2M.npy,
data/zsum_pairs_2M.npy.  Midpoints are formed from the 80-bit ordinates.
Runtime: about 25 minutes on one core (the prime sums dominate).
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tightpairs import data as D, euler as E, kernel as K  # noqa: E402
from tightpairs.constants import DELTAS  # noqa: E402


def main(which=("8p4e9", "ctrl", "2M")):
    t = time.time()
    pr = E.Primes(10 ** 7)
    if "8p4e9" in which or "ctrl" in which:
        hl = D.load_hilo()
        zl = D.load_longdouble(hl=hl)
        zd = hl[:, 0]
        if "8p4e9" in which:
            idx = D.tight_pairs(zd)
            t0 = (zl[idx] + zl[idx + 1]) / 2
            print("8p4e9: %d pairs" % len(idx), flush=True)
            np.save(D.path("logP_pairs_8p4e9.npy"), E.logP_exact(t0, pr=pr))
            print("  prime sums %.0fs" % (time.time() - t), flush=True)
            np.save(D.path("zsum_pairs_8p4e9.npy"), np.array([K.zsum(zd, zl, t0, Dl) for Dl in DELTAS]))
            print("  kernel sums %.0fs" % (time.time() - t), flush=True)
        if "ctrl" in which:
            sel = D.control_indices(zd)
            np.save(D.path("ctrl_idx_8p4e9.npy"), sel)
            np.save(D.path("logP_ctrl_8p4e9.npy"), E.logP_exact((zl[sel] + zl[sel + 1]) / 2, pr=pr))
            print("controls %.0fs" % (time.time() - t), flush=True)
    if "2M" in which:
        z = D.load_odlyzko("zeros6")
        zl = z.astype(np.longdouble)
        idx = D.tight_pairs(z)
        t0 = (zl[idx] + zl[idx + 1]) / 2
        np.save(D.path("logP_pairs_2M.npy"), E.logP_exact(t0, pr=pr))
        np.save(D.path("zsum_pairs_2M.npy"), np.array([K.zsum(z, zl, t0, Dl) for Dl in DELTAS]))
        print("2M %.0fs" % (time.time() - t), flush=True)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("8p4e9", "ctrl", "2M"))
