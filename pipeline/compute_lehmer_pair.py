#!/usr/bin/env python3
"""compute_lehmer_pair.py -- exact log|P_X| and kernel sums at the midpoint of
Lehmer's pair (gamma_6709, gamma_6710), Appendix C, Table C.2.
Writes data/lehmer_pair_7005.npz (gamma (2,), logP (6,), zsum (3, 6) for Delta = 100, 1000, 3000)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tightpairs import data as D, euler as E, kernel as K  # noqa: E402

z1 = D.load_odlyzko("zeros1")
n = 6708
ga, gb = z1[n], z1[n + 1]
t0 = np.longdouble(ga) / 2 + np.longdouble(gb) / 2
lp = E.logP_exact(np.array([t0]))[0]
zs = np.array([K.zsum(z1, z1.astype(np.longdouble), np.array([t0]), Dl)[0] for Dl in (100.0, 1000.0, 3000.0)])
np.savez(D.path("lehmer_pair_7005.npz"), gamma=np.array([ga, gb]), logP=lp, zsum=zs)
print("logP:", lp)
print("zsum (Delta=3000):", zs[2])
