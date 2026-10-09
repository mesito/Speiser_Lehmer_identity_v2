#!/usr/bin/env python3
"""compute_logP_allgaps.py -- exact log|P_X(1/2+it)| (X = 1e6) at the 193,179
gap midpoints of the block-averaging test (Section 6.4).

Phases: the base phase U_p = (T0 log p) mod 2pi is formed once in 80-bit
precision with T0 = 8436146000; the offset (t - T0) log p <= 3.2e6 is then
added in double precision, where its rounding error is below 1e-9 rad.
Reads data/moment_allgaps.npy, writes data/logP_allgaps_1e6.npy.  Runtime
about one minute."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tightpairs import data as D, euler as E  # noqa: E402
from tightpairs.constants import T0_PRIMARY, TWO_PI_LD  # noqa: E402

X = 10 ** 6
pr = E.Primes(X)
U = np.asarray((np.longdouble(T0_PRIMARY) * pr.lnp_ld) % TWO_PI_LD, dtype=float)
a = np.load(D.path("moment_allgaps.npy"))
tm = a[:, 1]
assert np.all(tm >= T0_PRIMARY)
t = time.time()
out = E.logP_offset_batch(tm - T0_PRIMARY, U, pr, X)
np.save(D.path("logP_allgaps_1e6.npy"), out)
print("done %d points in %.0fs; mean %.4f sd %.4f" % (len(out), time.time() - t, out.mean(), out.std()))
