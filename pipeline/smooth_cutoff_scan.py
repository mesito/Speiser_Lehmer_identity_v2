#!/usr/bin/env python3
"""smooth_cutoff_scan.py -- the Euler regression with smooth prime cutoffs
(Section 5.4 and Table 5.2 of the paper), on the full primary ensemble.

For every tight pair of the primary ensemble and every cutoff X in
{1e2, ..., 1e7} the prime sum

    L_X^Phi(t0) = sum_{p^k} Phi(p^k / X) p^{-k/2} cos(k t0 log p) / k

is evaluated exactly in k for three weights Phi:
    sharp : Phi(u) = 1{u <= 1}                        (the product of the paper)
    exp4  : Phi(u) = exp(-u^4), truncated at u <= 4    (Phi(4) = e^-256)
    bump  : Phi(u) = 1 (u <= 1/2), exp(1 - 1/(1-(2u-1)^2)) (1/2 < u < 1), 0 (u >= 1)
Primes to 4e7 are needed (exp4 at X = 1e7).  The regression
    log F = a log g + b L_X^Phi + c
is then fitted for each (Phi, X).  Writes data/logP_smooth_8p4e9.npy
(2736 x 3 x 6, order sharp/exp4/bump) and data/smooth_cutoff_scan.csv.
Runtime: about 10 minutes on one core.
"""
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tightpairs import data as D, euler as E, stats as S  # noqa: E402
from tightpairs.constants import XS  # noqa: E402

LABELS = ("sharp", "exp4", "bump")


def main():
    hl = D.load_hilo()
    zl = D.load_longdouble(hl=hl)
    idx = D.tight_pairs(hl[:, 0])
    t0 = (zl[idx] + zl[idx + 1]) / 2
    print("sieving primes to 4e7 ...", flush=True)
    pr = E.Primes(4 * 10 ** 7)
    print("  %d primes" % len(pr.p), flush=True)
    out = np.empty((len(idx), len(LABELS), len(XS)))
    t = time.time()
    for j, tt in enumerate(t0):
        ph = E.phases(tt, pr)
        for a, label in enumerate(LABELS):
            for b, X in enumerate(XS):
                out[j, a, b] = E.logP_smooth(ph, pr, X, label)
        if (j + 1) % 200 == 0:
            print("  %d/%d (%.0fs)" % (j + 1, len(idx), time.time() - t), flush=True)
    np.save(D.path("logP_smooth_8p4e9.npy"), out)
    # consistency: the sharp weight reproduces the shipped exact product
    if os.path.exists(D.path("logP_pairs_8p4e9.npy")):
        lp = np.load(D.path("logP_pairs_8p4e9.npy"))
        print("max |sharp - logP_pairs_8p4e9| = %.2e" % np.max(np.abs(out[:, 0, :] - lp)))
    fa = np.load(D.path("floors_all_8p4e9.npy"))
    assert np.array_equal(fa[:, 0].astype(int) - 1, idx)
    lg, lf = np.log(fa[:, 2]), np.log(fa[:, 5])
    rows = []
    for a, label in enumerate(LABELS):
        for b, X in enumerate(XS):
            (aa, bb, cc), se, r2, _ = S.ols(np.column_stack([lg, out[:, a, b]]), lf)
            rows.append((label, X, aa, se[0], bb, se[1], r2))
            print("%-6s 1e%d  a=%.4f (%.4f)  b=%.4f (%.4f)  R2=%.4f" % (label, int(np.log10(X)), aa, se[0], bb, se[1], r2))
    with open(D.path("smooth_cutoff_scan.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cutoff", "X", "a", "se_a", "b", "se_b", "R2"])
        for r in rows:
            w.writerow([r[0], int(r[1]), "%.6f" % r[2], "%.6f" % r[3], "%.6f" % r[4], "%.6f" % r[5], "%.6f" % r[6]])
    print("wrote smooth_cutoff_scan.csv (%.0fs)" % (time.time() - t))


if __name__ == "__main__":
    main()
