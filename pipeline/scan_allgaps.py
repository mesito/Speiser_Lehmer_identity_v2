#!/usr/bin/env python3
"""scan_allgaps.py -- floors of every fourth gap of the primary ensemble
(Section 6.5): max|Z| over 12 interior samples, with the fast instrument
(main sum + C_0, 80-bit base phase at T0 = 8436146000 and double-precision
offsets; realised error about 5e-9, adequate for floors of order 1).

The scan is split into two halves (offset from a fixed seed), each written as
data/moment_floors_half{0,1}.npy with columns (gap_idx, t_mid, g, floor); the
merge step writes data/moment_allgaps.npy with columns (gap_idx, t_mid, g, floor,
L_X) where the last column is filled by compute_logP_allgaps.py (kept for
compatibility: the paper's L_X at X=1e6 is data/logP_allgaps_1e6.npy).
Runtime: about 40 minutes per half.
Usage:  scan_allgaps.py 0 | 1 | merge
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tightpairs import data as D  # noqa: E402
from tightpairs.constants import PI_LD, TWO_PI_LD, T0_PRIMARY  # noqa: E402

K = 12
BATCH = 512
N = 36642                                   # RS main-sum length, constant over the ensemble
LN2PI_LD = np.log(TWO_PI_LD)
nn = np.arange(1, N + 1)
v = np.log(nn)
w = 1.0 / np.sqrt(nn)
u = np.asarray((np.longdouble(T0_PRIMARY) * np.log(nn.astype(np.longdouble))) % TWO_PI_LD, dtype=float)


def theta_mod(t):
    tld = np.longdouble(t)
    th = ((tld / 2) * (np.log(tld) - LN2PI_LD) - tld / 2 - PI_LD / 8 + 1 / (48 * tld) + 7 / (5760 * tld ** 3) + 31 / (80640 * tld ** 5))
    return float(th % TWO_PI_LD)


def Z_batch(ts):
    ts = np.asarray(ts, float)
    d = ts - T0_PRIMARY
    A = np.array([theta_mod(t) for t in ts])
    ph = A[:, None] - u[None, :] - np.outer(d, v)
    main = 2.0 * np.sum(np.cos(ph) * w[None, :], axis=1)
    tau = np.sqrt(ts / (2 * np.pi))
    p = tau - N
    psi = np.cos(2 * np.pi * (p * p - p - 1.0 / 16.0)) / np.cos(2 * np.pi * p)
    return main + ((-1.0) ** (N + 1)) * tau ** -0.5 * psi


def gap_indices():
    zl = D.load_parsed()
    rng = np.random.default_rng(20260808)
    off = int(rng.integers(0, 4))
    return zl, np.arange(off, len(zl) - 1, 4)


def scan(H):
    zl, gidx_all = gap_indices()
    gidx = gidx_all[H::2]
    print("half %d: %d gaps" % (H, len(gidx)), flush=True)
    out = np.empty((len(gidx), 4))
    us = (np.arange(K) + 0.5) / K
    t = time.time()
    for c0 in range(0, len(gidx), BATCH // K):
        gi = gidx[c0:c0 + BATCH // K]
        ts = (zl[gi][:, None] + np.diff(zl)[gi][:, None] * us[None, :]).ravel()
        zv = np.abs(Z_batch(ts)).reshape(len(gi), K).max(1)
        out[c0:c0 + len(gi)] = np.column_stack([gi, 0.5 * (zl[gi] + zl[gi + 1]), zl[gi + 1] - zl[gi], zv])
        if (c0 // (BATCH // K)) % 200 == 0:
            print("  %d/%d (%.0fs)" % (c0, len(gidx), time.time() - t), flush=True)
    np.save(D.path("moment_floors_half%d.npy" % H), out)


def merge():
    a = np.vstack([np.load(D.path("moment_floors_half%d.npy" % h)) for h in (0, 1)])
    a = a[np.argsort(a[:, 1])]
    np.save(D.path("moment_allgaps.npy"), a)
    print("merged %d gaps" % len(a))


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "merge"
    merge() if arg == "merge" else scan(int(arg))
