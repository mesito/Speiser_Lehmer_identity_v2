#!/usr/bin/env python3
"""compute_floors.py -- floors of every tight pair and of the control mid-gaps.

floor = max|Z| over the open gap, Z = Z_dd (main sum with double-double phases
and the Riemann-Siegel corrections C_0..C_3, realised error at most 9.1e-14 at t ~ 8.4e9),
33-point grid plus parabolic refinement (tightpairs.instrument.floor_max_dd);
the gap endpoints are the exact (double-double) ordinates.

Writes
  data/floors_all_8p4e9.npy   (idx 1-based, t0, g, s, far, floor)   2736 pairs, W = 250
  data/floors_all_2M.npy      (idx 1-based, t0, g, s, far, floor)   6088 pairs
  data/floors_all_100k.npy    (idx 1-based, t0, g, s, far, floor)   261 pairs
  data/floors_ctrl_8p4e9.npy  (idx 1-based, g, floor)               300 control mid-gaps
  data/lehmer_rows_<ens>.npy  (idx 1-based, t0, g, s, S_on, far, C, C_mid, h_thr, r)
                              for ens = 100k, 2M, 8p4e9 (the windows at 1e12 and
                              1e21: pipeline/compute_windows.py)
Runtime: about 15 minutes (primary), 10 minutes (2M) on one core.
"""
import os
import sys
import time

import numpy as np
import mpmath as mp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tightpairs import data as D, geometry as G, instrument as I  # noqa: E402
from tightpairs.constants import W  # noqa: E402


def mpz(hl, i):
    return mp.mpf(float(hl[i, 0])) + mp.mpf(float(hl[i, 1]))


def floors_for(zl, hl, idx, label):
    out = np.empty((len(idx), 6))
    t = time.time()
    for j, i in enumerate(idx):
        q = G.pair_quantities(zl, i, W)
        f, _ = I.floor_max_dd(mpz(hl, i), mpz(hl, i + 1), mp)
        out[j] = (i + 1, q["t0"], q["g"], q["s"], q["far"], f)
        if (j + 1) % 500 == 0:
            print("  %s %d/%d (%.0fs)" % (label, j + 1, len(idx), time.time() - t), flush=True)
    return out


def main(which=("8p4e9", "2M", "100k", "ctrl")):
    t = time.time()
    mp.mp.dps = 40
    if "8p4e9" in which or "ctrl" in which:
        hl = D.load_hilo()
        zl = D.load_longdouble(hl=hl)
        zd = hl[:, 0]
        idx = D.tight_pairs(zd)
        if "8p4e9" in which:
            fa = floors_for(zl, hl, idx, "8p4e9")
            np.save(D.path("floors_all_8p4e9.npy"), fa)
            np.save(D.path("lehmer_rows_8p4e9.npy"), G.rows_table(zl, idx, W))
            print("8p4e9: %d pairs, smallest floor %.4e (%.0fs)" % (len(idx), fa[:, 5].min(), time.time() - t), flush=True)
        if "ctrl" in which:
            sel = D.control_indices(zd)
            np.save(D.path("ctrl_idx_8p4e9.npy"), sel)
            fc = np.array([(i + 1, float(zl[i + 1] - zl[i]), I.floor_max_dd(mpz(hl, i), mpz(hl, i + 1), mp)[0]) for i in sel])
            np.save(D.path("floors_ctrl_8p4e9.npy"), fc)
            print("controls: %d, median floor %.4f (%.0fs)" % (len(sel), np.median(fc[:, 2]), time.time() - t), flush=True)
    for ens, fname in (("2M", "zeros6"), ("100k", "zeros1")):
        if ens not in which:
            continue
        z = D.load_odlyzko(fname)
        zl = z.astype(np.longdouble)
        hl = np.column_stack([z, np.zeros_like(z)])
        idx = D.tight_pairs(z)
        np.save(D.path("lehmer_rows_%s.npy" % ens), G.rows_table(zl, idx, W))
        fa = floors_for(zl, hl, idx, ens)
        np.save(D.path("floors_all_%s.npy" % ens), fa)
        print("%s: %d pairs, smallest floor %.4e (%.0fs)" % (ens, len(idx), fa[:, 5].min(), time.time() - t), flush=True)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("8p4e9", "2M", "100k", "ctrl"))
