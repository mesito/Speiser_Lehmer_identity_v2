#!/usr/bin/env python3
"""compute_windows.py -- the windows at heights 10^12 and 10^21 (Table 3.1, Table 6.1, Sections 7.1, 7.3, 7.4).

Reads Odlyzko's tables zeros3 (zeros 10^12 + 1 .. 10^12 + 10^4, base B = 267653395647,
accuracy 10^-8) and zeros4 (zeros 10^21 + 1 .. 10^21 + 10^4, base B = 144176897509546973000,
accuracy 'probably' 10^-6) through tightpairs.data.load_window: each table lists the offsets
gamma - B.  All pair geometry uses only differences of ordinates, so it is computed from the
offsets in 80-bit precision; the mean density is dens(B + t0).  A tight pair is s < 0.15
with s = g dens(B + t0), as in every other ensemble; the window W = 250 is clipped at the
ends of the table.

Writes
  data/lehmer_rows_1e12.npy, data/lehmer_rows_1e21.npy
        (idx 1-based within the table, t0 - B, g, s, S_on, far, C, C_mid, h_thr, r)
  data/floors_1e12.npy
        (idx 1-based, t0 - B, g, s, far, floor) for the tight pairs of the 10^12 window, floor
        = max|Z| over the open gap with the reference instrument Z_dd (C_0..C_3), the gap
        endpoints being B + offset exactly.  At 10^21 the main sum has 4.8e9 terms and no
        floors are computed.
Runtime: about five minutes (the floors).
"""
import os
import sys
import time

import numpy as np
import mpmath as mp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tightpairs import data as D, geometry as G, instrument as I  # noqa: E402
from tightpairs.constants import W, S_TIGHT, dens  # noqa: E402


def window_rows(name):
    """(base, offsets, tight-pair indices, rows table) of window `name`."""
    base, off = D.load_window(name)
    zl = off.astype(np.longdouble)
    g = np.diff(off)
    t0 = 0.5 * (off[:-1] + off[1:])
    s = g * dens(base + t0)
    idx = np.flatnonzero(s < S_TIGHT)
    rows = G.rows_table(zl, idx, W)
    rows[:, 3] = rows[:, 2] * dens(base + rows[:, 1])         # s with the true height
    return base, off, idx, rows


def floors_window(base, off, idx, rows):
    mp.mp.dps = 40
    out = np.empty((len(idx), 6))
    B = mp.mpf(base)
    for j, i in enumerate(idx):
        ga = B + mp.mpf(repr(float(off[i])))
        gb = B + mp.mpf(repr(float(off[i + 1])))
        f, _ = I.floor_max_dd(ga, gb, mp)
        out[j] = (i + 1, rows[j, 1], rows[j, 2], rows[j, 3], rows[j, 5], f)
    return out


def main():
    t = time.time()
    for ens, name in D.WINDOWS.items():
        base, off, idx, rows = window_rows(name)
        np.save(D.path("lehmer_rows_%s.npy" % ens), rows)
        print("%s: base %d, %d ordinates, %d tight pairs, min s %.4f (%.0fs)"
              % (ens, base, len(off), len(idx), rows[:, 3].min(), time.time() - t), flush=True)
        if ens == "1e12":
            fl = floors_window(base, off, idx, rows)
            np.save(D.path("floors_1e12.npy"), fl)
            print("1e12 floors: smallest %.4e, median %.4e (%.0fs)" % (fl[:, 5].min(), np.median(fl[:, 5]), time.time() - t), flush=True)


if __name__ == "__main__":
    main()
