#!/usr/bin/env python3
"""decode_platt.py -- decodes the partial LMFDB/Platt block file data/lmfdb_top.bin
(the first 10.07 MB of zeros_8436146000.dat) with the four integrity checks of
Section 2.2 and writes

  data/lmfdb_zeros_exact.npz   the exact 104-bit cumulative integers (cum_hi, cum_lo),
                               the block headers and the block index of each zero
  data/lmfdb_zeros_hilo.npy    every ordinate as a double-double (hi, lo), exact
                               to 3e-23, from which the 80-bit array is formed at
                               load time

Only complete blocks are used (a truncated final block is discarded).  If the
output files exist they are compared with the fresh decode instead of being
overwritten (pass --force to overwrite).  Runtime about 20 seconds.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tightpairs import data as D  # noqa: E402


def main(force=False):
    t = time.time()
    raw = open(D.path("lmfdb_top.bin"), "rb").read()
    ex = D.decode_platt(raw)
    n = len(ex["cum_lo"])
    print("file header: %d blocks; complete blocks decoded: %d; zeros: %d (no. %d to %d); checks (i)-(iv) passed (%.0fs)"
          % (ex["nblocks_header"], len(ex["block_t0"]), n, ex["block_n0"][0], ex["block_n1"][-1] - 1, time.time() - t), flush=True)
    hl = D.hilo_from_exact(ex)
    print("t range: %.2f .. %.2f" % (hl[0, 0], hl[-1, 0]))
    for name, obj in (("lmfdb_zeros_exact.npz", ex), ("lmfdb_zeros_hilo.npy", hl)):
        p = D.path(name)
        if os.path.exists(p) and not force:
            if name.endswith(".npz"):
                old = np.load(p)
                same = all(np.array_equal(old[k], ex[k]) for k in ("cum_lo", "cum_hi", "block_t0", "block_n0", "block_n1", "block_of"))
            else:
                same = np.array_equal(np.load(p), hl)
            print("%s exists: %s" % (name, "identical to the fresh decode" if same else "DIFFERS from the fresh decode"))
        else:
            if name.endswith(".npz"):
                np.savez(p, **{k: ex[k] for k in ("block_t0", "block_t1", "block_n0", "block_n1", "cum_lo", "cum_hi", "block_of")})
            else:
                np.save(p, hl)
            print("wrote", p)
    print("max |exact - (hi+lo)| = %.2e (%.0fs)" % (D.check_hilo(ex, hl), time.time() - t))


if __name__ == "__main__":
    main("--force" in sys.argv)
