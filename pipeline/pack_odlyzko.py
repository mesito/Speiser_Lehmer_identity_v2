#!/usr/bin/env python3
"""pack_odlyzko.py -- packs Odlyzko's text tables zeros1 / zeros6 (one ordinate per
line, nine decimals) into data/zeros1.npz, data/zeros6.npz: 10^9 x ordinate as
integers, the first value plus uint32 deltas (exception list for the four
deltas >= 2^32).  Lossless: tightpairs.data.load_odlyzko() returns from the
.npz exactly the array that numpy.loadtxt returns from the text, bit for bit
(checked here).  The packed zeros6 is 7.5 MB against 14.4 MB for the gzip text.
Usage: pack_odlyzko.py [zeros1] [zeros6]   (reads data/<name>.gz or data/<name>)
"""
import gzip
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tightpairs import data as D  # noqa: E402

for name in sys.argv[1:] or ("zeros1", "zeros6"):
    src = D.path(name + ".gz") if os.path.exists(D.path(name + ".gz")) else D.path(name)
    out = D.path(name + ".npz")
    D.pack_odlyzko(src, out)
    txt = np.loadtxt(gzip.open(src, "rt") if src.endswith(".gz") else open(src))
    same = np.array_equal(txt, D.unpack_odlyzko(out))
    print("%s: %d ordinates, %.1f MB -> %.1f MB, bit-identical to the text: %s" % (name, len(txt), os.path.getsize(src) / 1e6, os.path.getsize(out) / 1e6, same))
    assert same
