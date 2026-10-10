#!/usr/bin/env python3
"""verify_all.py -- independent recomputation of every numerical claim of

    M. Ismail, "Tight zero pairs of the Riemann zeta function at height 8.4e9:
    the Speiser-Lehmer identity, the displacement law and the Euler-Hadamard
    decomposition of the floor" (2026).

Thirteen verification groups V1-V13 (verify/v01_*.py ... v13_*.py) map to the
sections of the paper (Appendix A, Table A.1).  Each check prints
    name | computed | target | tolerance | PASS/FAIL
and the script exits with status 1 if any check fails.  Targets are the numbers
of the paper: the table values come from tables/*.json written by
make_tables.py (the same files that produce the LaTeX tables), the numbers
quoted in the text are written into the groups.

Usage:  python3 verify_all.py [--full] [--only V3,V5] [--list]
  default : all groups, with the shipped arrays checked against fresh
            recomputations on random subsets (about 3.5 minutes on one core)
  --full  : complete recomputations (every pair, every reference pair, the
            exact-ordinate check on every ordinate, the Monte-Carlo and
            importance-sampling tiers; about 28 minutes)

Data root: $RH_DATA (default ./data); tables: $RH_TABLES (default ./tables).
"""
import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tightpairs import report  # noqa: E402
from verify.context import Context  # noqa: E402

GROUPS = ["v01_data", "v02_identity", "v03_floorlaw", "v04_sweep", "v05_displacement", "v06_euler",
          "v07_decomposition", "v08_mediation", "v09_farfield", "v10_gue", "v11_certificate", "v12_tails",
          "v13_subsample"]


def main(argv):
    full = "--full" in argv
    only = None
    if "--only" in argv:
        only = {s.strip().upper() for s in argv[argv.index("--only") + 1].split(",")}
    if "--list" in argv:
        for g in GROUPS:
            print(g)
        return 0
    ctx = Context(full=full)
    t0 = time.time()
    print("verify_all.py: %s tier" % ("full" if full else "default"))
    for g in GROUPS:
        tag = "V" + str(int(g[1:3]))
        if only and tag not in only:
            continue
        t = time.time()
        importlib.import_module("verify." + g).run(ctx)
        print("    [%s done in %.1fs]" % (tag, time.time() - t))
    print("\ntotal %.1fs" % (time.time() - t0))
    return report.summary()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
