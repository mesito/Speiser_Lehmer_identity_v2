#!/usr/bin/env python3
"""gue_mc.py -- the GUE Monte-Carlo reference of Section 7 (Observation 7.2 and
Table 7.3): conditioned far-field sums E[sum 1/x_m^2 | tight pair at the origin]
from Dumitriu-Edelman tridiagonal GUE matrices (beta = 2), semicircle-CDF
unfolding, bulk conditioning (|midpoint| <= 0.1 N), pairs with normalised gap
s < 0.25, windows |x| <= 25, 50, 100, 200.  Also the bulk spacing distribution
(small-gap tail ratios against pi^2 s^3/9, with Poisson standard errors; compared
with the exact sine-kernel values in Section 7.4) and the lag-1 correlation over
pairs of consecutive bulk spacings of the same matrix (the reference row of
Table 7.3).

Plan: N = 200, 300, 500 with 40000, 40000, 45000 matrices (two processes).
Writes data/gue_mc_results.npz (per-N conditioned gaps and sums per window and
the first 100,000 bulk spacings per N, all in single precision) and
data/gue_mc_summary.npz (the numbers quoted in the paper, computed in double
precision from the complete arrays before truncation).
Runtime about 8 minutes on two cores.
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tightpairs import data as D, gue as GU  # noqa: E402

S0 = 0.25
WINDOWS = [25.0, 50.0, 100.0, 200.0]
PLAN = {200: 40000, 300: 40000, 500: 45000}


def worker(task):
    from scipy.linalg import eigh_tridiagonal
    N, nmat, seed = task
    rng = np.random.default_rng(seed)
    edge = N / 2.0
    Xb = max(10.0, edge * 0.2)
    maxwin = min(WINDOWS[-1], edge - Xb - 5)
    wins = [w for w in WINDOWS if w <= maxwin]
    gaps_cond, sums, all_sp, lag_a, lag_b = [], {w: [] for w in wins}, [], [], []
    for _ in range(nmat):
        d = rng.standard_normal(N)
        e = np.sqrt(rng.chisquare(2 * np.arange(N - 1, 0, -1)) / 2.0)
        x = GU.gue_unfold(eigh_tridiagonal(d, e, eigvals_only=True, check_finite=False), N)
        midx = 0.5 * (x[:-1] + x[1:])
        sp = np.diff(x)
        bulk = np.abs(midx) <= Xb
        all_sp.append(sp[bulk])
        lag_a.append(sp[bulk][:-1])           # consecutive bulk spacings of the same matrix
        lag_b.append(sp[bulk][1:])
        for i in np.nonzero(bulk & (sp < S0))[0]:
            rel = np.delete(x - midx[i], [i, i + 1])
            arel = np.abs(rel)
            gaps_cond.append(sp[i])
            for wv in wins:
                m = arel < wv
                sums[wv].append(np.sum(1.0 / rel[m] ** 2) if m.any() else 0.0)
    return (N, nmat, np.array(gaps_cond), {w: np.array(v) for w, v in sums.items()}, np.concatenate(all_sp),
            np.concatenate(lag_a), np.concatenate(lag_b))


def main():
    tasks, seed = [], 12345
    for N, total in PLAN.items():
        for _ in range(2):
            tasks.append((N, total // 2, seed))
            seed += 1
    t0 = time.time()
    out = {N: [] for N in PLAN}
    with Pool(2) as p:
        for res in p.imap_unordered(worker, tasks):
            out[res[0]].append(res)
            print("N=%d chunk done: %d matrices, %d pairs (%.0fs)" % (res[0], res[1], len(res[2]), time.time() - t0), flush=True)
    res, summary = {}, {}
    for N, chunks in out.items():
        gaps = np.concatenate([c[2] for c in chunks])
        wins = sorted(set().union(*[set(c[3].keys()) for c in chunks]))
        sums = {w: np.concatenate([c[3][w] for c in chunks if w in c[3]]) for w in wins}
        sp = np.concatenate([c[4] for c in chunks])
        la = np.concatenate([c[5] for c in chunks])
        lb = np.concatenate([c[6] for c in chunks])
        res[N] = (sum(c[1] for c in chunks), gaps, sums, sp, la, lb)
        for w in wins:
            summary["N%d_w%d_mean" % (N, int(w))] = sums[w].mean()
            summary["N%d_w%d_se" % (N, int(w))] = sums[w].std(ddof=1) / np.sqrt(len(sums[w]))
        summary["N%d_npairs" % N] = len(gaps)
        print("N=%d: %d matrices, %d conditioned pairs, %d bulk spacings" % (N, res[N][0], len(gaps), len(sp)), flush=True)
    sp = res[500][3]
    C9 = np.pi ** 2 / 9
    for s0, key in ((0.05, "tail_005"), (0.10, "tail_010"), (0.15, "tail_015")):
        cnt = int(np.sum(sp < s0))
        summary[key] = cnt / len(sp) / (C9 * s0 ** 3)
        summary[key + "_count"] = cnt
        summary[key + "_se"] = np.sqrt(cnt) / len(sp) / (C9 * s0 ** 3)
    summary["n_bulk_spacings"] = len(sp)
    # lag-1 correlation over pairs of consecutive bulk spacings of the same matrix
    la, lb = res[500][4], res[500][5]
    summary["lag1"] = float(np.corrcoef(la, lb)[0, 1])
    summary["lag1_npairs"] = len(la)
    summary["total_pairs"] = sum(len(v[1]) for v in res.values())
    np.savez_compressed(D.path("gue_mc_results.npz"),
             **{"N%d_gaps" % N: v[1].astype(np.float32) for N, v in res.items()},
             **{"N%d_sp" % N: v[3][:100000].astype(np.float32) for N, v in res.items()},
             **{"N%d_nmats" % N: np.array([v[0]]) for N, v in res.items()},
             **{"N%d_sums_%d" % (N, int(w)): v[2][w].astype(np.float32) for N, v in res.items() for w in v[2]})
    np.savez(D.path("gue_mc_summary.npz"), **summary)
    for k in sorted(summary):
        print("  %s = %s" % (k, summary[k]))
    print("saved (%.0fs)" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
