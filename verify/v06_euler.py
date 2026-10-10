"""V6 -- Delta_0(X) (Lemma 9.1), the Euler regression (Observation 5.1, Table 5.1),
smooth cutoffs (Table 5.2), the variance ladder."""
import time

import numpy as np

from tightpairs import euler as E, stats as S
from tightpairs.constants import XS
from tightpairs.report import close, rel, report, below, section


def _cmp3(name, got, tg, tol=5e-4):
    report(name, "%.4f/%.4f/%.4f" % got, "%.4f/%.4f/%.4f" % tg, tol, all(abs(a - b) <= tol for a, b in zip(got, tg)))


def run(ctx):
    section("V6: harmonic truncation, Euler regression, smooth cutoffs (Sections 5.2-5.4)")
    pr = E.Primes(10 ** 7)
    term = E.Delta0_terms(pr)
    for X, tgt in ((10 ** 2, 0.2381784258), (10 ** 7, 0.2386607628)):
        close("V6 Delta_0(1e%d)" % int(np.log10(X)), term[pr.p <= X].sum(), tgt, 1e-9, "%.10f", "%.10f")
    close("V6 p=2 term of Delta_0", term[0], 0.15298926591521, 1e-13, "%.14f", "%.14f")
    close("V6 p=2 share of Delta_0 (64%)", term[0] / term.sum(), 0.64, 0.005, "%.3f", "%.2f")
    below("V6 Delta_0 tail beyond 1e7 < 2e-9", 1 / (4 * 1e7 * np.log(1e7)) * 1.001, 2e-9)
    tf = ctx.table("tab_fits")
    lg8, lf8 = np.log(ctx.f8[:, 2]), np.log(ctx.f8[:, 5])
    lg2, lf2 = np.log(ctx.f2[:, 2]), np.log(ctx.f2[:, 5])
    for j, X in enumerate(XS):
        for ens, lg, lf, lp in (("8p4e9", lg8, lf8, ctx.lp8), ("2M", lg2, lf2, ctx.lp2)):
            (a, b, c), se, r2, _ = S.ols(np.column_stack([lg, lp[:, j]]), lf)
            tg = tf[ens][str(int(X))]
            _cmp3("V6 %s X=1e%d a/b/R2 (Table 5.1)" % (ens, int(np.log10(X))), (a, b, r2), (tg["a"], tg["b"], tg["R2"]), 1e-9)
            _cmp3("V6 %s X=1e%d se(a)/se(b)/-" % (ens, int(np.log10(X))), (se[0], se[1], 0.0), (tg["se_a"], tg["se_b"], 0.0), 1e-9)
    # numbers quoted in the text
    t7 = tf["8p4e9"]["10000000"]
    close("V6 a at X=1e7 (1.937)", t7["a"], 1.937, 5e-4, "%.4f", "%.3f")
    close("V6 b at X=1e7 (0.936)", t7["b"], 0.936, 5e-4, "%.4f", "%.3f")
    close("V6 R2 at X=1e7 (0.953)", t7["R2"], 0.953, 5e-4, "%.4f", "%.3f")
    close("V6 b at X=1e2 (0.746)", tf["8p4e9"]["100"]["b"], 0.746, 5e-4, "%.4f", "%.3f")
    close("V6 R2 at X=1e2 (0.938)", tf["8p4e9"]["100"]["R2"], 0.938, 5e-4, "%.4f", "%.3f")
    below("V6 |a - 2| <= 0.07 at every X", max(abs(tf["8p4e9"][str(int(X))]["a"] - 2) for X in XS), 0.07)
    bs = [tf["8p4e9"][str(int(X))]["b"] for X in XS]
    close("V6 b rises by ~30 s.e. over the scan", (bs[-1] - bs[0]) / tf["8p4e9"]["10000000"]["se_b"], 34.5, 5.0, "%.1f", "%.1f")
    close("V6 a drift at X=1e7 in s.e. (5)", (2 - t7["a"]) / t7["se_a"], 5.3, 1.0, "%.1f", "%.1f")
    t2 = tf["2M"]
    _cmp3("V6 2M X=1e4 a/b/R2 text (1.951/0.911/0.950)", (t2["10000"]["a"], t2["10000"]["b"], t2["10000"]["R2"]), (1.951, 0.911, 0.950), 5e-4)
    _cmp3("V6 2M X=1e7 -/b/R2 text (0.947/0.979)", (0.0, t2["10000000"]["b"], t2["10000000"]["R2"]), (0.0, 0.947, 0.979), 5e-4)
    lad = tf["ladder_1e4"]
    (_, _), _, r2g, _ = S.ols(lg8[:, None], lf8)
    close("V6 R2(g only) (0.467)", r2g, 0.467, 5e-4, "%.4f", "%.3f")
    rel("V6 R2(g only) equals table", r2g, lad["g"], 1e-9)
    close("V6 g-only log-log slope (1.95)", S.ols(lg8[:, None], lf8)[0][0], 1.953, 5e-3, "%.3f", "%.3f")
    close("V6 corr(log g, log F) (0.68)", S.corr(lg8, lf8), 0.684, 5e-3, "%.3f", "%.3f")
    close("V6 R2(g + log far) (0.747)", S.ols(np.column_stack([lg8, np.log(ctx.f8[:, 4])]), lf8)[2], 0.747, 5e-4, "%.4f", "%.3f")
    close("V6 R2(g + logP) X=1e4 (0.941)", lad["g_logP"], 0.941, 5e-4, "%.4f", "%.3f")
    close("V6 R2(g + logP + far) X=1e4 (0.942)", lad["g_logP_far"], 0.942, 5e-4, "%.4f", "%.3f")
    # smooth cutoffs
    ts = ctx.table("tab_smooth")
    sm = ctx.arr("logP_smooth_8p4e9.npy")
    below("V6 sharp weight reproduces the exact product (1e-9)", float(np.max(np.abs(sm[:, 0, :] - ctx.lp8))), 1e-9)
    for a_, lab in enumerate(("sharp", "exp4", "bump")):
        for j, X in enumerate(XS):
            (a, b, c), se, r2, _ = S.ols(np.column_stack([lg8, sm[:, a_, j]]), lf8)
            tg = ts[lab][str(int(X))]
            _cmp3("V6 smooth %s X=1e%d a/b/R2 (Table 5.2)" % (lab, int(np.log10(X))), (a, b, r2), (tg["a"], tg["b"], tg["R2"]), 1e-6)
    below("V6 max shift of b under smooth cutoffs", ts["max_shift_b"], ts["max_shift_b"] + 1e-12)
    for lab in ("exp4", "bump"):
        bl = [ts[lab][str(int(X))]["b"] for X in XS]
        report("V6 %s preserves the climb of b" % lab, "%.3f -> %.3f" % (bl[0], bl[-1]), "increasing", "-", bl[-1] > bl[0] + 0.1, "%s", "%s")
    # fresh smooth sums on a few pairs (primes to 4e7)
    if ctx.full:
        t = time.time()
        pr4 = E.Primes(4 * 10 ** 7)
        sub = ctx.rng.choice(len(ctx.idx8), 5, replace=False)
        worst = 0.0
        for j in sub:
            ph = E.phases(ctx.t0_8[j], pr4)
            for a_, lab in enumerate(("sharp", "exp4", "bump")):
                for jx, X in enumerate(XS):
                    worst = max(worst, abs(E.logP_smooth(ph, pr4, X, lab) - sm[j, a_, jx]))
        below("V6 smooth sums recomputed on 5 pairs, max dev", worst, 1e-9)
        print("    (smooth recomputation %.0fs)" % (time.time() - t))
    # shipped log|P_X|: fresh recomputation on a subset (all on --full)
    t = time.time()
    n8 = len(ctx.idx8) if ctx.full else 40
    sub = np.sort(ctx.rng.choice(len(ctx.idx8), n8, replace=False))
    below("V6 log|P_X| 8.4e9 recomputed on %d pairs, max dev" % n8, float(np.max(np.abs(E.logP_exact(ctx.t0_8[sub], pr=pr) - ctx.lp8[sub]))), 1e-9)
    n2 = len(ctx.idx2) if ctx.full else 40
    sub2 = np.sort(ctx.rng.choice(len(ctx.idx2), n2, replace=False))
    below("V6 log|P_X| 2M recomputed on %d pairs, max dev" % n2, float(np.max(np.abs(E.logP_exact(ctx.t0_2[sub2], pr=pr) - ctx.lp2[sub2]))), 1e-9)
    print("    (prime sums: %.1fs)" % (time.time() - t))
    ctx._cache["sub8"] = sub
