"""V12 -- tail bounds for the truncated Euler product (Section 9, Tables 9.1-9.3)."""
import numpy as np

from tightpairs import tails as T, euler as E
from tightpairs.report import close, rel, report, below, equal, section


def run(ctx):
    section("V12: tail bounds (Section 9)")
    tb = T.TailBounds(10 ** 7)
    c = tb.constants()
    tc = ctx.table("tab_chernoff")
    equal("V12 prime powers <= 1e7", c["n_prime_powers"], 665134)
    equal("V12 ... with k >= 2", c["n_k_ge_2"], 555)
    close("V12 sum a_p", c["sum_a_p"], 463.3111407, 1e-7, "%.9f", "%.7f")
    close("V12 sum sigma_p^2", c["sum_sigma2"], 1.5904554722, 1e-9, "%.10f", "%.10f")
    close("V12 sigma", c["sigma"], 1.26113, 1e-5, "%.6f", "%.5f")
    close("V12 Lambda_1", c["Lambda_1"], 6321.4118494, 1e-6, "%.7f", "%.7f")
    TB = 8.436e9
    e1, e2, e3 = tb.second_moment_terms(TB ** 2)
    rel("V12 2nd-moment term (i)", e1, 1.21e-7, 0.01)
    rel("V12 2nd-moment term (ii)", e2, 2.49e-8, 0.01)
    rel("V12 2nd-moment term (iii)", e3, 4.4e-15, 0.02)
    rel("V12 2nd-moment total", e1 + e2 + e3, 1.46e-7, 0.01)
    rel("V12 first-moment bound", tb.first_moment_bound(TB ** 2), 1.9e-17, 0.02)
    rel("V12 4th-moment crude bound", tb.fourth_moment_crude(TB ** 2), 2.6e19, 0.05)
    for u, s_tgt, d_tgt in ((5.0, 4.20, 7.06e-5), (6.0, 5.30, 6.13e-7), (7.0, 6.47, 1.71e-9), (8.0, 7.70, 1.44e-12), (9.0, 8.98, 3.46e-16), (10.0, 10.31, 2.25e-20)):
        d_, s_ = tb.best(u)
        tj = tc[str(u)]
        report("V12 certified d(%.0f) at s* (Table 9.2)" % u, "%.3e @ %.2f" % (d_, s_), "%.2e @ %.2f" % (d_tgt, s_tgt), "0.5%/0.02",
               abs(d_ / d_tgt - 1) <= 5e-3 and abs(s_ - s_tgt) <= 0.02)
        below("V12 ... equals the table", abs(d_ / tj["d"] - 1) + abs(s_ - tj["s_star"]), 1e-9)
    rel("V12 pure Bennett d(6.9)", tb.bennett_all(6.9), 1.24e-4, 0.01)
    v1k = tb.log_Mp_small(9.5, tb.term_grid(tb.small, 1024))
    v64k = tb.log_Mp_small(9.5, tb.term_grid(tb.small, 65536))
    close("V12 sum log M_p(-9.5), 4096 nodes", tb.log_Mp_small(9.5), 30.4273750585495, 1e-12, "%.13f", "%.13f")
    below("V12 mesh doubling 1024->65536 changes < 1e-13", abs(v1k - v64k), 1e-13)
    equal("V12 exact factors for p <= 1000: count", int(len(tb.small)), 168)
    # Lemma 9.1 and Table 9.1
    td0 = ctx.table("tab_delta0")
    pr0 = E.Primes(10 ** 7)
    terms = E.Delta0_terms(pr0)
    for X in (1e2, 1e3, 1e4, 1e5, 1e6, 1e7):
        close("V12 Delta_0(1e%d) (Table 9.1)" % int(np.log10(X)), float(terms[pr0.p <= X].sum()), td0[str(int(X))], 1e-15, "%.12f", "%.12f")
    close("V12 Delta_0(1e2) (0.2381784258)", td0["100"], 0.2381784258, 1e-10, "%.10f", "%.10f")
    close("V12 Delta_0(1e7) (0.2386607628)", td0["10000000"], 0.2386607628, 1e-10, "%.10f", "%.10f")
    close("V12 p = 2 term (0.15298926591521)", float(terms[0]), 0.15298926591521, 1e-14, "%.14f", "%.14f")
    close("V12 share of p = 2 (64%)", float(terms[0] / terms.sum()), 0.64, 5e-3, "%.4f", "%.2f")
    below("V12 variation of Delta_0 over 1e2..1e7 (< 5e-4)", td0["variation"], 5e-4)
    close("V12 variation (4.8e-4)", td0["variation"], 4.8e-4, 5e-6, "%.3e", "%.1e")
    below("V12 rigorous tail beyond 1e7 (< 3.4e-9)", td0["tail_bound"], 3.4e-9)
    close("V12 tail estimate 1/(4 X log X) (1.6e-9)", td0["tail_estimate"], 1.6e-9, 5e-11, "%.3e", "%.1e")
    rel("V12 rigorous tail bound formula", 2 * (1 + 1.5 / np.log(1e7)) / (1e7 * np.log(1e7)) / (4 * (1 - 1e7 ** -0.5)), td0["tail_bound"], 1e-12)
    Nn = tb.N_criterion(TB)
    rel("V12 N = H Lambda_1/(2m)", Nn, 5.33e13, 2e-3)
    tu = ctx.table("tab_uniform")
    close("V12 Table 9.3: H = T_0 (start of the primary block), m = 1/2", abs(tu["H"] - 8436146000.0) + abs(tu["m"] - 0.5), 0.0, 0.0, "%.1f", "%.1f")
    Nt = tb.N_criterion(tu["H"], tu["m"])
    rel("V12 N for Table 9.3", Nt, tu["N"], 1e-12)
    for u, tgt in ((9.0, 1.40), (9.36, 6.4e-2), (9.5, 1.85e-2), (10.0, 1.8e-4)):
        d_, s_ = tb.best(u - 0.5)
        rel("V12 N d(u-m) at u=%.2f" % u, Nt * d_, tgt, 0.03, "%.3e", "%.2e")
        rel("V12 ... equals Table 9.3", Nt * d_, tu[str(u)]["Nd"], 1e-9, "%.3e", "%.3e")
    rel("V12 d(8.5)/d(9.5) (four orders)", tu["9.0"]["d"] / tu["10.0"]["d"], 7.96e3, 0.01, "%.3e", "%.2e")
    close("V12 prod M_p(-1), X=100", tb.mean_value_product(100, 1.0), 1.559923, 1e-6, "%.6f", "%.6f")
    pr = E.Primes(10 ** 7)
    PR17 = np.flatnonzero(pr.pf <= 100)
    KP = pr.kmax(10 ** 7)
    rng17 = np.random.default_rng(19)
    t17 = rng17.uniform(0.0, 2e6, 300_000)
    L17 = np.zeros(len(t17))
    for i in PR17:
        ph = (t17 * np.log(pr.pf[i])) % (2 * np.pi)
        for k in range(1, KP[i] + 1):
            L17 += pr.pf[i] ** (-k / 2) * np.cos(k * ph) / k
    Y17 = np.exp(-L17)
    close("V12 MC mean (seed 19) (1.5612)", Y17.mean(), 1.5612, 5e-4, "%.4f", "%.4f")
    close("V12 MC std err (0.0030)", Y17.std(ddof=1) / np.sqrt(len(Y17)), 0.0030, 2e-4, "%.4f", "%.4f")
    sig4, pref, pred, s21, Kv, Kpp, TM21 = T.bahadur_rao(10 ** 4, 4.580)
    close("V12 sigma(X=1e4)", sig4, 1.14500, 1e-5, "%.5f", "%.5f")
    close("V12 BR prefactor", pref, 0.0960, 2e-4, "%.4f", "%.4f")
    rel("V12 BR prediction", pred, 1.97e-6, 5e-3)
    close("V12 4 sigma at X=1e4 = 4.580", 4 * sig4, 4.580, 1e-3, "%.4f", "%.3f")
    if ctx.full:
        est, se_ = T.importance_sampling(TM21, s21, Kv, 4.580, 2000, 19)
        rel("V12 IS estimate (seed 19, N=2000)", est, 1.88e-6, 0.01)
        rel("V12 IS std err", se_, 9e-8, 0.05)
    else:
        print("    (importance-sampling tier skipped; use --full)")
