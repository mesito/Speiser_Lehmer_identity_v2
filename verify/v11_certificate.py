"""V11 -- the certificate on the data (Observation 8.4) and the cutoff dependence of S_far."""
import numpy as np

from tightpairs import kernel as K
from tightpairs.constants import XS, GAMMA_E
from tightpairs.report import close, report, section


def run(ctx):
    section("V11: the certificate on the data (Section 8)")
    tgt = {1e2: (2.03, 3.08, 0.082), 1e3: (1.60, 2.47, 0.123), 1e4: (1.31, 2.01, 0.164), 1e5: (1.10, 1.78, 0.205), 1e6: (0.95, 1.59, 0.246), 1e7: (0.82, 1.43, 0.287)}
    g8, lf8 = ctx.f8[:, 2], np.log(ctx.f8[:, 5])
    eps = g8 / 2
    for j, X in enumerate(XS):
        Dq = lf8 - ctx.lp8[:, j] - 2 * np.log(eps * np.log(X)) - 2 * GAMMA_E
        s5 = np.percentile(Dq, 5)
        ratio = 0.5 * (np.exp((Dq - s5) / 2 - GAMMA_E) / (eps * np.log(X)) - 1)
        med, p95 = np.sqrt(np.median(ratio)), np.sqrt(np.percentile(ratio, 95))
        tg = tgt[X]
        report("V11 h0max/eps X=1e%d median/95%%/median eps logX" % int(np.log10(X)), "%.2f/%.2f/%.3f" % (med, p95, np.median(eps * np.log(X))),
               "%.2f/%.2f/%.3f" % tg, 5e-3, abs(med - tg[0]) <= 5e-3 and abs(p95 - tg[1]) <= 5e-3 and abs(np.median(eps * np.log(X)) - tg[2]) <= 5e-4)
    S7 = ctx.zs8[1, :, 5] - K.pair_term(g8, 1e7)
    S2 = ctx.zs8[1, :, 0] - K.pair_term(g8, 1e2)
    close("V11 median(S_far(1e7) - S_far(1e2)) (-0.58)", np.median(S7 - S2), -0.58, 5e-3, "%.3f", "%.2f")
    close("V11 sd(S_far(1e7) - S_far(1e2)) (0.41)", (S7 - S2).std(), 0.41, 5e-3, "%.3f", "%.2f")
    close("V11 inflation e^(6.9/4) (5.6)", np.exp(6.9 / 4), 5.6, 0.05, "%.2f", "%.1f")
    close("V11 mean log|P_X| at pairs, X=1e7 (-2.5)", ctx.lp8[:, 5].mean(), -2.54, 0.01, "%.3f", "%.2f")
    close("V11 median bound at X=1e7 inflated (4.6 half-gaps)", 0.82 * np.exp(6.9 / 4), 4.6, 0.05, "%.2f", "%.1f")
