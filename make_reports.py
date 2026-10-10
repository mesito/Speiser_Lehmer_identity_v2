#!/usr/bin/env python3
"""make_reports.py -- human-readable Markdown reports of the derived data, written
to data/reports/.  They summarise the shipped arrays (not the paper): the full-
ensemble fits, the smooth-cutoff scan, the detector sweep, the block-averaging
scan, the reference pairs, the rank subsample with the errors-in-variables
comparison, and the identity checks.  Requires tables/*.json
(run make_tables.py first).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tightpairs import data as D, stats as S, geometry as G  # noqa: E402
from tightpairs.constants import XS  # noqa: E402

OUT = D.path("reports")
TAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")


def J(name):
    return json.load(open(os.path.join(TAB, name + ".json")))


def w(name, lines):
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, name), "w").write("\n".join(lines) + "\n")
    print("wrote", os.path.join(OUT, name))


def full_pairs_report():
    tf, td, tdf = J("tab_fits"), J("tab_decomp"), J("tab_deficit")
    L = ["# Full-ensemble floor fits and statistics", "",
         "Data: `floors_all_8p4e9.npy` (2736 pairs, floor = max|Z_dd| over the gap, 33-point grid + parabolic refinement), "
         "`logP_pairs_8p4e9.npy` (exact log|P_X|, GHK convention), `floors_all_2M.npy` (6088 pairs), `logP_pairs_2M.npy`.", "",
         "## Table 5.1: log F = a log g + b log|P_X| + c", "",
         "| X | a (8.4e9) | se | b | se | R2 | a (2M) | se | b | se | R2 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for X in XS:
        p, q = tf["8p4e9"][str(int(X))], tf["2M"][str(int(X))]
        L.append("| 1e%d | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |" % (int(np.log10(X)), p["a"], p["se_a"], p["b"], p["se_b"], p["R2"], q["a"], q["se_a"], q["b"], q["se_b"], q["R2"]))
    lad = tf["ladder_1e4"]
    L += ["", "## Variance ladder at X = 1e4 (8.4e9)", "", "| model | R2 |", "|---|---|",
          "| g only | %.4f |" % lad["g"], "| g + log far | %.4f |" % lad["g_far"], "| g + log P_X | %.4f |" % lad["g_logP"], "| g + log P_X + log far | %.4f |" % lad["g_logP_far"],
          "", "g-only log-log slope %.3f, corr(log g, log F) %.3f, smallest floors %.4e (8.4e9) and %.4e (2M)." % (tf["g_only_slope"], tf["corr_logg_logF"], tf["smallest_floor_8p4e9"], tf["smallest_floor_2M"]),
          "", "## Table 5.6: decomposition residual R_Delta = log F - log|P_X| - S_Delta (8.4e9)", "",
          "| X | median (100) | sd (100) | max (100) | median (1000) | sd (1000) | max (1000) |", "|---|---|---|---|---|---|---|"]
    for X in XS:
        a, b = td["8p4e9"]["100"][str(int(X))], td["8p4e9"]["1000"][str(int(X))]
        L.append("| 1e%d | %+.4f | %.4f | %.3f | %+.4f | %.4f | %.3f |" % (int(np.log10(X)), a["median"], a["sd"], a["max"], b["median"], b["sd"], b["max"]))
    L += ["", "## Observation 5.10: the loading explained (8.4e9, Delta = 1000)", "",
          "| X | curvature in a | S_far slope on log P | S_far slope on log g | corr(S_far, log P) | corr(S_far, log far) | sd(S_far) |", "|---|---|---|---|---|---|---|"]
    for X in XS:
        a = td["loading"][str(int(X))]
        L.append("| 1e%d | %+.4f | %+.4f | %+.4f | %.3f | %.3f | %.3f |" % (int(np.log10(X)), a["curv_a"], a["Sfar_b"], a["Sfar_a"], a["corr_Sfar_logP"], a["corr_Sfar_logfar"], a["sd_Sfar"]))
    L += ["", "## Table 6.2: Euler deficit at tight-pair midpoints vs 300 controls", "",
          "| X | mean pairs | mean controls | sd controls | shift (sigma) | factor |", "|---|---|---|---|---|---|"]
    for X in XS:
        a = tdf[str(int(X))]
        L.append("| 1e%d | %.3f | %.3f | %.3f | %+.2f | %.1f |" % (int(np.log10(X)), a["mean_pairs"], a["mean_ctrl"], a["sd_ctrl"], a["shift"], a["factor"]))
    w("full_pairs_report.md", L)


def smooth_cutoff_report():
    ts = J("tab_smooth")
    L = ["# Smooth-cutoff scan (Section 5.4, Table 5.2)", "",
         "Weights on the prime powers n = p^k: sharp 1{n <= X}; exp4 exp(-(n/X)^4) summed to n <= 4X; C^inf bump (1 for n <= X/2, 0 for n >= X). "
         "Exact in k; phases in 80-bit precision; primes to 4e7. Full primary ensemble (2736 pairs). Data: `logP_smooth_8p4e9.npy`, `smooth_cutoff_scan.csv`.", "",
         "| X | a sharp | b sharp | R2 sharp | a exp4 | b exp4 | R2 exp4 | a bump | b bump | R2 bump |", "|---|---|---|---|---|---|---|---|---|---|"]
    for X in XS:
        c = [ts[l][str(int(X))] for l in ("sharp", "exp4", "bump")]
        L.append("| 1e%d | " % int(np.log10(X)) + " | ".join("%.4f | %.4f | %.4f" % (v["a"], v["b"], v["R2"]) for v in c) + " |")
    L += ["", "Maximal shift of b between the sharp and a smooth cutoff: %.4f." % ts["max_shift_b"]]
    w("smooth_cutoff_report.md", L)


def detector_sweep_summary():
    det = np.genfromtxt(D.path("detector_sweep_8p4e9.csv"), delimiter=",", names=True)
    hr = det["H"] / (det["g"] / 2)
    L = ["# Detector sweep (Section 3.6)", "",
         "Ghat(t) on 257 interior points of every tight pair of the primary ensemble, W = 250. Data: `detector_sweep_8p4e9.csv`.", "",
         "| quantity | value |", "|---|---|",
         "| pairs | %d |" % det.size, "| pairs with a non-negative value | %d |" % int((det["n_sign_changes"] > 0).sum()),
         "| largest Ghat | %.4e (pair %d, g = %.4e) |" % (det["max_Ghat"].max(), det["pair_idx"][np.argmax(det["max_Ghat"])], det["g"][np.argmax(det["max_Ghat"])]),
         "| r: min / median / max | %.6f / %.6f / %.6f |" % (det["r"].min(), np.median(det["r"]), det["r"].max()),
         "| H/(g/2): min / median / max | %.3f / %.3f / %.3f |" % (hr.min(), np.median(hr), hr.max()),
         "| h_thr: min / median / max | %.3e / %.3e / %.3e |" % (det["h_thr"].min(), np.median(det["h_thr"]), det["h_thr"].max())]
    w("detector_sweep_summary.md", L)


def moment_mediation_report():
    tb = J("tab_blocks")
    L = ["# Block-averaging (moment-level mediation) scan (Section 6.5, Table 6.3)", "",
         "Every fourth gap of the primary ensemble (%d gaps), floor = max|Z| over 12 interior samples (fast instrument), log|P_X| at X = 1e6. "
         "Blocks of B consecutive gaps sorted by t; OLS of block means. Data: `moment_allgaps.npy`, `logP_allgaps_1e6.npy`." % tb["n_gaps"], "",
         "| B | n blocks | a | b | se(b) | R2 | partial corr | sd of block-mean log P | corr(log F, log P) | corr(log P, log g) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for B in ("1", "16", "256", "4096"):
        a = tb[B]
        L.append("| %s | %d | %.3f | %.3f | %.3f | %.4f | %.3f | %.3f | %.3f | %.3f |" % (B, a["n"], a["a"], a["b"], a["se_b"], a["R2"], a["pcorr"], a["sd_logP"], a["corr_logF_logP"], a["corr_logP_logg"]))
    ts = tb["tight_subset"]
    L += ["", "Tight subset (s < 0.15) at B = 1: n = %d, a = %.3f, b = %.3f, R2 = %.3f." % (ts["n"], ts["a"], ts["b"], ts["R2"])]
    w("moment_mediation_report.md", L)


def reference_pairs_report():
    L = ["# Reference pairs: Theorem A(b),(c) and the displacement law", "",
         "30-digit mpmath references with exact ordinates. Columns: delta = t* - t0 (measured), pred = (g^2/8) lambda_far at W = 250, "
         "R_250 and R_all = F S_on(t*)/|Z''(t*)| - 1 with the far field over W = 250 and over the complete sum, "
         "curv = -(log|Z|)''(t0)/S_on(t0) - 1. Data: `reference_*.csv`.", ""]
    for ens in ("8p4e9", "8p4e9_sub", "100k", "2M", "lehmer"):
        a = np.atleast_1d(np.genfromtxt(D.path("reference_%s.csv" % ens), delimiter=",", names=True))
        L += ["## %s (%d pairs)" % (ens, a.size), "", "| rank | idx | g | delta | pred | err % | F | |Z''| | r^2 | R_250 | R_all | curv |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in a:
            L.append("| %d | %d | %.4e | %+.4e | %+.4e | %.3f | %.4e | %.3f | %.6f | %+.2e | %+.2e | %+.2e |" % (r["rank"], r["idx1"], r["g"], r["delta"], r["prediction"], r["abs_rel_err_pct"], r["F"], r["Zpp"], r["r2"], r["R250"], r["Rall"], r["curv_dev"]))
        if a.size > 1:
            L.append("")
            L.append("corr(delta, pred) = %.7f, median error %.3f %%, max %.3f %%, max |R_all| = %.2e, max |curv| = %.2e." % (np.corrcoef(a["delta"], a["prediction"])[0, 1], np.median(a["abs_rel_err_pct"]), a["abs_rel_err_pct"].max(), np.abs(a["Rall"]).max(), np.abs(a["curv_dev"]).max()))
        L.append("")
    w("reference_pairs_report.md", L)


def subsample_report():
    ts, te, tl = J("tab_sub"), J("tab_eiv"), J("tab_ladder")
    L = ["# The rank subsample, the variance ladder and errors in variables", "",
         "Rank subsample: `tightpairs.data.rank_subsample(g, 300)` -- the pairs at ranks round(k(N-1)/299), k = 0..299, of the stable ascending "
         "order of the gaps g (column 2 of `floors_all_8p4e9.npy`). First fifteen pairs (1-based index of gamma_n): %s." % ", ".join(str(i) for i in ts["sub_idx1"]), "",
         "| X | a (full) | b (full) | R2 (full) | a (sub) | b (sub) | R2 (sub) |", "|---|---|---|---|---|---|---|"]
    for X in XS:
        f, u = ts["full"][str(int(X))], ts["sub"][str(int(X))]
        L.append("| 1e%d | %.4f | %.4f | %.4f | %.4f +- %.4f | %.4f +- %.4f | %.4f |" % (int(np.log10(X)), f["a"], f["b"], f["R2"], u["a"], u["se_a"], u["b"], u["se_b"], u["R2"]))
    for k in ("g_only_full", "g_only_sub"):
        v = ts[k]
        L.append("")
        L.append("%s: slope %.4f, corr %.4f, residual sd %.4f nats (%.4f dex)." % (k, v["slope"], v["corr"], v["resid_sd_nats"], v["resid_sd_dex"]))
    L += ["", "## Variance ladder (R^2)", "", "| ensemble, X | g | g + log far | g + log|P_X| | g + log|P_X| + log far |", "|---|---|---|---|---|"]
    for k, v in tl.items():
        L.append("| %s | %.4f | %.4f | %.4f | %.4f |" % (k, v["g"], v["g_far"], v["g_P"], v["g_P_far"]))
    L += ["", "## Errors in variables: proxy L_X^(3) against the exact log|P_X|", "", "| ensemble | X | b | b3 | b3/b | lambda_meas | lambda_rand | lambda_sup | delta_X | max|L3 - logP| |", "|---|---|---|---|---|---|---|---|---|---|"]
    for ens in ("8p4e9", "2M"):
        for X in XS:
            v = te[ens][str(int(X))]
            L.append("| %s | 1e%d | %.4f | %.4f | %.4f | %.4f | %.4f | %.3f | %.4f | %.3f |" % (ens, int(np.log10(X)), v["b"], v["b3"], v["ratio"], v["lambda_meas"], v["lambda_rand"], v["lambda_sup"], v["delta_sup"], v["max_abs_diff"]))
    w("subsample_eiv_report.md", L)


def identity_report():
    ti = J("tab_identity")
    L = ["# The Speiser-Lehmer identity on five ensembles", "", "| ensemble | tight pairs | max |r^2(1+1/(8 C_mid)) - 1| |", "|---|---|---|"]
    for k, v in ti.items():
        L.append("| %s | %d | %.2e%s |" % (k, v["n"], v["max_dev_double"], (" (80-bit: %.2e)" % v["max_dev_80bit"]) if "max_dev_80bit" in v else ""))
    r8 = np.load(D.path("lehmer_rows_8p4e9.npy"))
    rr, Cm, s = r8[:, -1], r8[:, 7], r8[:, 3]
    L += ["", "Primary ensemble: r min/median/max = %.6f / %.6f / %.6f; C_mid min/median/max = %.3f / %.2f / %.1f; mean (1-r)/s^2 = %.4f; "
          "pairs with C > 5/4 (Lehmer pairs in the CSV sense): %d of %d." % (rr.min(), np.median(rr), rr.max(), Cm.min(), np.median(Cm), Cm.max(), ((1 - rr) / s ** 2).mean(), int((r8[:, 6] > 1.25).sum()), len(r8))]
    w("identity_report.md", L)


if __name__ == "__main__":
    for f in (full_pairs_report, smooth_cutoff_report, detector_sweep_summary, moment_mediation_report, reference_pairs_report, subsample_report, identity_report):
        try:
            f()
        except (FileNotFoundError, KeyError) as e:
            print("skipped %s: %s" % (f.__name__, e))
