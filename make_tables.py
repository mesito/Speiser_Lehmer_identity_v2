#!/usr/bin/env python3
"""make_tables.py -- generates every numerical table of the paper from the shipped
arrays, as LaTeX table bodies (paper/tables/tab_*.tex) and as JSON files with
the same numbers (tables/tab_*.json), which the verification groups use as
targets.  Usage:  python3 make_tables.py [--out DIR]   (default ./tables)

Tables (paper numbering):
  tab_identity   3.1  identity deviation on five ensembles
  tab_floorid    3.2  floor law and curvature on the 15 reference pairs
  tab_fits       5.1  Euler regression, two ensembles
  tab_smooth     5.2  smooth cutoffs
  tab_decomp     5.3  decomposition residuals
  tab_deficit    6.1  Euler deficit
  tab_blocks     6.2  block averaging
  tab_farlaw     7.1  far-field law
  tab_gue        7.2  GUE diagnostics
  tab_chernoff   9.1  certified Chernoff bounds
  tab_disp       B.1  displacement law, 15 pairs
  tab_lehmer     B.2  Lehmer's pair
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tightpairs import data as D, euler as E, kernel as K, stats as S, tails as T, geometry as G  # noqa: E402
from tightpairs.constants import XS, GAMMA_E, dens  # noqa: E402

OUT = "tables"
ENS = [("100k", "100k", r"Odlyzko 100k", r"$\approx7.5\times10^{4}$", 1.494),
       ("2M", "2M", r"Odlyzko 2M", r"$\approx1.2\times10^{6}$", 1.935),
       ("8p4e9", "LMFDB", r"LMFDB $8.4\times10^{9}$", r"$8.436\times10^{9}$", 3.3451),
       ("1e12", "1e12", r"$10^{12}$ window", r"$\approx2.7\times10^{11}$", 3.896),
       ("1e21", "1e21", r"$10^{21}$ window", r"$\approx1.4\times10^{20}$", 7.095)]


def sci(x, digits=1):
    """LaTeX scientific notation $m\times10^{e}$."""
    if x == 0:
        return "$0$"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    s = ("%%.%df" % digits) % m
    if float(s) == 10.0:
        m, e = m / 10, e + 1
        s = ("%%.%df" % digits) % m
    return "$%s\\times10^{%d}$" % (s, e)


def write(name, body, numbers):
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, name + ".tex"), "w").write(body)
    json.dump(numbers, open(os.path.join(OUT, name + ".json"), "w"), indent=1)
    print("wrote", name)


def tab_identity():
    rows, num = [], {}
    for key, short, label, height, _ in ENS:
        a = np.load(D.path("lehmer_rows_%s.npy" % key))
        dev = float(np.max(np.abs(G.identity_deviation(a[:, -1], a[:, 7]))))
        num[short] = {"n": int(len(a)), "max_dev_double": dev}
        extra = ""
        if key == "8p4e9":
            g = a[:, 2].astype(np.longdouble)
            far = a[:, 5].astype(np.longdouble)
            r = np.sqrt(2 / (far + 8 / g ** 2)) / (g / 2)
            Cm = (1 / g ** 2) / far
            dev80 = float(np.max(np.abs(G.identity_deviation(r, Cm))))
            num[short]["max_dev_80bit"] = dev80
            extra = " (%s in 80-bit precision)" % sci(dev80)
        rows.append("%s & %s & %d & %s%s\\\\" % (label, height, len(a), sci(dev), extra))
    body = ("\\begin{tabular}{@{}llrl@{}}\n\\toprule\nEnsemble & Height $t$ & Tight pairs & Max deviation\\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_identity", body, num)


def tab_floorid():
    a = np.genfromtxt(D.path("reference_8p4e9.csv"), delimiter=",", names=True)
    rows, num = [], {"rows": []}
    for k in range(len(a)):
        rows.append("%d & %s & %.6f & %s & %.2f & %s & %s & %s\\\\" % (
            a["rank"][k], sci(a["g"][k], 3), a["r2"][k], sci(a["F"][k], 4), a["Zpp"][k],
            sci(a["R250"][k], 2), sci(a["Rall"][k], 1), sci(a["curv_dev"][k], 1)))
        num["rows"].append({"rank": int(a["rank"][k]), "g": float(a["g"][k]), "r2": float(a["r2"][k]), "F": float(a["F"][k]),
                            "Zpp": float(a["Zpp"][k]), "R250": float(a["R250"][k]), "Rall": float(a["Rall"][k]), "curv_dev": float(a["curv_dev"][k])})
    num["max_abs_R250"] = float(np.abs(a["R250"]).max()); num["min_abs_R250"] = float(np.abs(a["R250"]).min())
    num["max_abs_Rall"] = float(np.abs(a["Rall"]).max()); num["max_abs_curv"] = float(np.abs(a["curv_dev"]).max())
    num["max_tail_explained"] = float(np.abs(a["R250"] + (a["farall"] - a["far250"]) / a["Sonall"]).max())
    for ens in ("100k", "2M", "lehmer"):
        b = np.genfromtxt(D.path("reference_%s.csv" % ens), delimiter=",", names=True)
        b = np.atleast_1d(b)
        num["max_abs_Rall_" + ens] = float(np.abs(b["Rall"]).max())
        num["max_abs_curv_" + ens] = float(np.abs(b["curv_dev"]).max())
    body = ("\\begin{tabular}{@{}rccccccc@{}}\n\\toprule\npair & $g$ & $r^{2}$ & $\\Fl$ & $|Z''(t^{*})|$ & $R_{250}$ & $R_{\\mathrm{all}}$ & $\\Delta_{\\mathrm{curv}}$\\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_floorid", body, num)


def _fits(fa, lp):
    lg, lf = np.log(fa[:, 2]), np.log(fa[:, 5])
    out = []
    for j, X in enumerate(XS):
        (a, b, c), se, r2, _ = S.ols(np.column_stack([lg, lp[:, j]]), lf)
        out.append((a, se[0], b, se[1], r2))
    return out


def tab_fits():
    f8 = np.load(D.path("floors_all_8p4e9.npy")); lp8 = np.load(D.path("logP_pairs_8p4e9.npy"))
    f2 = np.load(D.path("floors_all_2M.npy")); lp2 = np.load(D.path("logP_pairs_2M.npy"))
    F8, F2 = _fits(f8, lp8), _fits(f2, lp2)
    rows, num = [], {"8p4e9": {}, "2M": {}}
    for j, X in enumerate(XS):
        a1, sa1, b1, sb1, r1 = F8[j]; a2, sa2, b2, sb2, r2 = F2[j]
        rows.append("$10^{%d}$ & $%.4f\\,(%.4f)$ & $%.4f\\,(%.4f)$ & $%.4f$ & & $%.4f\\,(%.4f)$ & $%.4f\\,(%.4f)$ & $%.4f$\\\\"
                    % (int(np.log10(X)), a1, sa1, b1, sb1, r1, a2, sa2, b2, sb2, r2))
        num["8p4e9"][str(int(X))] = {"a": a1, "se_a": sa1, "b": b1, "se_b": sb1, "R2": r1}
        num["2M"][str(int(X))] = {"a": a2, "se_a": sa2, "b": b2, "se_b": sb2, "R2": r2}
    body = ("\\begin{tabular}{@{}c ccc c ccc@{}}\n\\toprule\n & \\multicolumn{3}{c}{$t\\approx8.436\\times10^{9}$} & & \\multicolumn{3}{c}{$t\\approx1.2\\times10^{6}$}\\\\\n"
            "\\cmidrule{2-4}\\cmidrule{6-8}\n$X$ & $a$ & $b$ & $R^{2}$ & & $a$ & $b$ & $R^{2}$\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    # variance ladder at X = 1e4 and the g-only statistics
    lg, lf, far = np.log(f8[:, 2]), np.log(f8[:, 5]), f8[:, 4]
    (_, _), _, r2g, _ = S.ols(lg[:, None], lf)
    num["ladder_1e4"] = {"g": r2g, "g_far": S.ols(np.column_stack([lg, np.log(far)]), lf)[2],
                         "g_logP": S.ols(np.column_stack([lg, lp8[:, 2]]), lf)[2],
                         "g_logP_far": S.ols(np.column_stack([lg, lp8[:, 2], np.log(far)]), lf)[2]}
    num["g_only_slope"] = float(S.ols(lg[:, None], lf)[0][0]); num["corr_logg_logF"] = S.corr(lg, lf)
    num["smallest_floor_8p4e9"] = float(f8[:, 5].min()); num["smallest_floor_2M"] = float(f2[:, 5].min())
    write("tab_fits", body, num)


def tab_smooth():
    a = np.genfromtxt(D.path("smooth_cutoff_scan.csv"), delimiter=",", names=True, dtype=None, encoding="utf-8")
    labels = ("sharp", "exp4", "bump")
    rows, num = [], {}
    for X in XS:
        cells = []
        for lab in labels:
            m = (a["cutoff"] == lab) & (a["X"] == X)
            b, r2 = float(a["b"][m][0]), float(a["R2"][m][0])
            num.setdefault(lab, {})[str(int(X))] = {"a": float(a["a"][m][0]), "b": b, "se_b": float(a["se_b"][m][0]), "R2": r2}
            cells.append("$%.4f$ & $%.4f$" % (b, r2))
        rows.append("$10^{%d}$ & " % int(np.log10(X)) + " & & ".join(cells) + "\\\\")
    bsh = np.array([num["sharp"][str(int(X))]["b"] for X in XS])
    num["max_shift_b"] = float(max(np.max(np.abs(np.array([num[l][str(int(X))]["b"] for X in XS]) - bsh)) for l in ("exp4", "bump")))
    body = ("\\begin{tabular}{@{}c cc c cc c cc@{}}\n\\toprule\n & \\multicolumn{2}{c}{sharp} & & \\multicolumn{2}{c}{$\\e^{-(n/X)^{4}}$} & & \\multicolumn{2}{c}{bump}\\\\\n"
            "\\cmidrule{2-3}\\cmidrule{5-6}\\cmidrule{8-9}\n$X$ & $b$ & $R^{2}$ & & $b$ & $R^{2}$ & & $b$ & $R^{2}$\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_smooth", body, num)


def tab_decomp():
    f8 = np.load(D.path("floors_all_8p4e9.npy")); lp8 = np.load(D.path("logP_pairs_8p4e9.npy")); zs8 = np.load(D.path("zsum_pairs_8p4e9.npy"))
    f2 = np.load(D.path("floors_all_2M.npy")); lp2 = np.load(D.path("logP_pairs_2M.npy")); zs2 = np.load(D.path("zsum_pairs_2M.npy"))
    lf8, lf2 = np.log(f8[:, 5]), np.log(f2[:, 5])
    rows, num = [], {"8p4e9": {"100": {}, "1000": {}}, "2M": {"1000": {}}}
    for j, X in enumerate(XS):
        cells = []
        for d, Dl in enumerate((100, 1000)):
            R = lf8 - lp8[:, j] - zs8[d, :, j]
            med, sd, mx = float(np.median(R)), float(R.std()), float(np.abs(R).max())
            num["8p4e9"][str(Dl)][str(int(X))] = {"median": med, "sd": sd, "max": mx}
            cells.append("$%+.4f$ & $%.4f$ & $%.3f$" % (med, sd, mx))
        R2 = lf2 - lp2[:, j] - zs2[1, :, j]
        num["2M"]["1000"][str(int(X))] = {"median": float(np.median(R2)), "sd": float(R2.std()), "max": float(np.abs(R2).max())}
        rows.append("$10^{%d}$ & " % int(np.log10(X)) + " & & ".join(cells) + "\\\\")
    num["2M_max_abs_median"] = max(abs(v["median"]) for v in num["2M"]["1000"].values())
    num["2M_max_sd"] = max(v["sd"] for v in num["2M"]["1000"].values())
    # loading accounting (Observation 5.8)
    g8, lg8 = f8[:, 2], np.log(f8[:, 2]); far8 = f8[:, 4]
    acc = {}
    for j, X in enumerate(XS):
        Sp = K.pair_term(g8, X); Sf = zs8[1, :, j] - Sp; Rr = lf8 - lp8[:, j] - zs8[1, :, j]
        (ap, bp, _), _, _, _ = S.ols(np.column_stack([lg8, lp8[:, j]]), Sp - 2 * lg8)
        (a2, b2, _), _, _, _ = S.ols(np.column_stack([lg8, lp8[:, j]]), Sf)
        (ar, br, _), _, _, _ = S.ols(np.column_stack([lg8, lp8[:, j]]), Rr)
        acc[str(int(X))] = {"curv_a": ap, "Sfar_b": b2, "Sfar_a": a2, "resid_max": max(abs(ar), abs(br)),
                            "corr_Sfar_logP": S.corr(Sf, lp8[:, j]), "corr_Sfar_logfar": S.corr(Sf, np.log(far8)), "sd_Sfar": float(Sf.std())}
    num["loading"] = acc
    body = ("\\begin{tabular}{@{}c ccc c ccc@{}}\n\\toprule\n & \\multicolumn{3}{c}{$\\Delta=100$} & & \\multicolumn{3}{c}{$\\Delta=1000$}\\\\\n"
            "\\cmidrule{2-4}\\cmidrule{6-8}\n$X$ & median & s.d. & $\\max|R|$ & & median & s.d. & $\\max|R|$\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_decomp", body, num)


def tab_deficit():
    lp8 = np.load(D.path("logP_pairs_8p4e9.npy")); lpc = np.load(D.path("logP_ctrl_8p4e9.npy"))
    rows, num = [], {}
    for j, X in enumerate(XS):
        mp_, mc_, sd_ = float(lp8[:, j].mean()), float(lpc[:, j].mean()), float(lpc[:, j].std(ddof=1))
        sh, fac = (mp_ - mc_) / sd_, float(np.exp(mc_ - mp_))
        num[str(int(X))] = {"mean_pairs": mp_, "mean_ctrl": mc_, "sd_ctrl": sd_, "shift": sh, "factor": fac}
        rows.append("$10^{%d}$ & $%.3f$ & $%.3f$ & $%.3f$ & $%+.2f\\sigma$ & $%.1f$\\\\" % (int(np.log10(X)), mp_, mc_, sd_, sh, fac))
    fc = np.load(D.path("floors_ctrl_8p4e9.npy")); yc = np.log(fc[:, 2])
    for j, X in ((2, 1e4), (5, 1e7)):
        (m_, c_), _, _, _ = S.ols(lpc[:, j][:, None], yc)
        num["ctrl_regression_1e%d" % int(np.log10(X))] = {"slope": m_, "intercept": c_, "corr": S.corr(lpc[:, j], yc)}
    body = ("\\begin{tabular}{@{}cccccc@{}}\n\\toprule\n$X$ & mean at pairs & mean at controls & control s.d. & shift & $\\e^{-\\Delta}$\\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_deficit", body, num)


def tab_blocks():
    am = np.load(D.path("moment_allgaps.npy")); lpa = np.load(D.path("logP_allgaps_1e6.npy"))
    lfA, lgA = np.log(am[:, 3]), np.log(am[:, 2])
    rows, num = [], {}
    for B in (1, 16, 256, 4096):
        lfb, lgb, Lb = S.block_means(lfA, B), S.block_means(lgA, B), S.block_means(lpa, B)
        (aa, bb, cc), se, r2, _ = S.ols(np.column_stack([lgb, Lb]), lfb)
        pc = S.partial_corr(lfb, Lb, lgb)
        num[str(B)] = {"n": int(len(lfb)), "a": aa, "b": bb, "se_b": se[1], "R2": r2, "pcorr": pc, "sd_logP": float(Lb.std()),
                       "corr_logF_logP": S.corr(lfb, Lb), "corr_logP_logg": S.corr(Lb, lgb)}
        rows.append("%d & %s & %.3f & %.3f & %.3f & %.4f & %.3f & %.3f\\\\" % (B, "{:,}".format(len(lfb)).replace(",", "{,}"), aa, bb, se[1], r2, pc, Lb.std()))
    dA = dens(am[:, 1].mean()); mA = am[:, 2] * dA < 0.15
    (aa, bb, cc), _, r2, _ = S.ols(np.column_stack([lgA[mA], lpa[mA]]), lfA[mA])
    num["tight_subset"] = {"n": int(mA.sum()), "a": aa, "b": bb, "R2": r2}
    num["n_gaps"] = int(len(am))
    body = ("\\begin{tabular}{@{}rrcccccc@{}}\n\\toprule\n$B$ & $n$ & $a$ & $b$ & s.e.$(b)$ & $R^{2}$ & pcorr & s.d.\\ of $\\log|\\PX|$ over blocks\\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_blocks", body, num)


def tab_farlaw():
    rows, num = [], {}
    xs_, ys_ = [], []
    for key, short, label, height, dn in ENS:
        a = np.load(D.path("lehmer_rows_%s.npy" % key))
        mf = float(a[:, 5].mean())
        num[short] = {"dens": dn, "dens2": dn ** 2, "mean_far": mf, "ratio": mf / dn ** 2, "n": int(len(a))}
        xs_.append(dn ** 2); ys_.append(mf)
        hl = {"100k": "$7.5\\times10^{4}$", "2M": "$1.2\\times10^{6}$", "LMFDB": "$8.436\\times10^{9}$", "1e12": "$2.68\\times10^{11}$", "1e21": "$1.44\\times10^{20}$"}[short]
        name = {"100k": "100k", "2M": "2M", "LMFDB": "LMFDB", "1e12": "$10^{12}$ window", "1e21": "$10^{21}$ window"}[short]
        rows.append("%s & %s & %.3f & %.3f & %.2f & %.3f\\\\" % (name, hl, dn, dn ** 2, mf, mf / dn ** 2))
    num["corr"] = S.corr(xs_, ys_)
    a8 = np.load(D.path("lehmer_rows_8p4e9.npy"))
    num["far_8p4e9"] = {"median": float(np.median(a8[:, 5])), "mean": float(a8[:, 5].mean()), "max": float(a8[:, 5].max())}
    body = ("\\begin{tabular}{@{}lccccc@{}}\n\\toprule\nensemble & $t$ & $\\dens$ & $\\dens^{2}$ & mean $\\far$ & $\\overline{\\far}/\\dens^{2}$\\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_farlaw", body, num)


def tab_gue():
    C9 = np.pi ** 2 / 9.0
    S0 = (0.05, 0.10, 0.15)
    rows, num = [], {}
    for name, loader in (("100k", lambda: D.load_odlyzko("zeros1")), ("2M", lambda: D.load_odlyzko("zeros6")), ("LMFDB $8.4\\times10^{9}$", lambda: D.load_parsed())):
        z = loader()
        s = D.normalised_spacings(z)
        rr = [float(np.mean(s < s0)) / (C9 * s0 ** 3) for s0 in S0]
        lag = S.corr(s[:-1], s[1:])
        key = name.split()[0]
        num[key] = {"tail_ratios": rr, "lag1": lag}
        rows.append("%s & %.3f & %.3f & %.3f & $%.3f$\\\\" % (name, rr[0], rr[1], rr[2], lag))
    m = np.load(D.path("gue_mc_summary.npz"))
    ref = {"tail_010": float(m["tail_010"]), "tail_015": float(m["tail_015"]), "lag1": float(m["lag1"]),
           "N500_w100": float(m["N500_w100_mean"]), "N500_w100_se": float(m["N500_w100_se"]), "total_pairs": int(m["total_pairs"])}
    num["GUE_MC"] = ref
    rows.append("GUE Monte Carlo & --- & %.2f & %.2f & $%.3f$\\\\" % (ref["tail_010"], ref["tail_015"], ref["lag1"]))
    body = ("\\begin{tabular}{@{}lcccc@{}}\n\\toprule\nensemble & ratio, $s_{0}=0.05$ & ratio, $s_{0}=0.10$ & ratio, $s_{0}=0.15$ & lag-1 corr.\\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_gue", body, num)


def tab_chernoff():
    tb = T.TailBounds(10 ** 7)
    rows, num = [], {"constants": tb.constants()}
    for u in (5.0, 6.0, 7.0, 8.0, 9.0, 10.0):
        d_, s_ = tb.best(u)
        num[str(u)] = {"s_star": s_, "d": d_}
        rows.append("%.2f & %.2f & %s\\\\" % (u, s_, sci(d_, 2)))
    body = ("\\begin{tabular}{@{}ccc@{}}\n\\toprule\n$u$ & $s^{*}$ & $d(u)\\le$\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_chernoff", body, num)


def tab_disp():
    a = np.genfromtxt(D.path("reference_8p4e9.csv"), delimiter=",", names=True)
    rows, num = [], {"rows": []}
    for k in range(len(a)):
        rows.append("%d & %s & $%.4f$ & %s & %s & %.2f\\\\" % (a["rank"][k], sci(a["g"][k], 3), a["lambda_far"][k], sci(a["delta"][k], 4), sci(a["prediction"][k], 4), a["abs_rel_err_pct"][k]))
        num["rows"].append({"rank": int(a["rank"][k]), "g": float(a["g"][k]), "lambda_far": float(a["lambda_far"][k]), "delta": float(a["delta"][k]),
                            "prediction": float(a["prediction"][k]), "err_pct": float(a["abs_rel_err_pct"][k])})
    num["corr"] = S.corr(a["delta"], a["prediction"]); num["median_err"] = float(np.median(a["abs_rel_err_pct"])); num["max_err"] = float(a["abs_rel_err_pct"].max())
    for ens in ("100k", "2M"):
        b = np.genfromtxt(D.path("reference_%s.csv" % ens), delimiter=",", names=True)
        num[ens] = {"n": int(len(b)), "corr": S.corr(b["delta"], b["prediction"]), "median_err": float(np.median(b["abs_rel_err_pct"])), "max_err": float(b["abs_rel_err_pct"].max())}
    body = ("\\begin{tabular}{@{}rccccc@{}}\n\\toprule\npair & $g$ & $\\lfar$ & $\\delta$ (measured) & $(g^{2}/8)\\lfar$ & rel.\\ err.\\ (\\%)\\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_disp", body, num)


def tab_lehmer():
    b = np.atleast_1d(np.genfromtxt(D.path("reference_lehmer.csv"), delimiter=",", names=True))[0]
    leh = np.load(D.path("lehmer_pair_7005.npz"))
    import mpmath as mp
    mp.mp.dps = 30
    t0 = mp.mpf(float(leh["gamma"][0]) + float(leh["gamma"][1])) / 2
    Zpp0 = float(abs(mp.diff(mp.siegelz, t0, 2)))
    g = float(b["g"]); F = float(b["F"])
    pred_floor = g ** 2 / 8 * Zpp0
    RL = np.log(F) - leh["logP"] - leh["zsum"][2]
    num = {"gamma": [float(x) for x in leh["gamma"]], "g": g, "lambda_far": float(b["lambda_far"]), "far250": float(b["far250"]),
           "delta_pred": float(b["prediction"]), "delta_meas": float(b["delta"]), "err_pct": float(b["abs_rel_err_pct"]),
           "floor_meas": F, "floor_pred": pred_floor, "floor_err_pct": 100 * abs(pred_floor / F - 1),
           "exact_law_dev": float(b["Rall"]), "R3000": {str(int(X)): float(RL[j]) for j, X in enumerate(XS)}}
    rows = ["displacement $\\delta$ (Cor.~\\ref{cor:B1}) & %s & %s & $%.1f\\%%$\\\\" % (sci(num["delta_pred"], 3), sci(num["delta_meas"], 3), num["err_pct"]),
            "floor, $(g^{2}/8)|Z''(t_{0})|$ (Prop.~\\ref{prop:C}) & %s & %s & $%.2f\\%%$\\\\" % (sci(pred_floor, 4), sci(F, 4), num["floor_err_pct"]),
            "floor, $|Z''(t^{*})|/\\Son(t^{*})$ (Thm.~\\ref{thm:A}(c)) & exact & $\\Fl$ & %s\\\\" % sci(abs(num["exact_law_dev"]), 1),
            "$R_{3000}(t_{0};X)$ of \\eqref{eq:resid}, $X=10^{2},10^{4},10^{7}$ & $0$ & $%+.4f$, $%+.4f$, $%+.4f$ & ---\\\\" % (RL[0], RL[2], RL[5])]
    body = ("\\begin{tabular}{@{}lccc@{}}\n\\toprule\nquantity & predicted & measured & rel.\\ err.\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_lehmer", body, num)


ALL = [tab_identity, tab_floorid, tab_fits, tab_smooth, tab_decomp, tab_deficit, tab_blocks, tab_farlaw, tab_gue, tab_chernoff, tab_disp, tab_lehmer]

if __name__ == "__main__":
    if "--out" in sys.argv:
        OUT = sys.argv[sys.argv.index("--out") + 1]
    only = [a for a in sys.argv[1:] if a.startswith("tab_")]
    for f in ALL:
        if only and f.__name__ not in only:
            continue
        try:
            f()
        except FileNotFoundError as e:
            print("skipped %s: %s" % (f.__name__, e))
