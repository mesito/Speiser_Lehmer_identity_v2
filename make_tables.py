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
from tightpairs import data as D, euler as E, kernel as K, stats as S, tails as T, geometry as G, gue as GU  # noqa: E402
from tightpairs.constants import XS, GAMMA_E, dens  # noqa: E402

OUT = "tables"
ENS = [("100k", "100k", r"Odlyzko 100k", r"$\le7.5\times10^{4}$", 1.494),
       ("2M", "2M", r"Odlyzko 2M", r"$\le1.13\times10^{6}$", 1.926),
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
    body = ("\\begin{tabular}{@{}c ccc c ccc@{}}\n\\toprule\n & \\multicolumn{3}{c}{LMFDB, $t\\approx8.436\\times10^{9}$} & & \\multicolumn{3}{c}{2M, $t\\le1.13\\times10^{6}$}\\\\\n"
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
    # loading accounting (Observation 5.10)
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
    f8 = np.load(D.path("floors_all_8p4e9.npy"))
    sub = _sub_index(f8)
    rows, num = [], {}
    for j, X in enumerate(XS):
        mp_, mc_, sd_ = float(lp8[:, j].mean()), float(lpc[:, j].mean()), float(lpc[:, j].std(ddof=1))
        sh, fac = (mp_ - mc_) / sd_, float(np.exp(mc_ - mp_))
        ms_ = float(lp8[sub, j].mean())
        shs, facs = (ms_ - mc_) / sd_, float(np.exp(mc_ - ms_))
        num[str(int(X))] = {"mean_pairs": mp_, "mean_ctrl": mc_, "sd_ctrl": sd_, "shift": sh, "factor": fac,
                            "mean_sub": ms_, "shift_sub": shs, "factor_sub": facs, "sd_pairs": float(lp8[:, j].std(ddof=1))}
        rows.append("$10^{%d}$ & $%.3f$ & $%.3f$ & $%.3f$ & $%+.2f\\sigma$ & $%.1f$ & & $%.3f$ & $%+.2f\\sigma$ & $%.1f$\\\\"
                    % (int(np.log10(X)), mp_, mc_, sd_, sh, fac, ms_, shs, facs))
    fc = np.load(D.path("floors_ctrl_8p4e9.npy")); yc = np.log(fc[:, 2])
    for j, X in ((2, 1e4), (5, 1e7)):
        (m_, c_), _, r2_, _ = S.ols(lpc[:, j][:, None], yc)
        num["ctrl_regression_1e%d" % int(np.log10(X))] = {"slope": m_, "intercept": c_, "corr": S.corr(lpc[:, j], yc), "R2": r2_}
    body = ("\\begin{tabular}{@{}cccccc c ccc@{}}\n\\toprule\n & \\multicolumn{5}{c}{all $2{,}736$ pairs} & & \\multicolumn{3}{c}{rank subsample}\\\\\n"
            "\\cmidrule(lr){2-6}\\cmidrule(lr){8-10}\n$X$ & mean, pairs & mean, controls & control s.d. & shift & $\\e^{-\\Delta}$ & & mean & shift & $\\e^{-\\Delta}$\\\\\n\\midrule\n"
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


BASES = {"1e12": 267653395647, "1e21": 144176897509546973000}


def tab_farlaw():
    """Table: the far-field law; far is normalised pair by pair with dens(t0)^2 at the pair's own height."""
    rows, num = [], {}
    xs_, ys_ = [], []
    for key, short, label, height, _ in ENS:
        a = np.load(D.path("lehmer_rows_%s.npy" % key))
        t0 = a[:, 1] + BASES.get(key, 0)
        d2 = dens(t0) ** 2
        phi = a[:, 5] / d2
        mf, md2 = float(a[:, 5].mean()), float(d2.mean())
        num[short] = {"mean_dens2": md2, "mean_far": mf, "ratio": float(phi.mean()), "se_ratio": float(phi.std(ddof=1) / np.sqrt(len(phi))),
                      "n": int(len(a)), "mean_log_t": float(np.mean(np.log(t0 / (2 * np.pi))))}
        xs_.append(md2); ys_.append(mf)
        name = {"100k": "100k", "2M": "2M", "LMFDB": "LMFDB", "1e12": "$10^{12}$ window", "1e21": "$10^{21}$ window"}[short]
        rows.append("%s & %d & %.3f & %.2f & $%.3f\\pm%.3f$\\\\" % (name, len(a), md2, mf, phi.mean(), num[short]["se_ratio"]))
    num["corr"] = S.corr(xs_, ys_)
    # one-parameter finite-height correction C_pop (1 - c / log(t/2pi)), weighted least squares
    C_pop = 2.610
    y = np.array([num[k]["ratio"] for k in ("100k", "2M", "LMFDB", "1e12", "1e21")])
    se = np.array([num[k]["se_ratio"] for k in ("100k", "2M", "LMFDB", "1e12", "1e21")])
    L = np.array([num[k]["mean_log_t"] for k in ("100k", "2M", "LMFDB", "1e12", "1e21")])
    w = 1 / se ** 2; xx = C_pop / L
    c = float(np.sum(w * xx * (C_pop - y)) / np.sum(w * xx * xx))
    num["logfit"] = {"C_pop": C_pop, "c": c, "chi2": float(np.sum(((y - C_pop * (1 - c / L)) / se) ** 2)), "dof": 4}
    a8 = np.load(D.path("lehmer_rows_8p4e9.npy"))
    num["far_8p4e9"] = {"median": float(np.median(a8[:, 5])), "mean": float(a8[:, 5].mean()), "max": float(a8[:, 5].max())}
    body = ("\\begin{tabular}{@{}lrccc@{}}\n\\toprule\nensemble & pairs & mean $\\dens(t_{0})^{2}$ & mean $\\far$ & mean $\\far/\\dens(t_{0})^{2}$\\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_farlaw", body, num)


def tab_gue():
    C9 = np.pi ** 2 / 9.0
    S0 = (0.05, 0.10, 0.15)
    rows, num = [], {}
    for name, loader in (("100k", lambda: D.load_odlyzko("zeros1")), ("2M", lambda: D.load_odlyzko("zeros6")), ("LMFDB $8.4\\times10^{9}$", lambda: D.load_longdouble())):
        z = loader()
        s = D.local_spacings(z)
        rr = [float(np.mean(s < s0)) / (C9 * s0 ** 3) for s0 in S0]
        lag = S.corr(s[:-1], s[1:])
        key = name.split()[0]
        num[key] = {"tail_ratios": rr, "counts": [int(np.sum(s < s0)) for s0 in S0], "n_spacings": int(len(s)), "lag1": lag}
        rows.append("%s & %.3f & %.3f & %.3f & $%.3f$\\\\" % (name, rr[0], rr[1], rr[2], lag))
    for name, wname in (("$10^{12}$ window", "zeros3"), ("$10^{21}$ window", "zeros4")):
        s, base, off = _window_spacings(wname)
        cnt = [int(np.sum(s < s0)) for s0 in S0]
        rr = [c / len(s) / (C9 * s0 ** 3) for c, s0 in zip(cnt, S0)]
        lag = S.corr(s[:-1], s[1:])
        key = "1e12" if wname == "zeros3" else "1e21"
        num[key] = {"tail_ratios": rr, "counts": cnt, "n_spacings": int(len(s)), "lag1": lag, "mean_s": float(s.mean())}
        rows.append("%s & %.2f (%d) & %.2f (%d) & %.2f (%d) & $%.3f$\\\\" % (name, rr[0], cnt[0], rr[1], cnt[1], rr[2], cnt[2], lag))
    m = np.load(D.path("gue_mc_summary.npz"))
    ref = {"lag1": float(m["lag1"]), "lag1_npairs": int(m["lag1_npairs"]), "n_bulk_spacings": int(m["n_bulk_spacings"]),
           "N500_w100": float(m["N500_w100_mean"]), "N500_w100_se": float(m["N500_w100_se"]), "total_pairs": int(m["total_pairs"])}
    for key in ("tail_005", "tail_010", "tail_015"):
        ref[key], ref[key + "_se"], ref[key + "_count"] = float(m[key]), float(m[key + "_se"]), int(m[key + "_count"])
    num["GUE_MC"] = ref
    exact = [GU.tail_ratio_exact(s0) for s0 in S0]
    num["sine_kernel"] = {"tail_ratios": exact, "series": [float(GU.tail_ratio_series(s0)) for s0 in S0]}
    rows.append("GUE (sine kernel) & %.3f & %.3f & %.3f & $%.3f$\\\\" % (exact[0], exact[1], exact[2], ref["lag1"]))
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



# ----------------------------------------------------------------- tables added with the rank subsample,
# the proxy L_X^(3), the windows at 1e12 / 1e21 and the certificate
def _fit(lg, L, lf):
    (a, b, c), se, r2, res = S.ols(np.column_stack([lg, L]), lf)
    return a, se[0], b, se[1], r2


def _sub_index(f8):
    """indices (into the 2,736 rows of floors_all_8p4e9) of the rank subsample of 300 pairs."""
    return D.rank_subsample(f8[:, 2], 300)


def tab_sub():
    """Table: Euler regression on the full primary ensemble and on its 300-pair rank subsample."""
    f8 = np.load(D.path("floors_all_8p4e9.npy")); lp8 = np.load(D.path("logP_pairs_8p4e9.npy"))
    sub = _sub_index(f8)
    lg, lf = np.log(f8[:, 2]), np.log(f8[:, 5])
    rows, num = [], {"full": {}, "sub": {}}
    for j, X in enumerate(XS):
        A = _fit(lg, lp8[:, j], lf); B = _fit(lg[sub], lp8[sub, j], lf[sub])
        num["full"][str(int(X))] = dict(zip(("a", "se_a", "b", "se_b", "R2"), A))
        num["sub"][str(int(X))] = dict(zip(("a", "se_a", "b", "se_b", "R2"), B))
        rows.append("$10^{%d}$ & $%.4f$ & $%.4f$ & $%.4f$ & & $%.4f\\,(%.4f)$ & $%.4f\\,(%.4f)$ & $%.4f$ & & $%+.3f$ & $%+.3f$\\\\"
                    % (int(np.log10(X)), A[0], A[2], A[4], B[0], B[1], B[2], B[3], B[4], B[0] - A[0], B[2] - A[2]))
    # the subsample itself and the gap-only statistics (Section 5.1)
    order = np.argsort(f8[:, 2], kind="stable")
    num["n_sub"] = int(len(sub)); num["first_is_tightest"] = bool(sub[0] == order[0])
    num["sub_idx1"] = [int(f8[i, 0]) for i in sub[:15]]
    num["sub_g_range"] = [float(f8[sub, 2].min()), float(f8[sub, 2].max())]
    for key, ii in (("g_only_full", np.arange(len(f8))), ("g_only_sub", sub)):
        (sl, c0), se, r2, res = S.ols(lg[ii][:, None], lf[ii])
        num[key] = {"slope": float(sl), "se": float(se[0]), "corr": S.corr(lg[ii], lf[ii]),
                    "resid_sd_nats": float(res.std()), "resid_sd_dex": float(res.std() / np.log(10))}
    (_, _), _, _, rs = S.ols(lg[sub][:, None], lf[sub])
    far = f8[:, 4]
    num["pcorr_sub_P_given_far"] = S.partial_corr(rs, lp8[sub, 2], far[sub])
    num["pcorr_sub_far_given_P"] = S.partial_corr(rs, far[sub], lp8[sub, 2])
    f2 = np.load(D.path("floors_all_2M.npy")); lp2 = np.load(D.path("logP_pairs_2M.npy"))
    sub2 = D.rank_subsample(f2[:, 2], 300)
    lg2, lf2 = np.log(f2[:, 2]), np.log(f2[:, 5])
    num["2M"] = {"full_1e4": dict(zip(("a", "se_a", "b", "se_b", "R2"), _fit(lg2, lp2[:, 2], lf2))),
                 "sub_1e4": dict(zip(("a", "se_a", "b", "se_b", "R2"), _fit(lg2[sub2], lp2[sub2, 2], lf2[sub2])))}
    body = ("\\begin{tabular}{@{}c ccc c ccc c cc@{}}\n\\toprule\n & \\multicolumn{3}{c}{full ensemble, $2{,}736$ pairs} & & \\multicolumn{3}{c}{rank subsample, $300$ pairs} & & \\multicolumn{2}{c}{difference}\\\\\n"
            "\\cmidrule{2-4}\\cmidrule{6-8}\\cmidrule{10-11}\n$X$ & $a$ & $b$ & $R^{2}$ & & $a$ & $b$ & $R^{2}$ & & $\\Delta a$ & $\\Delta b$\\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_sub", body, num)


def tab_ladder():
    """Table: the variance ladder, R^2 of log F on successive regressor sets, two ensembles, X = 1e4 and 1e7."""
    sets = (("g", "$\\log g$"), ("g_far", "$\\log g,\\ \\log\\far$"), ("g_P", "$\\log g,\\ \\log|\\PX|$"), ("g_P_far", "$\\log g,\\ \\log|\\PX|,\\ \\log\\far$"))
    num, cols = {}, []
    for ens in ("8p4e9", "2M"):
        fa = np.load(D.path("floors_all_%s.npy" % ens)); lp = np.load(D.path("logP_pairs_%s.npy" % ens))
        lg, lf, lfar = np.log(fa[:, 2]), np.log(fa[:, 5]), np.log(fa[:, 4])
        for j, X in ((2, 1e4), (5, 1e7)):
            L = lp[:, j]
            r = {"g": S.ols(lg[:, None], lf)[2], "g_far": S.ols(np.column_stack([lg, lfar]), lf)[2],
                 "g_P": S.ols(np.column_stack([lg, L]), lf)[2], "g_P_far": S.ols(np.column_stack([lg, L, lfar]), lf)[2]}
            num["%s_%s" % (ens, int(np.log10(X)))] = r
            cols.append(r)
    rows = []
    for key, lab in sets:
        rows.append(lab + " & " + " & ".join("$%.4f$" % c[key] for c in cols) + "\\\\")
    body = ("\\begin{tabular}{@{}l cc cc@{}}\n\\toprule\n & \\multicolumn{2}{c}{LMFDB} & \\multicolumn{2}{c}{2M}\\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nregressors & $X=10^{4}$ & $X=10^{7}$ & $X=10^{4}$ & $X=10^{7}$\\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_ladder", body, num)


def tab_eiv():
    """Table: errors in variables -- the regression on the proxy L_X^(3) against the regression on the exact
    log|P_X|, and three reliabilities of the proxy."""
    pr = E.Primes(10 ** 7)
    num, rows = {}, []
    for ens in ("8p4e9", "2M"):
        fa = np.load(D.path("floors_all_%s.npy" % ens)); lp = np.load(D.path("logP_pairs_%s.npy" % ens)); l3 = np.load(D.path("LX3_pairs_%s.npy" % ens))
        lg, lf = np.log(fa[:, 2]), np.log(fa[:, 5])
        num[ens] = {}
        for j, X in enumerate(XS):
            x, w = lp[:, j], l3[:, j]
            b = _fit(lg, x, lf)[2]; a3, _, b3, se3, r23 = _fit(lg, w, lf)
            rx, rw = S.resid_on(x, lg[:, None]), S.resid_on(w, lg[:, None])
            lam_meas = float(np.dot(rx, rw) / np.dot(rw, rw))
            dX = E.delta_sup(X, pr)
            lam_sup = float(1 - dX ** 2 / rw.var())
            lam_r = E.lambda_rand(X, pr)
            num[ens][str(int(X))] = {"b": b, "a3": a3, "b3": b3, "se_b3": se3, "R2_3": r23, "ratio": b3 / b, "lambda_meas": lam_meas,
                                     "lambda_rand": lam_r, "lambda_sup": lam_sup, "delta_sup": dX, "max_abs_diff": float(np.abs(w - x).max()),
                                     "sd_diff": float((w - x).std()), "corr_x_w": S.corr(x, w)}
            if ens == "8p4e9" or X in (1e4, 1e7):
                rows.append("%s & $10^{%d}$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.3f$ & $%.3f$\\\\"
                            % ("LMFDB" if (ens == "8p4e9" and j == 0) else ("2M" if (ens == "2M" and X == 1e4) else ""),
                               int(np.log10(X)), b, b3, b3 / b, lam_meas, lam_r, lam_sup, dX))
        if ens == "8p4e9":
            rows.append("\\midrule")
    body = ("\\begin{tabular}{@{}cc ccc ccc c@{}}\n\\toprule\nensemble & $X$ & $b$ (exact) & $b_{3}$ (proxy) & $b_{3}/b$ & $\\lambda_{\\mathrm{meas}}$ & $\\lambda_{\\mathrm{rand}}$ & $\\lambda_{\\sup}$ & $\\delta_{X}$\\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_eiv", body, num)


def tab_k5():
    """the correlation of the residual of log F after the gap law with log far, four ensembles."""
    num, rows = {}, []
    for key, label, fname in (("100k", "100k", "floors_all_100k.npy"), ("2M", "2M", "floors_all_2M.npy"),
                              ("8p4e9", "LMFDB $8.4\\times10^{9}$", "floors_all_8p4e9.npy"), ("1e12", "$10^{12}$ window", "floors_1e12.npy")):
        fa = np.load(D.path(fname))
        lg, lf, lfar = np.log(fa[:, 2]), np.log(fa[:, 5]), np.log(fa[:, 4])
        res = S.ols(lg[:, None], lf)[3]
        k5 = S.corr(res, lfar)
        n = len(fa)
        se = (1 - k5 ** 2) / np.sqrt(n - 3)
        num[key] = {"n": int(n), "K5": k5, "se": float(se), "g_slope": float(S.ols(lg[:, None], lf)[0][0])}
        rows.append("%s & %d & $%.3f$ & $%.3f$\\\\" % (label, n, k5, se))
    body = ("\\begin{tabular}{@{}lrcc@{}}\n\\toprule\nensemble & pairs & $\\operatorname{corr}(\\text{resid},\\log\\far)$ & s.e.\\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_k5", body, num)


def _window_spacings(name):
    base, off = D.load_window(name)
    mid = 0.5 * (off[:-1] + off[1:])
    return np.diff(off) * dens(base + mid), base, off


def tab_shield():
    """Table: the compensation shield on five ensembles, with standard errors."""
    num, rows = {}, []
    srcs = [("100k", "100k", D.load_odlyzko("zeros1")), ("2M", "2M", D.load_odlyzko("zeros6")), ("8p4e9", "LMFDB $8.4\\times10^{9}$", D.load_longdouble())]
    for key, label, z in srcs:
        gaps = np.asarray(np.diff(z), dtype=float)   # 80-bit differences for the primary ensemble
        ii = np.load(D.path("lehmer_rows_%s.npy" % key))[:, 0].astype(int) - 1
        num[key] = _shield(gaps, ii, 1.0 / dens(np.load(D.path("lehmer_rows_%s.npy" % key))[:, 1]))
        rows.append(_shield_row(label, num[key]))
    for key, label, name in (("1e12", "$10^{12}$ window", "zeros3"), ("1e21", "$10^{21}$ window", "zeros4")):
        base, off = D.load_window(name)
        gaps = np.diff(off)
        ii = np.load(D.path("lehmer_rows_%s.npy" % key))[:, 0].astype(int) - 1
        ok = (ii >= 1) & (ii + 1 < len(gaps))
        t0w = np.load(D.path("lehmer_rows_%s.npy" % key))[:, 1] + base
        num[key] = _shield(gaps, ii[ok], 1.0 / dens(t0w[ok]))
        num[key]["n_edge_excluded"] = int((~ok).sum())
        rows.append(_shield_row(label, num[key]))
    body = ("\\begin{tabular}{@{}lrcc@{}}\n\\toprule\nensemble & pairs & preceding gap & following gap\\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_shield", body, num)


def _shield(gaps, ii, gm):
    prev, nxt = gaps[ii - 1] / gm, gaps[ii + 1] / gm
    n = len(ii)
    return {"n": int(n), "prev": float(prev.mean()), "next": float(nxt.mean()), "se_prev": float(prev.std(ddof=1) / np.sqrt(n)),
            "se_next": float(nxt.std(ddof=1) / np.sqrt(n)), "both": float(0.5 * (prev.mean() + nxt.mean()))}


def _shield_row(label, r):
    return "%s & %d & $%.3f\\pm%.3f$ & $%.3f\\pm%.3f$\\\\" % (label, r["n"], r["prev"], r["se_prev"], r["next"], r["se_next"])


def tab_delta0():
    """Table: Delta_0(X), the uniform bound on the harmonics k >= 4 (Lemma 9.1)."""
    pr = E.Primes(10 ** 7)
    terms = E.Delta0_terms(pr)
    num, rows = {}, []
    for X in XS:
        v = float(terms[pr.p <= X].sum())
        num[str(int(X))] = v
        rows.append("$10^{%d}$ & $%.10f$\\\\" % (int(np.log10(X)), v))
    # the tail beyond X = 1e7: per prime sum_{k>=4} p^-k/2/k <= p^-2/(4(1-X^-1/2)), and by partial summation with
    # Rosser-Schoenfeld pi(x) < (x/log x)(1 + 3/(2 log x)) for x > 1, sum_{p>X} p^-2 <= 2(1 + 1.5/log X)/(X log X)
    X = 1e7
    tail = 2 * (1 + 1.5 / np.log(X)) / (X * np.log(X)) / (4 * (1 - X ** -0.5))
    num["tail_estimate"] = float(1 / (4 * X * np.log(X)))
    num["tail_bound"] = float(tail); num["p2"] = float(terms[0]); num["share_p2"] = float(terms[0] / terms.sum())
    num["variation"] = float(num["10000000"] - num["100"])
    rows.append("all primes & $%.10f+\\theta\\cdot%.1f\\times10^{-9}$\\\\" % (num["10000000"], tail * 1e9))
    body = "\\begin{tabular}{@{}cl@{}}\n\\toprule\n$X$ & $\\Delta_{0}(X)$\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    write("tab_delta0", body, num)


def tab_uniform():
    """Table: the excursion-measure criterion N d(u - m) of the conditional uniform bound, H = T, m = 0.5."""
    tb = T.TailBounds(10 ** 7)
    H = 8436146000.0
    m = 0.5
    N = tb.N_criterion(H, m)
    num, rows = {"N": N, "H": H, "m": m}, []
    for u in (9.0, 9.36, 9.5, 10.0):
        d_, s_ = tb.best(u - m)
        num[str(u)] = {"d": d_, "s": s_, "Nd": N * d_}
        rows.append("$%.2f$ & $%.2f$ & $%.2f$ & %s & %s\\\\" % (u, u - m, s_, sci(d_, 2), sci(N * d_, 2)))
    body = ("\\begin{tabular}{@{}ccccc@{}}\n\\toprule\n$u$ & $u-m$ & $s^{*}$ & $d(u-m)\\le$ & $N\\,d(u-m)$\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_uniform", body, num)


def tab_sweep():
    """Table: the detector sweep over the primary ensemble (Appendix C)."""
    a = np.genfromtxt(D.path("detector_sweep_8p4e9.csv"), delimiter=",", names=True)
    num = {"n": int(len(a)), "sign_changes": int(a["n_sign_changes"].sum()), "max_Ghat": float(a["max_Ghat"].max()),
           "g_at_max_Ghat": float(a["g"][np.argmax(a["max_Ghat"])]), "max_pair_term": float(np.max(-8 / a["g"] ** 2)),
           "r": [float(a["r"].min()), float(np.median(a["r"])), float(a["r"].max())],
           "H_rel": [float(a["H_over_halfgap"].min()), float(np.median(a["H_over_halfgap"])), float(a["H_over_halfgap"].max())],
           "far": [float(a["far"].min()), float(np.median(a["far"])), float(a["far"].max())]}
    rows = ["$r=\\hthr/(g/2)$ & $%.6f$ & $%.6f$ & $%.6f$\\\\" % tuple(num["r"]),
            "$H/(g/2)$, $H=\\sqrt{2/\\far}$ & $%.3f$ & $%.2f$ & $%.2f$\\\\" % tuple(num["H_rel"]),
            "$\\far$ ($W=250$) & $%.3f$ & $%.2f$ & $%.2f$\\\\" % tuple(num["far"])]
    body = "\\begin{tabular}{@{}lccc@{}}\n\\toprule\nstatistic & min & median & max\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    write("tab_sweep", body, num)


def tab_dispsub():
    """Table: displacement law on the first fifteen pairs of the rank subsample (Appendix B)."""
    a = np.genfromtxt(D.path("reference_8p4e9_sub.csv"), delimiter=",", names=True)
    rows, num = [], {"rows": []}
    for k in range(len(a)):
        rows.append("%d & %d & %s & $%+.4f$ & %s & %s & %.2f\\\\" % (a["rank"][k], a["idx1"][k], sci(a["g"][k], 4), a["lambda_far"][k],
                                                                  sci(a["delta"][k], 4), sci(a["prediction"][k], 4), a["abs_rel_err_pct"][k]))
        num["rows"].append({"rank": int(a["rank"][k]), "idx1": int(a["idx1"][k]), "g": float(a["g"][k]), "lambda_far": float(a["lambda_far"][k]),
                            "delta": float(a["delta"][k]), "prediction": float(a["prediction"][k]), "err_pct": float(a["abs_rel_err_pct"][k]),
                            "Rall": float(a["Rall"][k]), "curv_dev": float(a["curv_dev"][k])})
    num["corr"] = S.corr(a["delta"], a["prediction"]); num["median_err"] = float(np.median(a["abs_rel_err_pct"])); num["max_err"] = float(a["abs_rel_err_pct"].max())
    num["n_below_1pct"] = int((a["abs_rel_err_pct"] < 1).sum()); num["max_abs_Rall"] = float(np.abs(a["Rall"]).max())
    num["max_abs_err"] = float(np.abs(a["delta"] - a["prediction"]).max())
    body = ("\\begin{tabular}{@{}rrccccc@{}}\n\\toprule\nrank & $n$ & $g$ & $\\lfar$ & $\\delta$ (measured) & $(g^{2}/8)\\lfar$ & rel.\\ err.\\ (\\%)\\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_dispsub", body, num)


def demand(f8, lp8, zs8, j, X, h):
    """the demand Delta_*(h; X) of (8.5) on every pair: orbit local term + s_-(X) - (log F - log|P_X|)."""
    g = f8[:, 2]
    eps = g / 2
    Sfar = zs8[1, :, j] - K.pair_term(g, X)
    s_minus = float(np.percentile(Sfar, 5))
    obs = np.log(f8[:, 5]) - lp8[:, j]
    return K.orbit_local_term(eps, h, X) + s_minus - obs, s_minus, Sfar


def tab_demand():
    """Table: the demand of the depth certificate at depths g/2 and g, exact orbit term; the geometric part at H."""
    f8 = np.load(D.path("floors_all_8p4e9.npy")); lp8 = np.load(D.path("logP_pairs_8p4e9.npy")); zs8 = np.load(D.path("zsum_pairs_8p4e9.npy"))
    g, far = f8[:, 2], f8[:, 4]
    eps = g / 2
    H = np.sqrt(2 / far)
    geo = np.log((eps ** 2 + 2 * H ** 2) ** 2 / eps ** 2)
    num = {"geometric": {"min_pos": float(geo[geo > 0].min()), "median": float(np.median(geo)), "max": float(geo.max()), "n_pos": int((geo > 0).sum()),
                         "neg_values": sorted(float(v) for v in geo[geo <= 0])}}
    rows = []
    for j, X in enumerate(XS):
        scale = 2 * (GAMMA_E + np.log(np.log(X)))
        dE, s_minus, Sfar = demand(f8, lp8, zs8, j, X, eps)
        dG, _, _ = demand(f8, lp8, zs8, j, X, 2 * eps)
        zH = np.sqrt(eps ** 2 + 2 * H ** 2) * np.log(X)
        zE = np.sqrt(3) * eps * np.log(X)
        num[str(int(X))] = {"scale": float(scale), "s_minus": s_minus,
                            "eps": {"median": float(np.median(dE)), "min": float(dE.min()), "max": float(dE.max()), "n_pos": int((dE > 0).sum())},
                            "g": {"median": float(np.median(dG)), "min": float(dG.min()), "n_pos": int((dG > 0).sum())},
                            "n_zH_le_1": int((zH <= 1).sum()), "max_zE": float(zE.max()), "median_zE": float(np.median(zE))}
        rows.append("$10^{%d}$ & $%.2f$ & $%+.2f$ & $%+.2f$ & $%+.2f$ & %d & $%+.2f$ & %d & %d\\\\"
                    % (int(np.log10(X)), scale, s_minus, np.median(dE), dE.min(), (dE > 0).sum(), np.median(dG), (dG > 0).sum(), (zH <= 1).sum()))
    body = ("\\begin{tabular}{@{}c cc ccc cc c@{}}\n\\toprule\n & & & \\multicolumn{3}{c}{depth $h=g/2$} & \\multicolumn{2}{c}{depth $h=g$} & \\\\\n"
            "\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}\n$X$ & $2\\log(\\e^{\\gamma_{E}}\\log X)$ & $s_{-}(X)$ & median & min & $\\#>0$ & median & $\\#>0$ & $\\#\\{z_{H}\\le1\\}$\\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_demand", body, num)


ALL = [tab_identity, tab_floorid, tab_fits, tab_sub, tab_ladder, tab_eiv, tab_smooth, tab_decomp, tab_k5, tab_deficit, tab_blocks,
       tab_farlaw, tab_shield, tab_gue, tab_demand, tab_delta0, tab_chernoff, tab_uniform, tab_disp, tab_dispsub, tab_sweep, tab_lehmer]

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
