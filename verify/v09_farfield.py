"""V9 -- the far-field law, the shield, GUE diagnostics (Section 7, Observation 7.3, Tables 7.1-7.3)."""
import numpy as np

from tightpairs import data as D, stats as S
from tightpairs.constants import dens
from tightpairs.report import close, report, equal, below, section


def run(ctx):
    section("V9: far-field law, shield, GUE diagnostics (Section 7)")
    tfl = ctx.table("tab_farlaw")
    names = {"100k": "100k", "2M": "2M", "8p4e9": "LMFDB", "1e12": "1e12", "1e21": "1e21"}
    bases = {"1e12": D.load_window("zeros3")[0], "1e21": D.load_window("zeros4")[0]}
    xs, ys, ratio, se, L = [], [], {}, {}, {}
    for k in ("100k", "2M", "8p4e9", "1e12", "1e21"):
        a = ctx.rows[k]
        t0 = a[:, 1] + bases.get(k, 0)
        d2 = dens(t0) ** 2
        phi = a[:, 5] / d2
        tg = tfl[names[k]]
        close("V9 mean far %s (Table 7.1)" % k, float(a[:, 5].mean()), tg["mean_far"], 1e-9, "%.4f", "%.4f")
        close("V9 mean dens(t0)^2 %s (Table 7.1)" % k, float(d2.mean()), tg["mean_dens2"], 1e-9, "%.4f", "%.4f")
        close("V9 mean far/dens(t0)^2 %s (Table 7.1)" % k, float(phi.mean()), tg["ratio"], 1e-9, "%.4f", "%.4f")
        close("V9 s.e. of far/dens(t0)^2 %s (Table 7.1)" % k, float(phi.std(ddof=1) / np.sqrt(len(phi))), tg["se_ratio"], 1e-9, "%.4f", "%.4f")
        xs.append(float(d2.mean())); ys.append(float(a[:, 5].mean()))
        ratio[k], se[k] = float(phi.mean()), float(phi.std(ddof=1) / np.sqrt(len(phi)))
        L[k] = float(np.mean(np.log(t0 / (2 * np.pi))))
    for k, tgt in (("100k", 2.198), ("2M", 2.328), ("8p4e9", 2.501), ("1e12", 2.709), ("1e21", 2.552)):
        close("V9 far/dens(t0)^2 %s (text %.3f)" % (k, tgt), ratio[k], tgt, 5e-4, "%.4f", "%.3f")
    close("V9 corr(mean far, mean dens(t0)^2) (0.9997)", S.corr(xs, ys), 0.9997, 5e-5, "%.5f", "%.4f")
    m3 = np.mean([ratio[k] for k in ("8p4e9", "1e12", "1e21")])
    se3 = np.sqrt(sum(se[k] ** 2 for k in ("8p4e9", "1e12", "1e21"))) / 3
    report("V9 mean of the 3 high ensembles (2.59 +- 0.08)", "%.3f +- %.3f" % (m3, se3), "2.59 +- 0.08", 5e-3, abs(m3 - 2.59) <= 5e-3 and abs(se3 - 0.08) <= 5e-3, "%s", "%s")
    for k, tgt in (("100k", 16), ("2M", 11)):
        close("V9 deficit of %s below 2.610 (%d%%)" % (k, tgt), 100 * (1 - ratio[k] / 2.610), tgt, 0.5, "%.2f", "%d")
    # Observation 7.3: weighted one-parameter fit 2.610 (1 - c / log(t/2pi))
    ks = ("100k", "2M", "8p4e9", "1e12", "1e21")
    y = np.array([ratio[k] for k in ks]); w = 1 / np.array([se[k] for k in ks]) ** 2; xx = 2.610 / np.array([L[k] for k in ks])
    c = float(np.sum(w * xx * (2.610 - y)) / np.sum(w * xx * xx))
    z = (y - 2.610 * (1 - c / np.array([L[k] for k in ks]))) * np.sqrt(w)
    close("V9 log-fit c (1.23)", c, 1.23, 5e-3, "%.4f", "%.2f")
    close("V9 log-fit chi^2, 4 dof (6.6)", float(np.sum(z ** 2)), 6.6, 0.05, "%.3f", "%.1f")
    close("V9 log-fit residual, primary (+1.7)", float(z[2]), 1.7, 0.05, "%.3f", "%.1f")
    close("V9 log-fit residual, 100k (-1.5)", float(z[0]), -1.5, 0.05, "%.3f", "%.1f")
    report("V9 log-fit: largest |residuals| are primary and 100k", "%s" % np.round(z, 2).tolist(), "-", "-", set(np.argsort(-np.abs(z))[:2]) == {0, 2}, "%s", "%s")
    close("V9 log-fit c (table JSON)", c, tfl["logfit"]["c"], 1e-9, "%.4f", "%.4f")
    close("V9 far/dens(t0)^2 at 8.4e9 (2.5)", ratio["8p4e9"], 2.5, 5e-3, "%.3f", "%.1f")
    close("V9 far median 8.4e9 (24.84)", tfl["far_8p4e9"]["median"], 24.84, 0.01, "%.2f", "%.2f")
    close("V9 far max 8.4e9 (368.35)", tfl["far_8p4e9"]["max"], 368.35, 0.01, "%.2f", "%.2f")
    # the shield, in units of the local mean gap 1/dens(t0)
    tsh = ctx.table("tab_shield")
    srcs = (("100k", ctx.z1, None), ("2M", ctx.z6, None), ("8p4e9", ctx.zl, None), ("1e12", None, "zeros3"), ("1e21", None, "zeros4"))
    sh = {}
    for name, zz, wname in srcs:
        a = ctx.rows[name]
        ii = a[:, 0].astype(int) - 1
        if wname is None:
            gp = np.asarray(np.diff(zz), dtype=float); t0 = a[:, 1]      # 80-bit differences for the primary ensemble
        else:
            b, off = D.load_window(wname); gp = np.diff(off); t0 = a[:, 1] + b
        ok = (ii >= 1) & (ii + 1 < len(gp))
        ii, t0 = ii[ok], t0[ok]
        prev, nxt = gp[ii - 1] * dens(t0), gp[ii + 1] * dens(t0)
        tg = tsh[name]
        sh[name] = (prev.mean(), nxt.mean(), prev.std(ddof=1) / np.sqrt(len(ii)), nxt.std(ddof=1) / np.sqrt(len(ii)))
        good = all(abs(u - v) <= 1e-9 for u, v in zip(sh[name], (tg["prev"], tg["next"], tg["se_prev"], tg["se_next"])))
        report("V9 shield %s prev/next/se (Table 7.2)" % name, "%.4f/%.4f/%.4f/%.4f" % sh[name],
               "%.4f/%.4f/%.4f/%.4f" % (tg["prev"], tg["next"], tg["se_prev"], tg["se_next"]), 1e-9, good, "%s", "%s")
    for name, tgt in (("100k", (1.403, 1.373)), ("2M", (1.375, 1.378)), ("8p4e9", (1.354, 1.360)), ("1e12", (1.399, 1.236)), ("1e21", (1.476, 1.210))):
        report("V9 shield %s prev/next (text)" % name, "%.3f/%.3f" % sh[name][:2], "%.3f/%.3f" % tgt, 5e-4,
               abs(sh[name][0] - tgt[0]) <= 5e-4 and abs(sh[name][1] - tgt[1]) <= 5e-4)
    ses = [v for k in ("100k", "2M", "8p4e9") for v in sh[k][2:]]
    report("V9 shield s.e. on the large ensembles in [0.005, 0.024]", "%.4f..%.4f" % (min(ses), max(ses)), "[0.005,0.024]", "-", 0.0045 <= min(ses) and max(ses) <= 0.0245, "%s", "%s")
    sew = [v for k in ("1e12", "1e21") for v in sh[k][2:]]
    report("V9 shield s.e. in the windows in [0.07, 0.08]", "%.4f..%.4f" % (min(sew), max(sew)), "[0.07,0.08]", "-", 0.065 <= min(sew) and max(sew) <= 0.085, "%s", "%s")
    allv = [v for k in ("100k", "2M", "8p4e9") for v in sh[k][:2]]
    report("V9 shield in [1.35, 1.40] on the large ensembles", "%.3f..%.3f" % (min(allv), max(allv)), "[1.35,1.40]", "-", 1.35 <= min(allv) and max(allv) <= 1.405, "%s", "%s")
    for k, tgt in (("1e12", 1.32), ("1e21", 1.34)):
        close("V9 shield %s mean of the two sides (%.2f)" % (k, tgt), 0.5 * (sh[k][0] + sh[k][1]), tgt, 5e-3, "%.4f", "%.2f")
    for k, tgt in (("1e12", 1.5), ("1e21", 2.3)):
        dz = (sh[k][0] - sh[k][1]) / np.hypot(sh[k][2], sh[k][3])
        close("V9 shield %s prev-next in s.e. of the difference (%.1f)" % (k, tgt), dz, tgt, 0.05, "%.3f", "%.1f")
    both = {k: 0.5 * (sh[k][0] + sh[k][1]) for k in sh}
    report("V9 two-sided shield on the large ensembles in [1.36, 1.39] (rounded)", "%.4f/%.4f/%.4f" % (both["100k"], both["2M"], both["8p4e9"]), "[1.355,1.395]", "-",
           all(1.355 <= both[k] < 1.395 for k in ("100k", "2M", "8p4e9")), "%s", "%s")
    for k in ("1e12", "1e21"):
        sew_ = 0.5 * np.hypot(sh[k][2], sh[k][3])
        below("V9 window %s two-sided shield vs large-ensemble range, in s.e." % k, max(0.0, 1.355 - both[k], both[k] - 1.395) / sew_, 1.0, "%.2f")
        below("V9 window %s two-sided shield within 1.3 s.e. of each large-ensemble value" % k, max(abs(both[k] - both[e]) for e in ("100k", "2M", "8p4e9")) / sew_, 1.3, "%.2f")
    d2m = 0.5 * (sh["2M"][0] + sh["2M"][1]) - 0.5 * (sh["8p4e9"][0] + sh["8p4e9"][1])
    close("V9 shield: primary below 2M by 0.02", d2m, 0.02, 0.005, "%.4f", "%.2f")
    equal("V9 tight pairs in the 1e12 window", int(len(ctx.rows["1e12"])), 38)
    equal("V9 tight pairs in the 1e21 window", int(len(ctx.rows["1e21"])), 32)
    tg_ = ctx.table("tab_gue")
    C9 = np.pi ** 2 / 9.0
    S0 = (0.05, 0.10, 0.15)
    npairs = {"100k": 261, "2M": 6088, "LMFDB": 2736}
    for name, zz, key in (("100k", ctx.z1, "100k"), ("2M", ctx.z6, "2M"), ("8.4e9", ctx.zl, "LMFDB")):
        s = D.local_spacings(zz)          # s_i = g_i dens(midpoint), the normalisation of the tight-pair definition
        rr_ = [float(np.mean(s < s0)) / (C9 * s0 ** 3) for s0 in S0]
        report("V9 GUE tail ratios %s (Table 7.3)" % name, "/".join("%.3f" % v for v in rr_), "/".join("%.3f" % v for v in tg_[key]["tail_ratios"]), 1e-9,
               all(abs(a - b) <= 1e-9 for a, b in zip(rr_, tg_[key]["tail_ratios"])))
        close("V9 lag-1 corr %s (Table 7.3)" % name, S.corr(s[:-1], s[1:]), tg_[key]["lag1"], 1e-9, "%.4f", "%.4f")
        equal("V9 %s: spacings below 0.15 = tight pairs" % name, int(np.sum(s < 0.15)), npairs[key])
        if key == "LMFDB":
            sd = D.local_spacings(ctx.zd)
            equal("V9 LMFDB: counts below 0.05/0.10/0.15 unchanged with double ordinates", [int(np.sum(sd < s0)) for s0 in S0], [int(np.sum(s < s0)) for s0 in S0])
    from tightpairs import gue as GU
    L_ = tg_["LMFDB"]
    zs = [(GU.tail_ratio_exact(s0) - L_["tail_ratios"][k]) / (L_["tail_ratios"][k] / np.sqrt(L_["counts"][k])) for k, s0 in enumerate(S0)]
    report("V9 LMFDB tail ratios below the sine kernel, in Poisson s.e. (1.2/0.3/1.4)", "/".join("%.2f" % z for z in zs), "1.2/0.3/1.4", 0.05,
           all(abs(a - b) <= 0.05 for a, b in zip(zs, (1.2, 0.3, 1.4))), "%s", "%s")
    report("V9 counts below 0.05: 11/255/94 (100k/2M/LMFDB)", "%d/%d/%d" % tuple(tg_[k]["counts"][0] for k in ("100k", "2M", "LMFDB")), "11/255/94", 0,
           [tg_[k]["counts"][0] for k in ("100k", "2M", "LMFDB")] == [11, 255, 94], "%s", "%s")
    close("V9 LMFDB lag-1 minus reference (-0.024)", L_["lag1"] - tg_["GUE_MC"]["lag1"], -0.024, 5e-4, "%.4f", "%.3f")
    close("V9 LMFDB lag-1 standard error (1 - rho^2)/sqrt(n) (about 0.001)", (1 - L_["lag1"] ** 2) / np.sqrt(L_["n_spacings"]), 0.001, 2e-4, "%.5f", "%.3f")
    for k in ("1e12", "1e21"):
        close("V9 window %s lag-1 standard error (about 0.01)" % k, (1 - tg_[k]["lag1"] ** 2) / np.sqrt(tg_[k]["n_spacings"]), 0.01, 1.5e-3, "%.4f", "%.2f")
    pct = [100 / np.sqrt(tg_[k]["counts"][0]) for k in ("100k", "2M", "LMFDB")]
    report("V9 Poisson errors at s0=0.05: 30/6/10 %", "/".join("%.1f" % v for v in pct), "30/6/10", 0.5,
           all(abs(round(v) - t) <= 0.5 for v, t in zip(pct, (30, 6, 10))), "%s", "%s")
    pw = [100 / np.sqrt(c) for k in ("1e12", "1e21") for c in tg_[k]["counts"]]
    report("V9 Poisson errors in the windows 16-100 %", "%.1f..%.1f" % (min(pw), max(pw)), "16..100", 0.5, round(min(pw)) == 16 and round(max(pw)) == 100, "%s", "%s")
    equal("V9 LMFDB spacings (772,718)", L_["n_spacings"], 772718)
    for key, tgt, lag in (("100k", (0.802, 0.720, 0.705), -0.357), ("2M", (0.930, 0.831, 0.822), -0.349), ("LMFDB", (0.887, 0.981, 0.957), -0.332)):
        report("V9 GUE tail ratios %s (text %s)" % (key, "/".join("%.3f" % v for v in tgt)), "/".join("%.3f" % v for v in tg_[key]["tail_ratios"]), "/".join("%.3f" % v for v in tgt), 5e-4,
               all(abs(a - b) <= 5e-4 for a, b in zip(tg_[key]["tail_ratios"], tgt)))
        close("V9 lag-1 %s (text %.3f)" % (key, lag), tg_[key]["lag1"], lag, 5e-4, "%.4f", "%.3f")
    for key, wname in (("1e12", "zeros3"), ("1e21", "zeros4")):
        b, off = D.load_window(wname)
        sw = np.diff(off) * dens(b + 0.5 * (off[:-1] + off[1:]))
        cnt = [int(np.sum(sw < s0)) for s0 in S0]
        rr_ = [c / len(sw) / (C9 * s0 ** 3) for c, s0 in zip(cnt, S0)]
        tgw = tg_[key]
        report("V9 GUE window %s counts / ratios / lag-1 (Table 7.3)" % key, "%s / %s / %.4f" % (cnt, "/".join("%.3f" % v for v in rr_), S.corr(sw[:-1], sw[1:])),
               "%s / %s / %.4f" % (tgw["counts"], "/".join("%.3f" % v for v in tgw["tail_ratios"]), tgw["lag1"]), 1e-9,
               cnt == tgw["counts"] and abs(S.corr(sw[:-1], sw[1:]) - tgw["lag1"]) <= 1e-9, "%s", "%s")

