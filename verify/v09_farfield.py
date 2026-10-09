"""V9 -- the far-field law, the shield, GUE diagnostics (Section 7, Tables 7.1, 7.3)."""
import numpy as np

from tightpairs import data as D, stats as S
from tightpairs.report import close, report, equal, section


def run(ctx):
    section("V9: far-field law, shield, GUE diagnostics (Section 7)")
    tfl = ctx.table("tab_farlaw")
    names = {"100k": "100k", "2M": "2M", "8p4e9": "LMFDB", "1e12": "1e12", "1e21": "1e21"}
    for k, a in ctx.rows.items():
        mf = float(a[:, 5].mean())
        tg = tfl[names[k]]
        close("V9 mean far %s (Table 7.1)" % k, mf, tg["mean_far"], 1e-9, "%.4f", "%.4f")
        close("V9 far/dens^2 %s" % k, mf / tg["dens"] ** 2, tg["ratio"], 1e-9, "%.4f", "%.4f")
    close("V9 corr(mean far, dens^2) (0.9995)", tfl["corr"], 0.9995, 1e-4, "%.4f", "%.4f")
    close("V9 far/dens^2 at 8.4e9 (2.500)", tfl["LMFDB"]["ratio"], 2.500, 2e-3, "%.3f", "%.3f")
    close("V9 mean of 3 high ensembles (2.59)", np.mean([tfl[k]["ratio"] for k in ("LMFDB", "1e12", "1e21")]), 2.59, 0.01, "%.3f", "%.2f")
    close("V9 far median 8.4e9 (24.84)", tfl["far_8p4e9"]["median"], 24.84, 0.01, "%.2f", "%.2f")
    close("V9 far max 8.4e9 (368.35)", tfl["far_8p4e9"]["max"], 368.35, 0.01, "%.2f", "%.2f")
    zl = ctx.zl
    gaps = np.diff(zl)
    gm = gaps.mean()
    i0 = ctx.rows["8p4e9"][:, 0].astype(int) - 1
    close("V9 shield prev 8.4e9 (1.354)", float(gaps[i0 - 1].mean() / gm), 1.354, 1e-3, "%.4f", "%.3f")
    close("V9 shield next 8.4e9 (1.360)", float(gaps[i0 + 1].mean() / gm), 1.360, 1e-3, "%.4f", "%.3f")
    for name, zz, tg in (("100k", ctx.z1, (1.376, 1.347)), ("2M", ctx.z6, (1.369, 1.372))):
        ii = ctx.rows[name][:, 0].astype(int) - 1
        gp = np.diff(zz)
        sp_, sn_ = float(gp[ii - 1].mean() / gp.mean()), float(gp[ii + 1].mean() / gp.mean())
        report("V9 shield %s prev/next" % name, "%.4f/%.4f" % (sp_, sn_), "%.3f/%.3f" % tg, 1e-3, abs(sp_ - tg[0]) <= 1e-3 and abs(sn_ - tg[1]) <= 1e-3)
    equal("V9 tight pairs in the 1e12 window", int(len(ctx.rows["1e12"])), 38)
    equal("V9 tight pairs in the 1e21 window", int(len(ctx.rows["1e21"])), 32)
    tg_ = ctx.table("tab_gue")
    C9 = np.pi ** 2 / 9.0
    S0 = (0.05, 0.10, 0.15)
    for name, zz, key in (("100k", ctx.z1, "100k"), ("2M", ctx.z6, "2M"), ("8.4e9", ctx.zd, "LMFDB")):
        s = D.normalised_spacings(zz)
        rr_ = [float(np.mean(s < s0)) / (C9 * s0 ** 3) for s0 in S0]
        report("V9 GUE tail ratios %s (Table 7.2)" % name, "/".join("%.3f" % v for v in rr_), "/".join("%.3f" % v for v in tg_[key]["tail_ratios"]), 1e-9,
               all(abs(a - b) <= 1e-9 for a, b in zip(rr_, tg_[key]["tail_ratios"])))
        close("V9 lag-1 corr %s" % name, S.corr(s[:-1], s[1:]), tg_[key]["lag1"], 1e-9, "%.4f", "%.4f")
    report("V9 LMFDB tail ratios (0.887/0.981/0.957)", "/".join("%.3f" % v for v in tg_["LMFDB"]["tail_ratios"]), "0.887/0.981/0.957", 2e-3,
           all(abs(a - b) <= 2e-3 for a, b in zip(tg_["LMFDB"]["tail_ratios"], (0.887, 0.981, 0.957))))
    close("V9 LMFDB lag-1 (-0.332)", tg_["LMFDB"]["lag1"], -0.332, 1e-3, "%.4f", "%.3f")
