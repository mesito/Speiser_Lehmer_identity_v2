"""V11 -- the depth certificate on the data (Section 8.5) and its cutoff dependence (Theorem 9.3)."""
import numpy as np

from tightpairs import kernel as K
from tightpairs.constants import XS, GAMMA_E
from tightpairs.report import close, report, below, equal, section

# median and 95th percentile of h0max/eps and median of eps log X (Observation 8.3), by X
TGT_D = {1e2: (2.03, 3.08, 0.082), 1e3: (1.60, 2.47, 0.123), 1e4: (1.30, 2.00, 0.164), 1e5: (1.09, 1.77, 0.205), 1e6: (0.94, 1.57, 0.246), 1e7: (0.81, 1.42, 0.287)}
# Observation 8.2: median of the demand at h = g/2, number positive at g/2, number positive at h = g
TGT_DEM = {1e2: (-2.26, 0, 1241), 1e3: (None, 0, None), 1e4: (None, 83, None), 1e5: (None, 794, None), 1e6: (None, 1583, None), 1e7: (0.44, 2054, 2712)}


def run(ctx):
    section("V11: the depth certificate on the data (Section 8.5, Theorem 9.3)")
    orbit_local_term = K.orbit_local_term
    f8, lp8, zs8 = ctx.f8, ctx.lp8, ctx.zs8
    g8, lf8, far8 = f8[:, 2], np.log(f8[:, 5]), f8[:, 4]
    eps = g8 / 2
    H = np.sqrt(2 / far8)
    td = ctx.table("tab_demand")
    # geometric part of the demand at h = H (X-independent)
    geo = np.log((eps ** 2 + 2 * H ** 2) ** 2 / eps ** 2)
    tg = td["geometric"]
    report("V11 geometric part at H: n>0 / min / median / max", "%d/%.3f/%.3f/%.3f" % ((geo > 0).sum(), geo[geo > 0].min() if (geo > 0).any() else 0, np.median(geo), geo.max()),
           "%d/-/%.3f/%.3f" % (tg["n_pos"], tg["median"], tg["max"]), 1e-9, (geo > 0).sum() == tg["n_pos"] and abs(np.median(geo) - tg["median"]) <= 1e-9)
    equal("V11 geometric part positive <=> far < 4/(eps(1-eps))", bool(np.array_equal(geo > 0, far8 < 4 / (eps * (1 - eps)))), True)
    zmax_e, zmax_g = 0.0, 0.0
    for j, X in enumerate(XS):
        Sfar = zs8[1, :, j] - K.pair_term(g8, X)
        s_minus = float(np.percentile(Sfar, 5))
        obs = lf8 - lp8[:, j]
        dE = orbit_local_term(eps, eps, X) + s_minus - obs
        dG = orbit_local_term(eps, g8, X) + s_minus - obs
        t = td[str(int(X))]
        ok = abs(s_minus - t["s_minus"]) <= 1e-12 and abs(np.median(dE) - t["eps"]["median"]) <= 1e-9 and abs(dE.min() - t["eps"]["min"]) <= 1e-9 \
            and int((dE > 0).sum()) == t["eps"]["n_pos"] and abs(np.median(dG) - t["g"]["median"]) <= 1e-9 and int((dG > 0).sum()) == t["g"]["n_pos"]
        report("V11 demand X=1e%d s_-/med,min,#>0 (h=g/2)/med,#>0 (h=g) (Table 8.2)" % int(np.log10(X)),
               "%.3f/%.3f,%.3f,%d/%.3f,%d" % (s_minus, np.median(dE), dE.min(), (dE > 0).sum(), np.median(dG), (dG > 0).sum()),
               "%.3f/%.3f,%.3f,%d/%.3f,%d" % (t["s_minus"], t["eps"]["median"], t["eps"]["min"], t["eps"]["n_pos"], t["g"]["median"], t["g"]["n_pos"]), 1e-9, ok, "%s", "%s")
        zH = np.sqrt(eps ** 2 + 4 / far8) * np.log(X)
        equal("V11 #{z_H <= 1} X=1e%d (Table 8.2)" % int(np.log10(X)), int((zH <= 1).sum()), t["n_zH_le_1"])
        zmax_e = max(zmax_e, float(np.max(np.sqrt(3) * eps * np.log(X))))
        zmax_g = max(zmax_g, float(np.max(3 * eps * np.log(X))))
        md, npos, npos_g = TGT_DEM[X]
        if md is not None:
            close("V11 median demand at h=g/2, X=1e%d (%+.2f)" % (int(np.log10(X)), md), np.median(dE), md, 5e-3, "%.3f", "%+.2f")
        equal("V11 positive demands at h=g/2, X=1e%d (text)" % int(np.log10(X)), int((dE > 0).sum()), npos)
        if npos_g is not None:
            equal("V11 positive demands at h=g, X=1e%d (text)" % int(np.log10(X)), int((dG > 0).sum()), npos_g)
        if X == 1e2:
            equal("V11 pairs with z_H <= 1 at X=1e2 (21)", int((zH <= 1).sum()), 21)
        if X >= 1e5:
            equal("V11 pairs with z_H <= 1 at X=1e%d (0)" % int(np.log10(X)), int((zH <= 1).sum()), 0)
        close("V11 cutoff term 2 log(e^gamma log X) X=1e%d" % int(np.log10(X)), 2 * (GAMMA_E + np.log(np.log(X))), t["scale"], 1e-12, "%.4f", "%.4f")
        # the exact orbit term against its leading form for small z
        z = np.sqrt(eps ** 2 + 2 * eps ** 2) * np.log(X)
        lead = 4 * GAMMA_E + 2 * np.log((eps ** 2 + 2 * eps ** 2) * np.log(X) ** 2)
        below("V11 |L_orb - leading| <= z^2 e^z at h=g/2, X=1e%d" % int(np.log10(X)), float(np.max(np.abs(orbit_local_term(eps, eps, X) - lead) - z ** 2 * np.exp(z))), 0.0)
        zg = 3 * eps * np.log(X)
        leadg = 4 * GAMMA_E + 2 * np.log((9 * eps ** 2) * np.log(X) ** 2)
        below("V11 |L_orb - leading| <= z^2 e^z at h=g, X=1e%d" % int(np.log10(X)), float(np.max(np.abs(orbit_local_term(eps, g8, X) - leadg) - zg ** 2 * np.exp(zg))), 0.0)
        # Observation: the per-pair depth bound (s_- the 5th percentile of S_far)
        Dq = lf8 - lp8[:, j] - 2 * np.log(eps * np.log(X)) - 2 * GAMMA_E
        ratio = 0.5 * (np.exp((Dq - s_minus) / 2 - GAMMA_E) / (eps * np.log(X)) - 1)
        med, p95 = np.sqrt(np.median(ratio)), np.sqrt(np.percentile(ratio, 95))
        tgD = TGT_D[X]
        report("V11 h0max/eps X=1e%d median/95%%/median eps logX" % int(np.log10(X)), "%.2f/%.2f/%.3f" % (med, p95, np.median(eps * np.log(X))),
               "%.2f/%.2f/%.3f" % tgD, 5e-3, abs(med - tgD[0]) <= 5e-3 and abs(p95 - tgD[1]) <= 5e-3 and abs(np.median(eps * np.log(X)) - tgD[2]) <= 5e-4)
    close("V11 max z at h=g/2 over pairs and cutoffs (0.63)", zmax_e, 0.63, 5e-3, "%.4f", "%.2f")
    close("V11 max z at h=g over pairs and cutoffs (1.08)", zmax_g, 1.08, 5e-3, "%.4f", "%.2f")
    gneg = np.sort(geo[geo <= 0])
    report("V11 geometric part at H: 2734 positive, range 0.857..8.855, median 4.462, negatives -1.252/-0.372",
           "%d, %.3f..%.3f, %.3f, %s" % ((geo > 0).sum(), geo[geo > 0].min(), geo.max(), np.median(geo), "/".join("%.3f" % v for v in gneg)),
           "2734, 0.857..8.855, 4.462, -1.252/-0.372", 5e-4,
           (geo > 0).sum() == 2734 and abs(geo[geo > 0].min() - 0.857) <= 5e-4 and abs(geo.max() - 8.855) <= 5e-4 and abs(np.median(geo) - 4.462) <= 5e-4
           and len(gneg) == 2 and abs(gneg[0] + 1.252) <= 5e-4 and abs(gneg[1] + 0.372) <= 5e-4, "%s", "%s")
    ineg = np.flatnonzero(geo <= 0)
    # zero numbers: the array index k of the primary ensemble is the zero number N_t0 + 1 + k
    n0 = int(ctx.exact["block_n0"][0])
    members = [set((n0 + 1 + i, n0 + 2 + i)) for i in (f8[ineg, 0].astype(int) - 1)]
    shared = set.intersection(*members) if len(members) == 2 else set()
    report("V11 the two negative pairs have the two largest far fields and share gamma_26877795164", "%s" % sorted(shared), "[26877795164]", "-",
           set(ineg) == set(np.argsort(far8)[-2:]) and shared == {26877795164}, "%s", "%s")
    close("V11 2 log(e^gamma log 1e2) (4.21)", 2 * (GAMMA_E + np.log(np.log(1e2))), 4.21, 5e-3, "%.4f", "%.2f")
    close("V11 2 log(e^gamma log 1e7) (6.71)", 2 * (GAMMA_E + np.log(np.log(1e7))), 6.71, 5e-3, "%.4f", "%.2f")
    close("V11 2 log(log 1e7 / log 1e2) (2.51)", 2 * np.log(np.log(1e7) / np.log(1e2)), 2.51, 5e-3, "%.4f", "%.2f")
    S7 = zs8[1, :, 5] - K.pair_term(g8, 1e7)
    S2 = zs8[1, :, 0] - K.pair_term(g8, 1e2)
    close("V11 median(S_far(1e7) - S_far(1e2)) (-0.58)", np.median(S7 - S2), -0.58, 5e-3, "%.3f", "%.2f")
    close("V11 sd(S_far(1e7) - S_far(1e2)) (0.41)", (S7 - S2).std(), 0.41, 5e-3, "%.3f", "%.2f")
    close("V11 inflation e^(6.9/4) (5.6)", np.exp(6.9 / 4), 5.6, 0.05, "%.2f", "%.1f")
    close("V11 mean log|P_X| at pairs, X=1e7 (-2.54)", lp8[:, 5].mean(), -2.54, 0.005, "%.3f", "%.2f")
    close("V11 u + mean log|P_X| at X=1e7 (6.9)", 9.4 + lp8[:, 5].mean(), 6.9, 0.05, "%.3f", "%.1f")
    close("V11 median bound at X=1e7 inflated (4.5 half-gaps)", TGT_D[1e7][0] * np.exp(6.9 / 4), 4.5, 0.05, "%.2f", "%.1f")
    tdf = ctx.table("tab_deficit")
    dlt = tdf["10000000"]["mean_pairs"] - tdf["10000000"]["mean_ctrl"]
    close("V11 Corollary 8.5: exp(-Delta/4) at X=1e7", np.exp(-dlt / 4), np.exp(2.433 / 4), 5e-3, "%.3f", "%.3f")
    close("V11 log X / log T0 at X=1e7 (0.705)", np.log(1e7) / np.log(8.436e9), 0.705, 5e-4, "%.4f", "%.3f")
