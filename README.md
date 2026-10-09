# Speiser_Lehmer_identity_v2 — verification suite, pipeline and data

Reproducibility package for

> M. Ismail, *Tight zero pairs of the Riemann zeta function at height 8.4·10⁹:
> the Speiser–Lehmer identity, the displacement law and the Euler–Hadamard
> decomposition of the floor*, 2026.

Every number of the paper — every table, every constant quoted in the text, the
reference computations of Theorem A and of the displacement law, the certified
tail bounds — is recomputed from the raw ordinate files by `verify_all.py`
(twelve groups **V1–V12**, pass/fail criteria stated in the code), and every
table of the paper is generated from the shipped arrays by `make_tables.py`.

## 1. Layout

```
tightpairs/        library
  constants.py     conventions (W = 250, s < 0.15, cutoffs X, control draw)
  data.py          Platt block decoding (exact 104-bit integers), double-double
                   and 80-bit ordinates, Odlyzko tables, tight pairs, controls
  geometry.py      far field, first moment, S_on, S_end, r, C_mid, kappa, tails,
                   the Speiser–Lehmer identity, the transposed detector
  instrument.py    Z_rs2 (80-bit phases, C_0), Z_dd (double-double phases,
                   C_0..C_3; realised error 8e-14 at t ~ 8.4e9), floors,
                   mpmath references (extremum, Z'', curvature)
  ddarith.py       the double-double arithmetic behind Z_dd
  euler.py         primes, exact log|P_X| (all prime powers), smooth cutoffs,
                   Delta_0(X), the three-harmonic proxy
  kernel.py        the Gonek–Hughes–Keating kernel Ci on the line, S_Delta
  stats.py         least squares with standard errors, partial correlations
  gue.py           sine-kernel quantities of Theorem 7.2, GUE sampler
  tails.py         a_p, sigma_p, Lambda_1, hybrid Chernoff bound, Bahadur–Rao
  report.py        PASS/FAIL bookkeeping
verify/            v01_data ... v12_tails: one module per group (Appendix A)
verify_all.py      runs the groups; --full for the complete recomputations
make_tables.py     tables/tab_*.tex (the LaTeX tables of the paper) and
                   tables/tab_*.json (the same numbers; targets of the suite)
make_figures.py    the five figures of the paper
make_reports.py    Markdown reports of the derived data (data/reports/)
pipeline/          the scripts that produced every array in data/ (Section 3)
data/              raw ordinate files and derived arrays (Section 4)
tables/            generated tables (shipped so that the suite runs as is)
```

## 2. Requirements and use

Python ≥ 3.10 with `numpy`, `scipy`, `mpmath` (and `matplotlib` for the
figures). No network access is required; every input ships in `data/`.

```
python3 verify_all.py            # V1–V12, default tier, about 3 minutes
python3 verify_all.py --full     # complete recomputations, about 20 minutes
python3 verify_all.py --only V2,V3
python3 make_tables.py           # regenerates tables/ from data/
python3 make_figures.py          # regenerates ../paper/figs/
python3 make_reports.py          # data/reports/*.md
```

The data root is `$RH_DATA` (default `./data`), the tables `$RH_TABLES`
(default `./tables`). The suite exits with status 1 if any check fails; the
shipped run logs are `runlog_default.txt` and `runlog_full.txt`.

Map paper → groups (Appendix A, Table A.1): V1 data and instruments (§2);
V2 the identity and self-calibration (Thm A(a), Obs 3.9, 3.11); V3 the floor law
and curvature (Thm A(b),(c), Table 3.2, Cor 3.5); V4 the detector sweep
(Prop 3.12); V5 the displacement law (§4, Tables B.1–B.2); V6 Delta_0, the Euler
regression, smooth cutoffs (§5.2, Tables 5.1–5.2); V7 the decomposition and the
loading (§5.4, Table 5.3); V8 mediation (§6, Tables 6.1–6.2); V9 far field,
shield, GUE diagnostics (§7, Tables 7.1, 7.3); V10 Theorem 7.2 and the Monte
Carlo (§7.2); V11 the certificate on the data (§8); V12 tail bounds (§9,
Table 9.1).

## 3. `pipeline/` — provenance of the derived data

Scripts are run from the repository root; each writes into `data/` and states
its runtime in its docstring.

| script | produces | from |
|---|---|---|
| `decode_platt.py` | `lmfdb_zeros_exact.npz`, `lmfdb_zeros_hilo.npy` | `lmfdb_top.bin` (checks (i)–(iv) of §2.2); the same decode runs automatically on first use |
| `pack_odlyzko.py` | `zeros1.npz`, `zeros6.npz` | Odlyzko's text tables (lossless) |
| `compute_floors.py` | `floors_all_8p4e9.npy`, `floors_all_2M.npy`, `floors_ctrl_8p4e9.npy`, `ctrl_idx_8p4e9.npy`, `lehmer_rows_{100k,2M,8p4e9}.npy` | ordinates (Z_dd floors, 33-point grid + parabola) |
| `compute_logP.py` | `logP_pairs_*.npy`, `zsum_pairs_*.npy`, `logP_ctrl_8p4e9.npy` | ordinates, primes ≤ 10⁷ |
| `compute_lehmer_pair.py` | `lehmer_pair_7005.npz` | `zeros1` |
| `reference_pairs.py` | `reference_{8p4e9,100k,2M,lehmer}.csv` | ordinates, mpmath at 30 digits (Tables 3.2, B.1, B.2) |
| `detector_sweep.py` | `detector_sweep_8p4e9.csv` | ordinates (§3.6) |
| `smooth_cutoff_scan.py` | `logP_smooth_8p4e9.npy`, `smooth_cutoff_scan.csv` | ordinates, primes ≤ 4·10⁷ (Table 5.2) |
| `scan_allgaps.py 0|1|merge` | `moment_floors_half{0,1}.npy`, `moment_allgaps.npy` | ordinates (fast instrument; §6.4) |
| `compute_logP_allgaps.py` | `logP_allgaps_1e6.npy` | `moment_allgaps.npy` |
| `gue_mc.py` | `gue_mc_results.npz`, `gue_mc_summary.npz` | random matrices (§7.2, Table 7.2) |

The windows at 10¹² and 10²¹ (`lehmer_rows_1e12.npy`, `lehmer_rows_1e21.npy`)
were built with the same geometry code from Odlyzko's `zeros3` and `zeros4`
tables, which are not redistributed here; the arrays are shipped and are used
by V2 and V9 only.

## 4. Data

| file | content |
|---|---|
| `lmfdb_top.bin` | the first 10.07 MB of Platt's `zeros_8436146000.dat` (LMFDB); 110 complete blocks — the raw primary data |
| `lmfdb_zeros_exact.npz` | the exact 104-bit cumulative integers of the 772,719 ordinates (§2.2); **built from `lmfdb_top.bin` on first use** (25 s, bit-identical on every platform) |
| `lmfdb_zeros_hilo.npy` | the ordinates as double-doubles (hi, lo), exact to 3·10⁻²³; built together with the file above |
| `zeros1.npz`, `zeros6.npz` | Odlyzko's tables of the first 10⁵ and 2,001,052 zeros in a lossless packed form (10⁹·γ as integers; `pipeline/pack_odlyzko.py`); `load_odlyzko` returns bit for bit what `numpy.loadtxt` returns from the original text files, which it also accepts (`zeros1.gz`, `zeros6.gz`) |
| `floors_all_8p4e9.npy`, `floors_all_2M.npy` | (idx 1-based, t0, g, s, far, floor) for every tight pair |
| `lehmer_rows_*.npy` | (idx, t0, g, s, S_on, far, C, C_mid, h_thr, r) per ensemble |
| `logP_pairs_*.npy`, `zsum_pairs_*.npy` | exact log\|P_X\| (6 cutoffs) and S_Delta (Delta = 100, 1000) at the midpoints |
| `logP_ctrl_8p4e9.npy`, `floors_ctrl_8p4e9.npy`, `ctrl_idx_8p4e9.npy` | the 300 control mid-gaps |
| `logP_smooth_8p4e9.npy`, `smooth_cutoff_scan.csv` | the smooth-cutoff scan |
| `reference_*.csv` | the 30-digit references (Tables 3.2, B.1, B.2) |
| `detector_sweep_8p4e9.csv` | the sweep of §3.6 |
| `moment_allgaps.npy`, `logP_allgaps_1e6.npy` | the all-gaps scan of §6.4 (the merge of the two scan halves written by `scan_allgaps.py`; the halves themselves are not shipped) |
| `lehmer_pair_7005.npz` | prime sums and kernel sums at Lehmer's pair |
| `gue_mc_results.npz`, `gue_mc_summary.npz` | the GUE Monte Carlo |
| `reports/*.md` | human-readable summaries (make_reports.py) |

Conventions: tight pair s = g·log(t₀/2π)/2π < 0.15; floor = max|Z| over the
open gap (33-point grid + parabolic refinement, reference instrument Z_dd);
far, λ_far and S_end over W = 250 ordinates on each side; log|P_X| in the
Gonek–Hughes–Keating convention (all prime powers p^k ≤ X, exact in k); phases
reduced in 80-bit precision with a 40-digit π; controls: 300 mid-gaps with
normalised spacing in [0.95, 1.05], `numpy.random.default_rng(11)`.

`MANIFEST.sha256` lists the SHA-256 of every file in the package in its shipped
state (`sha256sum -c MANIFEST.sha256`); the two ordinate arrays built on first
use are not in it. Re-running a pipeline script rewrites its outputs bit for bit
on the same platform except for the Monte-Carlo and random-subset checks, which
are seeded.
