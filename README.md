# Speiser_Lehmer_identity_v2 — verification suite, pipeline and data

Reproducibility package for

> M. Ismail, *Tight zero pairs of the Riemann zeta function at height 8.4·10⁹:
> the Speiser–Lehmer identity, the displacement law and the Euler–Hadamard
> decomposition of the floor*, 2026.

Every number of the paper — every table, every constant quoted in the text, the
arbitrary-precision reference computations of Theorem A and of the displacement
law, the certified tail bounds — is recomputed from the raw ordinate files by
`verify_all.py` (thirteen groups **V1–V13**, pass/fail criteria stated in the
code), and every table of measured or computed quantities (Table 3.1–Table 9.3,
Table B.1–Table C.2) is generated from the shipped arrays by
`make_tables.py`; the descriptive Table 2.1, Table 2.2, Table 8.1 and
Table A.1 are written by hand, the numbers of the first two being checked by V1.

## 1. Layout

```
tightpairs/        library
  constants.py     conventions (W = 250, s < 0.15, cutoffs X, control draw)
  data.py          Platt block decoding (exact 104-bit integers), double-double
                   and 80-bit ordinates, Odlyzko tables and windows (gamma - B),
                   tight pairs, controls, the rank subsample
  geometry.py      far field, first moment, S_on, S_end, r, C_mid, kappa, tails,
                   the Speiser–Lehmer identity, the transposed detector
  instrument.py    Z_rs2 (80-bit phases, C_0), Z_dd (double-double phases,
                   C_0..C_3), floors, mpmath references (extremum, Z'', curvature)
  ddarith.py       the double-double arithmetic behind Z_dd
  euler.py         primes, exact log|P_X| (all prime powers), smooth cutoffs,
                   Delta_0(X), the three-harmonic proxy L_X^(3), delta_X, lambda_rand
  kernel.py        the Gonek–Hughes–Keating kernel Ci on the line, S_Delta,
                   the exact kernel term of an off-line orbit
  stats.py         least squares with standard errors, partial correlations
  gue.py           sine-kernel quantities of Theorem E, exact spacing tail
                   (Fredholm determinant), GUE sampler
  tails.py         a_p, sigma_p, Lambda_1, hybrid Chernoff bound, Bahadur–Rao
  report.py        PASS/FAIL bookkeeping
verify/            v01_data ... v13_subsample: one module per group (Appendix A)
verify_all.py      runs the groups; --full for the complete recomputations
make_tables.py     tables/tab_*.tex (the LaTeX tables of the paper) and
                   tables/tab_*.json (the same numbers; targets of the suite)
make_figures.py    the five figures of the paper
make_reports.py    Markdown reports of the derived data (data/reports/)
pipeline/          the scripts that produced every array in data/ (Section 3 below)
data/              raw ordinate files and derived arrays (Section 4 below)
tables/            generated tables (shipped so that the suite runs as is)
```

## 2. Requirements and use

Python ≥ 3.10 with `numpy`, `scipy`, `mpmath` (and `matplotlib` for the
figures). No network access is required; every input ships in `data/`.

```
python3 verify_all.py            # V1–V13, default tier, about 2 minutes
python3 verify_all.py --full     # complete recomputations, about 20.5 minutes
python3 verify_all.py --only V2,V3
python3 make_tables.py           # regenerates tables/ from data/
python3 make_figures.py          # regenerates ../paper/figs/
python3 make_reports.py          # data/reports/*.md
```

The data root is `$RH_DATA` (default `./data`), the tables `$RH_TABLES`
(default `./tables`). The suite exits with status 1 if any check fails; the
shipped run logs are `runlog_default.txt` and `runlog_full.txt`.

Map paper → groups (Appendix A, Table A.1):

| group | paper | content |
|---|---|---|
| V1 | Section 2, Table 2.1, Table 2.2 | decoding and exact ordinates; the windows at 1e12 and 1e21; tight-pair counts; instrument errors |
| V2 | Theorem A(a), Corollary A.2, Corollary A.3, Proposition A.4, Observation 3.2, Observation 3.4, Table 3.1 | the identity on five ensembles; r, C_mid, kappa, h_flip; self-calibration |
| V3 | Theorem A(b),(c), Corollary A.1, Observation 3.3, Table 3.2 | floor law and midpoint curvature on the reference pairs |
| V4 | Proposition 3.5, Table C.1 | the sweep of the transposed detector |
| V5 | Section 4, Observation 4.2, Table B.2, Table C.2 | the displacement law; Lehmer's pair |
| V6 | Section 5.2–Section 5.3, Observation 5.1, Table 5.1, Table 5.2 | the Euler regression at two heights; smooth cutoffs |
| V7 | Observation 5.9, Observation 5.10, Table 5.6 | the decomposition of the floor; the loading |
| V8 | Section 6, Table 6.1, Table 6.2, Table 6.3 | K5; partial correlations; Euler deficit; controls; block averages |
| V9 | Section 7, Observation 7.3, Table 7.1, Table 7.2, Table 7.3 | far-field law (per-pair normalisation) and its finite-height fit; shield; spacing tails and lag-1 |
| V10 | Theorem E, Observation 7.2, Table 7.3 | the GUE constant: rho(x), the integrals, C(a), Monte Carlo; exact sine-kernel tail ratios |
| V11 | Section 8.5, Observation 8.2, Observation 8.3, Table 8.2, Theorem 9.3 | the depth certificate on the data and its cutoff dependence |
| V12 | Section 9, Lemma 9.1, Table 9.1, Table 9.2, Table 9.3 | Delta_0(X); constants; Chernoff table; Lipschitz; uniform criterion; Bahadur–Rao |
| V13 | Section 5.1, Section 5.4, Table 5.3, Table 5.4, Table 5.5, Table B.1 | rank subsample; gap-only regression; variance ladder; errors in variables; Table B.1 |

## 3. `pipeline/` — provenance of the derived data

Scripts are run from the repository root; each writes into `data/` and states
its runtime in its docstring.

| script | produces | from |
|---|---|---|
| `decode_platt.py` | `lmfdb_zeros_exact.npz`, `lmfdb_zeros_hilo.npy` | `lmfdb_top.bin` (checks (i)–(iv) of Section 2.2); the same decode runs automatically on first use |
| `pack_odlyzko.py` | `zeros1.npz`, `zeros6.npz` | Odlyzko's text tables (lossless) |
| `compute_floors.py` | `floors_all_{8p4e9,2M,100k}.npy`, `floors_ctrl_8p4e9.npy`, `ctrl_idx_8p4e9.npy`, `lehmer_rows_{100k,2M,8p4e9}.npy` | ordinates (Z_dd floors, 33-point grid + parabola) |
| `compute_windows.py` | `lehmer_rows_{1e12,1e21}.npy`, `floors_1e12.npy` | `zeros3.gz`, `zeros4.gz` (offsets gamma − B; floors with Z_dd at B + offset) |
| `compute_logP.py` | `logP_pairs_*.npy`, `zsum_pairs_*.npy`, `logP_ctrl_8p4e9.npy` | ordinates, primes ≤ 10⁷ |
| `compute_LX3.py` | `LX3_pairs_{8p4e9,2M}.npy`, `LX3_ctrl_8p4e9.npy` | ordinates, primes ≤ 10⁷ (the three-harmonic proxy of Section 5.4.2) |
| `compute_lehmer_pair.py` | `lehmer_pair_7005.npz` | `zeros1` |
| `reference_pairs.py` | `reference_{8p4e9,8p4e9_sub,100k,2M,lehmer}.csv` | ordinates, mpmath at 30 digits (Table 3.2, Table B.1, Table B.2, Table C.2) |
| `detector_sweep.py` | `detector_sweep_8p4e9.csv` | ordinates (Section 3.6) |
| `smooth_cutoff_scan.py` | `logP_smooth_8p4e9.npy`, `smooth_cutoff_scan.csv` | ordinates, primes ≤ 4·10⁷ (Table 5.2) |
| `scan_allgaps.py 0|1|merge` | `moment_floors_half{0,1}.npy`, `moment_allgaps.npy` | ordinates (fast instrument; Section 6.5) |
| `compute_logP_allgaps.py` | `logP_allgaps_1e6.npy` | `moment_allgaps.npy` |
| `gue_mc.py` | `gue_mc_results.npz`, `gue_mc_summary.npz` | random matrices (Observation 7.2, Table 7.3) |

**The rank subsample.** The 300-pair subsample used as a robustness check
(Section 5.4.1) is `tightpairs.data.rank_subsample(g, 300)`: the pairs at the
ranks round(k(N−1)/299), k = 0..299, of the stable ascending order of the gaps g
of the N = 2,736 tight pairs (column 2 of `floors_all_8p4e9.npy`, i.e. the gaps
formed from the 80-bit ordinates). Its first entry is the tightest pair, and its
first fifteen pairs are the pairs of Table B.1. Because the subsample is
defined through the gaps, its membership depends on how the gaps are formed: with
the ordinates rounded to doubles, 280 of the 300 pairs are the same.

**The windows.** Odlyzko's `zeros3` and `zeros4` list, for the zeros numbered
10¹²+1 … 10¹²+10⁴ and 10²¹+1 … 10²¹+10⁴, the offsets gamma − B from the integer B
given in the header (B = 267653395647 and B = 144176897509546973000); the stated
accuracies are 10⁻⁸ and, probably, 10⁻⁶. `tightpairs.data.load_window` returns
(B, offsets); all geometry is formed from the offsets, the density at B + t.

## 4. Data

| file | content |
|---|---|
| `lmfdb_top.bin` | the first 10.07 MB of Platt's `zeros_8436146000.dat` (LMFDB); 110 complete blocks — the raw primary data |
| `lmfdb_zeros_exact.npz` | the exact 104-bit cumulative integers of the 772,719 ordinates; **built from `lmfdb_top.bin` on first use** (25 s, bit-identical on every platform) |
| `lmfdb_zeros_hilo.npy` | the ordinates as double-doubles (hi, lo), exact to 3·10⁻²³; built together with the file above |
| `zeros1.npz`, `zeros6.npz` | Odlyzko's tables of the first 10⁵ and 2,001,052 zeros in a lossless packed form (10⁹·γ as integers; `pipeline/pack_odlyzko.py`); `load_odlyzko` returns bit for bit what `numpy.loadtxt` returns from the original text files, which it also accepts (`zeros1.gz`, `zeros6.gz`) |
| `zeros3.gz`, `zeros4.gz` | Odlyzko's windows at 10¹² and 10²¹, gzip of the original text tables |
| `floors_all_8p4e9.npy`, `floors_all_2M.npy`, `floors_all_100k.npy`, `floors_1e12.npy` | (idx 1-based, t0, g, s, far, floor) for every tight pair (in the 10¹² window t0 is the offset t0 − B); idx is the position of γ_n in the ensemble, so that in the primary ensemble γ_n is the zero number 26,877,139,498 + idx |
| `lehmer_rows_*.npy` | (idx, t0, g, s, S_on, far, C, C_mid, h_thr, r) per ensemble (windows: t0 − B) |
| `logP_pairs_*.npy`, `zsum_pairs_*.npy` | exact log\|P_X\| (6 cutoffs) and S_Delta (Delta = 100, 1000) at the midpoints |
| `LX3_pairs_*.npy`, `LX3_ctrl_8p4e9.npy` | the three-harmonic proxy L_X^(3) at the same points |
| `logP_ctrl_8p4e9.npy`, `floors_ctrl_8p4e9.npy`, `ctrl_idx_8p4e9.npy` | the 300 control mid-gaps |
| `logP_smooth_8p4e9.npy`, `smooth_cutoff_scan.csv` | the smooth-cutoff scan |
| `reference_*.csv` | the 30-digit references |
| `detector_sweep_8p4e9.csv` | the sweep of Section 3.6 |
| `moment_allgaps.npy`, `logP_allgaps_1e6.npy` | the all-gaps scan of Section 6.5 (the merge of the two scan halves written by `scan_allgaps.py`; the halves themselves are not shipped) |
| `lehmer_pair_7005.npz` | prime sums and kernel sums at Lehmer's pair |
| `gue_mc_results.npz`, `gue_mc_summary.npz` | the GUE Monte Carlo |
| `reports/*.md` | human-readable summaries (`make_reports.py`) |

Conventions: tight pair s = g·log(t₀/2π)/2π < 0.15; floor = max|Z| over the
open gap (33-point grid + parabolic refinement, reference instrument Z_dd);
far, λ_far and S_end over W = 250 ordinates on each side, clipped at the ends of
an ensemble; log|P_X| in the Gonek–Hughes–Keating convention (all prime powers
p^k ≤ X, exact in k); phases reduced in 80-bit precision with a 40-digit π;
controls: 300 mid-gaps with normalised spacing in [0.95, 1.05] (density at the
mean height of the primary block, double-precision spacings),
`numpy.random.default_rng(11)`; spacing statistics (Table 7.3) normalised with
the density at the midpoint of each spacing, as in the tight-pair definition; the
far field of Table 7.1 normalised pair by pair with dens(t₀)²; the shield of
Table 7.2 in units of the local mean gap 1/dens(t₀).

`MANIFEST.sha256` lists the SHA-256 of every file in the package in its shipped
state (`sha256sum -c MANIFEST.sha256`); the two ordinate arrays built on first
use are not in it. Re-running a pipeline script rewrites its outputs bit for bit
on the same platform; the Monte Carlo (`gue_mc.py`, two worker processes) is
seeded and reproduces its summary up to the order of floating-point summation.
