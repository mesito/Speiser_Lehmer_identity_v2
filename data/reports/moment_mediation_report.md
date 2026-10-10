# Block-averaging (moment-level mediation) scan (Section 6.5, Table 6.3)

Every fourth gap of the primary ensemble (193179 gaps), floor = max|Z| over 12 interior samples (fast instrument), log|P_X| at X = 1e6. Blocks of B consecutive gaps sorted by t; OLS of block means. Data: `moment_allgaps.npy`, `logP_allgaps_1e6.npy`.

| B | n blocks | a | b | se(b) | R2 | partial corr | sd of block-mean log P | corr(log F, log P) | corr(log P, log g) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 193179 | 0.644 | 0.693 | 0.001 | 0.9216 | 0.841 | 1.407 | 0.950 | 0.818 |
| 16 | 12073 | 1.581 | 0.215 | 0.004 | 0.9543 | 0.440 | 0.277 | 0.921 | 0.901 |
| 256 | 754 | 1.762 | 0.149 | 0.013 | 0.9817 | 0.386 | 0.087 | 0.951 | 0.943 |
| 4096 | 47 | 1.861 | 0.075 | 0.051 | 0.9762 | 0.215 | 0.015 | 0.919 | 0.917 |

Tight subset (s < 0.15) at B = 1: n = 680, a = 1.914, b = 0.886, R2 = 0.945.
