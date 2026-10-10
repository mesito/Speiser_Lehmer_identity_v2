# Smooth-cutoff scan (Section 5.4, Table 5.2)

Weights on the prime powers n = p^k: sharp 1{n <= X}; exp4 exp(-(n/X)^4) summed to n <= 4X; C^inf bump (1 for n <= X/2, 0 for n >= X). Exact in k; phases in 80-bit precision; primes to 4e7. Full primary ensemble (2736 pairs). Data: `logP_smooth_8p4e9.npy`, `smooth_cutoff_scan.csv`.

| X | a sharp | b sharp | R2 sharp | a exp4 | b exp4 | R2 exp4 | a bump | b bump | R2 bump |
|---|---|---|---|---|---|---|---|---|---|
| 1e2 | 1.9906 | 0.7460 | 0.9377 | 1.9927 | 0.7673 | 0.9468 | 1.9948 | 0.7578 | 0.9400 |
| 1e3 | 1.9959 | 0.7401 | 0.9415 | 1.9934 | 0.7523 | 0.9497 | 1.9928 | 0.7440 | 0.9448 |
| 1e4 | 1.9923 | 0.7814 | 0.9407 | 1.9973 | 0.7871 | 0.9473 | 1.9972 | 0.7781 | 0.9435 |
| 1e5 | 1.9645 | 0.8326 | 0.9434 | 1.9684 | 0.8402 | 0.9498 | 1.9724 | 0.8340 | 0.9469 |
| 1e6 | 1.9596 | 0.9024 | 0.9471 | 1.9567 | 0.9080 | 0.9519 | 1.9597 | 0.8995 | 0.9480 |
| 1e7 | 1.9371 | 0.9356 | 0.9534 | 1.9399 | 0.9433 | 0.9576 | 1.9387 | 0.9371 | 0.9540 |

Maximal shift of b between the sharp and a smooth cutoff: 0.0213.
