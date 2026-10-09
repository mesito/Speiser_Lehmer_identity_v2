"""tightpairs -- library behind the verification suite and the data pipeline of

    M. Ismail, "Tight zero pairs of the Riemann zeta function at height 8.4e9:
    the Speiser-Lehmer identity, the displacement law and the Euler-Hadamard
    decomposition of the floor" (2026).

Modules
    constants   numerical constants and conventions (W, s < 0.15, cutoffs X)
    data        ordinate files: Platt block decoding, exact/80-bit/double
                loaders, Odlyzko tables, tight-pair selection, control gaps
    geometry    pair geometry: far field, first moment, S_on, S_end, r, C_mid,
                kappa, tail corrections, Speiser-Lehmer identity
    instrument  Riemann-Siegel instrument Z_rs2, floor over a gap, mpmath
                references (extremum, Z'', curvature)
    euler       primes, truncated Euler product log|P_X| (exact in k),
                smooth cutoffs, Delta_0(X), three-harmonic proxy
    kernel      the Gonek-Hughes-Keating kernel Ci on the line, S_Delta sums
    stats       least squares with standard errors, partial correlations
    gue         sine-kernel quantities: rho(x), C(a), the two integrals,
                Dumitriu-Edelman sampler
    tails       prime constants a_p, sigma_p, Lambda_1, hybrid Chernoff bound,
                Bahadur-Rao prefactor, importance sampling
    report      PASS/FAIL bookkeeping shared by the verification groups
"""
from . import constants  # noqa: F401

__version__ = "4.0"
