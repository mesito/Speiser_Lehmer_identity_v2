"""The Gonek-Hughes-Keating kernel on the critical line (Lemma 5.7 of the paper):

    log|Z_X(1/2+it)| = sum_gamma Ci_u(|t - gamma| log X),

Ci the cosine integral.  S_Delta(t0; X) sums the unsmoothed Ci over all
ordinates within Delta of t0 (the pair included); the pair term alone is
2 Ci((g/2) log X); S_far = S_Delta - pair term.
"""
import numpy as np
from scipy.special import sici

from .constants import XS, GAMMA_E


def Ci(y):
    return sici(np.asarray(y, dtype=float))[1]


def zsum(zeros, zeros_ld, t0_ld, delta, xs=XS):
    """array (len(t0), len(xs)) of S_Delta(t0; X).  `zeros` double (for searching),
    `zeros_ld` 80-bit (for the distances), `t0_ld` 80-bit midpoints."""
    t0_ld = np.atleast_1d(np.asarray(t0_ld, dtype=np.longdouble))
    out = np.empty((len(t0_ld), len(xs)))
    for j, tt in enumerate(t0_ld):
        lo = np.searchsorted(zeros, float(tt) - delta)
        hi = np.searchsorted(zeros, float(tt) + delta)
        d = np.abs(np.asarray(zeros_ld[lo:hi] - tt, dtype=float))
        for jx, X in enumerate(xs):
            out[j, jx] = np.sum(Ci(d * np.log(X)))
    return out


def pair_term(g, X):
    """2 Ci((g/2) log X): the kernel term of the pair itself at its midpoint."""
    return 2 * Ci((np.asarray(g, dtype=float) / 2) * np.log(X))


def pair_term_leading(g, X):
    """2 gamma_E + 2 log((g/2) log X): the leading term of pair_term."""
    return 2 * GAMMA_E + 2 * np.log((np.asarray(g, dtype=float) / 2) * np.log(X))


def orbit_local_term(eps_app, h, X):
    """sum over the four zeros 1/2 +- h + i(t0 +- sqrt(eps_app^2 + h^2)) of -Re E_1((s - rho) log X) at s = 1/2 + i t0:
    the exact kernel term of the orbit configuration of Section 4.4, the term L_orb of (8.1) (without the u-averaging)."""
    from scipy.special import exp1
    eps_app, h = np.asarray(eps_app, float), np.asarray(h, float)
    et = np.sqrt(eps_app ** 2 + h ** 2)
    L = np.log(X)
    tot = 0.0
    for sg in (1, -1):
        for tg in (1, -1):
            z = (-sg * h - 1j * tg * et) * L
            tot = tot - np.real(exp1(z))
    return tot
