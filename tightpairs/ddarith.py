"""Minimal double-double arithmetic (numpy, vectorised) for the phases of the
Riemann-Siegel sum at large heights.

A double-double x = (hi, lo) represents hi + lo with |lo| <= ulp(hi)/2, about
106 bits.  Only the operations needed for  (t * log n) mod 2pi  are provided:
exact products by Dekker's splitting (no fma needed), exact sums by two_sum,
and the reduction modulo 2pi with 2pi carried to three doubles.
"""
import numpy as np

_SPLIT = 134217729.0          # 2^27 + 1
TWO_PI_3 = (6.283185307179586, 2.4492935982947064e-16, -5.989539619436679e-33)   # 2pi as three doubles


def two_sum(a, b):
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return s, e


def split(a):
    c = _SPLIT * a
    hi = c - (c - a)
    return hi, a - hi


def two_prod(a, b):
    p = a * b
    ah, al = split(a)
    bh, bl = split(b)
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e


def dd_mul(a_hi, a_lo, b_hi, b_lo):
    """(a_hi + a_lo) * (b_hi + b_lo) as a double-double."""
    p, e = two_prod(a_hi, b_hi)
    e = e + (a_hi * b_lo + a_lo * b_hi)
    return two_sum(p, e)


def mod_two_pi(x_hi, x_lo):
    """(x_hi + x_lo) mod 2pi for |x| < 2^53 * 2pi, as a double-double; accurate to ~1e-16 * (number of terms)."""
    q = np.floor(x_hi / TWO_PI_3[0])
    p, e = two_prod(q, TWO_PI_3[0])
    r = (x_hi - p) - e                              # x_hi - q*2pi_hi exactly (Sterbenz) minus the product error
    r = r + x_lo - q * TWO_PI_3[1] - q * TWO_PI_3[2]
    # bring into [0, 2pi)
    r = np.where(r < 0, r + TWO_PI_3[0], r)
    r = np.where(r >= TWO_PI_3[0], r - TWO_PI_3[0], r)
    return r
