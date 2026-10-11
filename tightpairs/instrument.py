"""The Riemann-Siegel instrument and the arbitrary-precision references.

Z_rs2(t) = 2 sum_{n<=N} n^{-1/2} cos(theta(t) - t log n)
           + (-1)^{N-1} (t/2pi)^{-1/4} cos(2pi(p^2 - p - 1/16)) / cos(2pi p),

tau = sqrt(t/2pi), N = floor(tau), p = tau - N: the main sum with the first
correction term.  Every phase theta(t) - t log n is formed and reduced modulo
2pi in 80-bit precision with the 40-digit pi of constants.py; the cosines are
then taken in double precision.  Realised error against mpmath.siegelz at
t ~ 8.4e9: at most 3.6e-8, median 8e-9 (verify group V1).  With the
double-precision 2pi in the reduction the error is up to 4e-6 (Section 2.3 of
the paper).

Z_dd(t) keeps the corrections C_0..C_3 and forms every phase in double-double
arithmetic (ddarith.py); realised error at most 8.4e-14 on random mid-gaps and
9.1e-14 at the floor extrema of Table 3.2 at t ~ 8.4e9, 1.3e-13 at t ~ 2.7e11.

The floor of a gap is max|Z| on a 33-point grid of the open gap, refined by a
parabola through the three largest neighbouring samples (floor_max).  The
reference computations use mpmath (extremum_mp, floor_identity_mp).
"""
import numpy as np

from .constants import PI_LD, TWO_PI_LD, GRID_FLOOR

_LNN = np.log(np.arange(1, 40001).astype(np.longdouble))
_SQN = np.sqrt(np.arange(1.0, 40001.0))


def _theta_terms(tld):
    tau = np.sqrt(tld / TWO_PI_LD)
    N = int(np.floor(tau))
    p = float(tau - N)
    thcorr = 1 / (48 * tld) + 7 / (5760 * tld ** 3) + 31 / (80640 * tld ** 5)
    return tau, N, p, thcorr


def Z_rs2(t):
    """Z(t) for a scalar t (float or long double)."""
    tld = np.longdouble(t)
    tau, N, p, thcorr = _theta_terms(tld)
    ph = (tld * (np.log(tau) - _LNN[:N]) - tld / 2 - PI_LD / 8 + thcorr) % TWO_PI_LD
    main = 2.0 * float(np.sum(np.cos(np.asarray(ph, dtype=float)) / _SQN[:N]))
    psi = np.cos(2 * np.pi * (p * p - p - 1.0 / 16.0)) / np.cos(2 * np.pi * p)
    return main + (-1) ** (N + 1) * float(tau) ** -0.5 * psi


def Z_rs2_batch(ts):
    """Z at an array of long-double points that share the same N (points of one gap)."""
    ts = np.asarray(ts, dtype=np.longdouble)
    tau = np.sqrt(ts / TWO_PI_LD)
    N = int(np.floor(tau[0]))
    assert int(np.floor(tau[-1])) == N, "points must share N; use Z_rs2 pointwise"
    p = np.asarray(tau - N, dtype=float)
    thcorr = 1 / (48 * ts) + 7 / (5760 * ts ** 3) + 31 / (80640 * ts ** 5)
    ph = (ts[:, None] * (np.log(tau)[:, None] - _LNN[None, :N]) - ts[:, None] / 2 - PI_LD / 8 + thcorr[:, None]) % TWO_PI_LD
    main = 2.0 * (np.cos(np.asarray(ph, dtype=float)) @ (1.0 / _SQN[:N]))
    psi = np.cos(2 * np.pi * (p * p - p - 1.0 / 16.0)) / np.cos(2 * np.pi * p)
    return main + (-1) ** (N + 1) * np.asarray(tau, dtype=float) ** -0.5 * psi


def floor_max(ga, gb, npts=GRID_FLOOR):
    """max|Z| over the open gap (ga, gb) (long doubles): grid + parabolic refinement.

    Returns (floor, u_hat) with u_hat the refined position as a fraction of the gap.
    """
    us = np.linspace(0.0, 1.0, npts + 2)[1:-1]
    ts = ga + (gb - ga) * us.astype(np.longdouble)
    try:
        vals = np.abs(Z_rs2_batch(ts))
    except AssertionError:
        vals = np.abs(np.array([Z_rs2(t) for t in ts]))
    j = int(np.argmax(vals))
    f, u = vals[j], us[j]
    if 0 < j < len(us) - 1:
        y0, y1, y2 = vals[j - 1], vals[j], vals[j + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            h = us[1] - us[0]
            du = 0.5 * (y0 - y2) / den
            u = us[j] + du * h
            f = max(y1, y1 - 0.25 * (y0 - y2) * du)
    return float(f), float(u)


# ------------------------------------------------------------ mpmath references
def extremum_mp(mp, t0_mp):
    """critical point of Z nearest t0 (mpmath, at the current precision): findroot on Z'."""
    return mp.findroot(lambda x: mp.diff(mp.siegelz, x), t0_mp)


def floor_identity_mp(mp, tstar):
    """(F, |Z''(t*)|) at a critical point t* of Z."""
    return abs(mp.siegelz(tstar)), abs(mp.diff(mp.siegelz, tstar, 2))


def log_curvature_mp(mp, t):
    """-(d^2/dt^2) log|Z(t)| by numerical differentiation at the current precision."""
    return -mp.diff(lambda x: mp.log(abs(mp.siegelz(x))), t, 2)


def archimedean_curvature(t):
    """(log omega)''(t) for omega(t) = (1/2)(t^2+1/4) pi^{-1/4} |Gamma(1/4+it/2)|; equals -7/(4t^2)(1+O(t^-2))."""
    t = float(t)
    return -7.0 / (4.0 * t * t)


# ------------------------------------------------------------ double-double instrument
from . import ddarith as _dd  # noqa: E402

_LN_DD = None


def _ln_dd(nmax=40000):
    """log n, n <= nmax, as double-doubles (built once with mpmath at 40 digits)."""
    global _LN_DD
    if _LN_DD is None or len(_LN_DD[0]) < nmax + 1:
        import mpmath as mp
        with mp.workdps(40):
            vals = [mp.log(n) if n > 0 else mp.mpf(0) for n in range(nmax + 1)]
            hi = np.array([float(v) for v in vals])
            lo = np.array([float(v - mp.mpf(h)) for v, h in zip(vals, hi.tolist())])
        _LN_DD = (hi, lo)
    return _LN_DD


def _rs_coefficients(mp, p, nterms=4):
    """C_0..C_{nterms-1} of the Riemann-Siegel formula at p (Edwards, Section 7.4)."""
    pi = mp.pi
    Psi = lambda x: mp.cos(2 * pi * (x * x - x - mp.mpf(1) / 16)) / mp.cos(2 * pi * x)
    d = lambda k: mp.diff(Psi, p, k) if k > 0 else Psi(p)
    C = [d(0)]
    if nterms > 1:
        C.append(-d(3) / (96 * pi ** 2))
    if nterms > 2:
        C.append(d(2) / (64 * pi ** 2) + d(6) / (18432 * pi ** 4))
    if nterms > 3:
        C.append(-d(1) / (64 * pi ** 2) - d(5) / (3840 * pi ** 4) - d(9) / (5308416 * pi ** 6))
    return C


def Z_dd(t_mp, mp, nterms=4):
    """Z(t) for an mpmath t (exact to the working precision), phases in double-double arithmetic,
    Riemann-Siegel correction terms C_0..C_{nterms-1}.  Returns a float.

    The phase theta(t) - t log n is formed as theta mod 2pi (mpmath) minus (t log n) mod 2pi
    (double-double, from the exact double-double of t and of log n), so that no rounding
    of a phase of size 1e11 enters; the remaining error is the omitted C_{nterms} term,
    (t/2pi)^{-(2 nterms + 1)/4} in order of magnitude.
    """
    with mp.workdps(40):
        t = mp.mpf(t_mp)
        tau = mp.sqrt(t / (2 * mp.pi))
        N = int(mp.floor(tau))
        p = tau - N
        theta = (t * mp.log(tau) - t / 2 - mp.pi / 8 + 1 / (48 * t) + 7 / (5760 * t ** 3) + 31 / (80640 * t ** 5)
                 + 127 / (430080 * t ** 7)) % (2 * mp.pi)
        th_hi = float(theta)
        th_lo = float(theta - mp.mpf(th_hi))
        t_hi = float(t)
        t_lo = float(t - mp.mpf(t_hi))
        C = _rs_coefficients(mp, p, nterms)
        corr = float(sum(C[k] * tau ** (-k) for k in range(len(C))) * tau ** (-mp.mpf(1) / 2))
    ln_hi, ln_lo = _ln_dd(max(N, 1))
    ph_hi, ph_lo = _dd.dd_mul(t_hi, t_lo, ln_hi[1:N + 1], ln_lo[1:N + 1])     # t log n
    red = _dd.mod_two_pi(ph_hi, ph_lo)                                         # (t log n) mod 2pi
    phase = (th_hi - red) + th_lo
    main = 2.0 * float(np.sum(np.cos(phase) / np.sqrt(np.arange(1.0, N + 1.0))))
    return main + (-1) ** (N - 1) * corr


def floor_max_dd(ga_mp, gb_mp, mp, npts=GRID_FLOOR):
    """max|Z| over the open gap with Z_dd: grid + parabolic refinement.  Returns (floor, u_hat)."""
    us = np.linspace(0.0, 1.0, npts + 2)[1:-1]
    vals = np.array([abs(Z_dd(ga_mp + (gb_mp - ga_mp) * mp.mpf(u), mp)) for u in us])
    j = int(np.argmax(vals))
    f, u = vals[j], us[j]
    if 0 < j < len(us) - 1:
        y0, y1, y2 = vals[j - 1], vals[j], vals[j + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            h = us[1] - us[0]
            du = 0.5 * (y0 - y2) / den
            u = us[j] + du * h
            f = max(y1, y1 - 0.25 * (y0 - y2) * du)
    return float(f), float(u)
