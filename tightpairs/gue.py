"""Sine-kernel quantities of Theorem 7.2 and the GUE Monte-Carlo sampler.

    S(x) = sin(pi x)/(pi x),   rho(x) = 1 - S^2 - (3/pi^2) S'^2,
    C(a) = int_{|x|>a} rho_3(x,a,-a)/rho_2(a,-a) dx/x^2,   C_GUE = 4 pi^2/15.
"""
import warnings

import numpy as np
from scipy.integrate import quad

C_GUE = 4 * np.pi ** 2 / 15
PTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def S(x):
    return np.sinc(x)


def Sp(x):
    return (np.cos(np.pi * x) - np.sinc(x)) / x if x != 0 else 0.0


def rho(x):
    return 1 - S(x) ** 2 - 3 / np.pi ** 2 * Sp(x) ** 2


def _piecewise(f, a0, pts=PTS, tail=True):
    v = sum(quad(f, a_, b_, limit=400)[0] for a_, b_ in zip([a0] + pts[:-1], pts))
    if tail:
        v += quad(f, pts[-1], np.inf, limit=400)[0]
    return v


def integrals():
    """(int (1-S^2)/x^2, int S'^2/x^2, C_GUE from them)."""
    I1 = 2 * _piecewise(lambda x: (1 - S(x) ** 2) / x ** 2, 1e-12)
    I2 = 2 * _piecewise(lambda x: Sp(x) ** 2 / x ** 2, 1e-12)
    return I1, I2, I1 - 3 / np.pi ** 2 * I2


def Kdet(x, a):
    M = np.array([[1, S(x - a), S(x + a)], [S(x - a), 1, S(2 * a)], [S(x + a), S(2 * a), 1]])
    return np.linalg.det(M) / (1 - S(2 * a) ** 2)


def C_of_a(a, Xmax=1024.0):
    warnings.filterwarnings("ignore")
    f = lambda x: Kdet(x, a) / x ** 2
    v = 0.0
    edges = [a] + [e for e in PTS if a < e < Xmax] + [Xmax]
    for a_, b_ in zip(edges[:-1], edges[1:]):
        v += quad(f, a_, b_, limit=800, epsabs=1e-13, epsrel=1e-13)[0]
    return 2 * v + 2 / Xmax


def window_integral(Xi):
    pts = [p for p in PTS if p < Xi]
    return 2 * sum(quad(lambda x: rho(x) / x ** 2, a_, b_, limit=400)[0] for a_, b_ in zip([1e-12] + pts, pts + [Xi]))


def small_s_law(sg=np.linspace(0.01, 0.15, 15)):
    """linear fit of (C_GUE - C(s))/s^2 and the cubic fit of C_GUE - C(s); returns (sg, Cg, lin, cubic)."""
    Cg = np.array([C_of_a(s_ / 2, 1024.0) for s_ in sg])
    lin = np.polyfit(sg, (C_GUE - Cg) / sg ** 2, 1)
    cub = np.polyfit(sg, C_GUE - Cg, 3)
    return sg, Cg, lin, cub


# ----------------------------------------------------------------- Monte Carlo
def gue_unfold(w, N):
    R = 2.0 * np.sqrt(N)
    zc = np.clip(w / R, -1.0, 1.0)
    Fc = 0.5 + (zc * np.sqrt(1 - zc * zc) + np.arcsin(zc)) / np.pi
    return N * (Fc - 0.5)


def sample_pairs(N, want, seed=2024, windows=(25.0, 50.0), s_max=0.25, bulk_frac=0.2):
    """conditioned tight pairs from Dumitriu-Edelman tridiagonal GUE matrices of size N.
    Returns dict: sums[w] (far sums per pair), lag (array of consecutive bulk spacings), tails (bulk spacings)."""
    from scipy.linalg import eigh_tridiagonal
    Xb = bulk_frac * (N / 2.0)
    rng = np.random.default_rng(seed)
    sums = {w: [] for w in windows}
    lag, tails, npair = [], [], 0
    while npair < want:
        dd_ = rng.standard_normal(N)
        ee_ = np.sqrt(rng.chisquare(2 * np.arange(N - 1, 0, -1)) / 2.0)
        xx = gue_unfold(eigh_tridiagonal(dd_, ee_, eigvals_only=True, check_finite=False), N)
        sp = np.diff(xx)
        midx = 0.5 * (xx[:-1] + xx[1:])
        bulk = np.abs(midx) <= Xb
        sb = sp[bulk]
        lag.append(np.column_stack([sb[:-1], sb[1:]]))
        tails.append(sb)
        for i in np.nonzero(bulk & (sp < s_max))[0]:
            relp = np.delete(xx - midx[i], [i, i + 1])
            for w in sums:
                m = np.abs(relp) < w
                sums[w].append(np.sum(1.0 / relp[m] ** 2) if m.any() else 0.0)
            npair += 1
    return dict(sums={w: np.array(v[:want]) for w, v in sums.items()}, lag=np.vstack(lag), tails=np.concatenate(tails))
