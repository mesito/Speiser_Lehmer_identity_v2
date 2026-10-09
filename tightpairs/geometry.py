"""Pair geometry: the quadratic form of the zero set at a tight pair.

For a pair (gamma_n, gamma_{n+1}) with midpoint t0 and gap g (Section 3 of the
paper):

    far(t)   = sum_{m not in {n, n+1}, |m-n| <= W} (t - gamma_m)^-2
    lam(t)   = sum_{...} (t - gamma_m)^-1                 (first moment)
    S_on     = far(t0) + 8/g^2                            (on-line sum at the midpoint)
    h_thr    = sqrt(2 / S_on)                             (Speiser threshold)
    r        = h_thr / (g/2)                              (threshold ratio)
    C_mid    = (1/g^2) / far(t0)                          (midpoint Lehmer constant)
    S_end    = sum_m [(gamma_m - gamma_n)^-2 + (gamma_m - gamma_{n+1})^-2]
    C        = (1/g^2) / S_end                            (Csordas-Smith-Varga constant)
    kappa    = g^2 / (4 d_min^2),  d_min = min_m |t0 - gamma_m|

The Speiser-Lehmer identity is r^2 (1 + 1/(8 C_mid)) = 1 (Theorem A(a)).

All sums take the ordinates as an 80-bit array `zl` (long double) and are
evaluated in long double; `W=None` uses every ordinate in the file.  The
ensemble edges clip the window.
"""
import numpy as np

from .constants import W as W_DEFAULT, dens


def window(i, n, W=W_DEFAULT):
    """index arrays (lo, hi) of the W ordinates below gamma_i and above gamma_{i+1}."""
    if W is None:
        return np.arange(0, i), np.arange(i + 2, n)
    return np.arange(max(0, i - W), i), np.arange(i + 2, min(n, i + 2 + W))


def far_lambda(zl, i, t, W=W_DEFAULT):
    """(far(t), lam(t)) for the pair at index i, at the point t (long double)."""
    lo, hi = window(i, len(zl), W)
    d = np.concatenate([t - zl[lo], t - zl[hi]])
    return float(np.sum(1 / d ** 2)), float(np.sum(1 / d))


def mirror_sum(zl, t):
    """sum over the file of (t + gamma)^-2: the lower-half-plane terms (negligible at large t)."""
    return float(np.sum(1 / (t + zl) ** 2))


def tail_second_moment(zl, t, lower_complete=False):
    """mean-density estimate of sum (t-gamma)^-2 over the ordinates beyond the ends of the
    file, dens/L on each truncated side (plus the mirror tail dens/(t + gamma_last))."""
    tl, tr = float(t - zl[0]), float(zl[-1] - t)
    d = dens(float(zl[-1]))
    tail = d / tr + d / (float(zl[-1]) + float(t))
    if not lower_complete:
        tail += dens(float(zl[0])) / tl
    return tail


def pair_quantities(zl, i, W=W_DEFAULT):
    """dict of the geometric quantities of the pair at index i (long double in, float out)."""
    ga, gb = zl[i], zl[i + 1]
    t0 = (ga + gb) / 2
    g = gb - ga
    lo, hi = window(i, len(zl), W)
    d = np.concatenate([t0 - zl[lo], t0 - zl[hi]])
    far = float(np.sum(1 / d ** 2))
    lam = float(np.sum(1 / d))
    others = np.concatenate([zl[lo], zl[hi]])
    Send = float(np.sum(1 / (others - ga) ** 2) + np.sum(1 / (others - gb) ** 2))
    gf = float(g)
    Son = far + 8.0 / gf ** 2
    hthr = np.sqrt(2.0 / Son)
    r = hthr / (gf / 2)
    Cmid = (1.0 / gf ** 2) / far
    C = (1.0 / gf ** 2) / Send
    dmin = float(np.min(np.abs(d)))
    kappa = gf ** 2 / (4 * dmin ** 2)
    return dict(idx=i, t0=float(t0), g=gf, s=gf * dens(float(t0)), far=far, lam=lam, Son=Son,
                hthr=hthr, r=r, Cmid=Cmid, Send=Send, C=C, dmin=dmin, kappa=kappa)


def identity_deviation(r, Cmid):
    """r^2 (1 + 1/(8 C_mid)) - 1, elementwise, in the dtype of the inputs."""
    return r ** 2 * (1 + 1 / (8 * Cmid)) - 1


def endpoint_correction_series(zl, i, W=W_DEFAULT):
    """the positive series of Corollary A.1: S_end - 2 far = (g^2/2) sum (3a^2 - g^2/4)/(a^2 (a^2-g^2/4)^2)."""
    ga, gb = zl[i], zl[i + 1]
    t0 = (ga + gb) / 2
    g = gb - ga
    lo, hi = window(i, len(zl), W)
    a = np.concatenate([t0 - zl[lo], t0 - zl[hi]])
    e2 = g ** 2 / 4
    return float(g ** 2 / 2 * np.sum((3 * a ** 2 - e2) / (a ** 2 * (a ** 2 - e2) ** 2)))


def rows_table(zl, idx, W=W_DEFAULT):
    """the lehmer_rows_* layout: (idx 1-based, t0, g, s, S_on, far, C, C_mid, h_thr, r)."""
    out = np.empty((len(idx), 10))
    for j, i in enumerate(idx):
        q = pair_quantities(zl, i, W)
        out[j] = (i + 1, q["t0"], q["g"], q["s"], q["Son"], q["far"], q["C"], q["Cmid"], q["hthr"], q["r"])
    return out


def Ghat_on_gap(zl, i, npts=257, W=W_DEFAULT):
    """the transposed detector of Section 3.6 on `npts` interior points of the gap; returns its maximum."""
    ga, gb = zl[i], zl[i + 1]
    lo, hi = window(i, len(zl), W)
    gmax = -np.inf
    for u in np.linspace(0, 1, npts + 2)[1:-1]:
        t = ga + (gb - ga) * np.longdouble(u)
        G = float(np.sum(1 / (t - zl[lo]) ** 2) + np.sum(1 / (t - zl[hi]) ** 2) + 2 / ((t - ga) * (t - gb)))
        gmax = max(gmax, G)
    return gmax
