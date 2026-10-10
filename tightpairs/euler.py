"""The truncated Euler product.

    log|P_X(1/2+it)| = sum_{p^k <= X} p^{-k/2} cos(k t log p) / k         (GHK convention)

evaluated exactly in k.  Phases t log p (up to 1.3e11 on the primary ensemble)
are reduced modulo 2pi in 80-bit precision with the 40-digit pi.  Smooth
cutoffs Phi(p^k/X) and the three-harmonic proxy L_X^(3) of (5.3) are also
provided, as is Delta_0(X), the uniform bound on the harmonics k >= 4 (Lemma 9.1).
"""
import numpy as np

from .constants import TWO_PI_LD, XS


def sieve(n):
    b = np.ones(n + 1, dtype=bool)
    b[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if b[i]:
            b[i * i::i] = False
    return np.flatnonzero(b)


class Primes:
    """prime table to pmax with the 80-bit logarithms; built once per process."""
    _cache = {}

    def __new__(cls, pmax=10 ** 7):
        if pmax not in cls._cache:
            obj = super().__new__(cls)
            obj.pmax = pmax
            obj.p = sieve(pmax)
            obj.pf = obj.p.astype(float)
            obj.lnp_ld = np.log(obj.p.astype(np.longdouble))
            obj.lnp = np.log(obj.pf)
            cls._cache[pmax] = obj
        return cls._cache[pmax]

    def kcut(self, X):
        """[(k, m_k)]: the first m_k primes have p^k <= X."""
        return [(k, int(np.searchsorted(self.p, X ** (1.0 / k), side="right")))
                for k in range(1, int(np.log(X) / np.log(2)) + 1)]

    def kmax(self, X):
        """K_p = floor(log X / log p) for every prime (0 for p > X)."""
        return np.floor(np.log(X) / self.lnp + 1e-12).astype(int)


def phases(t_ld, pr):
    """(t log p) mod 2pi in 80-bit precision, returned as doubles."""
    return np.asarray((np.longdouble(t_ld) * pr.lnp_ld) % TWO_PI_LD, dtype=float)


def logP_from_phases(ph, pr, X):
    s = 0.0
    for k, m in pr.kcut(X):
        if m == 0:
            break
        s += np.dot(np.cos(k * ph[:m]), pr.pf[:m] ** (-k / 2.0)) / k
    return s


def logP_exact(t_ld, xs=XS, pr=None):
    """array (len(t), len(xs)) of log|P_X(1/2+it)| for 80-bit t values."""
    pr = pr or Primes(max(xs))
    t_ld = np.atleast_1d(np.asarray(t_ld, dtype=np.longdouble))
    out = np.empty((len(t_ld), len(xs)))
    for j, tt in enumerate(t_ld):
        ph = phases(tt, pr)
        for jx, X in enumerate(xs):
            out[j, jx] = logP_from_phases(ph, pr, X)
    return out


def logP_offset_batch(dt, base_phase, pr, X, chunk=256):
    """log|P_X| at t = T0 + dt for many dt, from the base phases U_p = (T0 log p) mod 2pi
    (80-bit) and double-precision offsets dt log p (Section 6.5 scan)."""
    KP = pr.kmax(X)
    m1 = KP >= 1
    PR, LNP, U = pr.pf[m1], pr.lnp[m1], base_phase[m1]
    KPm = KP[m1]
    out = np.empty(len(dt))
    w1 = PR ** -0.5
    for c0 in range(0, len(dt), chunk):
        d = dt[c0:c0 + chunk]
        ph = (U[None, :] + d[:, None] * LNP[None, :]) % (2 * np.pi)
        s = np.cos(ph) @ w1
        for k in range(2, int(KPm.max()) + 1):
            m = KPm >= k
            s += (np.cos(k * ph[:, m]) @ (PR[m] ** (-k / 2.0))) / k
        out[c0:c0 + chunk] = s
    return out


# ----------------------------------------------------------------- smooth cutoffs
def phi_sharp(u):
    return (u <= 1.0).astype(float)


def phi_exp4(u):
    return np.exp(-(u ** 4))


def phi_bump(u):
    out = np.zeros_like(u)
    out[u <= 0.5] = 1.0
    mid = (u > 0.5) & (u < 1.0)
    x = 2.0 * u[mid] - 1.0
    out[mid] = np.exp(1.0 - 1.0 / (1.0 - x ** 2))
    return out


CUTOFFS = {"sharp": (phi_sharp, 1.0), "exp4": (phi_exp4, 4.0), "bump": (phi_bump, 1.0)}


def logP_smooth(ph, pr, X, label):
    """sum_{p^k <= umax X} Phi(p^k/X) p^{-k/2} cos(k t log p)/k with the weights of CUTOFFS."""
    phi, umax = CUTOFFS[label]
    s = 0.0
    for k in range(1, int(np.log(umax * X) / np.log(2)) + 1):
        m = int(np.searchsorted(pr.p, (umax * X) ** (1.0 / k), side="right"))
        if m == 0:
            break
        w = phi(pr.pf[:m] ** k / X)
        s += np.dot(np.cos(k * ph[:m]), w * pr.pf[:m] ** (-k / 2.0)) / k
    return s


# ----------------------------------------------------------------- Lemma 9.1
def Delta0_terms(pr):
    x = pr.pf ** -0.5
    return -np.log1p(-x) - x - x ** 2 / 2 - x ** 3 / 3


def Delta0(X, pr=None):
    pr = pr or Primes(10 ** 7)
    return float(Delta0_terms(pr)[pr.p <= X].sum())


def LX3(ph, pr, X):
    """the three-harmonic proxy of the sharp product prod_{p<=X}(1-p^{-s})^{-1}."""
    m = int(np.searchsorted(pr.p, X, side="right"))
    c1 = np.cos(ph[:m])
    c2 = 2 * c1 ** 2 - 1
    c3 = 4 * c1 ** 3 - 3 * c1
    p = pr.pf[:m]
    return float(np.dot(c1, p ** -0.5) + np.dot(c2, p ** -1.0) / 2 + np.dot(c3, p ** -1.5) / 3)


# ----------------------------------------------------------------- Section 5.4.2 (errors in variables)
def delta_sup(X, pr):
    """sup over t of |L_X^(3)(t) - log|P_X(1/2+it)||: the k = 2, 3 terms of the primes with p^k > X
    plus the k >= 4 terms with p^k <= X, each bounded by its modulus."""
    p = pr.pf[pr.p <= X]
    s = 0.5 * np.sum(1 / p[p > np.sqrt(X)]) + np.sum(p[p > X ** (1 / 3)] ** -1.5) / 3
    for k in range(4, int(np.log(X) / np.log(2)) + 1):
        q = p[p ** k <= X]
        s += np.sum(q ** (-k / 2)) / k
    return float(s)


def lambda_rand(X, pr):
    """reliability of L_X^(3) as a proxy of log|P_X| under independent uniform phases:
    cov(x, w)/var(w) = sum_{p^k <= X, k <= 3} p^-k/k^2 / sum_{p <= X, k <= 3} p^-k/k^2."""
    p = pr.pf[pr.p <= X]
    num = sum(np.sum(p[p ** k <= X] ** (-k)) / k ** 2 for k in (1, 2, 3))
    den = sum(np.sum(p ** (-k)) / k ** 2 for k in (1, 2, 3))
    return float(num / den)
