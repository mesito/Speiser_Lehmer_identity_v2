"""Tail bounds for the truncated Euler product (Section 9 of the paper).

For X = 1e7 and each prime p <= X, K_p = floor(log X / log p),
    term_p(theta) = sum_{k<=K_p} p^{-k/2} cos(k theta)/k,   |term_p| <= a_p = sum_k p^{-k/2}/k,
    sigma_p^2 = (1/2) sum_k p^{-k}/k^2,   Lambda_1 = sum_{p^k<=X} p^{-k/2} log p.
Hybrid Chernoff bound (Theorem 9.7): exact factors M_p(-s) for p <= 1000 (midpoint
rule, 4096 nodes), Bennett's inequality for p > 1000.
"""
import numpy as np

from .euler import Primes


class TailBounds:
    def __init__(self, X=10 ** 7, nq=4096, p_split=1000):
        self.X = X
        self.pr = Primes(X)
        P = self.pr.pf
        self.P = P
        self.KP = self.pr.kmax(X)
        ks = np.arange(1, int(self.KP.max()) + 1)
        mask = ks[None, :] <= self.KP[:, None]
        self.Pk = np.where(mask, P[:, None] ** (-ks[None, :] / 2.0), 0.0)
        self.a_p = (self.Pk / ks[None, :]).sum(1)
        self.sig2_p = 0.5 * np.where(mask, P[:, None] ** (-ks[None, :].astype(float)) / ks[None, :] ** 2, 0.0).sum(1)
        self.Lam1 = float(np.sum(np.log(P) * self.Pk.sum(1)))
        self.M = float((1.0 / ks[None, :] * mask).sum())
        self.nq = nq
        self.small = np.flatnonzero(P <= p_split)
        self.big = np.flatnonzero(P > p_split)
        self.TM = self.term_grid(self.small, nq)
        self.cp = self.a_p[self.big]
        self.sig2_big = self.sig2_p[self.big]

    def constants(self):
        return {"n_prime_powers": int(self.KP.sum()), "n_k_ge_2": int((self.KP - 1).sum()), "sum_a_p": float(self.a_p.sum()),
                "sum_sigma2": float(self.sig2_p.sum()), "sigma": float(np.sqrt(self.sig2_p.sum())), "Lambda_1": self.Lam1, "M": self.M}

    def term_grid(self, idx, nq=None):
        nq = nq or self.nq
        th = (np.arange(nq) + 0.5) * (2 * np.pi / nq)
        out = np.zeros((len(idx), nq))
        for j, i in enumerate(idx):
            for k in range(1, int(self.KP[i]) + 1):
                out[j] += self.P[i] ** (-k / 2) * np.cos(k * th) / k
        return out

    def second_moment_terms(self, H):
        A = self.a_p.sum()
        return (A ** 2 * 4 * self.X / H, 4 * self.M ** 2 / H, A ** 2 / (H * np.log(2)))

    def first_moment_bound(self, H):
        return 2 * self.a_p.sum() / (H * np.log(2))

    def fourth_moment_crude(self, H):
        return 4 * self.a_p.sum() ** 4 * self.X ** 4 / H

    def log_Mp_small(self, s, TM=None):
        TM = self.TM if TM is None else TM
        return float(np.sum(np.log(np.exp(-s * TM).mean(1))))

    def cert(self, u, s, margin=1e-6):
        """the hybrid Chernoff bound on d(u) at tilt s."""
        return float(np.exp(-(s * u - self.log_Mp_small(s) - np.sum(self.sig2_big * (np.expm1(s * self.cp) - s * self.cp) / self.cp ** 2) - margin)))

    def best(self, u):
        sv = np.linspace(1, 14, 261)
        vals = [self.cert(u, s_) for s_ in sv]
        j = int(np.argmin(vals))
        sv2 = np.linspace(sv[j] - 0.05, sv[j] + 0.05, 41)
        vals2 = [self.cert(u, s_) for s_ in sv2]
        j2 = int(np.argmin(vals2))
        return vals2[j2], float(sv2[j2])

    def bennett_all(self, u):
        """Bennett over all primes (no exact small-prime factors)."""
        with np.errstate(over="ignore"):
            return min(float(np.exp(-(s_ * u - np.sum(self.sig2_p * (np.expm1(s_ * self.a_p) - s_ * self.a_p) / self.a_p ** 2))))
                       for s_ in np.linspace(1e-6, 12, 1201))

    def N_criterion(self, H, m=0.5):
        """N = H Lambda_1 / m: the number of disjoint intervals of length m/Lambda_1 in a window of length H
        (Theorem 9.10; the one-sided interval that the window hypothesis guarantees)."""
        return H * self.Lam1 / m

    def mean_value_product(self, Xsmall=100, s=1.0):
        """prod_{p <= Xsmall} M_p(-s) for the product P_{Xsmall}, i.e. with K_p = floor(log Xsmall / log p)
        (the harmonics of P_{Xsmall} itself, not those of P_X)."""
        idx = np.flatnonzero(self.P <= Xsmall)
        KP = self.pr.kmax(Xsmall)
        th = (np.arange(self.nq) + 0.5) * (2 * np.pi / self.nq)
        TM = np.zeros((len(idx), self.nq))
        for j, i in enumerate(idx):
            for k in range(1, int(KP[i]) + 1):
                TM[j] += self.P[i] ** (-k / 2) * np.cos(k * th) / k
        return float(np.exp(np.sum(np.log(np.exp(-s * TM).mean(1)))))


def bahadur_rao(X=10 ** 4, u=4.580, nq=4096):
    """the Bahadur-Rao prefactor and prediction at X with exact factors for all primes; returns (sigma, prefactor, prediction, s*, K, Kpp, TM)."""
    pr = Primes(X)
    P = pr.pf
    KP = pr.kmax(X)
    th = (np.arange(nq) + 0.5) * (2 * np.pi / nq)
    sig = np.sqrt(0.5 * sum(sum(P[i] ** (-k) / k ** 2 for k in range(1, KP[i] + 1)) for i in range(len(P))))
    TM = np.zeros((len(P), nq))
    for i in range(len(P)):
        for k in range(1, KP[i] + 1):
            TM[i] += P[i] ** (-k / 2) * np.cos(k * th) / k

    def K(s_):
        E = np.exp(-s_ * TM)
        Mp = E.mean(1)
        w = E / Mp[:, None]
        m1 = (w * TM).mean(1)
        m2 = (w * TM ** 2).mean(1)
        return float(np.log(Mp).sum()), float(-m1.sum()), float((m2 - m1 ** 2).sum())

    s = 3.0
    for _ in range(100):
        Kv, Kp_, Kpp = K(s)
        sn = s - (Kp_ - u) / Kpp
        if abs(sn - s) < 1e-13:
            s = sn
            break
        s = sn
    Kv, Kp_, Kpp = K(s)
    I = s * u - Kv
    pref = 1.0 / (s * np.sqrt(2 * np.pi * Kpp))
    return float(sig), float(pref), float(pref * np.exp(-I)), float(s), float(Kv), float(Kpp), TM


def importance_sampling(TM, s, Kv, u, n=2000, seed=19):
    """tilted importance-sampling estimate of d(u) (exact factors), returns (estimate, std err)."""
    nq = TM.shape[1]
    E = np.exp(-s * TM)
    CDF = np.cumsum(E / E.mean(1)[:, None] / nq, axis=1)
    rng = np.random.default_rng(seed)
    U = rng.random((n, TM.shape[0]))
    idx = np.empty((n, TM.shape[0]), dtype=np.int32)
    for p_ in range(TM.shape[0]):
        idx[:, p_] = np.searchsorted(CDF[p_], U[:, p_])
    Lt = TM[np.arange(TM.shape[0])[None, :], idx].sum(1)
    w = np.exp(s * Lt) * (Lt <= -u)
    return float(np.exp(Kv) * w.mean()), float(np.exp(Kv) * w.std(ddof=1) / np.sqrt(n))
