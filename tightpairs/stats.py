"""Least squares and partial correlations."""
import numpy as np


def ols(Xm, y):
    """y = Xm @ coef[:-1] + coef[-1].  Returns (coef, se, R2, residuals)."""
    Xm = np.atleast_2d(np.asarray(Xm, dtype=float))
    if Xm.shape[0] != len(y):
        Xm = Xm.T
    A = np.column_stack([Xm, np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    r2 = 1.0 - float(res @ res / np.sum((y - y.mean()) ** 2))
    k = A.shape[1]
    sig2 = float(res @ res / (len(y) - k))
    se = np.sqrt(np.diag(sig2 * np.linalg.inv(A.T @ A)))
    return coef, se, r2, res


def resid_on(x, c):
    """residual of x after a linear fit on the columns c (with intercept)."""
    c = np.atleast_2d(np.asarray(c, dtype=float))
    if c.shape[0] != len(x):
        c = c.T
    A = np.column_stack([c, np.ones(len(x))])
    cf, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ cf


def partial_corr(x, y, c):
    """corr(x, y | c)."""
    return float(np.corrcoef(resid_on(x, c), resid_on(y, c))[0, 1])


def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def block_means(v, B):
    nb = len(v) // B
    return v[:nb * B].reshape(nb, B).mean(1)
