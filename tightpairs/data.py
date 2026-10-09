"""Ordinate files.

Platt / LMFDB block format (little-endian).  The file starts with an 8-byte
unsigned block count.  Each block has a 32-byte header <ddQQ> = (t0, t1, N_t0,
N_t1): bounding ordinates as doubles and the corresponding zero indices as
64-bit unsigned integers.  It is followed by N_t1 - N_t0 records <QIB>, each
the 104-bit unsigned integer Z_i = q_i + 2^64 I_i + 2^96 B_i, and

        gamma_{N_t0 + j} = t0 + (Z_1 + ... + Z_j) * 2^-101 .

The cumulative sums are exact integers.  decode_platt() keeps them as such
(cum_hi * 2^64 + cum_lo) and derives from them
  * the exact ordinate as an mpmath number (exact_ordinate),
  * the double-double representation (hi, lo) shipped as lmfdb_zeros_hilo.npy,
    which reproduces the exact value to 1e-22 (checked by verify group V1),
  * the 80-bit array used by the vector code (hi + lo in long double).

Integrity checks on decoding (Section 2.2 of the paper): (i) t1 of block k ==
t0 of block k+1; (ii) N_t1 of block k == N_t0 of block k+1; (iii) strict
monotonicity inside each block; (iv) global strict monotonicity.
"""
import gzip
import os
import struct
from fractions import Fraction

import numpy as np

from .constants import S_TIGHT, N_CTRL, CTRL_SEED, CTRL_BAND, dens

DATA = os.environ.get("RH_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))


def path(name, data_dir=None):
    return os.path.join(data_dir or DATA, name)


# ----------------------------------------------------------------- Platt format
def decode_platt(raw):
    """Decode a (possibly truncated) Platt block file given as bytes.

    Returns a dict with block_t0, block_t1, block_n0, block_n1 (arrays over the
    complete blocks), cum_lo, cum_hi (uint64 arrays over the zeros: the exact
    104-bit cumulative deltas split at 2^64) and block_of (block index of each
    zero).  A truncated final block is discarded.  The checks (i)-(iv) are
    applied and raise AssertionError on failure.
    """
    nblocks = struct.unpack("<Q", raw[:8])[0]
    pos = 8
    bt0, bt1, bn0, bn1 = [], [], [], []
    cum_lo, cum_hi = [], []
    while pos + 32 <= len(raw):
        t0, t1, n0, n1 = struct.unpack("<ddQQ", raw[pos:pos + 32])
        nrec = n1 - n0
        end = pos + 32 + 13 * nrec
        if end > len(raw):
            break                                   # truncated final block
        rec = np.frombuffer(raw[pos + 32:end], dtype=np.dtype([("q", "<u8"), ("I", "<u4"), ("B", "u1")]))
        cum = 0
        prev = -1
        for a, b, c in zip(rec["q"].tolist(), rec["I"].tolist(), rec["B"].tolist()):
            z = a + (b << 64) + (c << 96)
            assert z > 0, "check (iii) failed in block %d" % len(bt0)
            cum += z
            cum_lo.append(cum & ((1 << 64) - 1))
            cum_hi.append(cum >> 64)
            prev = cum
        assert prev > 0
        bt0.append(t0); bt1.append(t1); bn0.append(n0); bn1.append(n1)
        pos = end
    for k in range(len(bt0) - 1):
        assert bt1[k] == bt0[k + 1], "check (i) failed at block %d" % k
        assert bn1[k] == bn0[k + 1], "check (ii) failed at block %d" % k
    out = dict(nblocks_header=nblocks,
               block_t0=np.array(bt0), block_t1=np.array(bt1),
               block_n0=np.array(bn0, dtype=np.int64), block_n1=np.array(bn1, dtype=np.int64),
               cum_lo=np.array(cum_lo, dtype=np.uint64), cum_hi=np.array(cum_hi, dtype=np.uint64))
    out["block_of"] = np.repeat(np.arange(len(bt0)), out["block_n1"] - out["block_n0"]).astype(np.int32)
    # check (iv): global monotonicity, exact
    e = exact_fractions(out)
    assert all(e[i] < e[i + 1] for i in range(len(e) - 1)), "check (iv) failed"
    return out


def exact_fractions(ex):
    """all ordinates as exact Fractions (slow: ~10 s for 7.7e5 zeros)."""
    scale = Fraction(1, 2 ** 101)
    t0f = [Fraction(float(t)) for t in ex["block_t0"]]
    lo, hi, bo = ex["cum_lo"].tolist(), ex["cum_hi"].tolist(), ex["block_of"].tolist()
    return [t0f[b] + Fraction(h * 2 ** 64 + l) * scale for l, h, b in zip(lo, hi, bo)]


def exact_ordinate(ex, i, mp):
    """ordinate i (0-based) as an mpmath number, exact at mp.mp.dps >= 35."""
    c = (int(ex["cum_hi"][i]) << 64) | int(ex["cum_lo"][i])
    return mp.mpf(float(ex["block_t0"][ex["block_of"][i]])) + mp.mpf(c) / mp.mpf(2) ** 101


def hilo_from_exact(ex):
    """double-double (hi, lo) for every ordinate, from the exact integers."""
    fr = exact_fractions(ex)
    hi = np.array([float(f) for f in fr])
    lo = np.array([float(f - Fraction(h)) for f, h in zip(fr, hi.tolist())])
    return np.column_stack([hi, lo])


def check_hilo(ex, hl):
    """max |exact - (hi + lo)| over all ordinates, exactly (returns a float)."""
    fr = exact_fractions(ex)
    worst = Fraction(0)
    for f, h, l in zip(fr, hl[:, 0].tolist(), hl[:, 1].tolist()):
        d = abs(f - Fraction(h) - Fraction(l))
        if d > worst:
            worst = d
    return float(worst)


# ----------------------------------------------------------------- loaders
EXACT_KEYS = ("block_t0", "block_t1", "block_n0", "block_n1", "cum_lo", "cum_hi", "block_of")


def build_ordinates(data_dir=None, verbose=True):
    """Decode data/lmfdb_top.bin (checks (i)-(iv)) and write the derived ordinate arrays
    lmfdb_zeros_exact.npz and lmfdb_zeros_hilo.npy next to it.  Deterministic integer
    arithmetic: the files are bit-identical on every platform.  About 25 seconds."""
    if verbose:
        print("decoding %s (first use; about 25 s) ..." % path("lmfdb_top.bin", data_dir), flush=True)
    ex = decode_platt(open(path("lmfdb_top.bin", data_dir), "rb").read())
    hl = hilo_from_exact(ex)
    np.savez(path("lmfdb_zeros_exact.npz", data_dir), **{k: ex[k] for k in EXACT_KEYS})
    np.save(path("lmfdb_zeros_hilo.npy", data_dir), hl)
    return ex, hl


def load_exact(data_dir=None):
    f = path("lmfdb_zeros_exact.npz", data_dir)
    if not os.path.exists(f):
        return build_ordinates(data_dir)[0]
    return dict(np.load(f))


def load_hilo(data_dir=None):
    f = path("lmfdb_zeros_hilo.npy", data_dir)
    if not os.path.exists(f):
        return build_ordinates(data_dir)[1]
    return np.load(f)


def load_longdouble(data_dir=None, hl=None):
    """80-bit ordinates of the primary ensemble (hi + lo in long double)."""
    if hl is None:
        hl = load_hilo(data_dir)
    return hl[:, 0].astype(np.longdouble) + hl[:, 1].astype(np.longdouble)


def load_parsed(data_dir=None, hl=None):
    """double-precision ordinates: the hi column.  Only for selection and coarse statistics."""
    p = path("lmfdb_zeros_parsed.npy", data_dir)
    if os.path.exists(p):
        return np.load(p)
    if hl is None:
        hl = load_hilo(data_dir)
    return hl[:, 0].copy()


def pack_odlyzko(text_file, out_file):
    """Pack an Odlyzko table (one ordinate per line, exactly nine decimals) losslessly:
    integers v = 10^9 * ordinate, stored as v[0] plus uint32 deltas with an exception list for
    the few deltas >= 2^32.  unpack_odlyzko() returns exactly the array np.loadtxt would."""
    op = gzip.open if text_file.endswith(".gz") else open
    v = np.array([int(l.strip().replace(".", "")) for l in op(text_file, "rt") if l.strip()], dtype=np.int64)
    d = np.diff(v)
    big = np.flatnonzero(d >= 2 ** 32)
    np.savez_compressed(out_file, first=np.array([v[0]], dtype=np.int64), delta=(d % 2 ** 32).astype(np.uint32),
                        exc_idx=big.astype(np.int64), exc_val=d[big])
    return v


def unpack_odlyzko(npz_file):
    z = np.load(npz_file)
    d = z["delta"].astype(np.int64)
    d[z["exc_idx"]] = z["exc_val"]
    v = np.concatenate([z["first"], z["first"][0] + np.cumsum(d)])
    return v / 1e9


def load_odlyzko(name, data_dir=None):
    """Odlyzko's table `name` (zeros1, zeros6, zeros3, zeros4): the packed .npz of pack_odlyzko(),
    or the original text table, gzip-compressed or plain.  All three give the same array."""
    f = path(name + ".npz", data_dir)
    if os.path.exists(f):
        return unpack_odlyzko(f)
    for fn, op in ((name + ".gz", lambda f: gzip.open(f, "rt")), (name, lambda f: open(f, "rt"))):
        f = path(fn, data_dir)
        if os.path.exists(f):
            return np.loadtxt(op(f))
    raise FileNotFoundError("%s.npz / %s(.gz) not found in %s: download the table from "
                            "https://www-users.cse.umn.edu/~odlyzko/zeta_tables/" % (name, name, data_dir or DATA))


def load_odlyzko_deltas(name, data_dir=None):
    """a table stored as successive differences (zeros4): cumulative sum in 80-bit precision."""
    d = load_odlyzko(name, data_dir)
    return np.cumsum(d.astype(np.longdouble))


# ----------------------------------------------------------------- selections
def tight_pairs(z, s_max=S_TIGHT):
    """0-based indices n of consecutive pairs (gamma_n, gamma_{n+1}) with s < s_max."""
    z = np.asarray(z, dtype=float)
    g = np.diff(z)
    t0 = 0.5 * (z[:-1] + z[1:])
    s = g * dens(t0)
    return np.flatnonzero(s < s_max)


def normalised_spacings(z):
    z = np.asarray(z, dtype=float)
    return np.diff(z) * dens(z.mean())


def control_indices(z, n=N_CTRL, seed=CTRL_SEED, band=CTRL_BAND):
    """indices of the control mid-gaps: gaps with normalised spacing in `band`,
    normalised with the ensemble-mean density, drawn without replacement with
    numpy's default_rng(seed)."""
    s = normalised_spacings(z)
    elig = np.flatnonzero((s >= band[0]) & (s <= band[1]))
    return np.random.default_rng(seed).choice(elig, n, replace=False)


# ----------------------------------------------------------------- the all-gaps scan
def load_allgaps(data_dir=None, hl=None):
    """the all-gaps scan of Section 6.4 as (gap_idx, t_mid, g, floor), 193,179 rows sorted by t.
    The file moment_allgaps.npy stores (gap_idx, floor); t_mid and g are the midpoint and
    length of gap gap_idx formed from the double-precision ordinates, as in the scan."""
    a = np.load(path("moment_allgaps.npy", data_dir))
    if a.shape[1] == 4:
        return a
    z = (hl if hl is not None else load_hilo(data_dir))[:, 0]
    gi = a[:, 0].astype(int)
    return np.column_stack([a[:, 0], 0.5 * (z[gi] + z[gi + 1]), z[gi + 1] - z[gi], a[:, 1]])
