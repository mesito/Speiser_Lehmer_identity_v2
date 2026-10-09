"""Shared, lazily loaded data for the verification groups."""
import json
import os

import numpy as np

from tightpairs import data as D
from tightpairs.constants import XS

TABLES = os.environ.get("RH_TABLES", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tables"))


class Context:
    def __init__(self, full=False):
        self.full = full
        self.XS = XS
        self.rng = np.random.default_rng(7)
        self._cache = {}

    def table(self, name):
        """numbers of a paper table, as written by make_tables.py (tables/<name>.json)."""
        if name not in self._cache:
            self._cache[name] = json.load(open(os.path.join(TABLES, name + ".json")))
        return self._cache[name]

    def _get(self, key, loader):
        if key not in self._cache:
            self._cache[key] = loader()
        return self._cache[key]

    # ordinates
    @property
    def hl(self):
        return self._get("hl", D.load_hilo)

    @property
    def zl(self):
        return self._get("zl", lambda: D.load_longdouble(hl=self.hl))

    @property
    def zd(self):
        return self._get("zd", lambda: self.hl[:, 0].copy())

    @property
    def exact(self):
        return self._get("exact", D.load_exact)

    @property
    def z1(self):
        return self._get("z1", lambda: D.load_odlyzko("zeros1"))

    @property
    def z6(self):
        return self._get("z6", lambda: D.load_odlyzko("zeros6"))

    @property
    def z6l(self):
        return self._get("z6l", lambda: self.z6.astype(np.longdouble))

    # derived arrays
    def arr(self, name):
        return self._get("arr:" + name, lambda: np.load(D.path(name)))

    def csv(self, name):
        return self._get("csv:" + name, lambda: np.atleast_1d(np.genfromtxt(D.path(name), delimiter=",", names=True)))

    @property
    def rows(self):
        return self._get("rows", lambda: {k: np.load(D.path("lehmer_rows_%s.npy" % k)) for k in ("100k", "2M", "8p4e9", "1e12", "1e21")})

    @property
    def f8(self):
        return self.arr("floors_all_8p4e9.npy")

    @property
    def f2(self):
        return self.arr("floors_all_2M.npy")

    @property
    def idx8(self):
        return self.f8[:, 0].astype(int) - 1

    @property
    def idx2(self):
        return self.f2[:, 0].astype(int) - 1

    @property
    def t0_8(self):
        return (self.zl[self.idx8] + self.zl[self.idx8 + 1]) / 2

    @property
    def t0_2(self):
        return (self.z6l[self.idx2] + self.z6l[self.idx2 + 1]) / 2

    @property
    def lp8(self):
        return self.arr("logP_pairs_8p4e9.npy")

    @property
    def lp2(self):
        return self.arr("logP_pairs_2M.npy")

    @property
    def zs8(self):
        return self.arr("zsum_pairs_8p4e9.npy")

    @property
    def zs2(self):
        return self.arr("zsum_pairs_2M.npy")

    @property
    def lpc(self):
        return self.arr("logP_ctrl_8p4e9.npy")

    @property
    def fc(self):
        return self.arr("floors_ctrl_8p4e9.npy")

    @property
    def ctrl_idx(self):
        return self.arr("ctrl_idx_8p4e9.npy")
