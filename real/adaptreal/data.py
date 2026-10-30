# -*- coding: utf-8 -*-
"""Real so'rov oqimini yuklash va indekslash (Amazon Reviews 2023, Video_Games).

Oqim vaqt bo'yicha tartiblangan, 5-core filtrlangan. Bu yerda hech qanday
model yo'q, faqat ma'lumot tuzilmalari.
"""
import csv, os, numpy as np, collections

DATA = os.environ.get("VG_DATA", "/data/vg")


class Oqim:
    """Vaqt bo'yicha tartiblangan hodisalar va item metadatasi."""

    def __init__(self, min_ts=None, max_ts=None):
        u_idx, i_idx = {}, {}
        U, I, R, T = [], [], [], []
        with open(os.path.join(DATA, "vg_stream.csv"), encoding="utf-8") as f:
            r = csv.reader(f); next(r)
            for u, i, rt, ts in r:
                ts = int(ts)
                if min_ts is not None and ts < min_ts: continue
                if max_ts is not None and ts > max_ts: continue
                U.append(u_idx.setdefault(u, len(u_idx)))
                I.append(i_idx.setdefault(i, len(i_idx)))
                R.append(float(rt)); T.append(ts)
        self.u = np.asarray(U, np.int32)
        self.i = np.asarray(I, np.int32)
        self.r = np.asarray(R, np.float32)
        self.t = np.asarray(T, np.int64)
        self.n_user, self.n_item = len(u_idx), len(i_idx)
        self.u_idx, self.i_idx = u_idx, i_idx
        self.i_rev = {v: k for k, v in i_idx.items()}
        self._meta = None

    def __len__(self):
        return len(self.u)

    @property
    def meta(self):
        """item_id -> (sarlavha, kategoriyalar, do'kon). Kerak bo'lganda yuklanadi."""
        if self._meta is None:
            m = {}
            with open(os.path.join(DATA, "vg_items.csv"), encoding="utf-8") as f:
                r = csv.reader(f); next(r)
                for row in r:
                    asin = row[0]
                    j = self.i_idx.get(asin)
                    if j is not None:
                        m[j] = (row[1], row[4], row[3])
            self._meta = m
        return self._meta

    def matn(self):
        """Har bir item uchun matnli tavsif (kontent komponenti uchun)."""
        md = self.meta
        out = [""] * self.n_item
        for j in range(self.n_item):
            t, c, s = md.get(j, ("", "", ""))
            out[j] = (t + " " + c.replace("|", " ") + " " + s).strip()
        return out

    def xulosa(self):
        return dict(hodisa=len(self), foydalanuvchi=self.n_user, item=self.n_item,
                    boshlanish=int(self.t[0]), tugash=int(self.t[-1]))
