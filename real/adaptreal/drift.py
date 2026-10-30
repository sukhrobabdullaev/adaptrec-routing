# -*- coding: utf-8 -*-
"""Komponent hissasi oqimlarida drift aniqlash (1-maqola, 3.5-bo'lim).

Sintetik `adaptrec/drift.py` ning real datadagi ekvivalenti. Har bir
komponentning kuzatilgan sifat hissasi alohida oqim hosil qiladi va
har biri Page-Hinkley statistikasi bilan kuzatiladi. K ta oqimni bir
vaqtda kuzatish ko'p taqqoslash muammosi, shuning uchun Bonferroni
tuzatishi qo'llanadi.
"""
import numpy as np


class PageHinkley:
    """Bir tomonlama Page-Hinkley: o'rtachaning PASAYISHINI aniqlaydi."""

    def __init__(self, delta=0.002, lam=0.35, oyna=500):
        self.delta, self.lam, self.oyna = delta, lam, oyna
        self.reset()

    def reset(self):
        self.n = 0; self.mean = 0.0; self.m = 0.0; self.M = 0.0

    def yangilash(self, x):
        self.n += 1
        self.mean += (x - self.mean) / min(self.n, self.oyna)
        self.m += (self.mean - x - self.delta)     # pasayish uchun ishora
        self.M = min(self.M, self.m)
        return (self.m - self.M) > self.lam

    def PH(self):
        return self.m - self.M


class Monitor:
    """K ta komponent oqimini bir vaqtda kuzatish."""

    def __init__(self, K, delta=0.002, lam=0.35, oyna=500, bonferroni=False, alfa=0.05):
        # Bonferroni: har bir oqim uchun qat'iyroq chegara
        self.lam = lam * (1.0 + np.log(K)) if bonferroni else lam
        self.det = [PageHinkley(delta, self.lam, oyna) for _ in range(K)]
        self.K = K

    def yangilash(self, q):
        """q: (K,) sifat hissalari. Qaytadi: aniqlangan komponentlar ro'yxati."""
        return [k for k in range(self.K) if self.det[k].yangilash(float(q[k]))]

    def reset(self, k):
        self.det[k].reset()
