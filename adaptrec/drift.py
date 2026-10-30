# -*- coding: utf-8 -*-
"""A2/A3 — komponent darajasidagi driftni aniqlash (dissertatsiya, 3.2-bo'lim).

Model darajasidagi driftdan farqli o'laroq, har bir komponentning hissa
oqimi alohida kuzatiladi (Page–Hinkley statistikasi, W-oynali siljuvchi
o'rtacha bo'yicha). K komponent uchun chegara Bonferroni bo'yicha
tuzatiladi: tuzatishsiz yolg'on signal 5% -> 18% ga ko'tariladi
(nazariy kutilma 1 - 0.95^5 = 22.6%), tuzatish bilan 3.2% (4.3-bo'lim).

Drift indeksi d(t) — komponentlar reytingining Kendall-tau masofasi
asosidagi ko'rsatkich; u holat vektorining tashkil etuvchisi sifatida
A1 ga uzatiladi.
"""
import numpy as np
from itertools import combinations


class PageHinkleyKomponentli:
    """K ta parallel PH detektori + Bonferroni tuzatilgan chegara."""

    def __init__(self, K, delta=0.010, h_bitta=0.60, alpha=0.05,
                 oyna=100, refrakter=2000):
        self.K = int(K)
        self.delta = float(delta)
        # Bonferroni: har komponent uchun alpha/K darajaga mos chegara.
        # Amalda chegarani kalibrlash simulyatsiyada bajariladi (4.3),
        # bu yerda konservativ ko'paytma ishlatiladi.
        self.h = float(h_bitta) * (1.0 + np.log(self.K))
        self.oyna = int(oyna)
        self.refrakter = int(refrakter)
        self.ph = np.zeros(self.K)
        self.qbar = np.full(self.K, np.nan)
        self.bufer = []
        self.oxirgi_signal = -10**9
        self.t = 0

    def qadam(self, hissa_vektori):
        """Bir raundlik yangilanish. hissa_vektori: (K,) — joriy baholangan
        marjinal hissalar. Signal bo'lsa True qaytaradi."""
        self.t += 1
        self.bufer.append(np.asarray(hissa_vektori, float))
        if len(self.bufer) > self.oyna:
            self.bufer.pop(0)
        if len(self.bufer) < self.oyna:
            return False
        qk = np.mean(self.bufer, axis=0)
        self.qbar = np.where(np.isnan(self.qbar), qk,
                             0.99 * self.qbar + 0.01 * qk)
        self.ph = np.maximum(0.0, self.ph + (self.qbar - qk - self.delta))
        if self.ph.max() > self.h and (self.t - self.oxirgi_signal) > self.refrakter:
            self.oxirgi_signal = self.t
            self.ph[:] = 0.0
            self.qbar[:] = np.nan
            return True
        return False


def kendall_tau_indeksi(reyting_oldin, reyting_hozir):
    """Komponentlar reytingi orasidagi normallashtirilgan Kendall-tau
    masofasi — drift indeksi d(t) in [0, 1]."""
    a = list(reyting_oldin)
    b = list(reyting_hozir)
    K = len(a)
    if K < 2:
        return 0.0
    pos_b = {k: i for i, k in enumerate(b)}
    diskordant = sum(
        1 for i, j in combinations(range(K), 2)
        if (pos_b[a[i]] > pos_b[a[j]]) != (i > j) and pos_b[a[i]] != pos_b[a[j]]
    )
    return 2.0 * diskordant / (K * (K - 1))
