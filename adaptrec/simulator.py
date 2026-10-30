# -*- coding: utf-8 -*-
"""Nazorat qilinadigan simulyatsiya muhiti (dissertatsiya, 4.2-bo'lim).

Muhitning haqiqiy tuzilishi to'liq ma'lum, shuning uchun afsuslanish
BEVOSITA o'lchanadi (oracle bilan taqqoslash). Sifat submodulyar:
    M(A, phi) = 1 - prod_{k in A} (1 - alpha_k(phi))
mukofot esa narx va SLO jarimalari ayirilgan holda shakllanadi (4.4-formula).
"""
import numpy as np
from .feasible import mask_matritsa
from .state import D


class Muhit:
    def __init__(self, profillar, toplamlar, budget=8.0, slo=100.0,
                 c_cost=0.030, b_pen=2.0, shovqin=0.05, seed=0):
        self.K = len(profillar)
        self.gamma = np.array([p.gamma for p in profillar])
        self.lat = np.array([p.lat95 for p in profillar])
        self.M = mask_matritsa(toplamlar, self.K)
        self.budget, self.slo = budget, slo
        self.c_cost, self.b_pen, self.shovqin = c_cost, b_pen, shovqin
        self.rng = np.random.default_rng(seed)
        cost_s = self.M @ self.gamma
        lat_s = np.where(self.M, self.lat, -np.inf).max(1)
        self.pen = c_cost * cost_s + b_pen * np.maximum(0.0, lat_s / slo - 1.0)
        self.cost_s = cost_s

    def holat(self):
        """Tasodifiy holat vektori [1, n_u, drift, Q, yuklama, eps]."""
        r = self.rng
        return np.array([1.0, r.beta(1.4, 2.2), r.beta(1.2, 6.0),
                         float(np.clip(r.beta(5.0, 2.0), 0.05, 1.0)),
                         r.uniform(0.2, 0.9), r.uniform(0.3, 1.0)])

    def alphas(self, phi, zarba=False):
        """Komponentlarning HAQIQIY marjinal hissasi (4.1–4.2 formulalar)."""
        n_, d_, q_ = phi[1], phi[2], phi[3]
        a = np.array([0.05 + 0.30 * n_,
                      (0.28 - 0.18 * n_) * q_,
                      0.15 + 0.05 * d_,
                      0.10 + 0.20 * n_,
                      (0.70 - 0.55 * n_) * (0.5 + 0.5 * q_)])
        a = np.clip(a, 0.0, 0.95)
        if zarba:  # katalog zarbasi ssenariysi
            a[0] *= 0.30
            a[3] *= 0.55
            a[1] = min(a[1] * 1.60, 0.95)
        return a

    def ortacha_mukofot(self, phi, zarba=False):
        """(nA,) — har bir to'plam uchun shovqinsiz o'rtacha mukofot."""
        L = np.log1p(-self.alphas(phi, zarba))
        return (1.0 - np.exp(self.M @ L)) - self.pen

    def qadam(self, a_idx, phi, zarba=False):
        """Tanlangan to'plam uchun (mukofot, afsuslanish, narx)."""
        tm = self.ortacha_mukofot(phi, zarba)
        r = tm[a_idx] + self.rng.normal(0.0, self.shovqin)
        return r, float(tm.max() - tm[a_idx]), float(self.cost_s[a_idx])


assert D == 6, "holat vektori o'lchami 4.2-jadvalga mos bo'lishi kerak"
