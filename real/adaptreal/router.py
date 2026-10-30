# -*- coding: utf-8 -*-
"""Byudjet cheklovi ostida komponent to'plamini tanlash (1-maqola, 3.3-3.4).

Bu modul sintetik `adaptrec/routing.py` ning real datadagi ekvivalenti.
Farqi: mukofot oracle'dan emas, kuzatilgan reytingdan olinadi.

Baholash protokoli REPLAY: har bir so'rovda barcha komponent ballari
hisoblanadi, so'ng har bir siyosat o'z tanlagan to'plamini shu ballardan
yig'adi va O'LCHANGAN narxni to'laydi. Bu siyosatlarni bir xil
so'rovlarda taqqoslash imkonini beradi (aks holda counterfactual
taqqoslash mumkin emas).
"""
import numpy as np
from itertools import combinations


def toplamlar(K):
    """Bo'sh bo'lmagan barcha to'plamlar (2^K - 1)."""
    return [c for r in range(1, K + 1) for c in combinations(range(K), r)]


def yol_qoyiladigan(toplam_royxati, narx, lat, budget, slo):
    """Byudjet va SLO ostida yo'l qo'yiladigan to'plamlar.

    Cheklovlar to'plam panjarasida monoton (Taklif 1): narx additiv,
    kechikish maksimum, ikkalasi ham qo'shilish bilan kamaymaydi.
    """
    out = []
    for A in toplam_royxati:
        c = sum(narx[k] for k in A)
        l = max(lat[k] for k in A)
        if c <= budget and l <= slo:
            out.append(A)
    return out


class LinUCB:
    """Yo'l qo'yiladigan to'plamlar ustida kontekstli bandit (3.4-bo'lim)."""

    def __init__(self, n_arm, d, beta=0.35, lam=1.0):
        self.A_inv = np.array([np.eye(d) / lam for _ in range(n_arm)])
        self.b = np.zeros((n_arm, d), np.float64)
        self.theta = np.zeros((n_arm, d), np.float64)
        self.beta = beta

    def tanla(self, phi):
        p = self.theta @ phi
        # Sherman-Morrison bilan saqlangan A_inv dan ishonch radiusi
        u = np.einsum("aij,j->ai", self.A_inv, phi)
        rad = np.sqrt(np.maximum(np.einsum("ai,i->a", u, phi), 0.0))
        return int(np.argmax(p + self.beta * rad))

    def yangilash(self, a, phi, r):
        Ai = self.A_inv[a]
        u = Ai @ phi
        self.A_inv[a] = Ai - np.outer(u, u) / (1.0 + phi @ u)
        self.b[a] += r * phi
        self.theta[a] = self.A_inv[a] @ self.b[a]


def holat(hist, t, oxirgi_t, n_item):
    """Oqim darajasidagi holat vektori (3.1-bo'lim).

    Barcha signallar so'rov vaqtida mavjud: tarix chuqurligi, oxirgi
    hodisadan o'tgan vaqt, faollik.
    """
    n = len(hist)
    dt = 0.0 if oxirgi_t is None else min(np.log1p(max(0, t - oxirgi_t) / 3.6e6) / 10.0, 1.0)
    return np.array([
        1.0,
        min(n / 20.0, 1.0),
        1.0 if n <= 2 else 0.0,          # sovuq start
        dt,
        min(len(set(hist[-10:])) / 10.0, 1.0),
        1.0 if n > 10 else 0.0,
    ], np.float64)
