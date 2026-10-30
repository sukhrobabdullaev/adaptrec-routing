# -*- coding: utf-8 -*-
"""Baholash metrikalari (dissertatsiya, 4.2-bo'lim).

Metodologik talablar (1.4-bo'lim): vaqt bo'yicha bo'linish, element
namunalashsiz TO'LIQ reytinglash, beshta urug' va Holm tuzatishi.
"""
import numpy as np


def ndcg_at_k(reyting, nishon, k=10):
    """reyting: element indekslarining kamayish tartibidagi ro'yxati;
    nishon: haqiqiy (test) elementlar to'plami."""
    dcg = sum(1.0 / np.log2(i + 2) for i, it in enumerate(reyting[:k])
              if it in nishon)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(nishon))))
    return dcg / ideal if ideal > 0 else 0.0


def hr_at_k(reyting, nishon, k=10):
    return float(any(it in nishon for it in reyting[:k]))


def mrr(reyting, nishon):
    for i, it in enumerate(reyting):
        if it in nishon:
            return 1.0 / (i + 1)
    return 0.0


def ece(ehtimollar, natijalar, n_bin=10):
    """Kutilayotgan kalibrlash xatosi."""
    p = np.asarray(ehtimollar, float)
    y = np.asarray(natijalar, float)
    chegara = np.linspace(0, 1, n_bin + 1)
    xato, n = 0.0, len(p)
    for i in range(n_bin):
        m = (p >= chegara[i]) & (p < chegara[i + 1] if i < n_bin - 1 else p <= 1.0)
        if m.sum() == 0:
            continue
        xato += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return float(xato)


def toplangan_afsuslanish(optimal_mukofotlar, olingan_mukofotlar):
    r = np.asarray(optimal_mukofotlar, float) - np.asarray(olingan_mukofotlar, float)
    return np.cumsum(r)


def holm_tuzatish(p_qiymatlar, alpha=0.05):
    """Holm–Bonferroni: rad etilgan gipotezalar maskasi."""
    p = np.asarray(p_qiymatlar, float)
    tartib = np.argsort(p)
    m = len(p)
    rad = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(tartib):
        if p[idx] <= alpha / (m - rank):
            rad[idx] = True
        else:
            break
    return rad
