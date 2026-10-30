# -*- coding: utf-8 -*-
"""Baho darajasida birlashtirish (dissertatsiya, 2.2-bo'lim).

Komponent baholari avval taqqoslanadigan shkala keltiriladi
(z-normallashtirish yoki min-max), so'ng holatga bog'liq vaznlar bilan
birlashtiriladi:
    s(u, i) = sum_k [k in A] * w_k(phi) * s~_k(u, i)
Statik gibrid — w_k(phi) = const xususiy holati; demak to'g'ri
o'qitilgan adaptiv model statikdan yomon ishlamaydi (2.1-tasdiq).
"""
import numpy as np


def z_normallash(s):
    """Komponent bahosini nomzodlar to'plami bo'yicha z-normallashtirish."""
    s = np.asarray(s, float)
    std = s.std()
    return (s - s.mean()) / std if std > 1e-12 else np.zeros_like(s)


def minmax_normallash(s):
    s = np.asarray(s, float)
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(s)


def softmax_vazn(logits):
    z = np.asarray(logits, float)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def birlashtirish(baholar, faol_toplam, vaznlar, normalizator=z_normallash):
    """
    baholar:     dict {k: (n_nomzod,) massiv} — komponent baholari.
    faol_toplam: A1 tanlagan komponent indekslari.
    vaznlar:     (K,) — w_k(phi), faqat faol komponentlarniki ishlatiladi.
    """
    n = len(next(iter(baholar.values())))
    yig = np.zeros(n)
    w_sum = 0.0
    for k in faol_toplam:
        yig += vaznlar[k] * normalizator(baholar[k])
        w_sum += vaznlar[k]
    return yig / w_sum if w_sum > 1e-12 else yig
