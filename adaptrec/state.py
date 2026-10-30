# -*- coding: utf-8 -*-
"""Holat vektori phi(u, c, t) ni shakllantirish (dissertatsiya, 2.3-bo'lim).

Holat vektori tizim darajasidagi signallardan tuziladi:
    [1, n_u, d(t), Q, yuklama, eps_qoldiq]
bunda n_u — foydalanuvchi tarixining normallashtirilgan chuqurligi,
d(t) — komponent darajasidagi drift indeksi, Q — kontent ishonchliligi,
yuklama — klaster yuklamasi, eps_qoldiq — maxfiylik byudjeti qoldig'i.
"""
import numpy as np

D = 6  # holat vektori o'lchami


def holat_vektori(n_u, drift, Q, yuklama, eps_qoldiq):
    """Bitta so'rov uchun holat vektori."""
    return np.array([1.0,
                     float(np.clip(n_u, 0, 1)),
                     float(np.clip(drift, 0, 1)),
                     float(np.clip(Q, 0, 1)),
                     float(np.clip(yuklama, 0, 1)),
                     float(np.clip(eps_qoldiq, 0, 1))])


def tarix_chuqurligi(n_interaksiya, n0=30):
    """Interaksiyalar sonini [0,1] oraliqqa normallashtirish: n/(n+n0)."""
    n = float(max(n_interaksiya, 0))
    return n / (n + n0)
