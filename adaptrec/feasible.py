# -*- coding: utf-8 -*-
"""Yo'l qo'yiladigan to'plamlarni filtrlash (dissertatsiya, 3.1-tasdiq).

Byudjet, kechikish va huquqiy cheklovlar to'plamlar panjarasi bo'yicha
monoton: agar A to'plam cheklovni buzsa, uning har qanday ustto'plami ham
buzadi. Shu sababli filtrlash shoxchalarni kesish bilan bajariladi.
Asosiy tajribada 31 to'plamdan 19 tasi qoladi (fazo 38,7% ga qisqaradi).
"""
from itertools import combinations
import numpy as np


def yol_qoyiladigan_toplamlar(profillar, budget, slo, ruxsat_toifalar=None):
    """Cheklovlarni qanoatlantiradigan barcha bo'sh bo'lmagan to'plamlar.

    ruxsat_toifalar: agar berilsa, faqat shu huquqiy toifalardagi
    komponentlar qatnashishi mumkin (A7 joylashtirish rejasi bilan moslik).
    """
    K = len(profillar)
    indekslar = [k for k in range(K)
                 if profillar[k].faol and
                 (ruxsat_toifalar is None or profillar[k].rho in ruxsat_toifalar)]
    natija = []
    # monotonlik bo'yicha kesish: buzuvchi to'plamning ustto'plamlari tekshirilmaydi
    buzuvchi = set()
    for r in range(1, len(indekslar) + 1):
        for c in combinations(indekslar, r):
            if any(set(b) <= set(c) for b in buzuvchi):
                continue
            narx = sum(profillar[k].gamma for k in c)
            lat = max(profillar[k].lat95 for k in c)
            if narx > budget or lat > slo:
                buzuvchi.add(c)
                continue
            natija.append(tuple(c))
    return natija


def mask_matritsa(toplamlar, K):
    """To'plamlar ro'yxatini (nA, K) mantiqiy matritsaga aylantirish —
    vektorlashtirilgan hisob-kitoblar uchun."""
    M = np.zeros((len(toplamlar), K), dtype=bool)
    for i, c in enumerate(toplamlar):
        M[i, list(c)] = True
    return M
