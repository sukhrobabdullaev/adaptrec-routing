# -*- coding: utf-8 -*-
"""1-maqola kodlari: so'rovlar oqimida byudjet cheklovli kontekstli bandit
marshrutlash va komponent darajasidagi driftni aniqlash.

Modullar:
    profiles   — komponent profillari pi_k = (alpha, gamma, lambda, rho)
    state      — holat vektori phi(u, c, t)
    feasible   — yo'l qo'yiladigan to'plamlarni monotonlik bo'yicha filtrlash
    routing    — LinUCB marshrutlagich (Sherman–Morrison yangilanishi)
    drift      — Page–Hinkley + Bonferroni, Kendall-tau drift indeksi
    fusion     — baho darajasida birlashtirish
    metrics    — jamg'arilgan yo'qotish (regret), NDCG/HR/MRR/ECE, Holm
    simulator  — nazorat qilinadigan so'rovlar oqimi muhiti
"""
__version__ = "1.0.0"

from . import (profiles, state, feasible, routing, drift, fusion,
               metrics, simulator)  # noqa: F401
