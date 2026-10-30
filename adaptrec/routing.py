# -*- coding: utf-8 -*-
"""A1 — byudjet va kechikish cheklovlari ostida kontekstli bandit
marshrutlash (dissertatsiya, 3.1-bo'lim).

LinUCB sxemasi: har bir yo'l qo'yiladigan to'plam uchun chiziqli model
r ~ theta_A . phi saqlanadi. V_A^{-1} Sherman–Morrison formulasi bilan
inkremental yangilanadi, shuning uchun bir raundning murakkabligi
O(|A_yol| * d^2). Afsuslanish O(d * sqrt(T log(N T / delta))) bilan
chegaralanadi (3.3-teorema).

Empirik tavsiyalar (4.3-bo'lim):
  * beta_c = 0.35 (U-shakl bog'liqlikning minimumi);
  * drift signalida statistikani QISQARTIRMASLIK — faqat qadamni oshirish;
  * oflayn ilitish n0 ~ 50 kuzatuv (ortiqcha ilitish zarar keltiradi).
"""
import numpy as np


class LinUCBMarshrutlash:
    def __init__(self, n_toplam, d, beta_c=0.35, lam=0.5):
        self.nA, self.d = int(n_toplam), int(d)
        self.beta_c, self.lam = float(beta_c), float(lam)
        self.theta = np.zeros((self.nA, self.d))
        self.Vinv = np.tile(np.eye(self.d) / self.lam, (self.nA, 1, 1))
        self.b = np.zeros((self.nA, self.d))

    def tanlash(self, phi):
        """UCB qoidasi bo'yicha to'plam indeksini tanlash."""
        mu = self.theta @ phi
        s2 = np.einsum('aij,i,j->a', self.Vinv, phi, phi)
        return int(np.argmax(mu + self.beta_c * np.sqrt(np.maximum(s2, 0.0))))

    def yangilash(self, a, phi, mukofot):
        """Sherman–Morrison bilan inkremental yangilash."""
        u = self.Vinv[a] @ phi
        self.Vinv[a] -= np.outer(u, u) / (1.0 + phi @ u)
        self.b[a] += mukofot * phi
        self.theta[a] = self.Vinv[a] @ self.b[a]

    def ilitish(self, kuzatuvlar):
        """Oflayn loglardan ilitish: (a, phi, r) uchliklari ro'yxati.
        4.3-bo'lim: n0 ~ 50 da dastlabki afsuslanish 41% kamaydi."""
        for a, phi, r in kuzatuvlar:
            self.yangilash(int(a), np.asarray(phi, float), float(r))

    def drift_signali(self, kuchaytirish=3.0):
        """Drift qayd etilganda TAVSIYA ETILGAN javob: statistikani
        qisqartirish EMAS (bu empirik zarar keltirdi, 4.3-bo'lim,
        97.5 vs 127.1), balki qadam kattaligini vaqtincha oshirish."""
        self.beta_c *= float(kuchaytirish)

    def drift_signali_tugadi(self, kuchaytirish=3.0):
        self.beta_c /= float(kuchaytirish)
