# -*- coding: utf-8 -*-
"""Faza 3 ning qoldig'i: drift, Bonferroni va Holm statistikasi.

Chegara TAXMIN QILINMAYDI, nol gipotezada kalibrlanadi. Nol oqim
sifatida hodisalar tartibi aralashtiriladi: vaqt tuzilishi buziladi,
shuning uchun har qanday aniqlash soxta signal. Page-Hinkley
statistikasining segment bo'yicha maksimumi yozib olinadi, so'ng
chegara kerakli soxta signal darajasiga moslanadi.
"""
import sys, os, json, collections, time
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptreal.data import Oqim
from adaptreal.components import Ommaboplik, ItemKNN, Kontent, P3Alpha, Ketmaket
from adaptreal.router import toplamlar, LinUCB, holat

LIMIT = int(os.environ.get("LIMIT", "120000"))
WARM = int(os.environ.get("WARM", "40000"))
BUDGET, C_COST, BETA = 2.0, 0.010, 0.35
NOM = ["m0_pop", "m1_itemknn", "m2_kontent", "m3_graf", "m4_ketmaket"]
K = 5
SEG = 4000
BLOK = 2000
PH_DELTA = 0.0005
IZO = json.load(open("natijalar/izolyatsiya_kechikish.json"))["kechikish"]
NARX = np.array([max(IZO[n]["p50_ms"], 1e-4) for n in NOM])

s = Oqim()
T = toplamlar(K)
MASK = np.zeros((len(T), K), np.float32)
for r, A in enumerate(T): MASK[r, list(A)] = 1.0
T_NARX = MASK @ NARX
idx = np.flatnonzero(T_NARX <= BUDGET)
yakka = [[r_ for r_, v in enumerate(idx) if T[v] == (k,)][0] for k in range(K)]
print("byudjet %.1f -> %d to'plam" % (BUDGET, len(idx)), flush=True)

def norm(x):
    m = x.max(); return x / m if m > 0 else x

class PH:
    """Bir tomonlama Page-Hinkley (o'rtachaning pasayishi)."""
    def __init__(self, delta=PH_DELTA, oyna=500):
        self.delta, self.oyna = delta, oyna; self.reset()
    def reset(self):
        self.n = 0; self.mean = 0.0; self.m = 0.0; self.M = 0.0
    def yangilash(self, x):
        self.n += 1
        self.mean += (x - self.mean) / min(self.n, self.oyna)
        self.m += (self.mean - x - self.delta)
        self.M = min(self.M, self.m)
        return self.m - self.M

# ---- checkpoint qatlami (mantiq run_drift2.otish bilan aynan bir xil) ----
import dill, pickle
CKD = os.environ.get("CKDIR", "ck"); os.makedirs(CKD, exist_ok=True)
BUDJ = float(os.environ.get("BUDJ", "140")); TC = time.time()

def otish(key, tartib, rejim="yoq", lam=None):
    rf = os.path.join(CKD, "res_%s.pkl" % key); cf = os.path.join(CKD, "ck_%s.pkl" % key)
    if os.path.exists(rf):
        return pickle.load(open(rf, "rb"))
    if os.path.exists(cf):
        S_ = dill.load(open(cf, "rb"))
    else:
        S_ = dict(pop=Ommaboplik(s.n_item), knn=ItemKNN(s.n_item), kon=Kontent(s.matn()),
                  gr=P3Alpha(s.n_item), seq=Ketmaket(s.n_item), bandit=LinUCB(len(idx), 6, beta=BETA),
                  det=[PH() for _ in range(K)], hist=collections.defaultdict(list), oxirgi={}, oxirgi_t={},
                  ph_max={}, bloklar=[], joriy=[], n_det=0, statik=np.zeros(len(idx)), n_st=0, pos=0)
    pop, knn, kon, gr, seq = S_["pop"], S_["knn"], S_["kon"], S_["gr"], S_["seq"]
    det, hist, oxirgi, oxirgi_t, ph_max = S_["det"], S_["hist"], S_["oxirgi"], S_["oxirgi_t"], S_["ph_max"]
    bloklar, joriy = S_["bloklar"], S_["joriy"]
    pos = S_["pos"]
    while pos < len(tartib):
        if time.time() - TC > BUDJ:
            S_["pos"] = pos; S_["joriy"] = joriy
            dill.dump(S_, open(cf + ".tmp", "wb")); os.replace(cf + ".tmp", cf)
            print("checkpoint %s pos=%d" % (key, pos), flush=True); sys.exit(3)
        e = int(tartib[pos])
        u = int(s.u[e]); i = int(s.i[e]); t = int(s.t[e]); h = hist[u]
        if pos >= WARM and h:
            hn = np.asarray(h, np.int32)
            N = np.empty((K, s.n_item), np.float32)
            N[0] = norm(pop.baholash(u, h)); N[1] = norm(knn.baholash(u, h))
            N[2] = norm(kon.baholash(u, h)); N[3] = norm(gr.baholash(u, h))
            N[4] = norm(seq.baholash(u, h))
            S = MASK[idx] @ N; S[:, hn] = -1e30
            rank = 1 + (S > S[:, i:i+1]).sum(1)
            sig = 1.0 / np.log2(rank + 1.0)
            ndcg = np.where(rank <= 10, sig, 0.0)
            S_["statik"] += ndcg; S_["n_st"] += 1
            phi = holat(h, t, oxirgi_t.get(u), s.n_item)
            a = S_["bandit"].tanla(phi)
            S_["bandit"].yangilash(a, phi, float(sig[a] - C_COST * T_NARX[idx][a]))
            joriy.append(float(ndcg[a]))
            if len(joriy) >= BLOK:
                bloklar.append(float(np.mean(joriy))); joriy = []
            seg = (pos - WARM) // SEG
            if seg not in ph_max: ph_max[seg] = np.zeros(K)
            for k in range(K):
                v = det[k].yangilash(float(sig[yakka[k]]))
                if v > ph_max[seg][k]: ph_max[seg][k] = v
                if lam is not None and v > lam:
                    S_["n_det"] += 1
                    if rejim == "reset":
                        S_["bandit"] = LinUCB(len(idx), 6, beta=BETA)
                    det[k].reset()
        knn.yangilash(u, i, t, oldingi=h); seq.yangilash(u, i, t, oxirgi=oxirgi.get(u))
        gr.yangilash(u, i, t); pop.yangilash(u, i, t)
        h.append(i); oxirgi[u] = i; oxirgi_t[u] = t
        pos += 1
    segs = np.array([ph_max[k] for k in sorted(ph_max)])
    res = (np.asarray(bloklar), segs, S_["n_det"], S_["statik"] / max(1, S_["n_st"]))
    pickle.dump(res, open(rf, "wb"))
    if os.path.exists(cf): os.remove(cf)
    print("tugadi %s" % key, flush=True)
    return res
