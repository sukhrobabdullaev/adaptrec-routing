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

def otish(tartib, rejim="yoq", lam=None):
    pop = Ommaboplik(s.n_item); knn = ItemKNN(s.n_item); kon = Kontent(s.matn())
    gr = P3Alpha(s.n_item); seq = Ketmaket(s.n_item)
    bandit = LinUCB(len(idx), 6, beta=BETA)
    det = [PH() for _ in range(K)]
    hist = collections.defaultdict(list); oxirgi = {}; oxirgi_t = {}
    ph_max = collections.defaultdict(lambda: np.zeros(K))
    bloklar = []; joriy = []; n_det = 0
    statik = np.zeros(len(idx)); n_st = 0
    for pos, e in enumerate(tartib):
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
            statik += ndcg; n_st += 1
            phi = holat(h, t, oxirgi_t.get(u), s.n_item)
            a = bandit.tanla(phi)
            bandit.yangilash(a, phi, float(sig[a] - C_COST * T_NARX[idx][a]))
            joriy.append(float(ndcg[a]))
            if len(joriy) >= BLOK:
                bloklar.append(float(np.mean(joriy))); joriy = []
            seg = (pos - WARM) // SEG
            for k in range(K):
                v = det[k].yangilash(float(sig[yakka[k]]))
                if v > ph_max[seg][k]: ph_max[seg][k] = v
                if lam is not None and v > lam:
                    n_det += 1
                    if rejim == "reset":
                        bandit = LinUCB(len(idx), 6, beta=BETA)
                    det[k].reset()
        knn.yangilash(u, i, t, oldingi=h); seq.yangilash(u, i, t, oxirgi=oxirgi.get(u))
        gr.yangilash(u, i, t); pop.yangilash(u, i, t)
        h.append(i); oxirgi[u] = i; oxirgi_t[u] = t
    segs = np.array([ph_max[k] for k in sorted(ph_max)])
    return np.asarray(bloklar), segs, n_det, statik / max(1, n_st)

T0 = time.time()
haqiqiy = np.arange(LIMIT)
rng = np.random.default_rng(0)
aralash = np.concatenate([np.arange(WARM), rng.permutation(np.arange(WARM, LIMIT))])

_, nol_seg, _, _ = otish(aralash, "yoq", None)
print("nol oqim: %d segment x %d komponent (%.0fs)" % (nol_seg.shape[0], K, time.time()-T0), flush=True)

def tizim(segs, lam): return float((segs.max(1) > lam).mean())
def komp(segs, lam): return float((segs > lam).mean())

lam_5 = float(np.quantile(nol_seg, 0.95))
lam_bonf = float(np.quantile(nol_seg, 1.0 - 0.05 / K))
soxta = {"lam_tuzatishsiz": round(lam_5, 4), "lam_bonferroni": round(lam_bonf, 4),
         "komponent_tuzatishsiz": round(komp(nol_seg, lam_5), 4),
         "tizim_tuzatishsiz": round(tizim(nol_seg, lam_5), 4),
         "nazariy_1_minus_095_pow_K": round(1 - 0.95 ** K, 4),
         "komponent_bonferroni": round(komp(nol_seg, lam_bonf), 4),
         "tizim_bonferroni": round(tizim(nol_seg, lam_bonf), 4),
         "n_segment": int(nol_seg.shape[0])}
print("\nkalibrlash (%d segment):" % nol_seg.shape[0], flush=True)
print("  tuzatishsiz: lam=%.3f komponent %.1f%% TIZIM %.1f%% (nazariy %.1f%%)"
      % (lam_5, 100*soxta["komponent_tuzatishsiz"], 100*soxta["tizim_tuzatishsiz"],
         100*soxta["nazariy_1_minus_095_pow_K"]), flush=True)
print("  bonferroni : lam=%.3f komponent %.1f%% TIZIM %.1f%%"
      % (lam_bonf, 100*soxta["komponent_bonferroni"], 100*soxta["tizim_bonferroni"]), flush=True)

bl_s, _, det_s, statik = otish(haqiqiy, "saqla", lam_bonf)
print("\nhaqiqiy saqla: %d aniqlash, NDCG %.4f (%.0fs)" % (det_s, bl_s.mean(), time.time()-T0), flush=True)
bl_r, _, det_r, _ = otish(haqiqiy, "reset", lam_bonf)
print("haqiqiy reset: %d aniqlash, NDCG %.4f (%.0fs)" % (det_r, bl_r.mean(), time.time()-T0), flush=True)

n = min(len(bl_s), len(bl_r))
t_st, p_t = stats.ttest_rel(bl_s[:n], bl_r[:n])
w_st, p_w = stats.wilcoxon(bl_s[:n], bl_r[:n])
jarima = (bl_s[:n].mean() / max(bl_r[:n].mean(), 1e-9) - 1) * 100
eng = int(np.argmax(statik))

oila = {"saqla vs reset (t)": float(p_t), "saqla vs reset (Wilcoxon)": float(p_w)}
nomlar = list(oila); pl = [oila[x] for x in nomlar]
tartib = np.argsort(pl); m_test = len(pl); holm = {}; oldingi = 0.0
for r_, j in enumerate(tartib):
    p_adj = min(1.0, max(oldingi, (m_test - r_) * pl[j])); oldingi = p_adj
    holm[nomlar[j]] = {"p_xom": pl[j], "p_holm": p_adj, "ahamiyatli_005": bool(p_adj < 0.05)}

out = {"limit": LIMIT, "warm": WARM, "budget": BUDGET, "blok": BLOK, "segment": SEG,
       "ph_delta": PH_DELTA, "vaqt_s": round(time.time()-T0, 1), "soxta_signal": soxta,
       "reset_vs_saqla": {"NDCG_saqla": round(float(bl_s[:n].mean()), 4),
                          "NDCG_reset": round(float(bl_r[:n].mean()), 4),
                          "reset_jarimasi_foiz": round(float(jarima), 1),
                          "n_blok": int(n), "aniqlash_saqla": int(det_s),
                          "aniqlash_reset": int(det_r)},
       "holm": holm,
       "eng_yaxshi_statik": {"toplam": list(T[idx[eng]]), "NDCG": round(float(statik[eng]), 4)}}
json.dump(out, open("natijalar/drift.json", "w"), ensure_ascii=False, indent=1)
print("\n" + json.dumps(out, ensure_ascii=False, indent=1))
