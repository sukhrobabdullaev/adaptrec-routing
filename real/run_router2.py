# -*- coding: utf-8 -*-
"""Router, mukofot signali variantlari bilan (m5 siz, tez).

Gipoteza: LinUCB NDCG@10 mukofoti bilan o'rgana olmaydi, chunki u
so'rovlarning ~80% ida NOL. MRR (1/rank) esa har doim nolmas va ancha
zichroq signal beradi. Bu skript ikkalasini bir xil oqimda solishtiradi.
"""
import sys, os, json, collections, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptreal.data import Oqim
from adaptreal.components import Ommaboplik, ItemKNN, Kontent, P3Alpha, Ketmaket
from adaptreal.router import toplamlar, LinUCB, holat

LIMIT = int(os.environ.get("LIMIT", "120000"))
WARM = int(os.environ.get("WARM", "40000"))
C_COST = float(os.environ.get("C_COST", "0.010"))
BETA = float(os.environ.get("BETA", "0.35"))
BUDGET = float(os.environ.get("BUDGET", "2.0"))
IZO = json.load(open("natijalar/izolyatsiya_kechikish.json"))["kechikish"]
NOM = ["m0_pop", "m1_itemknn", "m2_kontent", "m3_graf", "m4_ketmaket"]
NARX = np.array([max(IZO[n]["p50_ms"], 1e-4) for n in NOM]); K = 5

s = Oqim()
pop = Ommaboplik(s.n_item); knn = ItemKNN(s.n_item); kon = Kontent(s.matn())
gr = P3Alpha(s.n_item); seq = Ketmaket(s.n_item)
T = toplamlar(K)
MASK = np.zeros((len(T), K), np.float32)
for r, A in enumerate(T): MASK[r, list(A)] = 1.0
T_NARX = MASK @ NARX
idx = np.flatnonzero(T_NARX <= BUDGET)
print("byudjet %.1f ms -> %d to'plam" % (BUDGET, len(idx)), flush=True)

MUKOFOTLAR = ["ndcg10", "mrr", "mrr_log"]
qur = {m: {"bandit": LinUCB(len(idx), 6, beta=BETA),
           "q": 0.0, "narx": 0.0, "n": 0, "tanlov": collections.Counter()}
       for m in MUKOFOTLAR}
statik_q = np.zeros(len(idx)); statik_n = 0; oracle = 0.0

def norm(x):
    m = x.max(); return x / m if m > 0 else x

hist = collections.defaultdict(list); oxirgi = {}; oxirgi_t = {}
T0 = time.time()
for e in range(LIMIT):
    u = int(s.u[e]); i = int(s.i[e]); t = int(s.t[e]); h = hist[u]
    if e >= WARM and h:
        hn = np.asarray(h, np.int32)
        N = np.empty((K, s.n_item), np.float32)
        N[0] = norm(pop.baholash(u, h)); N[1] = norm(knn.baholash(u, h))
        N[2] = norm(kon.baholash(u, h)); N[3] = norm(gr.baholash(u, h))
        N[4] = norm(seq.baholash(u, h))
        S = (MASK[idx] @ N); S[:, hn] = -1e30
        rank = 1 + (S > S[:, i:i+1]).sum(1)
        ndcg = np.where(rank <= 10, 1.0 / np.log2(rank + 1.0), 0.0)
        mrr = 1.0 / rank
        mrr_log = 1.0 / np.log2(rank + 1.0)          # har doim nolmas, NDCG shakli
        cc = T_NARX[idx]
        phi = holat(h, t, oxirgi_t.get(u), s.n_item)
        statik_q += ndcg; statik_n += 1; oracle += ndcg.max()
        for m, sig in (("ndcg10", ndcg), ("mrr", mrr), ("mrr_log", mrr_log)):
            g = qur[m]
            r = sig - C_COST * cc
            a = g["bandit"].tanla(phi)
            g["bandit"].yangilash(a, phi, float(r[a]))
            g["q"] += ndcg[a]; g["narx"] += cc[a]; g["n"] += 1
            g["tanlov"][T[idx[a]]] += 1
    knn.yangilash(u, i, t, oldingi=h); seq.yangilash(u, i, t, oxirgi=oxirgi.get(u))
    gr.yangilash(u, i, t); pop.yangilash(u, i, t)
    h.append(i); oxirgi[u] = i; oxirgi_t[u] = t
    if e and e % 40000 == 0: print("  [%4.0fs] %d/%d" % (time.time()-T0, e, LIMIT), flush=True)

st = statik_q / max(1, statik_n)
best = int(np.argmax(st))
out = {"n": statik_n, "byudjet": BUDGET, "n_toplam": int(len(idx)),
       "eng_yaxshi_statik": {"toplam": list(T[idx[best]]), "NDCG@10": round(float(st[best]), 4),
                             "narx_ms": round(float(T_NARX[idx[best]]), 4)},
       "per_sorov_oracle_NDCG": round(float(oracle / max(1, statik_n)), 4),
       "mukofot_variantlari": {}}
for m in MUKOFOTLAR:
    g = qur[m]; n = max(1, g["n"])
    out["mukofot_variantlari"][m] = {
        "NDCG@10": round(g["q"] / n, 4), "narx_ms": round(g["narx"] / n, 4),
        "statikka_nisbatan_foiz": round((g["q"]/n) / st[best] * 100 - 100, 1),
        "top3_tanlov": [[list(k), v] for k, v in g["tanlov"].most_common(3)]}
json.dump(out, open("natijalar/router_mukofot.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
