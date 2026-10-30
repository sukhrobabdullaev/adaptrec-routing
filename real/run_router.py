# -*- coding: utf-8 -*-
"""Faza 3: real oqimda adaptiv marshrutlash (1-maqola, Tier B).

REPLAY protokoli. Har bir so'rovda:
  1. oltita komponentning to'liq katalog ballari hisoblanadi
  2. 63 ta to'plamning birlashtirilgan reytingi va rank'i bir vaqtda
     hisoblanadi (mask matritsasi bilan vektorlashtirilgan)
  3. har bir siyosat o'z to'plamini tanlaydi va O'LCHANGAN narxni to'laydi

Shu tarzda barcha siyosatlar bir xil so'rovlarda taqqoslanadi.
Sintetik tajribadan farqi: mukofot oracle'dan emas, kuzatilgan
reytingdan keladi (real datada oracle yo'q).
"""
import sys, os, json, collections, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptreal.data import Oqim
from adaptreal.components import (Ommaboplik, ItemKNN, Kontent, P3Alpha,
                                  Ketmaket, CrossEncoderRerank)
from adaptreal.router import toplamlar, yol_qoyiladigan, LinUCB, holat

LIMIT = int(os.environ.get("LIMIT", "90000"))
WARM = int(os.environ.get("WARM", "60000"))
SLO = float(os.environ.get("SLO", "60.0"))
C_COST = float(os.environ.get("C_COST", "0.010"))
B_PEN = float(os.environ.get("B_PEN", "2.0"))
BETA = float(os.environ.get("BETA", "0.35"))
MODEL = os.environ.get("MINILM", "/data/minilm")
BUDGETLAR = [0.05, 0.1, 0.5, 2.0, 5.0, 50.0]

# izolyatsiyada O'LCHANGAN qiymatlar (natijalar/izolyatsiya_kechikish.json)
IZO = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "natijalar", "izolyatsiya_kechikish.json")))["kechikish"]
NOM = ["m0_pop", "m1_itemknn", "m2_kontent", "m3_graf", "m4_ketmaket", "m5_crossenc"]
NARX = np.array([max(IZO[n]["p50_ms"], 1e-4) for n in NOM])
LAT = np.array([max(IZO[n]["p95_ms"], 1e-4) for n in NOM])
K = len(NOM)

s = Oqim(); md = s.meta
titles = [(md.get(j, ("", "", ""))[0] or "item") for j in range(s.n_item)]
pop = Ommaboplik(s.n_item); knn = ItemKNN(s.n_item); kon = Kontent(s.matn())
gr = P3Alpha(s.n_item); seq = Ketmaket(s.n_item)
m5 = CrossEncoderRerank(s.n_item, titles, MODEL, top_k=20, max_len=32, backend="onnx")
print("narx (p50 ms):", np.round(NARX, 4).tolist(), flush=True)
print("kechikish (p95 ms):", np.round(LAT, 3).tolist(), flush=True)

T = toplamlar(K)
MASK = np.zeros((len(T), K), np.float32)
for r, A in enumerate(T): MASK[r, list(A)] = 1.0
T_NARX = MASK @ NARX
T_LAT = np.where(MASK > 0, LAT[None, :], -np.inf).max(1)

SIYOSAT = ["eng_yaxshi_yakka", "oracle_statik", "tasodifiy", "eng_qimmat", "adaptiv"]
qurilma = {}
for b in BUDGETLAR:
    idx = [r for r in range(len(T)) if T_NARX[r] <= b and T_LAT[r] <= SLO]
    qurilma[b] = {
        "idx": np.asarray(idx, np.int32),
        "bandit": LinUCB(len(idx), 6, beta=BETA),
        "yigindi": {p: dict(q=0.0, narx=0.0, lat=0.0, slo=0, r=0.0, n=0) for p in SIYOSAT},
        "arm_r": np.zeros(len(idx)), "arm_n": np.zeros(len(idx)),
        "tanlov": collections.Counter(),
    }
    print("byudjet %-6.2f -> %d ta yo'l qo'yiladigan to'plam" % (b, len(idx)), flush=True)

def norm(x):
    m = x.max(); return x / m if m > 0 else x

rng = np.random.default_rng(0)
hist = collections.defaultdict(list); oxirgi = {}; oxirgi_t = {}
T0 = time.time(); baholangan = 0

for e in range(LIMIT):
    u = int(s.u[e]); i = int(s.i[e]); t = int(s.t[e]); h = hist[u]

    if e >= WARM and h:
        baholangan += 1
        hn = np.asarray(h, np.int32)
        N = np.empty((K, s.n_item), np.float32)
        N[0] = norm(pop.baholash(u, h)); N[1] = norm(knn.baholash(u, h))
        N[2] = norm(kon.baholash(u, h)); N[3] = norm(gr.baholash(u, h))
        N[4] = norm(seq.baholash(u, h))
        asos = N[1] + N[3] + N[4]
        sc5 = m5.baholash(u, h, asos)
        mk = sc5 != 0
        N[5] = 0.0
        if mk.any():
            top = np.flatnonzero(mk)
            N[5][top] = 1.0 + norm(sc5[top] - sc5[top].min())

        S = MASK @ N                      # (63, n_item)
        S[:, hn] = -1e30
        rank = 1 + (S > S[:, i:i+1]).sum(1)        # (63,)
        q = np.where(rank <= 10, 1.0 / np.log2(rank + 1.0), 0.0)
        phi = holat(h, t, oxirgi_t.get(u), s.n_item)

        for b in BUDGETLAR:
            g = qurilma[b]; idx = g["idx"]
            qq = q[idx]; cc = T_NARX[idx]; ll = T_LAT[idx]
            rr = qq - C_COST * cc - B_PEN * np.maximum(0.0, ll / SLO - 1.0)
            g["arm_r"] += rr; g["arm_n"] += 1.0

            tanlov = {}
            m4_pos = [k for k, r_ in enumerate(idx) if T[r_] == (4,)]
            tanlov["eng_yaxshi_yakka"] = m4_pos[0] if m4_pos else int(np.argmax(cc <= cc.min()))
            tanlov["oracle_statik"] = int(np.argmax(g["arm_r"] / np.maximum(g["arm_n"], 1)))
            tanlov["tasodifiy"] = int(rng.integers(len(idx)))
            tanlov["eng_qimmat"] = int(np.argmax(cc))
            a = g["bandit"].tanla(phi)
            tanlov["adaptiv"] = a
            g["bandit"].yangilash(a, phi, float(rr[a]))
            g["tanlov"][T[idx[a]]] += 1

            for p, k in tanlov.items():
                y = g["yigindi"][p]
                y["q"] += qq[k]; y["narx"] += cc[k]; y["lat"] += ll[k]
                y["slo"] += int(ll[k] > SLO); y["r"] += rr[k]; y["n"] += 1

    knn.yangilash(u, i, t, oldingi=h); seq.yangilash(u, i, t, oxirgi=oxirgi.get(u))
    gr.yangilash(u, i, t); pop.yangilash(u, i, t)
    h.append(i); oxirgi[u] = i; oxirgi_t[u] = t

    if e and e % 10000 == 0:
        print("  [%5.0fs] %d/%d baholangan=%d" % (time.time()-T0, e, LIMIT, baholangan), flush=True)

out = {"limit": LIMIT, "warm": WARM, "baholangan": baholangan, "slo_ms": SLO,
       "c_cost": C_COST, "b_pen": B_PEN, "beta": BETA,
       "narx_ms": dict(zip(NOM, NARX.round(4).tolist())),
       "lat_ms": dict(zip(NOM, LAT.round(3).tolist())),
       "vaqt_s": round(time.time()-T0, 1), "byudjetlar": {}}
for b in BUDGETLAR:
    g = qurilma[b]
    d = {"n_toplam": int(len(g["idx"])), "siyosatlar": {}}
    for p in SIYOSAT:
        y = g["yigindi"][p]; n = max(1, y["n"])
        d["siyosatlar"][p] = {"NDCG@10": round(y["q"]/n, 4), "narx_ms": round(y["narx"]/n, 4),
                              "lat_ms": round(y["lat"]/n, 3), "slo_buzilish": round(y["slo"]/n, 4),
                              "mukofot": round(y["r"]/n, 4)}
    d["adaptiv_top5_tanlov"] = [[list(k), v] for k, v in g["tanlov"].most_common(5)]
    out["byudjetlar"][str(b)] = d
json.dump(out, open("natijalar/router.json", "w"), ensure_ascii=False, indent=1)
print("\nsaqlandi: natijalar/router.json")
