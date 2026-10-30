# -*- coding: utf-8 -*-
"""Diagnostika: per-so'rov oracle shifti bormi?

Adaptiv marshrutlash faqat eng yaxshi komponent KONTEKSTGA QARAB
o'zgarganda foyda beradi. Bu skript shu o'zgarishni bevosita o'lchaydi:
har bir so'rovda 31 ta arzon to'plamning eng yaxshisini topib,
per-so'rov oracle bilan eng yaxshi STATIK to'plamni solishtiradi.

m5 chiqarib tashlangan (u so'rov boshiga 43 ms va sifatni buzishi
allaqachon ko'rsatilgan), shuning uchun ancha ko'p so'rov qamraladi.
"""
import sys, os, json, collections, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptreal.data import Oqim
from adaptreal.components import Ommaboplik, ItemKNN, Kontent, P3Alpha, Ketmaket
from adaptreal.router import toplamlar

LIMIT = int(os.environ.get("LIMIT", "120000"))
WARM = int(os.environ.get("WARM", "40000"))
NOM = ["m0_pop", "m1_itemknn", "m2_kontent", "m3_graf", "m4_ketmaket"]
K = 5

s = Oqim()
pop = Ommaboplik(s.n_item); knn = ItemKNN(s.n_item); kon = Kontent(s.matn())
gr = P3Alpha(s.n_item); seq = Ketmaket(s.n_item)
T = toplamlar(K)
MASK = np.zeros((len(T), K), np.float32)
for r, A in enumerate(T): MASK[r, list(A)] = 1.0
print("to'plamlar:", len(T), flush=True)

def norm(x):
    m = x.max(); return x / m if m > 0 else x

hist = collections.defaultdict(list); oxirgi = {}
q_yig = np.zeros(len(T)); n = 0
oracle_q = 0.0
eng_yaxshi_kim = collections.Counter()
T0 = time.time()

for e in range(LIMIT):
    u = int(s.u[e]); i = int(s.i[e]); t = int(s.t[e]); h = hist[u]
    if e >= WARM and h:
        hn = np.asarray(h, np.int32)
        N = np.empty((K, s.n_item), np.float32)
        N[0] = norm(pop.baholash(u, h)); N[1] = norm(knn.baholash(u, h))
        N[2] = norm(kon.baholash(u, h)); N[3] = norm(gr.baholash(u, h))
        N[4] = norm(seq.baholash(u, h))
        S = MASK @ N; S[:, hn] = -1e30
        rank = 1 + (S > S[:, i:i+1]).sum(1)
        q = np.where(rank <= 10, 1.0 / np.log2(rank + 1.0), 0.0)
        q_yig += q; n += 1
        oracle_q += q.max()
        best = np.flatnonzero(q == q.max())
        # nol bo'lmagan holatlarda kim yutdi
        if q.max() > 0:
            eng_yaxshi_kim[T[int(best[0])]] += 1
    knn.yangilash(u, i, t, oldingi=h); seq.yangilash(u, i, t, oxirgi=oxirgi.get(u))
    gr.yangilash(u, i, t); pop.yangilash(u, i, t)
    h.append(i); oxirgi[u] = i
    if e and e % 25000 == 0:
        print("  [%4.0fs] %d/%d n=%d" % (time.time()-T0, e, LIMIT, n), flush=True)

statik = q_yig / max(1, n)
eng_statik = int(np.argmax(statik))
out = {"n": n,
       "eng_yaxshi_statik_toplam": list(T[eng_statik]),
       "eng_yaxshi_statik_NDCG": round(float(statik[eng_statik]), 4),
       "per_sorov_oracle_NDCG": round(float(oracle_q / max(1, n)), 4),
       "shift_foiz": round(float(oracle_q/max(1,n) / statik[eng_statik] - 1) * 100, 1),
       "top10_statik": [[list(T[j]), round(float(statik[j]), 4)]
                        for j in np.argsort(-statik)[:10]],
       "per_sorov_eng_yaxshi_ulush": [[list(k), v, round(v/max(1,sum(eng_yaxshi_kim.values()))*100, 1)]
                                      for k, v in eng_yaxshi_kim.most_common(8)]}
json.dump(out, open("natijalar/ceiling.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
