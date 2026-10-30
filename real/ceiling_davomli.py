# -*- coding: utf-8 -*-
"""run_ceiling.py ning checkpoint varianti (mantiq aynan bir xil)."""
import sys, os, json, collections, time
import numpy as np, dill
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptreal.data import Oqim
from adaptreal.components import Ommaboplik, ItemKNN, Kontent, P3Alpha, Ketmaket
from adaptreal.router import toplamlar
LIMIT = int(os.environ.get("LIMIT", "120000")); WARM = int(os.environ.get("WARM", "40000"))
CK = os.environ.get("CKPT", "ceiling.ckpt"); BUDJ = float(os.environ.get("BUDJ", "140"))
K = 5
T = toplamlar(K)
MASK = np.zeros((len(T), K), np.float32)
for r, A in enumerate(T): MASK[r, list(A)] = 1.0
def norm(x):
    m = x.max(); return x / m if m > 0 else x
if os.path.exists(CK):
    with open(CK, "rb") as f: ST = dill.load(f)
else:
    s = Oqim()
    ST = dict(s=s, pop=Ommaboplik(s.n_item), knn=ItemKNN(s.n_item), kon=Kontent(s.matn()), gr=P3Alpha(s.n_item),
              seq=Ketmaket(s.n_item), hist=collections.defaultdict(list), oxirgi={}, q_yig=np.zeros(len(T)), n=0,
              oracle_q=0.0, kim=collections.Counter(), e=0)
s, pop, knn, kon, gr, seq = ST["s"], ST["pop"], ST["knn"], ST["kon"], ST["gr"], ST["seq"]
hist, oxirgi, kim = ST["hist"], ST["oxirgi"], ST["kim"]
TC = time.time(); e = ST["e"]
while e < LIMIT:
    if time.time() - TC > BUDJ:
        ST["e"] = e
        with open(CK + ".tmp", "wb") as f: dill.dump(ST, f)
        os.replace(CK + ".tmp", CK); print("checkpoint e=%d n=%d" % (e, ST["n"]), flush=True); sys.exit(3)
    u = int(s.u[e]); i = int(s.i[e]); t = int(s.t[e]); h = hist[u]
    if e >= WARM and h:
        hn = np.asarray(h, np.int32)
        N = np.empty((K, s.n_item), np.float32)
        N[0] = norm(pop.baholash(u, h)); N[1] = norm(knn.baholash(u, h))
        N[2] = norm(kon.baholash(u, h)); N[3] = norm(gr.baholash(u, h)); N[4] = norm(seq.baholash(u, h))
        S = MASK @ N; S[:, hn] = -1e30
        rank = 1 + (S > S[:, i:i+1]).sum(1)
        q = np.where(rank <= 10, 1.0 / np.log2(rank + 1.0), 0.0)
        ST["q_yig"] += q; ST["n"] += 1; ST["oracle_q"] += q.max()
        best = np.flatnonzero(q == q.max())
        if q.max() > 0: kim[T[int(best[0])]] += 1
    knn.yangilash(u, i, t, oldingi=h); seq.yangilash(u, i, t, oxirgi=oxirgi.get(u))
    gr.yangilash(u, i, t); pop.yangilash(u, i, t)
    h.append(i); oxirgi[u] = i
    e += 1
n = ST["n"]; statik = ST["q_yig"] / max(1, n); es = int(np.argmax(statik)); oq = ST["oracle_q"]
out = {"n": n, "eng_yaxshi_statik_toplam": list(T[es]), "eng_yaxshi_statik_NDCG": round(float(statik[es]), 4),
       "per_sorov_oracle_NDCG": round(float(oq / max(1, n)), 4),
       "shift_foiz": round(float(oq/max(1,n) / statik[es] - 1) * 100, 1),
       "top10_statik": [[list(T[j]), round(float(statik[j]), 4)] for j in np.argsort(-statik)[:10]],
       "per_sorov_eng_yaxshi_ulush": [[list(k), v, round(v/max(1,sum(kim.values()))*100, 1)] for k, v in kim.most_common(8)]}
json.dump(out, open("natijalar/ceiling_qayta.json", "w"), ensure_ascii=False, indent=1)
print("TAYYOR")
if os.path.exists(CK): os.remove(CK)
