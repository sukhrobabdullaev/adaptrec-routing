# -*- coding: utf-8 -*-
"""run_sifat.py ning qayta tiklanadigan (checkpoint) varianti.
Mantiq adaptreal/harness.py dagi ishga_tushirish bilan aynan bir xil;
faqat sikl har ~140 s da dill bilan saqlanadi va keyingi chaqiruvda davom etadi."""
import sys, os, json, time, collections
import numpy as np, dill
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptreal.data import Oqim
from adaptreal.components import (Ommaboplik, ItemKNN, Kontent, P3Alpha, Ketmaket, CrossEncoderRerank)
from scipy.stats import kendalltau
LIMIT = int(os.environ.get("LIMIT", "120000")); WARM = int(os.environ.get("WARM", "40000"))
M5HAR = int(os.environ.get("M5HAR", "8")); MODEL = os.environ.get("MINILM", "/data/minilm")
CK = os.environ.get("CKPT", "sifat.ckpt"); BUDJ = float(os.environ.get("BUDJ", "140"))

def _norm(x):
    m = x.max(); return x / m if m > 0 else x

if os.path.exists(CK):
    with open(CK, "rb") as f: ST = dill.load(f)
    s = ST["s"]
else:
    s = Oqim(); print("oqim:", s.xulosa(), flush=True)
    comps = [Ommaboplik(s.n_item), ItemKNN(s.n_item), Kontent(s.matn()), P3Alpha(s.n_item), Ketmaket(s.n_item)]
    nomlar = [c.nom for c in comps] + ["f_asos", "m5_crossenc"]
    ST = dict(s=s, comps=comps, hist=collections.defaultdict(list), oxirgi={}, e=0, baholangan=0,
              met={n: dict(hr=0.0, ndcg=0.0, mrr=0.0, n=0) for n in nomlar},
              tau={"m2_kontent": [], "m4_ketmaket": [], "f_asos": []}, nomlar=nomlar)
md = s.meta
titles = [(md.get(j, ("", "", ""))[0] or "item") for j in range(s.n_item)]
m5 = CrossEncoderRerank(s.n_item, titles, MODEL, top_k=20, max_len=32, backend="onnx")
comps, hist, oxirgi, met, tau = ST["comps"], ST["hist"], ST["oxirgi"], ST["met"], ST["tau"]
knn = next(c for c in comps if c.nom == "m1_itemknn"); seq = next(c for c in comps if c.nom == "m4_ketmaket")
asos_uchun = ("m1_itemknn", "m3_graf", "m4_ketmaket")
N = min(LIMIT, len(s))

def qayd(nom, sc, hn, i):
    sc = sc.copy(); sc[hn] = -1e30
    rank = 1 + int((sc > sc[i]).sum())
    m = met[nom]; m["n"] += 1
    if rank <= 10:
        m["hr"] += 1.0; m["ndcg"] += 1.0 / np.log2(rank + 1)
    m["mrr"] += 1.0 / rank

TC = time.time()
e = ST["e"]
while e < N:
    if time.time() - TC > BUDJ:
        ST["e"] = e
        with open(CK + ".tmp", "wb") as f: dill.dump(ST, f)
        os.replace(CK + ".tmp", CK)
        print("checkpoint e=%d baholangan=%d" % (e, ST["baholangan"]), flush=True); sys.exit(3)
    u = int(s.u[e]); i = int(s.i[e]); t = int(s.t[e]); h = hist[u]
    if e >= WARM and h:
        ST["baholangan"] += 1
        hn = np.asarray(h, np.int32)
        ballar = {c.nom: c.baholash(u, h) for c in comps}
        for n_, sc in ballar.items(): qayd(n_, sc, hn, i)
        asos = sum(_norm(ballar[n_]) for n_ in asos_uchun if n_ in ballar)
        qayd("f_asos", asos, hn, i)
        if ST["baholangan"] % M5HAR == 0:
            sc5 = m5.baholash(u, h, asos); mask = sc5 != 0; birlash = asos.copy()
            if mask.any():
                top = np.flatnonzero(mask)
                birlash[top] = asos[top].max() + 1.0 + _norm(sc5[top] - sc5[top].min())
                for nom in tau:
                    ref = ballar.get(nom) if nom in ballar else asos
                    if ref is None: continue
                    tv = kendalltau(sc5[top], ref[top]).statistic
                    if tv == tv: tau[nom].append(float(tv))
            qayd("m5_crossenc", birlash, hn, i)
    knn.yangilash(u, i, t, oldingi=h); seq.yangilash(u, i, t, oxirgi=oxirgi.get(u))
    for c in comps:
        if c is knn or c is seq: continue
        c.yangilash(u, i, t)
    h.append(i); oxirgi[u] = i
    e += 1

out = {"hodisa": N, "baholangan": ST["baholangan"], "katalog": s.n_item, "m5_har": M5HAR, "komponentlar": {}}
for n_ in ST["nomlar"]:
    m = met[n_]; b = max(1, m["n"])
    out["komponentlar"][n_] = {"n_baho": m["n"], "HR@10": round(m["hr"]/b, 4), "NDCG@10": round(m["ndcg"]/b, 4), "MRR": round(m["mrr"]/b, 4)}
out["m5_kendall_tau"] = {k: (round(float(np.mean(v)), 4) if v else None) for k, v in tau.items()}
out["oqim"] = s.xulosa()
json.dump(out, open("natijalar/sifat_qayta.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
if os.path.exists(CK): os.remove(CK)
