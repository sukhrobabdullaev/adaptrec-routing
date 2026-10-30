# -*- coding: utf-8 -*-
"""Kechikishni IZOLYATSIYADA o'lchash (1-maqolaning 1-jadvali uchun).

Nega alohida skript: barcha komponentlarni bitta sikl ichida o'lchaganda
ular 2 yadro uchun raqobatlashadi va bir-birining kechikishini
buzadi (o'lchovda m2 ning p95 si 0,485 ms dan 8,0 ms ga sakragan edi).
Bu yerda avval holat qurib olinadi, keyin har bir komponent BOSHQALAR
ISHLAMAYOTGAN paytda, takrorlar bilan o'lchanadi.
"""
import sys, os, json, time, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptreal.data import Oqim
from adaptreal.components import (Ommaboplik, ItemKNN, Kontent, P3Alpha,
                                  Ketmaket, CrossEncoderRerank, Fuzya)

WARM = int(os.environ.get("WARM", "100000"))
NAMUNA = int(os.environ.get("NAMUNA", "120"))
TAKROR = int(os.environ.get("TAKROR", "7"))
MODEL = os.environ.get("MINILM", "/data/minilm")

s = Oqim(); md = s.meta
titles = [(md.get(j, ("", "", ""))[0] or "item") for j in range(s.n_item)]
pop = Ommaboplik(s.n_item); knn = ItemKNN(s.n_item); kon = Kontent(s.matn())
gr = P3Alpha(s.n_item); seq = Ketmaket(s.n_item)
m5 = CrossEncoderRerank(s.n_item, titles, MODEL, top_k=20, max_len=32, backend="onnx")
asos = Fuzya(s.n_item, [knn, gr, seq])

print("holat qurilmoqda (%d hodisa)..." % WARM, flush=True)
hist = collections.defaultdict(list); oxirgi = {}
for e in range(WARM):
    u = int(s.u[e]); i = int(s.i[e]); h = hist[u]
    knn.yangilash(u, i, 0, oldingi=h); seq.yangilash(u, i, 0, oxirgi=oxirgi.get(u))
    gr.yangilash(u, i, 0); pop.yangilash(u, i, 0)
    h.append(i); oxirgi[u] = i
print("holat tayyor", flush=True)

# namuna nuqtalari: tarixi bor foydalanuvchilar
rng = np.random.default_rng(0)
nuqtalar = []
e = WARM
while len(nuqtalar) < NAMUNA and e < len(s):
    u = int(s.u[e])
    if hist[u]: nuqtalar.append((u, list(hist[u])))
    e += 1

komponentlar = [pop, knn, kon, gr, seq]
natija = {}

def o_lchov(nom, fn):
    fn(*nuqtalar[0])                       # ilitish
    lat = []
    for u, h in nuqtalar:
        for _ in range(TAKROR):
            t0 = time.perf_counter(); fn(u, h); lat.append((time.perf_counter()-t0)*1000)
    L = np.asarray(lat)
    natija[nom] = {"n": len(L),
                   "p50_ms": round(float(np.percentile(L, 50)), 4),
                   "p95_ms": round(float(np.percentile(L, 95)), 4),
                   "p99_ms": round(float(np.percentile(L, 99)), 4),
                   "ortacha_ms": round(float(L.mean()), 4)}
    print("  %-14s p50=%8.3f  p95=%8.3f  p99=%8.3f ms" %
          (nom, natija[nom]["p50_ms"], natija[nom]["p95_ms"], natija[nom]["p99_ms"]), flush=True)

print("\nizolyatsiyada o'lchash (%d nuqta x %d takror):" % (len(nuqtalar), TAKROR), flush=True)
for c in komponentlar:
    o_lchov(c.nom, lambda u, h, c=c: c.baholash(u, h))
o_lchov("f_asos", lambda u, h: asos.baholash(u, h))

# m5 uchun asos ball kerak
asos_kesh = {}
def m5_fn(u, h):
    k = (u, len(h))
    if k not in asos_kesh: asos_kesh[k] = asos.baholash(u, h)
    return m5.baholash(u, h, asos_kesh[k])
o_lchov("m5_crossenc", m5_fn)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "natijalar",
                   "izolyatsiya_kechikish.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({"warm": WARM, "namuna": len(nuqtalar), "takror": TAKROR,
           "yadro": os.cpu_count(), "kechikish": natija},
          open(out, "w"), ensure_ascii=False, indent=1)
print("\nsaqlandi:", out)
