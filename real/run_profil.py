# -*- coding: utf-8 -*-
"""Komponent profillarini O'LCHASH (1-maqolaning 1-jadvali uchun)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptreal.data import Oqim
from adaptreal.components import (Ommaboplik, ItemKNN, Kontent, P3Alpha,
                                  Ketmaket, CrossEncoderRerank)
from adaptreal.harness import ishga_tushirish

LIMIT = int(os.environ.get("LIMIT", "120000"))
WARM = int(os.environ.get("WARM", "40000"))
MODEL = os.environ.get("MINILM", "/data/minilm")
BACKEND = os.environ.get("BACKEND", "onnx")

s = Oqim(); print("oqim:", s.xulosa(), flush=True)
md = s.meta
titles = [(md.get(j, ("", "", ""))[0] or "item") for j in range(s.n_item)]
comps = [Ommaboplik(s.n_item), ItemKNN(s.n_item), Kontent(s.matn()),
         P3Alpha(s.n_item), Ketmaket(s.n_item),
         CrossEncoderRerank(s.n_item, titles, MODEL, top_k=20, max_len=32,
                            backend=BACKEND)]
r = ishga_tushirish(s, comps, limit=LIMIT, ilitish=WARM)
r["oqim"] = s.xulosa(); r["m5_backend"] = BACKEND
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "natijalar",
                   "komponent_profillari.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(r, open(out, "w"), ensure_ascii=False, indent=1)
print(json.dumps(r, ensure_ascii=False, indent=1))
