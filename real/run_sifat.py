# -*- coding: utf-8 -*-
"""Sifat o'lchovi: asos fuzya bilan va m5 namunasi bilan."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptreal.data import Oqim
from adaptreal.components import (Ommaboplik, ItemKNN, Kontent, P3Alpha,
                                  Ketmaket, CrossEncoderRerank)
from adaptreal.harness import ishga_tushirish

LIMIT = int(os.environ.get("LIMIT", "120000"))
WARM = int(os.environ.get("WARM", "40000"))
M5HAR = int(os.environ.get("M5HAR", "8"))
MODEL = os.environ.get("MINILM", "/data/minilm")

s = Oqim(); print("oqim:", s.xulosa(), flush=True)
md = s.meta
titles = [(md.get(j, ("", "", ""))[0] or "item") for j in range(s.n_item)]
comps = [Ommaboplik(s.n_item), ItemKNN(s.n_item), Kontent(s.matn()),
         P3Alpha(s.n_item), Ketmaket(s.n_item)]
m5 = CrossEncoderRerank(s.n_item, titles, MODEL, top_k=20, max_len=32, backend="onnx")
r = ishga_tushirish(s, comps, m5=m5, limit=LIMIT, ilitish=WARM, m5_har=M5HAR)
r["oqim"] = s.xulosa()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "natijalar", "sifat.json")
json.dump(r, open(out, "w"), ensure_ascii=False, indent=1)
print(json.dumps(r, ensure_ascii=False, indent=1))
