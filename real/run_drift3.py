# -*- coding: utf-8 -*-
"""Reset jarimasi chegara funksiyasi sifatida.

Bonferroni bilan kalibrlangan chegarada real Amazon oqimida
aniqlash umuman bo'lmadi, ya'ni bu oqimda Page-Hinkley tutadigan
turdagi keskin drift yo'q. Nol natijani foydali qilish uchun chegara
pasaytirilib, "detektor ishlaganda reset zararlimi?" degan savol
bir necha ish nuqtasida o'lchanadi.

Muhim tejamkorlik: 'saqla' rejimi chegaraga UMUMAN bog'liq emas
(aniqlashlar yoziladi, lekin hech narsa o'zgarmaydi), shuning uchun
u bir marta ishga tushiriladi.
"""
import sys, os, json, collections, time
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_drift2 import (otish, aralash, haqiqiy, nol_seg, s, T, idx, K,
                        lam_bonf, lam_5, soxta, BLOK)

T0 = time.time()
KVANT = [0.95, 0.80, 0.60, 0.40, 0.20]
LAMS = [float(np.quantile(nol_seg, q)) for q in KVANT]
print("chegara nomzodlari:", [round(x, 2) for x in LAMS], flush=True)

# saqla bir marta (chegaradan mustaqil)
bl_s, _, _, statik = otish(haqiqiy, "saqla", LAMS[-1])
print("saqla: NDCG %.4f (%.0fs)" % (bl_s.mean(), time.time()-T0), flush=True)

qatorlar = []
for q, lam in zip(KVANT, LAMS):
    bl_r, _, det_r, _ = otish(haqiqiy, "reset", lam)
    n = min(len(bl_s), len(bl_r))
    a, b = bl_s[:n], bl_r[:n]
    jar = (a.mean() / max(b.mean(), 1e-12) - 1) * 100
    if np.allclose(a, b):
        p_t = p_w = float("nan")
    else:
        p_t = float(stats.ttest_rel(a, b).pvalue)
        p_w = float(stats.wilcoxon(a, b).pvalue)
    qatorlar.append({"nol_kvantil": q, "lam": round(lam, 3), "aniqlash": int(det_r),
                     "NDCG_saqla": round(float(a.mean()), 4),
                     "NDCG_reset": round(float(b.mean()), 4),
                     "reset_jarimasi_foiz": round(float(jar), 2),
                     "p_t": p_t, "p_wilcoxon": p_w, "n_blok": int(n)})
    print("  kvantil %.2f lam=%.1f -> %4d aniqlash, saqla %.4f reset %.4f, jarima %+.2f%%"
          % (q, lam, det_r, a.mean(), b.mean(), jar), flush=True)

# Holm: faqat aniqlash bo'lgan va NaN bo'lmagan testlar oilasi
oila = {}
for r in qatorlar:
    if r["aniqlash"] > 0 and r["p_t"] == r["p_t"]:
        oila["lam=%.1f (t)" % r["lam"]] = r["p_t"]
        oila["lam=%.1f (W)" % r["lam"]] = r["p_wilcoxon"]
holm = {}
if oila:
    nomlar = list(oila); pl = [oila[x] for x in nomlar]
    tartib = np.argsort(pl); m = len(pl); oldingi = 0.0
    for r_, j in enumerate(tartib):
        p_adj = min(1.0, max(oldingi, (m - r_) * pl[j])); oldingi = p_adj
        holm[nomlar[j]] = {"p_xom": float(pl[j]), "p_holm": float(p_adj),
                           "ahamiyatli_005": bool(p_adj < 0.05)}

eng = int(np.argmax(statik))
out = {"soxta_signal": soxta, "blok": BLOK,
       "bonferroni_lam": round(lam_bonf, 3),
       "bonferroni_aniqlash_haqiqiy_oqimda": 0,
       "izoh": ("Bonferroni bilan kalibrlangan chegarada real oqimda aniqlash bo'lmadi. "
                "Quyidagi jadval chegara pasaytirilganda reset jarimasi qanday "
                "o'zgarishini ko'rsatadi."),
       "chegara_sweep": qatorlar, "holm": holm,
       "eng_yaxshi_statik": {"toplam": list(T[idx[eng]]), "NDCG": round(float(statik[eng]), 4)},
       "vaqt_s": round(time.time()-T0, 1)}
json.dump(out, open("natijalar/drift.json", "w"), ensure_ascii=False, indent=1)
print("\nsaqlandi: natijalar/drift.json", flush=True)
