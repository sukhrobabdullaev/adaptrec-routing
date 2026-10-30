# -*- coding: utf-8 -*-
"""Prequential (test-then-train) baholash, to'liq reyting bilan.

Ikkita muhim tafsilot:

1. ASOS. m5 (cross-encoder) butun katalogni baholay olmaydi, u
   re-ranking bosqichi va arzon komponentlar fuzyasining top-K sini
   qayta tartiblaydi. Shuning uchun `f_asos` alohida komponent sifatida
   o'lchanadi: m5 ni faqat shunga nisbatan baholash mantiqiy.

2. NAMUNA. m5 so'rov boshiga ~45 ms turadi, ya'ni to'liq oqimda
   soatlar ketadi. U har `m5_har` so'rovda bir marta baholanadi va
   o'z hisoblagichi bo'yicha o'rtachalanadi. Kechikish bu yerda
   O'LCHANMAYDI (komponentlar bir-biriga xalaqit beradi) —
   buning uchun run_izolyatsiya.py bor.
"""
import numpy as np, collections, time


def _norm(x):
    m = x.max()
    return x / m if m > 0 else x


def ishga_tushirish(oqim, komponentlar, m5=None, asos_uchun=("m1_itemknn", "m3_graf", "m4_ketmaket"),
                    limit=None, ilitish=40000, m5_har=8, log_har=25000):
    hist = collections.defaultdict(list)
    oxirgi = {}
    N = len(oqim) if limit is None else min(limit, len(oqim))
    knn = next((c for c in komponentlar if c.nom == "m1_itemknn"), None)
    seq = next((c for c in komponentlar if c.nom == "m4_ketmaket"), None)

    nomlar = [c.nom for c in komponentlar] + ["f_asos"] + (["m5_crossenc"] if m5 else [])
    met = {n: dict(hr=0.0, ndcg=0.0, mrr=0.0, n=0) for n in nomlar}
    # m5 ning mazmunga moyilligini tekshirish uchun: top-20 ichida
    # m5 tartibi qaysi arzon komponentga yaqinroq?
    tau = {"m2_kontent": [], "m4_ketmaket": [], "f_asos": []}
    baholangan = 0
    T0 = time.time()

    def qayd(nom, sc, hn, i):
        sc = sc.copy(); sc[hn] = -1e30
        rank = 1 + int((sc > sc[i]).sum())
        m = met[nom]; m["n"] += 1
        if rank <= 10:
            m["hr"] += 1.0
            m["ndcg"] += 1.0 / np.log2(rank + 1)
        m["mrr"] += 1.0 / rank

    for e in range(N):
        u = int(oqim.u[e]); i = int(oqim.i[e]); t = int(oqim.t[e])
        h = hist[u]

        if e >= ilitish and h:
            baholangan += 1
            hn = np.asarray(h, np.int32)
            ballar = {c.nom: c.baholash(u, h) for c in komponentlar}
            for n, sc in ballar.items():
                qayd(n, sc, hn, i)

            asos = sum(_norm(ballar[n]) for n in asos_uchun if n in ballar)
            qayd("f_asos", asos, hn, i)

            if m5 is not None and baholangan % m5_har == 0:
                sc5 = m5.baholash(u, h, asos)
                mask = sc5 != 0
                birlash = asos.copy()
                if mask.any():
                    top = np.flatnonzero(mask)
                    birlash[top] = asos[top].max() + 1.0 + _norm(sc5[top] - sc5[top].min())
                    # Kendall-tau: m5 tartibi qaysi signalga yaqin?
                    from scipy.stats import kendalltau
                    for nom in tau:
                        ref = ballar.get(nom) if nom in ballar else asos
                        if ref is None: continue
                        tv = kendalltau(sc5[top], ref[top]).statistic
                        if tv == tv: tau[nom].append(float(tv))
                qayd("m5_crossenc", birlash, hn, i)

        if knn is not None: knn.yangilash(u, i, t, oldingi=h)
        if seq is not None: seq.yangilash(u, i, t, oxirgi=oxirgi.get(u))
        for c in komponentlar:
            if c is knn or c is seq: continue
            c.yangilash(u, i, t)
        h.append(i); oxirgi[u] = i

        if log_har and e and e % log_har == 0:
            print("  [%5.0fs] %d/%d baholangan=%d" % (time.time()-T0, e, N, baholangan), flush=True)

    out = {"hodisa": N, "baholangan": baholangan, "katalog": oqim.n_item,
           "m5_har": m5_har, "vaqt_s": round(time.time() - T0, 1), "komponentlar": {}}
    for n in nomlar:
        m = met[n]; b = max(1, m["n"])
        out["komponentlar"][n] = {
            "n_baho": m["n"],
            "HR@10": round(m["hr"] / b, 4),
            "NDCG@10": round(m["ndcg"] / b, 4),
            "MRR": round(m["mrr"] / b, 4)}
    out["m5_kendall_tau"] = {k: (round(float(np.mean(v)), 4) if v else None) for k, v in tau.items()}
    return out
