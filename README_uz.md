# 1-maqola kodlari (real ma'lumot bilan)

**Maqola:** "Adaptive Routing of Heterogeneous Recommendation Components over
E-Commerce Request Streams: A Budget-Constrained Contextual Bandit Approach
with Stream-Level Drift Detection"

Paket ikki qismdan iborat: `experiments/` sintetik tajribani (4.1-4.7),
`real/` esa Amazon Reviews 2023 oqimidagi o'lchovlarni (4.8-4.11) qayta
hosil qiladi.

## real/ tuzilishi

```
adaptreal/
  data.py          — Oqim: vg_stream.csv, item metama'lumoti, matn
  components.py    — m0 ommaboplik, m1 itemKNN, m2 kontent, m3 P3alpha,
                     m4 ketma-ket, f_asos fuziya, m5 cross-encoder
  minilm.py        — torch'siz numpy BERT forward + WordPiece tokenizator
  harness.py       — prequential (test-then-train) sikl, to'liq reyting
  router.py        — yo'l qo'yiladigan to'plamlar, LinUCB, holat vektori
  drift.py         — Page-Hinkley + Bonferroni monitor
build_stream.py    — xom JSONL dan 5-core oqim yasash
extract.py         — item matni va metama'lumotini ajratish
run_sifat.py       — sifat o'lchovi (NDCG@10 / HR@10 / MRR)
run_izolyatsiya.py — kechikish o'lchovi (izolyatsiyada, 120 holat x 7 takror)
run_profil.py      — 1-jadval (o'lchangan profillar)
run_router.py      — LinUCB marshrutlash, byudjet supurmasi
run_ceiling.py     — statik eng yaxshi to'plam va so'rov darajasidagi oracle
run_drift2.py      — null oqimda PH chegarasini kalibrlash
run_drift3.py      — reset jarimasi supurmasi (5-jadval)
make_fig5.py       — Figure 5 (real oqim: sifat-narx va shift)
natijalar/*.json   — barcha o'lchov chiqishlari
figures/a1_fig5.png
```

## Ma'lumot

Amazon Reviews 2023 (McAuley Lab, UCSD), `Video_Games` toifasi, 5-core:
857 505 hodisa, 98 906 foydalanuvchi, 26 354 item, 1999-10 dan 2023-09 gacha.
Xom fayl ochiq; `build_stream.py` uni qayta yasaydi. Barcha reyting TO'LIQ
katalog bo'yicha (namuna olingan negativlarsiz).

## Maqoladagi asosiy real raqamlar

| Komponent | NDCG@10 | HR@10 | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|
| m0 ommaboplik  | 0.0140 | 0.0274 | 0.0152 | 0.000 | 0.000 |
| m1 itemKNN     | 0.0448 | 0.0815 | 0.0438 | 0.071 | 0.296 |
| m2 kontent     | 0.0090 | 0.0175 | 0.0091 | 0.390 | 0.532 |
| m3 graf        | 0.0572 | 0.1032 | 0.0530 | 1.296 | 3.794 |
| m4 ketma-ket   | 0.1330 | 0.2564 | 0.1170 | 0.014 | 0.053 |
| f_asos fuziya  | 0.0617 | 0.1056 | 0.0574 | 1.544 | 3.467 |
| m5 cross-enc.  | 0.0427 | 0.0800 | 0.0414 | 43.385| 52.056 |

- Statik fuziya eng yaxshi yagona komponentdan **54% pastroq**.
- So'rov darajasidagi oracle 0.1903, ya'ni yana **+43%** imkoniyat;
  m4 faqat **57.3%** so'rovda eng yaxshi.
- Cross-encoder 984x qimmat va sifatni **31%** pasaytiradi;
  sabab Kendall-tau bilan ko'rsatilgan (m2 0.3495, m4 0.0045, f_asos 0.1756).
- Drift: tuzatishsiz 25.0% soxta signal, Bonferroni bilan 5.0%
  (nazariya 22.6%); himoyalanadigan chegaralarda aniqlash nol.
- Reset jarimasi +54.6% / +80.7%, Holm p < 1e-9.

Sintetik tajriba raqamlari o'zgarmagan (regret 80.0 +/- 1.8 adaptiv,
558.7 +/- 7.5 statik; log-log qiyalik 0.454; 18% -> 3.2% Bonferroni).

## Ishga tushirish

```bash
pip install numpy matplotlib
python build_stream.py          # oqimni yasash (xom JSONL kerak)
python run_sifat.py
python run_izolyatsiya.py
python run_ceiling.py
python run_drift2.py && python run_drift3.py
python make_fig5.py
```

Barcha urug'lar kodda qayd etilgan (rng seed = 7).

## Rasmlar qayerdan chiqadi

| Maqoladagi rasm | Skript |
|---|---|
| Figure 1-4 (sintetik oqim) | `experiments/make_figures.py` |
| Figure 5 (real oqim: sifat-narx va so'rov darajasidagi shift) | `real/make_fig5.py` |
