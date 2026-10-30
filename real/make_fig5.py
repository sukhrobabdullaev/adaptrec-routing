# -*- coding: utf-8 -*-
"""1-maqola, Figure 5: real oqimdagi sifat-narx chegarasi va so'rov darajasidagi shift.

(a) NDCG@10 ga qarshi p95 kechikish (izolyatsiyada o'lchangan), har komponent
    uchun bitta nuqta; so'rov darajasidagi oracle gorizontal chiziq.
(b) Har so'rovda qaysi to'plam eng yaxshi bo'lgani (ulush), plus statik
    eng yaxshi / statik fuziya / oracle ustunlari.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
N = os.path.join(ROOT, "natijalar")
T = json.load(open(os.path.join(N, "jadval1_olchangan.json")))
C = json.load(open(os.path.join(N, "ceiling.json")))

C1, C2, C3 = "#0072B2", "#D55E00", "#009E73"
INK, MUTED, GRID = "#1A1A1A", "#5A5A5A", "#D9D9D9"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 200,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight"})

NOM = {"m0_pop": "$m_0$ popularity", "m1_itemknn": "$m_1$ item-kNN",
       "m2_kontent": "$m_2$ content", "m3_graf": "$m_3$ graph",
       "m4_ketmaket": "$m_4$ sequential", "f_asos": "static fusion",
       "m5_crossenc": "$m_5$ cross-encoder"}
fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 3.0),
                           gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.33})

# --- (a) sifat / kechikish -------------------------------------------
oracle = C["per_sorov_oracle_NDCG"]
a.axhline(oracle, color=C3, lw=1.2, ls="--")
a.text(0.00048, oracle * 1.04, f"per-request ceiling  {oracle:.4f}",
       fontsize=7.8, color=C3)
OFS = {"m0_pop": (1.6, 1.00, "left", "center"),
       "m1_itemknn": (0.62, 0.86, "right", "center"),
       "m2_kontent": (1.6, 1.00, "left", "center"),
       "m3_graf": (0.42, 0.94, "right", "center"),
       "m4_ketmaket": (1.7, 1.00, "left", "center"),
       "f_asos": (1.0, 1.42, "center", "center"),
       "m5_crossenc": (1.0, 0.70, "center", "center")}
for q in T["qatorlar"]:
    x = max(q["p95_ms"], 5e-4); y = q["NDCG10"]
    col = C2 if q["komponent"] == "m5_crossenc" else (
        C3 if q["komponent"] == "f_asos" else C1)
    a.scatter([x], [y], s=42, color=col, zorder=5,
              marker="s" if q["komponent"] == "f_asos" else "o")
    dx, dy, ha, va = OFS[q["komponent"]]
    a.annotate(NOM[q["komponent"]], xy=(x, y), xytext=(x * dx, y * dy),
               fontsize=7.8, color=INK, ha=ha, va=va)
a.set_xscale("log"); a.set_yscale("log")
a.set_xlim(4e-4, 400); a.set_ylim(0.006, 0.33)
a.set_xlabel("Isolated p95 latency (ms, log)")
a.set_ylabel("NDCG@10 (log)")
a.set_title("(a) measured quality vs. cost", fontsize=9, pad=8)

# --- (b) ustunlar -----------------------------------------------------
labels = ["static\nfusion", "best single\ncomponent", "per-request\noracle"]
fus = [q for q in T["qatorlar"] if q["komponent"] == "f_asos"][0]["NDCG10"]
vals = [fus, C["eng_yaxshi_statik_NDCG"], oracle]
cols = [C3, C1, C2]
xs = np.arange(3)
b.bar(xs, vals, color=cols, width=0.6)
for x, v in zip(xs, vals):
    b.text(x, v + 0.004, f"{v:.4f}", ha="center", fontsize=8)
b.set_xticks(xs); b.set_xticklabels(labels, fontsize=8)
b.set_ylabel("NDCG@10")
b.set_ylim(0, oracle * 1.40)
top = oracle * 1.25
b.annotate("", xy=(0, top), xytext=(1, top),
           arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.9))
b.text(0.5, top * 1.015, f"$-${100*(1-fus/C['eng_yaxshi_statik_NDCG']):.0f}%",
       ha="center", fontsize=8, color=MUTED)
top2 = oracle * 1.12
b.annotate("", xy=(1, top2), xytext=(2, top2),
           arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.9))
b.text(1.5, top2 * 1.015, f"$+${C['shift_foiz']:.0f}%",
       ha="center", fontsize=8, color=MUTED)
ulush = C["per_sorov_eng_yaxshi_ulush"][0][2]
b.set_title(f"(b) the gap static routing leaves\n($m_4$ is best on only {ulush:.0f}% of requests)",
            fontsize=9, pad=8)

out = os.path.join(ROOT, "figures"); os.makedirs(out, exist_ok=True)
plt.savefig(os.path.join(out, "a1_fig5.png")); plt.close()
print("a1_fig5.png saqlandi; oracle", oracle, "fusion", fus, "ulush", ulush)
