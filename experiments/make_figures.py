# -*- coding: utf-8 -*-
"""1-maqolaning to'rtala rasmini qayta hosil qilish (Figure 1–4).

Kirish: natijalar/sim_data.npz (run_simulation.py yaratadi) va
natijalar/sim2_results.json (run_sensitivity.py yaratadi).
Chiqish: figures/a1_fig1.png ... a1_fig4.png

Ishga tushirish (paket ildizidan):  python experiments/make_figures.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C1, C2, C3 = "#0072B2", "#D55E00", "#009E73"
INK, MUTED, GRID = "#1A1A1A", "#5A5A5A", "#D9D9D9"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 200,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight"})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = np.load(os.path.join(ROOT, "natijalar", "sim_data.npz"))
S2 = json.load(open(os.path.join(ROOT, "natijalar", "sim2_results.json")))
OUT = os.path.join(ROOT, "figures"); os.makedirs(OUT, exist_ok=True)
T = int(D["T"]); tt = np.arange(1, T + 1)

def save(name):
    plt.savefig(os.path.join(OUT, f"{name}.png")); plt.close()

# Figure 1: jamg'arilgan yo'qotish + log-log
fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.8))
a.plot(tt, D["cum_rand"], color=MUTED, lw=1.2, label="Random feasible")
a.plot(tt, D["cum_stat"], color=C2, lw=1.4, label="Static hybrid (oracle-picked)")
a.plot(tt, D["cum_a1"], color=C1, lw=1.6, label="Adaptive routing (ours)")
a.fill_between(tt, D["cum_a1"] - D["std_a1"], D["cum_a1"] + D["std_a1"],
               color=C1, alpha=0.20, lw=0)
a.set_xlabel("Round $t$"); a.set_ylabel("Cumulative regret $R(t)$")
a.legend(frameon=False, fontsize=7.5)
m = tt > 400
b.loglog(tt[m], D["cum_a1"][m], color=C1, lw=1.6, label="Adaptive routing")
sl = np.polyfit(np.log(tt[m]), np.log(np.maximum(D["cum_a1"][m], 1e-9)), 1)
b.loglog(tt[m], np.exp(sl[1]) * tt[m] ** 0.5, color=MUTED, ls="--", lw=1.0,
         label=r"$\propto \sqrt{t}$ reference")
b.set_xlabel("Round $t$ (log)"); b.set_ylabel("$R(t)$ (log)")
b.text(0.05, 0.86, f"fitted slope = {sl[0]:.3f}", transform=b.transAxes, fontsize=8)
b.legend(frameon=False, fontsize=7.5, loc="lower right")
save("a1_fig1")

# Figure 2: ochko'z kafolat auditi (gistogramma)
fig, ax = plt.subplots(figsize=(4.6, 2.7))
ax.hist(D["ratios"], bins=48, color=C1, alpha=0.85, edgecolor="white", lw=0.3)
ax.axvline(1 - 1 / np.e, color=C2, ls="--", lw=1.3)
ax.text(1 - 1 / np.e - 0.015, ax.get_ylim()[1] * 0.62, r"$1-e^{-1}$",
        rotation=90, fontsize=8, color=C2, ha="right")
ax.set_xlabel("Greedy value / optimal value")
ax.set_ylabel("Number of states")
ax.set_yscale("log")
save("a1_fig2")

# Figure 3: drift ssenariysi (salbiy natija)
fig, ax = plt.subplots(figsize=(4.6, 2.8))
ax.plot(tt, D["drift_no"], color=C1, lw=1.6, label="No forgetting (step-size only)")
ax.plot(tt, D["drift_det"], color=C2, lw=1.4, label="Detector-triggered forgetting")
ax.axvline(T // 2, color=MUTED, ls=":", lw=1.0)
ax.text(T // 2 + 150, ax.get_ylim()[1] * 0.05, "catalogue shock", fontsize=8, color=MUTED)
ax.set_xlabel("Round $t$"); ax.set_ylabel("Cumulative regret $R(t)$")
ax.legend(frameon=False, fontsize=8, loc="upper left")
save("a1_fig3")

# Figure 4: sezgirlik (beta_c U-shakl; byudjet ustunligi)
bs = S2["beta_sezgirlik"]; keys = sorted(bs, key=float)
x = [float(k) for k in keys]; y = [bs[k]["afsuslanish"] for k in keys]
e = [bs[k]["std"] for k in keys]
gg = S2["byudjet_sezgirlik"]; gk = sorted(gg, key=float)
gx = [float(k) for k in gk]; gy = [gg[k]["nisbat"] for k in gk]
fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.7))
a.errorbar(x, y, yerr=e, color=C1, lw=1.5, marker="o", ms=4, capsize=3)
a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel(r"Exploration coefficient $\beta_c$")
a.set_ylabel("Final regret $R(T)$")
a.axvline(0.35, color=C3, ls="--", lw=1.0)
a.text(0.30, y[2] * 2.6, r"$\beta_c^{*}=0.35$", fontsize=8, color=C3,
       ha="right")
b.plot(gx, gy, color=C1, lw=1.5, marker="s", ms=4)
b.set_xlabel(r"Compute budget $\Gamma$")
b.set_ylabel("Static / adaptive regret ratio")
b.axhline(1.0, color=MUTED, ls=":", lw=0.8)
i_max = int(np.argmax(gy))
b.annotate("advantage peaks when\nbudget binds",
           xy=(gx[i_max], gy[i_max]), xytext=(gx[i_max] + 1.6, gy[i_max] - 0.35),
           fontsize=8, color=MUTED, va="top",
           arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
save("a1_fig4")

print("Rasmlar tayyor:", OUT)
