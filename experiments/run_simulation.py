# -*- coding: utf-8 -*-
"""IV bob uchun nazorat qilinadigan simulyatsiya tadqiqoti (vektorlashtirilgan)."""
import json
import os
import numpy as np
from itertools import combinations

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NATIJA = os.path.join(ROOT, "natijalar")
os.makedirs(NATIJA, exist_ok=True)

OUT = {}
K = 5
NAMES = ["m1 CF/graf", "m2 kontent", "m3 KG", "m4 ketma-ket", "m5 LLM"]
GAMMA = np.array([1.0, 1.5, 2.0, 2.5, 6.0])
LAT95 = np.array([15.0, 20.0, 25.0, 35.0, 85.0])
BUDGET, SLO, D = 8.0, 100.0, 6
C_COST, B_PEN, NOISE = 0.030, 2.0, 0.05   # narx jarimasi mutlaq narxga proporsional

# ---- to'plamlar va filtrlash ----
ALL = [c for r in range(1, K + 1) for c in combinations(range(K), r)]
M_all = np.zeros((len(ALL), K), dtype=bool)
for i, c in enumerate(ALL):
    M_all[i, list(c)] = True
cost_all = M_all @ GAMMA
lat_all = np.where(M_all, LAT95, -np.inf).max(1)
feas = (cost_all <= BUDGET) & (lat_all <= SLO)
M = M_all[feas]                       # (nA, K)
SETS = [ALL[i] for i in np.where(feas)[0]]
nA = M.shape[0]
cost_s = M @ GAMMA
lat_s = np.where(M, LAT95, -np.inf).max(1)
pen_s = C_COST * cost_s + B_PEN * np.maximum(0.0, lat_s / SLO - 1.0)
OUT["jami_toplamlar"] = int(len(ALL))
OUT["yol_qoyiladigan"] = int(nA)


def states(n, rng):
    """Holat vektorlari (n, D): [1, n_u, drift, Q, yuklama, eps]."""
    return np.column_stack([
        np.ones(n), rng.beta(1.4, 2.2, n), rng.beta(1.2, 6.0, n),
        np.clip(rng.beta(5.0, 2.0, n), 0.05, 1.0),
        rng.uniform(0.2, 0.9, n), rng.uniform(0.3, 1.0, n)])


def alphas(P, swap=False):
    """(n, K) — komponentlarning haqiqiy marjinal hissasi."""
    n_, d_, q_ = P[:, 1], P[:, 2], P[:, 3]
    A = np.column_stack([
        0.05 + 0.30 * n_,
        (0.28 - 0.18 * n_) * q_,
        0.15 + 0.05 * d_,
        0.10 + 0.20 * n_,
        (0.70 - 0.55 * n_) * (0.5 + 0.5 * q_)])
    A = np.clip(A, 0.0, 0.95)
    if swap:                       # katalog zarbasi: CF zaiflashadi, kontent kuchayadi
        A = A.copy()
        A[:, 0] *= 0.30
        A[:, 3] *= 0.55
        A[:, 1] = np.clip(A[:, 1] * 1.60, 0, 0.95)
    return A


def true_mean(P, swap=False):
    """(n, nA) — har bir holat va to'plam uchun haqiqiy o'rtacha mukofot."""
    L = np.log1p(-alphas(P, swap))          # (n, K)
    Q = 1.0 - np.exp(L @ M.T)               # (n, nA) submodulyar sifat
    return Q - pen_s[None, :]


# ---- statik gibrid: o'rtacha bo'yicha eng yaxshi qat'iy to'plam ----
rng0 = np.random.default_rng(999)
P0 = states(6000, rng0)
static_idx = int(true_mean(P0).mean(0).argmax())
OUT["statik_toplam"] = [NAMES[i] for i in SETS[static_idx]]


def run(T, seed, policy="linucb", beta_c=0.35, lam=0.5, drift_at=None, detect=True):
    rng = np.random.default_rng(seed)
    P = states(T, rng)
    theta = np.zeros((nA, D))
    Vinv = np.tile(np.eye(D) / lam, (nA, 1, 1))
    bvec = np.zeros((nA, D))
    reg = np.zeros(T); cst = np.zeros(T); resets = 0; last_reset = -10**9
    ph = np.zeros(K); qbar = np.full(K, np.nan)
    TM_a = true_mean(P, False)
    TM_b = true_mean(P, True) if drift_at is not None else None
    Aq = alphas(P, False)
    Aq_b = alphas(P, True) if drift_at is not None else None
    for t in range(T):
        swap = (drift_at is not None and t >= drift_at)
        TM = TM_b[t] if swap else TM_a[t]
        phi = P[t]
        if policy == "linucb":
            mu = theta @ phi
            s = np.sqrt(np.maximum(np.einsum('aij,i,j->a', Vinv, phi, phi), 0.0))
            a_t = int(np.argmax(mu + beta_c * s))
        elif policy == "random":
            a_t = int(rng.integers(nA))
        else:
            a_t = policy
        r = TM[a_t] + rng.normal(0, NOISE)
        reg[t] = TM.max() - TM[a_t]
        cst[t] = cost_s[a_t]
        if policy == "linucb":
            u = Vinv[a_t] @ phi
            Vinv[a_t] -= np.outer(u, u) / (1.0 + phi @ u)
            bvec[a_t] += r * phi
            theta[a_t] = Vinv[a_t] @ bvec[a_t]
            if detect and drift_at is not None:
                W = 100
                if t >= W:
                    lo = t - W + 1
                    qk = (Aq_b[lo:t+1] if swap else Aq[lo:t+1]).mean(0)  # oyna bo'yicha sifat
                    qbar = np.where(np.isnan(qbar), qk, 0.99 * qbar + 0.01 * qk)
                    ph = np.maximum(0.0, ph + (qbar - qk - 0.010))
                if ph.max() > 0.60 and (t - last_reset) > 2000:
                    Vinv = Vinv / 0.15          # V <- 0.15*V : o'tmishni unutish
                    bvec *= 0.15
                    last_reset = t
                    theta = np.einsum('aij,aj->ai', Vinv, bvec)
                    ph[:] = 0.0; qbar[:] = np.nan
                    resets += 1
    return reg, cst, resets


# ================= 1-tajriba: afsuslanish =================
T, SEEDS = 12000, [11, 22, 33, 44, 55]
res = {}
for label, pol in [("A1 (adaptiv marshrutlash)", "linucb"),
                   ("Statik gibrid", static_idx), ("Tasodifiy tanlov", "random")]:
    C = np.zeros((len(SEEDS), T)); costs = []
    for j, sd in enumerate(SEEDS):
        r, c, _ = run(T, sd, policy=pol)
        C[j] = np.cumsum(r); costs.append(c.mean())
    res[label] = dict(cum=C.mean(0), std=C.std(0), cost=float(np.mean(costs)),
                      final=float(C.mean(0)[-1]), fstd=float(C.std(0)[-1]))
OUT["afsuslanish"] = {k: dict(yakuniy=round(v["final"], 1), std=round(v["fstd"], 1),
                              ortacha_narx=round(v["cost"], 2)) for k, v in res.items()}
cum = res["A1 (adaptiv marshrutlash)"]["cum"]
tt = np.arange(1, T + 1); m = tt > 400
OUT["afsuslanish_qiya"] = round(float(np.polyfit(np.log(tt[m]), np.log(np.maximum(cum[m], 1e-9)), 1)[0]), 3)
OUT["narx_tejash_foiz"] = round(100 * (1 - res["A1 (adaptiv marshrutlash)"]["cost"]
                                       / res["Statik gibrid"]["cost"]), 1)

# ================= 2-tajriba: drift =================
dr = {}
for label, det in [("Drift detektori bilan", True), ("Detektorsiz", False)]:
    C = np.zeros((len(SEEDS), T)); rs = []
    for j, sd in enumerate(SEEDS):
        r, c, k_ = run(T, sd + 500, drift_at=T // 2, detect=det)
        C[j] = np.cumsum(r); rs.append(k_)
    dr[label] = dict(cum=C.mean(0), final=float(C.mean(0)[-1]),
                     post=float(C.mean(0)[-1] - C.mean(0)[T // 2]), resets=float(np.mean(rs)))
OUT["drift"] = {k: dict(yakuniy=round(v["final"], 1), driftdan_keyin=round(v["post"], 1),
                        qayta_boshlashlar=v["resets"]) for k, v in dr.items()}
OUT["drift_yaxshilanish_foiz"] = round(
    100 * (1 - dr["Drift detektori bilan"]["post"] / dr["Detektorsiz"]["post"]), 1)
# ufq bo'yicha kesim: driftdan keyin necha raundda foyda paydo bo'ladi
horizons = [500, 1000, 2000, 3000, 4000, 6000]
cut = {}
for H in horizons:
    a = float(dr["Drift detektori bilan"]["cum"][T//2 + H - 1] - dr["Drift detektori bilan"]["cum"][T//2])
    b = float(dr["Detektorsiz"]["cum"][T//2 + H - 1] - dr["Detektorsiz"]["cum"][T//2])
    cut[H] = {"detektor_bilan": round(a, 1), "detektorsiz": round(b, 1),
              "farq_foiz": round(100*(1 - a/b), 1) if b > 0 else None}
OUT["drift_ufq_kesimi"] = cut

# ================= 3-tajriba: Bonferroni =================
def false_alarm_rate(h, n_runs=600, T2=3000, delta=0.004, seed=7):
    rng = np.random.default_rng(seed)
    q = 0.30 + rng.normal(0, 0.03, (n_runs, T2, K))
    qbar = np.full((n_runs, K), 0.30); ph = np.zeros((n_runs, K))
    fired = np.zeros(n_runs, dtype=bool)
    for t in range(T2):
        qt = q[:, t, :]
        qbar = 0.99 * qbar + 0.01 * qt
        ph = np.maximum(0.0, ph + (qbar - qt - delta))
        fired |= (ph.max(1) > h)
    return float(fired.mean())

def single_comp_rate(h, n_runs=600, T2=3000, delta=0.004, seed=7):
    rng = np.random.default_rng(seed)
    q = 0.30 + rng.normal(0, 0.03, (n_runs, T2))
    qbar = np.full(n_runs, 0.30); ph = np.zeros(n_runs)
    fired = np.zeros(n_runs, dtype=bool)
    for t in range(T2):
        qbar = 0.99 * qbar + 0.01 * q[:, t]
        ph = np.maximum(0.0, ph + (qbar - q[:, t] - delta))
        fired |= (ph > h)
    return float(fired.mean())

# bitta komponent uchun ~5% darajaga sozlangan chegara
grid = np.linspace(0.05, 1.2, 60)
h_single = float(grid[np.argmin([abs(single_comp_rate(h) - 0.05) for h in grid])])
r_no = false_alarm_rate(h_single)
grid2 = np.linspace(h_single, 2.5, 40)
h_bonf = float(grid2[np.argmin([abs(false_alarm_rate(h) - 0.05) for h in grid2])])
OUT["bonferroni"] = {
    "h_bitta_komponent": round(h_single, 3),
    "K_komponentda_tuzatishsiz_yolgon_signal": round(r_no, 3),
    "nazariy_kutilma_1_minus_095_pow_K": round(1 - 0.95 ** K, 3),
    "h_bonferroni": round(h_bonf, 3),
    "bonferroni_bilan_yolgon_signal": round(false_alarm_rate(h_bonf), 3)}

# ================= 4-tajriba: ochko'z va to'liq saralash =================
rngg = np.random.default_rng(4242)
Pg = states(4000, rngg)
TMg = true_mean(Pg)
Ag = alphas(Pg)
ratios = np.zeros(len(Pg))
for i in range(len(Pg)):
    sel, rem = [], BUDGET
    while True:
        best, bv = None, 0.0
        for k in range(K):
            if k in sel or GAMMA[k] > rem:
                continue
            cand = sel + [k]
            if LAT95[cand].max() > SLO:
                continue
            base = (1 - np.prod(1 - Ag[i, sel])) - C_COST*GAMMA[sel].sum() if sel else 0.0
            new = (1 - np.prod(1 - Ag[i, cand])) - C_COST*GAMMA[cand].sum()
            g = (new - base) / GAMMA[k]        # jarimani hisobga olgan marjinal hissa
            if g > bv:
                bv, best = g, k
        if best is None:
            break
        sel.append(best); rem -= GAMMA[best]
    idx = SETS.index(tuple(sorted(sel)))
    ratios[i] = TMg[i, idx] / TMg[i].max()
OUT["ochkoz"] = {"ortacha_nisbat": round(float(ratios.mean()), 4),
                 "eng_past_nisbat": round(float(ratios.min()), 4),
                 "nazariy_kafolat": round(float(1 - 1/np.e), 4),
                 "kafolatdan_past_ulush": round(float((ratios < 1 - 1/np.e).mean()), 4),
                 "optimalga_teng_ulush": round(float((ratios > 0.999).mean()), 4)}

# ================= 5-tajriba: 2.3-tasdiqning sharti =================
rngc = np.random.default_rng(31337)
Ac = alphas(states(30000, rngc))
best_k = Ac.argmax(1)
vals, cnts = np.unique(best_k, return_counts=True)
OUT["eng_foydali_komponent_ulushi"] = {NAMES[int(v)]: round(float(c / len(best_k)), 4)
                                       for v, c in zip(vals, cnts)}
OUT["turli_komponent_ustun_boladigan_holatlar"] = int(len(vals))

np.savez(os.path.join(NATIJA, "sim_data.npz"), cum_a1=res["A1 (adaptiv marshrutlash)"]["cum"],
         std_a1=res["A1 (adaptiv marshrutlash)"]["std"],
         cum_stat=res["Statik gibrid"]["cum"], cum_rand=res["Tasodifiy tanlov"]["cum"],
         drift_det=dr["Drift detektori bilan"]["cum"], drift_no=dr["Detektorsiz"]["cum"],
         ratios=ratios, T=T)
json.dump(OUT, open(os.path.join(NATIJA, "sim_results.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(OUT, ensure_ascii=False, indent=1))
