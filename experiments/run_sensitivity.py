# -*- coding: utf-8 -*-
"""IV bob: sezgirlik tahlili (haqiqiy hisoblangan)."""
import json, os, numpy as np
import importlib.util, sys, io, contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NATIJA = os.path.join(ROOT, "natijalar")
os.makedirs(NATIJA, exist_ok=True)

spec = importlib.util.spec_from_file_location("simmod", os.path.join(HERE, "run_simulation.py"))
sim = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(sim)

OUT = {}
T, SEEDS = 12000, [11, 22, 33]

# ---- 1) o'rganish koeffitsiyentiga sezgirlik ----
beta_grid = [0.05, 0.15, 0.35, 0.60, 1.00]
r_beta = {}
for bc in beta_grid:
    fin = []
    for sd in SEEDS:
        r, c, _ = sim.run(T, sd, policy="linucb", beta_c=bc)
        fin.append(float(np.sum(r)))
    r_beta[bc] = dict(afsuslanish=round(float(np.mean(fin)), 1),
                      std=round(float(np.std(fin)), 1))
OUT["beta_sezgirlik"] = r_beta
OUT["beta_optimal"] = min(r_beta, key=lambda k: r_beta[k]["afsuslanish"])

# ---- 2) byudjetga sezgirlik ----
import numpy as _np
orig_budget = sim.BUDGET
budget_grid = [4.0, 6.0, 8.0, 10.0, 14.0]
r_bud = {}
for G in budget_grid:
    sim.BUDGET = G
    feas = (sim.cost_all <= G) & (sim.lat_all <= sim.SLO)
    sim.M = sim.M_all[feas]
    sim.SETS = [sim.ALL[i] for i in _np.where(feas)[0]]
    sim.nA = sim.M.shape[0]
    sim.cost_s = sim.M @ sim.GAMMA
    sim.lat_s = _np.where(sim.M, sim.LAT95, -_np.inf).max(1)
    sim.pen_s = sim.C_COST * sim.cost_s + sim.B_PEN * _np.maximum(0.0, sim.lat_s/sim.SLO - 1.0)
    fin, cost = [], []
    for sd in SEEDS:
        r, c, _ = sim.run(T, sd, policy="linucb")
        fin.append(float(np.sum(r))); cost.append(float(c.mean()))
    # shu byudjetdagi statik gibrid
    P0 = sim.states(4000, _np.random.default_rng(999))
    st = int(sim.true_mean(P0).mean(0).argmax())
    fs = []
    for sd in SEEDS:
        r, c, _ = sim.run(T, sd, policy=st)
        fs.append(float(np.sum(r)))
    r_bud[G] = dict(yol_qoyiladigan=int(sim.nA),
                    A1_afsuslanish=round(float(np.mean(fin)), 1),
                    statik_afsuslanish=round(float(np.mean(fs)), 1),
                    nisbat=round(float(np.mean(fs) / max(np.mean(fin), 1e-9)), 2),
                    A1_ortacha_narx=round(float(np.mean(cost)), 2))
OUT["byudjet_sezgirlik"] = r_bud

# byudjetni tiklash
sim.BUDGET = orig_budget
feas = (sim.cost_all <= orig_budget) & (sim.lat_all <= sim.SLO)
sim.M = sim.M_all[feas]; sim.SETS = [sim.ALL[i] for i in _np.where(feas)[0]]
sim.nA = sim.M.shape[0]; sim.cost_s = sim.M @ sim.GAMMA
sim.lat_s = _np.where(sim.M, sim.LAT95, -_np.inf).max(1)
sim.pen_s = sim.C_COST*sim.cost_s + sim.B_PEN*_np.maximum(0.0, sim.lat_s/sim.SLO - 1.0)

# ---- 3) submodulyarlik farazini empirik tekshirish ----
rng = np.random.default_rng(777)
P = sim.states(2000, rng)
A_ = sim.alphas(P)
viol, tot = 0, 0
for i in range(len(P)):
    a = A_[i]
    for k in range(sim.K):
        for _ in range(6):
            sz = rng.integers(0, sim.K)
            pool = [j for j in range(sim.K) if j != k]
            B = list(rng.choice(pool, size=min(sz, len(pool)), replace=False))
            Aset = list(set(B) | set(rng.choice(pool, size=1)))
            if not set(B) <= set(Aset):
                continue
            qB = 1 - np.prod(1 - a[B]) if B else 0.0
            qA = 1 - np.prod(1 - a[Aset]) if Aset else 0.0
            gB = (1 - np.prod(1 - a[B + [k]])) - qB
            gA = (1 - np.prod(1 - a[Aset + [k]])) - qA
            tot += 1
            if gA > gB + 1e-12:      # B ⊆ A bo'lsa gA ≤ gB bo'lishi kerak
                viol += 1
OUT["submodulyarlik"] = {"tekshirilgan_juftliklar": int(tot),
                         "buzilishlar": int(viol),
                         "buzilish_ulushi": round(viol / max(tot, 1), 6)}

# ---- 4) oflayn ilitishning ta'siri ----
def run_warm(T, seed, n0, beta_c=0.35, lam=0.5):
    rng = np.random.default_rng(seed)
    P = sim.states(T, rng)
    theta = np.zeros((sim.nA, sim.D))
    Vinv = np.tile(np.eye(sim.D)/lam, (sim.nA, 1, 1))
    bvec = np.zeros((sim.nA, sim.D))
    if n0 > 0:                                   # oflayn baho asosida ilitish
        P0 = sim.states(3000, np.random.default_rng(4321))
        TM0 = sim.true_mean(P0)
        phibar = P0.mean(0)
        for a_ in range(sim.nA):
            rbar = float(TM0[:, a_].mean()) * 0.85   # oflayn baho — noaniq (15% siljish)
            V = lam*np.eye(sim.D) + n0*np.outer(phibar, phibar)
            Vinv[a_] = np.linalg.inv(V)
            bvec[a_] = n0*rbar*phibar
            theta[a_] = Vinv[a_] @ bvec[a_]
    TM = sim.true_mean(P)
    reg = np.zeros(T)
    for t in range(T):
        phi = P[t]
        mu = theta @ phi
        s = np.sqrt(np.maximum(np.einsum('aij,i,j->a', Vinv, phi, phi), 0.0))
        a_t = int(np.argmax(mu + beta_c*s))
        r = TM[t, a_t] + rng.normal(0, sim.NOISE)
        reg[t] = TM[t].max() - TM[t, a_t]
        u = Vinv[a_t] @ phi
        Vinv[a_t] -= np.outer(u, u)/(1.0 + phi @ u)
        bvec[a_t] += r*phi
        theta[a_t] = Vinv[a_t] @ bvec[a_t]
    return reg

warm = {}
for n0 in [0, 50, 200, 1000]:
    early, fin = [], []
    for sd in SEEDS:
        r = run_warm(T, sd + 77, n0)
        early.append(float(np.sum(r[:1000]))); fin.append(float(np.sum(r)))
    warm[n0] = dict(dastlabki_1000_raund=round(float(np.mean(early)), 1),
                    yakuniy=round(float(np.mean(fin)), 1))
OUT["oflayn_ilitish"] = warm

json.dump(OUT, open(os.path.join(NATIJA, "sim2_results.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(OUT, ensure_ascii=False, indent=1))
