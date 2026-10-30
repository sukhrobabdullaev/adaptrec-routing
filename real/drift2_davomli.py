from drift_asos import *
T0 = time.time()
haqiqiy = np.arange(LIMIT)
rng = np.random.default_rng(0)
aralash = np.concatenate([np.arange(WARM), rng.permutation(np.arange(WARM, LIMIT))])

_, nol_seg, _, _ = otish("nol", aralash, "yoq", None)
print("nol oqim: %d segment x %d komponent (%.0fs)" % (nol_seg.shape[0], K, time.time()-T0), flush=True)

def tizim(segs, lam): return float((segs.max(1) > lam).mean())
def komp(segs, lam): return float((segs > lam).mean())

lam_5 = float(np.quantile(nol_seg, 0.95))
lam_bonf = float(np.quantile(nol_seg, 1.0 - 0.05 / K))
soxta = {"lam_tuzatishsiz": round(lam_5, 4), "lam_bonferroni": round(lam_bonf, 4),
         "komponent_tuzatishsiz": round(komp(nol_seg, lam_5), 4),
         "tizim_tuzatishsiz": round(tizim(nol_seg, lam_5), 4),
         "nazariy_1_minus_095_pow_K": round(1 - 0.95 ** K, 4),
         "komponent_bonferroni": round(komp(nol_seg, lam_bonf), 4),
         "tizim_bonferroni": round(tizim(nol_seg, lam_bonf), 4),
         "n_segment": int(nol_seg.shape[0])}
print("\nkalibrlash (%d segment):" % nol_seg.shape[0], flush=True)
print("  tuzatishsiz: lam=%.3f komponent %.1f%% TIZIM %.1f%% (nazariy %.1f%%)"
      % (lam_5, 100*soxta["komponent_tuzatishsiz"], 100*soxta["tizim_tuzatishsiz"],
         100*soxta["nazariy_1_minus_095_pow_K"]), flush=True)
print("  bonferroni : lam=%.3f komponent %.1f%% TIZIM %.1f%%"
      % (lam_bonf, 100*soxta["komponent_bonferroni"], 100*soxta["tizim_bonferroni"]), flush=True)

bl_s, _, det_s, statik = otish("saqla_bonf", haqiqiy, "saqla", lam_bonf)
print("\nhaqiqiy saqla: %d aniqlash, NDCG %.4f (%.0fs)" % (det_s, bl_s.mean(), time.time()-T0), flush=True)
bl_r, _, det_r, _ = otish("reset_bonf", haqiqiy, "reset", lam_bonf)
print("haqiqiy reset: %d aniqlash, NDCG %.4f (%.0fs)" % (det_r, bl_r.mean(), time.time()-T0), flush=True)

n = min(len(bl_s), len(bl_r))
t_st, p_t = stats.ttest_rel(bl_s[:n], bl_r[:n])
w_st, p_w = stats.wilcoxon(bl_s[:n], bl_r[:n])
jarima = (bl_s[:n].mean() / max(bl_r[:n].mean(), 1e-9) - 1) * 100
eng = int(np.argmax(statik))

oila = {"saqla vs reset (t)": float(p_t), "saqla vs reset (Wilcoxon)": float(p_w)}
nomlar = list(oila); pl = [oila[x] for x in nomlar]
tartib = np.argsort(pl); m_test = len(pl); holm = {}; oldingi = 0.0
for r_, j in enumerate(tartib):
    p_adj = min(1.0, max(oldingi, (m_test - r_) * pl[j])); oldingi = p_adj
    holm[nomlar[j]] = {"p_xom": pl[j], "p_holm": p_adj, "ahamiyatli_005": bool(p_adj < 0.05)}

out = {"limit": LIMIT, "warm": WARM, "budget": BUDGET, "blok": BLOK, "segment": SEG,
       "ph_delta": PH_DELTA, "vaqt_s": round(time.time()-T0, 1), "soxta_signal": soxta,
       "reset_vs_saqla": {"NDCG_saqla": round(float(bl_s[:n].mean()), 4),
                          "NDCG_reset": round(float(bl_r[:n].mean()), 4),
                          "reset_jarimasi_foiz": round(float(jarima), 1),
                          "n_blok": int(n), "aniqlash_saqla": int(det_s),
                          "aniqlash_reset": int(det_r)},
       "holm": holm,
       "eng_yaxshi_statik": {"toplam": list(T[idx[eng]]), "NDCG": round(float(statik[eng]), 4)}}
json.dump(out, open("natijalar/drift2_qayta.json", "w"), ensure_ascii=False, indent=1)
print("\n" + json.dumps(out, ensure_ascii=False, indent=1))
