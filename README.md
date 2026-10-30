# adaptrec-routing

Reproducibility package for the article

> **Adaptive Routing of Heterogeneous Recommendation Components over E-Commerce Request Streams: A Budget-Constrained Contextual Bandit Approach with Stream-Level Drift Detection**
> Sukhrobjon Abdullaev, Tashkent University of Information Technologies named after Muhammad al-Khwarizmi.

The package regenerates every number, table and figure in the article. It has two parts:

| Part | Article sections | What it does |
|---|---|---|
| `experiments/` + `adaptrec/` | 4.1 to 4.7 | Synthetic request streams with known ground truth: regret of the LinUCB router over feasible component subsets, static and random baselines, lattice pruning, Bonferroni-corrected Page-Hinkley drift detection, sensitivity sweeps. |
| `real/` | 4.8 to 4.11 | Real request stream built from Amazon Reviews 2023 (Video_Games, 5-core): measured quality and latency of six components, static fusion vs. best single component vs. per-request oracle, budget sweep of the router, reward-density ablation, drift calibration on a null stream and reset penalty. |

Code comments and some identifiers are in Uzbek (`natijalar` = results, `oqim` = stream, `komponent` = component, `byudjet` = budget, `qayta` = re-run). An Uzbek README is in `README_uz.md`.

## Install

```bash
python -m pip install -r requirements.txt     # Python 3.10+
```

## Data

* **Amazon Reviews 2023** (McAuley Lab, UCSD; Hou et al., 2024), category `Video_Games`: `Video_Games.jsonl.gz` and `meta_Video_Games.jsonl.gz` from https://amazon-reviews-2023.github.io/. No data are redistributed here.
* **Cross-encoder (m5)**: `cross-encoder/ms-marco-MiniLM-L6-v2` from Hugging Face, file `onnx/model_qint8_avx512.onnx` (official int8 export) plus `vocab.txt`. Place them in one directory as `model_qint8.onnx` and `vocab.txt`.

```bash
export VG_DATA=/path/to/vg        # holds the two .jsonl.gz files; stream files are written here
export MINILM=/path/to/minilm     # holds model_qint8.onnx and vocab.txt
cd "$VG_DATA" && python /path/to/adaptrec-routing/real/extract.py && python /path/to/adaptrec-routing/real/build_stream.py
```

`extract.py` writes `vg_events_raw.csv` (4,624,615 reviews) and `vg_items.csv` (137,269 items); `build_stream.py` applies iterative 5-core filtering and writes `vg_stream.csv`: **857,505 events, 98,906 users, 26,354 items**, October 1999 to September 2023. All ranking is over the **full catalogue** (no sampled negatives).

## Run

```bash
make sintetik      # experiments/: Sections 4.1-4.7, Figures 1-4 (about 30 s)
make real          # real/: Sections 4.8-4.11, Table 4, Figure 5 (several hours on 4 CPU cores)
```

Every run is seeded (`rng seed = 7` for the synthetic part, `default_rng(0)` and fixed component seeds for the real part).

### Resumable runners

The real-stream scripts are long (the router sweep runs about 45 minutes, the quality pass about 15 minutes). `real/*_davomli.py` are line-for-line equivalents that checkpoint their full state with `dill` every `BUDJ` seconds and exit with code 3; calling them again resumes from the checkpoint. They write to `real/natijalar/*_qayta.json`.

```bash
cd real
until python sifat_davomli.py;  do [ $? -eq 3 ] || break; done   # Table 4 quality
until python ceiling_davomli.py; do [ $? -eq 3 ] || break; done  # Section 4.10 ceiling
until python router_davomli.py; do [ $? -eq 3 ] || break; done   # budget sweep
until python router2_davomli.py; do [ $? -eq 3 ] || break; done  # reward-density ablation
until python drift3_davomli.py; do [ $? -eq 3 ] || break; done   # drift calibration and reset penalty
```

## Independent re-run (30 October 2026)

All real-stream results were regenerated from the raw `.jsonl.gz` files on a different machine (Linux VM, Apple-silicon host, 4 cores, Python 3.10, numpy 2.2.6, scipy 1.15.3, scikit-learn 1.7.2, onnxruntime 1.23.2). Outputs are in `real/natijalar/qayta_2026-10-30/` next to the original outputs in `real/natijalar/`.

| Result | Original | Re-run | Status |
|---|---|---|---|
| Synthetic regret, sensitivity (`sim_results.json`, `sim2_results.json`) | | | byte-identical |
| Stream size | 857,505 / 98,906 / 26,354 | same | identical |
| NDCG@10, HR@10, MRR of m0 to m4 and static fusion (Table 4) | m4 0.1330, fusion 0.0617 | same | identical to 4 decimals |
| Cross-encoder m5 NDCG@10 | 0.0427 | 0.0426 | int8 ONNX arithmetic differs across CPU architectures in the 4th decimal |
| Per-request oracle, best static subset, share of requests where {m4} wins | 0.1903, {m4} 0.1330, 57.3% | same | identical |
| Router budget sweep (adaptive 0.1290 at 0.05 ms to 0.1108 at 50 ms) | | same | identical for the adaptive router; two baseline values at the 50 ms budget differ by 1e-4 (m5 arithmetic) |
| Reward-density ablation (86.3%, 93.2%, 98.8% of static) | | same | identical |
| Drift false alarms: 25.0% uncorrected, 5.0% Bonferroni (theory 22.6%) | | same | identical |
| Reset penalty +54.6% / +80.7%, Holm p | | same | identical (p-values agree to 1e-15 relative) |

Latencies (`run_izolyatsiya.py`, Table 1 p50/p95) are wall-clock measurements of the original machine and are hardware dependent by nature; the cost model of the router reads them from `real/natijalar/izolyatsiya_kechikish.json` so that routing results are reproducible on any hardware.

## Layout

```
adaptrec/            routing, feasibility lattice, drift, simulator (synthetic part)
experiments/         run_simulation.py, run_sensitivity.py, make_figures.py
real/adaptreal/      data stream, six components, prequential harness, LinUCB router, Page-Hinkley
real/run_*.py        original real-stream scripts
real/*_davomli.py    resumable equivalents
real/natijalar/      original outputs (August 2026) and qayta_2026-10-30/ re-run outputs
figures/             Figures 1-5 of the article
```

## Citation

See `CITATION.cff`. License: MIT.
