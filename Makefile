PY ?= python3

.PHONY: all sintetik real clean
all: sintetik real

# 4.1-4.7: sintetik oqim
sintetik:
	$(PY) experiments/run_simulation.py
	$(PY) experiments/run_sensitivity.py
	$(PY) experiments/make_figures.py

# 4.8-4.11: real oqim (avval real/build_stream.py bilan oqim yasang)
real:
	cd real && $(PY) run_sifat.py
	cd real && $(PY) run_izolyatsiya.py
	cd real && $(PY) run_profil.py
	cd real && $(PY) run_ceiling.py
	cd real && $(PY) run_drift2.py
	cd real && $(PY) run_drift3.py
	cd real && $(PY) make_fig5.py

clean:
	rm -rf natijalar figures real/natijalar real/figures
