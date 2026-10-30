export PYTHONUTF8 := 1

PYTHON ?= python

# VoltageLimits preset used to flag out-of-band states (models/VoltageLimits.py PRESETS)
VOLTAGE_PRESET ?= default

CASE := cases/glover37.json
POWERWORLD_DATA := data/powerworld/glover37

RESULTS := results/glover37
RAW := $(RESULTS)/raw
ANALYSIS := $(RESULTS)/analysis
CONTINGENCIES := $(RAW)/contingency_set.csv

FIGURES := figures/glover37


.PHONY: help case strength validate screen simulate analyze report layout \
	explorer test clean


help:
	@echo "System-strength metrics study (SCR, WSCR, ESCR, SDSCR), Glover 37-bus"
	@echo
	@echo "  make strength          validate -> screen -> simulate -> analyze -> report (about 8 min)"
	@echo
	@echo "Stages:"
	@echo "  make validate          E0: PowerWorld agreement, base + A1-A3, B1-B3"
	@echo "  make screen            Fix the N-1 line set (islanding, convergence)"
	@echo "  make simulate          168 h metrics run: every state (E1, E2, E5), every ESCR MIIF variant"
	@echo "  make analyze           Analysis of the existing raw results (claims C1-C9)"
	@echo "  make report            Figures from the existing analysis"
	@echo "  make layout            Network one-line diagram"
	@echo
	@echo "Other:"
	@echo "  make case              Rebuild $(CASE) from the PowerWorld exports"
	@echo "  make explorer          Open the results explorer (Ctrl-C to stop)"
	@echo "  make test              Run the checks in tests/"
	@echo "  make clean             Remove generated results and figures"
	@echo
	@echo "Variables:"
	@echo "  PYTHON=$(PYTHON)  VOLTAGE_PRESET=$(VOLTAGE_PRESET)"


case:
	$(PYTHON) experiments/glover37/pw2json.py \
		$(POWERWORLD_DATA)/base \
		-o $(CASE) \
		--report $(POWERWORLD_DATA)/conversion_report.md


strength:
	$(MAKE) validate
	$(MAKE) screen
	$(MAKE) simulate
	$(MAKE) analyze
	$(MAKE) report


validate:
	$(PYTHON) -m experiments.glover37.validate


screen:
	$(PYTHON) -m experiments.glover37.simulate screen --output $(CONTINGENCIES)


simulate:
	$(PYTHON) -m experiments.glover37.simulate run \
		--contingencies $(CONTINGENCIES) \
		--voltage-preset $(VOLTAGE_PRESET) \
		--output $(RAW)


analyze:
	$(PYTHON) -m experiments.glover37.analyze \
		--input-dir $(RAW) \
		--output-dir $(ANALYSIS)


report:
	$(PYTHON) -m experiments.glover37.report


layout:
	$(PYTHON) -m experiments.glover37.layout


# tools/explorer.py serves the repository on localhost and opens tools/explorer.html,
# which reads results/glover37/raw/ at runtime.
explorer:
	$(PYTHON) tools/explorer.py


test:
	$(PYTHON) -m unittest discover -s tests -v


clean:
	rm -rf $(RESULTS)
	rm -rf $(FIGURES)
