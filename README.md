# POPs Sediment Quality Guidelines — ML-refined Empirical TEL/PEL Derivation

Explainable machine learning (XGBoost + SHAP) pipeline for deriving empirical
sediment quality guidelines (TEL/PEL) for persistent organic pollutants (POPs)
in marine sediments.

## Overview

This repository reproduces the full derivation chain for the ML-refined
empirical TEL/PEL values reported in the manuscript:

| Substance | TEL (µg/kg dw) | PEL (µg/kg dw) |
|-----------|----------------|----------------|
| ΣDDTs     | 4.98           | 21.82          |
| ΣCHLs     | 3.91           | 23.80          |
| ΣPCBs     | 8.56           | 51.61          |

The pipeline integrates two North American sediment toxicity databases
(NOAA SEDDB and SCCWRP SQO), filters non-causal noise using SHAP feature
attributions, and derives TEL/PEL following the CCME geometric-mean convention
(Method B: percentiles computed in original units, then geometric mean).

## Repository structure

```
.
├── code/
│   ├── build_integrated_db_v2.py   # Step 0: integrate NOAA + SCCWRP → matching DB
│   ├── run_pipeline_v3.py          # Steps 1–4: curation → SHAP filtering → TEL/PEL → reliability
│   ├── derive_tcdd_empirical.py    # TCDD empirical SQG (TEQ-based, separate filter)
│   ├── recalc_tel_pel_methodB.py   # Method B verification (original-unit geometric mean)
│   └── sqg_config.py               # Single source of truth for all SQG values
├── data/                           # Input databases (NOAA, SCCWRP SQO)
└── output/                         # Generated results
```

## Pipeline steps

### Step 0 — Database integration (`build_integrated_db_v2.py`)
- Loads NOAA SEDDB (`US_Sediment_Risk_Analytical_Set_Mainland.csv`) and
  SCCWRP SQO database (`SQO_database/`).
- Expands SCCWRP to sample × species format (NOAA-compatible).
- Produces `integrated_matching_db_v2.csv` (20,640 rows).

### Step 1 — DB curation (`run_pipeline_v3.py`)
- Negative flags (−9) → NA; `Mean_Survival ≥ 0` and `TOC > 0` filters.
- Non-benthic species excluded; *Chironomus* unified.
- Heavy-metal mPELQ pre-filter (mean metal quotient > 0.5 → suspected
  co-toxicity, removed).
- Isomer weighted summation (ND = MDL/2, OC-normalized log10).
- EqP implausible-toxicity filter (Di Toro 1991: C_pw < CCC & survival < 80%).

### Step 2 — Species selection (XGBoost + SHAP)
- Per-species: N ≥ 30 and toxic N ≥ 10.
- XGBoost regression: `Mean_Survival ~ Conc_log + TOC_pct` (100 rounds).
- SHAP direction (Spearman ≤ −0.3) and dominance (TOC < chem × 1.5) checks.

### Step 2.5 — Dose–response quality
- DRC quality evaluation; poor-DRC species excluded from TEL/PEL.

### Step 2.7 — Multi-contaminant confounder filtering
- Per sample: `[target_conc, mPELQ_Metals, TOC_pct]` → XGBoost → survival.
- `|SHAP_mPELQ| > |SHAP_target|` → metal-dominated → dropped.

### Step 3 — TEL/PEL derivation (Method B)
- Final XGBoost → `SHAP_Chem`.
- EDS = `survival < 80` & `SHAP_Chem ≤ −0.1`; NEDS = `survival ≥ 80`.
- **Method B (CCME convention)**: percentiles in original units, then geometric mean:
  - `TEL = sqrt(EDS_P15 × NEDS_P50)`
  - `PEL = sqrt(EDS_P50 × NEDS_P85)`

### Step 4 — Reliability assessment
- Benchmark against CCME ISQG/PEL and NOAA TEL/PEL.
- ROC-AUC and TEL sensitivity/specificity.

## Reproducing the results

### Requirements

```bash
pip install pandas numpy xgboost scipy scikit-learn
```

### Run

```bash
# Step 0: build integrated database
python code/build_integrated_db_v2.py

# Steps 1–4: full pipeline (TEL/PEL derivation)
python code/run_pipeline_v3.py

# TCDD empirical SQG (separate TEQ-based filter)
python code/derive_tcdd_empirical.py

# Method B verification (optional)
python code/recalc_tel_pel_methodB.py
```

The final Integrated TEL/PEL values are written to
`output/Step3_TEL_PEL_Integrated.csv` and match the manuscript values
(DDTs 4.98/21.82, CHLs 3.91/23.80, PCBs 8.56/51.61 µg/kg dw).

## Method A vs Method B

- **Method A (log-space)**: percentiles computed on `log10(conc + 1)`, then
  back-transformed. This was the original `run_pipeline_v3.py` implementation.
- **Method B (original units)**: percentiles computed on original
  concentrations (µg/kg dw), then geometric mean. This follows the CCME
  guideline convention and is the method used for the manuscript values.

The current `run_pipeline_v3.py` implements **Method B** so that the code
directly reproduces the reported values.

## License

MIT
