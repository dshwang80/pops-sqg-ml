# POPs Sediment Quality Guidelines — ML-refined Empirical TEL/PEL Derivation

Explainable machine learning (XGBoost + SHAP) pipeline for deriving empirical
sediment quality guidelines (TEL/PEL) for persistent organic pollutants (POPs)
in marine sediments.

## Overview

This repository reproduces the full derivation chain for the ML-refined
empirical TEL/PEL values reported in the manuscript. The **Primary** pipeline
(`run_pipeline_v4.py`) is **SHAP-independent**: species selection and
confounder filtering rely on dose–response quality and survival-based
EDS/NEDS classification, not on SHAP attributions. SHAP is retained only as a
diagnostic/interpretability layer.

The pipeline integrates two North American sediment toxicity databases
(NOAA SEDDB and SCCWRP SQO), filters non-causal noise, and derives TEL/PEL
following the CCME geometric-mean convention (Method B: percentiles computed
in original units, then geometric mean).

## Repository structure

```
.
├── code/
│   ├── build_integrated_db_v2.py        # Step 0: integrate NOAA + SCCWRP → matching DB
│   ├── run_pipeline_v4.py               # Steps 1–4: curation → species selection → TEL/PEL → reliability
│   ├── pipeline_v4_config.json          # Reproducibility config (single source of truth for run settings)
│   ├── sqg_config.py                    # Single source of truth for all SQG values
│   ├── bootstrap_tel_pel_ci.py          # Cluster bootstrap CI for TEL/PEL
│   ├── mpelq_cutoff_sensitivity.py      # mPELQ cutoff sensitivity (0.25 / 0.5 / 1.0)
│   ├── anchor_species_sensitivity.py    # Anchor-species (L. plumulosus) sensitivity
│   ├── recalc_tel_pel_methodB.py        # Method B verification (original-unit geometric mean)
│   └── verify_pipeline.py               # Automated 8-point validation checklist
├── data/                                # Input databases (NOAA, SCCWRP SQO)
├── output/                              # Generated results (gitignored)
└── verification/                        # Frozen validation checkpoint (committed)
```

## Pipeline steps

### Step 0 — Database integration (`build_integrated_db_v2.py`)
- Loads NOAA SEDDB (`US_Sediment_Risk_Analytical_Set_Mainland.csv`) and
  SCCWRP SQO database (`SQO_database/`).
- Expands SCCWRP to sample × species format (NOAA-compatible).
- Produces `integrated_matching_db_v2.csv`.

### Step 1 — DB curation (`run_pipeline_v4.py`)
- Negative flags (−9) → NA; `Mean_Survival ≥ 0` and `TOC > 0` filters.
- Non-benthic species excluded; *Chironomus* unified.
- Heavy-metal mPELQ pre-filter (mean metal quotient > threshold → suspected
  co-toxicity, removed). Threshold from `pipeline_v4_config.json`
  (`mpelq_metals_threshold`, default 0.5). Requires ≥ `min_metals_measured`
  (default 3) measured metals for a valid mPELQ.
- Isomer weighted summation (ND = MDL/2, OC-normalized log10).

### Step 2 — Species selection (XGBoost, SHAP-independent)
- Per-species: N ≥ 30 and toxic N ≥ 10.
- XGBoost regression: `Mean_Survival ~ Conc_log + TOC_pct`.
- Direction (Spearman ≤ −0.3) and dominance (TOC < chem × 1.5) checks.

### Step 2.5 — Dose–response quality
- DRC quality evaluation; poor-DRC species excluded from TEL/PEL.

### Step 2.7 — Multi-contaminant confounder filtering
- Repeated shuffled GroupKFold (10 folds, `shuffle=True`) with leakage checks
  (`group_kfold_train_test_overlap = 0`, `n_unique_fold_splits = 10`).
- Per sample: `[target_conc, mPELQ_Metals, TOC_pct]` → XGBoost → survival.
- TreeSHAP and Interventional SHAP stability criteria are evaluated
  **separately**; a sample is removed only when the Interventional assignment
  is stable across ≥ 8 of 10 OOF assignments.

### Step 3 — TEL/PEL derivation (Method B, SHAP-independent)
- EDS = `survival < 80`; NEDS = `survival ≥ 80` (survival-based, no SHAP).
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

# Steps 1–4: full Primary pipeline (SHAP-independent TEL/PEL derivation)
python code/run_pipeline_v4.py

# Sensitivity analyses (each writes to its own output subdirectory)
python code/bootstrap_tel_pel_ci.py          # → output/TEL_PEL_bootstrap_CI.csv
python code/mpelq_cutoff_sensitivity.py      # → output/mpelq_sensitivity/cutoff_*/
python code/anchor_species_sensitivity.py    # → output/anchor_sensitivity/{primary,with_anchor}/

# Automated validation (8-point checklist, exit 0 = all pass)
python code/verify_pipeline.py
```

### Output paths

| Analysis | Output location |
|----------|-----------------|
| Primary TEL/PEL | `output/Step3_TEL_PEL_{Source}.csv` |
| EDS/NEDS raw | `output/Step3_EDSNEDS_{Source}_{Substance}.csv` |
| Fold signatures | `output/Step2_7v4_FoldSignatures_{Source}_{Substance}.csv` |
| Diagnosis metadata | `output/Step2_7v4_Diagnosis_All.csv` |
| Bootstrap CI | `output/TEL_PEL_bootstrap_CI.csv` |
| mPELQ cutoff sensitivity | `output/mpelq_sensitivity/cutoff_{0.25,0.5,1.0}/` |
| Anchor-species sensitivity | `output/anchor_sensitivity/{primary,with_anchor}/` |

Sensitivity analyses write to **separate subdirectories** so they never
overwrite the Primary results.

## Validation checkpoint

`verification/` holds the frozen validation results for the analysis-freeze
commit. The automated checklist (`code/verify_pipeline.py`) asserts:

1. `n_unique_fold_splits = 10`
2. `group_kfold_train_test_overlap = 0`
3. OOF count = 10/10
4. Interventional stability ≥ 8/10 (absolute count)
5. Config values match actual run settings
6. Primary / anchor-species / mPELQ sensitivity outputs are separated
7. Bootstrap and mPELQ sensitivity results are reproducible
8. TEL/PEL regression checks pass for DDTs, CHLs, and PCBs

## Method A vs Method B

- **Method A (log-space)**: percentiles computed on `log10(conc + 1)`, then
  back-transformed.
- **Method B (original units)**: percentiles computed on original
  concentrations (µg/kg dw), then geometric mean. This follows the CCME
  guideline convention and is the method used for the manuscript values.

The current `run_pipeline_v4.py` implements **Method B**.

## License

MIT
