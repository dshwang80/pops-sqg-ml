# pops-sqg-ml

ML-refined TEL/PEL derivation for persistent organic pollutant (POP) sediment quality guidelines (SQGs).

This repository contains the code and input data used to derive **ML-refined TEL (Threshold Effect Level)** and **PEL (Probable Effect Level)** values for five POP classes (DDTs, CHLs, PCBs, PAHs, Dieldrin) via an **XGBoost + SHAP** framework.

## Overview

The pipeline refines empirical sediment quality guidelines by separating **causal effect data (EDS)** from **noise (NEDS)** using SHAP feature-attribution values, then derives TEL/PEL from the geometric-mean of the classified concentration distributions.

### Pipeline (3 steps)

1. **Step 1 — DB curation**: negative-flag handling, isomer weighted summation, 1% organic-carbon normalization, heavy-metal mPELQ filtering.
2. **Step 2 — Species selection**: XGBoost SHAP directionality validation (SHAP ≤ −0.3) and TOC-dominance check.
3. **Step 3 — Final SQG derivation**: SHAP-based EDS/NEDS classification → TEL/PEL derivation.

## Requirements

```bash
pip install numpy pandas xgboost shap scipy matplotlib
```

## Usage

```bash
python code/sqg_empirical_xgboost_shap.py
```

The script reads `data/US_Sediment_Risk_Analytical_Set_Mainland.csv` and writes:
- `python/output/Master_Table_Python_Final.csv` — final TEL/PEL values
- `python/output/Step3_All_Classified_Data.csv` — per-point EDS/NEDS classification
- `python/figures/*.png` — SQG scatter, SHAP summary, R-vs-Python comparison

## Key parameters

| Parameter | Value |
|-----------|-------|
| Model | XGBoost (reg:squarederror) |
| eta / max_depth | 0.1 / 3 |
| Random seed | 42 |
| EDS threshold | Mean_Survival < 80% & SHAP_Chem ≤ −0.1 |
| NEDS threshold | Mean_Survival ≥ 80% |
| TEL (log) | √(P15_EDS × P50_NEDS) |
| PEL (log) | √(P50_EDS × P85_NEDS) |

## License

MIT
