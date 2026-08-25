#!/usr/bin/env python3
"""
mPELQ cutoff 민감도 분석
========================

mPELQ 중금속 사전 필터의 cutoff를 0.25 / 0.5 / 1.0으로 바꿔가며
전체 파이프라인(Step 1 → 2 → 2.5 → 3)을 재실행하여 TEL/PEL의 민감도를 평가한다.

- cutoff 0.5 = Primary (기본값)
- cutoff 0.25 = 더 보수적 (금속 독성 의심 샘플 더 많이 제거)
- cutoff 1.0 = 사실상 필터 해제 (모든 샘플 유지)

각 cutoff에서 N_total / N_EDS / N_NEDS / TEL / PEL을 산출하고,
Primary(0.5) 대비 변화율(%)을 계산한다.

재현성: run_pipeline_v4.py의 함수를 import하여 동일 로직 재사용.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# run_pipeline_v4.py를 import (같은 code/ 디렉토리)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_pipeline_v4 as rp

OUTPUT_DIR = rp.OUTPUT_DIR
CUTOFFS = [0.25, 0.5, 1.0]
SOURCES = [
    ("NOAA", rp.NOAA_DB_PATH),
    ("SCCWRP", rp.SCCWRP_DB_PATH),
    ("Integrated", rp.INTEGRATED_DB_PATH),
]


def run_pipeline_for_cutoff(source_name, db_path, cutoff):
    """특정 mPELQ cutoff로 전체 파이프라인 실행 → TEL/PEL 결과 DataFrame 반환."""
    df_raw = pd.read_csv(db_path, low_memory=False)
    df_base, substance_dfs = rp.step1_db_curation(df_raw, source_name,
                                                  mpelq_threshold=cutoff)
    df_eval = rp.step2_species_selection(substance_dfs, source_name)
    cleaned_dfs, drc_df, drc_ok_species = rp.step2_5_drc_evaluation(
        substance_dfs, df_eval, source_name)
    final, edsneds_data = rp.step3_tel_pel(cleaned_dfs, drc_ok_species, source_name)
    return final


def main():
    rows = []
    for source_name, db_path in SOURCES:
        for cutoff in CUTOFFS:
            print(f"\n=== {source_name} / mPELQ cutoff = {cutoff} ===")
            final = run_pipeline_for_cutoff(source_name, db_path, cutoff)
            for _, r in final.iterrows():
                rows.append({
                    "Source": source_name,
                    "mPELQ_cutoff": cutoff,
                    "Substance": r["Substance"],
                    "N_total": r["Total_N"],
                    "N_EDS": r["N_EDS"],
                    "N_NEDS": r["N_NEDS"],
                    "TEL": r["Our_TEL_dw"],
                    "PEL": r["Our_PEL_dw"],
                })

    df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "mPELQ_cutoff_sensitivity.csv"
    df.to_csv(out_path, index=False)

    # Primary(0.5) 대비 변화율 계산
    print(f"\n{'='*70}")
    print("mPELQ cutoff 민감도 요약 (Primary=0.5 대비 변화율)")
    print(f"{'='*70}")
    summary_rows = []
    for source_name, _ in SOURCES:
        for substance in rp.TARGET_SUBSTANCES:
            base = df[(df["Source"] == source_name) &
                      (df["Substance"] == substance) &
                      (df["mPELQ_cutoff"] == 0.5)]
            if base.empty:
                continue
            base_tel = base["TEL"].iloc[0]
            base_pel = base["PEL"].iloc[0]
            base_n = base["N_total"].iloc[0]
            for cutoff in CUTOFFS:
                row = df[(df["Source"] == source_name) &
                         (df["Substance"] == substance) &
                         (df["mPELQ_cutoff"] == cutoff)]
                if row.empty:
                    continue
                tel = row["TEL"].iloc[0]
                pel = row["PEL"].iloc[0]
                n = row["N_total"].iloc[0]
                tel_chg = (tel - base_tel) / base_tel * 100 if base_tel else np.nan
                pel_chg = (pel - base_pel) / base_pel * 100 if base_pel else np.nan
                n_chg = (n - base_n) / base_n * 100 if base_n else np.nan
                summary_rows.append({
                    "Source": source_name, "Substance": substance,
                    "mPELQ_cutoff": cutoff,
                    "N_total": n, "N_EDS": row["N_EDS"].iloc[0],
                    "N_NEDS": row["N_NEDS"].iloc[0],
                    "TEL": round(tel, 4), "PEL": round(pel, 4),
                    "TEL_chg_pct": round(tel_chg, 2),
                    "PEL_chg_pct": round(pel_chg, 2),
                    "N_chg_pct": round(n_chg, 2),
                })

    summary = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "mPELQ_cutoff_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"\n저장: {out_path}")
    print(f"저장: {summary_path}")


if __name__ == "__main__":
    main()
