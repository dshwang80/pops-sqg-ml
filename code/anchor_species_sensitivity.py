#!/usr/bin/env python3
"""
Anchor species (표준 시험종) 민감도 분석
========================================

Primary에서는 Leptocheirus plumulosus 강제 추가를 하지 않는다
(모든 종에 동일한 Step 2·2.5 선정 기준 적용).

이 스크립트는 Sensitivity 분석으로, 표준 시험종 L. plumulosus를
anchor로 강제 추가했을 때 TEL/PEL이 어떻게 변하는지 평가한다.

- Primary (anchor 없음): Step 2·2.5 선정 기준 통과 종만 사용
- Sensitivity (anchor 있음): L. plumulosus를 DRC 불량이어도 강제 포함

결과가 거의 같다면 robustness 근거로 Supplementary에 보고한다.

★ 출력 분리: Primary와 Sensitivity 산출물이 서로 덮어쓰지 않도록
  output/anchor_sensitivity/primary/ 와 output/anchor_sensitivity/with_anchor/
  로 분리 저장한다. 비교표(AnchorSpecies_Sensitivity_Comparison.csv)는
  output/ 루트에 저장한다.

재현성: run_pipeline_v4.py의 함수를 import하여 동일 로직 재사용.
anchor_species 파라미터는 step2_5_drc_evaluation에만 전달된다.
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# run_pipeline_v4.py를 import (같은 code/ 디렉토리)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_pipeline_v4 as rp

OUTPUT_DIR = rp.OUTPUT_DIR
CONFIG_PATH = Path(__file__).resolve().parent / "pipeline_v4_config.json"

# ★ 출력 분리 디렉토리
PRIMARY_OUT = OUTPUT_DIR / "anchor_sensitivity" / "primary"
ANCHOR_OUT = OUTPUT_DIR / "anchor_sensitivity" / "with_anchor"

SOURCES = [
    ("NOAA", rp.NOAA_DB_PATH),
    ("SCCWRP", rp.SCCWRP_DB_PATH),
    ("Integrated", rp.INTEGRATED_DB_PATH),
]


def load_anchor_config():
    """config에서 anchor_species_sensitivity 설정을 읽는다."""
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    anchor_cfg = cfg.get("anchor_species_sensitivity", {})
    if not anchor_cfg.get("enabled", False):
        return None
    species = anchor_cfg.get("species", [])
    substances = anchor_cfg.get("substances", [])
    # {substance: [species, ...]} 형태로 변환
    return {sub: species for sub in substances}


def run_pipeline(source_name, db_path, anchor_species, out_dir):
    """전체 파이프라인 실행 → (TEL/PEL 결과, DRC 품질 DataFrame) 반환.

    out_dir: 산출물 저장 디렉토리 (Primary/Anchor 분리용).
    """
    # ★ rp.OUTPUT_DIR을 임시로 out_dir로 전환 (덮어쓰기 방지)
    original_out = rp.OUTPUT_DIR
    rp.OUTPUT_DIR = out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        df_raw = pd.read_csv(db_path, low_memory=False)
        df_base, substance_dfs = rp.step1_db_curation(df_raw, source_name)
        df_eval = rp.step2_species_selection(substance_dfs, source_name)
        cleaned_dfs, drc_df, drc_ok_species = rp.step2_5_drc_evaluation(
            substance_dfs, df_eval, source_name, anchor_species=anchor_species)
        final, edsneds_data = rp.step3_tel_pel(cleaned_dfs, drc_ok_species, source_name)
    finally:
        rp.OUTPUT_DIR = original_out
    return final, drc_df


def anchor_status(drc_df, anchor_species_list):
    """drc_df에서 anchor 종의 DRC 상태와 N을 추출.

    반환: (N_total, DRC_status)
      - N_total: anchor 종의 총 샘플 수 (drc_df에 없으면 NaN)
      - DRC_status: "DRC_OK" / "DRC_FAIL" / "Not_Evaluated" / "Not_Selected"
    """
    if drc_df is None or drc_df.empty:
        return np.nan, "Not_Evaluated"
    anchor_rows = drc_df[drc_df["Species"].isin(anchor_species_list)]
    if anchor_rows.empty:
        return np.nan, "Not_Selected"
    n_total = anchor_rows["N_total"].sum()
    if (anchor_rows["DRC_OK"] == True).any():
        status = "DRC_OK"
    else:
        status = "DRC_FAIL"
    return n_total, status


def main():
    anchor_species = load_anchor_config()
    if anchor_species is None:
        print("anchor_species_sensitivity.enabled = false → 분석 건너뜀")
        return

    # anchor 종 전체 목록 (상태 조회용)
    anchor_species_list = sorted(set(
        sp for spp in anchor_species.values() for sp in spp))

    print("=" * 70)
    print("Anchor species 민감도 분석: L. plumulosus 포함 vs 미포함")
    print(f"anchor 설정: {anchor_species}")
    print(f"Primary 출력: {PRIMARY_OUT}")
    print(f"Anchor 출력: {ANCHOR_OUT}")
    print("=" * 70)

    rows = []
    for source_name, db_path in SOURCES:
        # Primary (anchor 없음)
        print(f"\n=== {source_name} / Primary (anchor 없음) ===")
        final_primary, drc_primary = run_pipeline(
            source_name, db_path, anchor_species=None, out_dir=PRIMARY_OUT)

        # Sensitivity (anchor 있음)
        print(f"\n=== {source_name} / Sensitivity (anchor 있음) ===")
        final_sens, drc_sens = run_pipeline(
            source_name, db_path, anchor_species=anchor_species, out_dir=ANCHOR_OUT)

        # 비교
        for _, rp_row in final_primary.iterrows():
            sub = rp_row["Substance"]
            rs_row = final_sens[final_sens["Substance"] == sub]
            if rs_row.empty:
                continue
            rs = rs_row.iloc[0]
            tel_primary = rp_row["Our_TEL_dw"]
            pel_primary = rp_row["Our_PEL_dw"]
            tel_sens = rs["Our_TEL_dw"]
            pel_sens = rs["Our_PEL_dw"]
            tel_delta_pct = (tel_sens - tel_primary) / tel_primary * 100 if tel_primary else np.nan
            pel_delta_pct = (pel_sens - pel_primary) / pel_primary * 100 if pel_primary else np.nan

            # anchor 종의 Primary 상태 (N, DRC status)
            anchor_n_primary, anchor_drc_primary = anchor_status(
                drc_primary, anchor_species_list)
            anchor_n_sens, _ = anchor_status(drc_sens, anchor_species_list)

            rows.append({
                "Source": source_name,
                "Substance": sub,
                "N_primary": rp_row["Total_N"],
                "N_sens": rs["Total_N"],
                "N_EDS_primary": rp_row["N_EDS"],
                "N_EDS_sens": rs["N_EDS"],
                "N_NEDS_primary": rp_row["N_NEDS"],
                "N_NEDS_sens": rs["N_NEDS"],
                "Anchor_N_primary": anchor_n_primary,
                "Anchor_N_sensitivity": anchor_n_sens,
                "Anchor_DRC_status_primary": anchor_drc_primary,
                "TEL_primary": tel_primary,
                "TEL_sens": tel_sens,
                "TEL_delta_pct": round(tel_delta_pct, 3),
                "PEL_primary": pel_primary,
                "PEL_sens": pel_sens,
                "PEL_delta_pct": round(pel_delta_pct, 3),
            })

    out_df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "AnchorSpecies_Sensitivity_Comparison.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n{'=' * 70}")
    print(f"결과 저장: {out_path}")
    print(f"{'=' * 70}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
