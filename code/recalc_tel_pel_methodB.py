#!/usr/bin/env python3
"""
Method B 재계산: 원래 단위(µg/kg dw)에서 TEL/PEL 도출
=====================================================
run_pipeline_v3.py Step 3는 로그공간(Method A)으로 TEL/PEL을 계산한다:
    tel_ml = sqrt(quantile(EDS_log, 0.15) * quantile(NEDS_log, 0.50))
    our_tel = 10 ** tel_ml - 1

그러나 논문 최종값(sqg_config.py EMP_TEL/EMP_PEL)은 CCME 지침(Method B)에 따라
"원래 단위(µg/kg dw)에서 percentile 계산 후 기하평균"한 값이다:
    TEL = sqrt(EDS_P15_원래단위 × NEDS_P50_원래단위)
    PEL = sqrt(EDS_P50_원래단위 × NEDS_P85_원래단위)

이 스크립트는 run_pipeline_v3.py가 저장한 EDS/NEDS 데이터
(output_v3/edsneds_data/Integrated_{sub}_edsneds.csv)를 읽어
Method B로 재계산하고, 논문 확정값(sqg_config.py)과 대조 검증한다.

입력: output_v3/edsneds_data/Integrated_{DDTs,CHLs,PCBs}_edsneds.csv
출력: output_v3/TEL_PEL_MethodB_final.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent  # repo 루트
EDSNEDS_DIR = BASE / "output" / "edsneds_data"
OUT_DIR = BASE / "output"

# 논문 확정값 (sqg_config.py EMP_TEL / EMP_PEL, 단위 µg/kg dw = ng/g dw)
# DDTs/CHLs/PCBs: Method B 재계산값 (이 스크립트의 검증 대상)
PAPER = {
    "DDTs": {"TEL": 4.9834, "PEL": 21.8168},
    "CHLs": {"TEL": 3.9143, "PEL": 23.8046},
    "PCBs": {"TEL": 8.5557, "PEL": 51.6125},
}

# Method B 재계산 대상 물질
METHOD_B_SUBSTANCES = ["DDTs", "CHLs", "PCBs"]


def method_b_tel_pel(conc_log_eds, conc_log_neds):
    """
    Method B (CCME 지침): 원래 단위에서 percentile 계산 후 기하평균.
    Conc_log = log10(원래농도 + 1) → 원래농도 = 10^Conc_log - 1
    """
    eds_orig = 10 ** conc_log_eds - 1
    neds_orig = 10 ** conc_log_neds - 1
    tel = np.sqrt(np.quantile(eds_orig, 0.15) * np.quantile(neds_orig, 0.50))
    pel = np.sqrt(np.quantile(eds_orig, 0.50) * np.quantile(neds_orig, 0.85))
    return tel, pel


def method_a_tel_pel(conc_log_eds, conc_log_neds):
    """Method A (로그공간): run_pipeline_v3.py Step 3 원래 로직."""
    tel_ml = np.sqrt(np.quantile(conc_log_eds, 0.15) * np.quantile(conc_log_neds, 0.50))
    pel_ml = np.sqrt(np.quantile(conc_log_eds, 0.50) * np.quantile(conc_log_neds, 0.85))
    return 10 ** tel_ml - 1, 10 ** pel_ml - 1


def main():
    print("=" * 70)
    print("Method B 재계산: 원래 단위(µg/kg dw) TEL/PEL 도출")
    print("=" * 70)

    rows = []
    for sub in METHOD_B_SUBSTANCES:
        f = EDSNEDS_DIR / f"Integrated_{sub}_edsneds.csv"
        if not f.exists():
            print(f"  [SKIP] {f.name} 없음")
            continue

        df = pd.read_csv(f)
        eds = df[df["Data_Type"] == "EDS"]["Conc_log"].values
        neds = df[df["Data_Type"] == "NEDS"]["Conc_log"].values
        noise = int((df["Data_Type"] == "Noise").sum())

        tel_b, pel_b = method_b_tel_pel(eds, neds)
        tel_a, pel_a = method_a_tel_pel(eds, neds)

        p_tel = PAPER[sub]["TEL"]
        p_pel = PAPER[sub]["PEL"]
        match_tel = abs(tel_b - p_tel) < 0.0001
        match_pel = abs(pel_b - p_pel) < 0.0001

        print(f"\n▶ {sub}  (EDS={len(eds)}, NEDS={len(neds)}, Noise={noise})")
        print(f"  Method A (로그공간): TEL={tel_a:.4f}, PEL={pel_a:.4f}")
        print(f"  Method B (원래단위): TEL={tel_b:.4f}, PEL={pel_b:.4f}")
        print(f"  논문 확정값:         TEL={p_tel:.4f}, PEL={p_pel:.4f}")
        print(f"  Method B 일치: TEL={'✓' if match_tel else '✗'}, "
              f"PEL={'✓' if match_pel else '✗'}")

        rows.append({
            "Substance": sub,
            "N_EDS": len(eds),
            "N_NEDS": len(neds),
            "N_Noise": noise,
            "TEL_MethodA_log": round(tel_a, 4),
            "PEL_MethodA_log": round(pel_a, 4),
            "TEL_MethodB_orig": round(tel_b, 4),
            "PEL_MethodB_orig": round(pel_b, 4),
            "TEL_paper": p_tel,
            "PEL_paper": p_pel,
            "TEL_match": match_tel,
            "PEL_match": match_pel,
        })

    out = pd.DataFrame(rows)
    out_path = OUT_DIR / "TEL_PEL_MethodB_final.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out_path}")

    all_match = out["TEL_match"].all() and out["PEL_match"].all()
    print(f"\n{'='*70}")
    print(f"최종 판정: Method B 재계산 {'전부 논문값과 일치 ✓' if all_match else '불일치 존재 ✗'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
