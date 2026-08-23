#!/usr/bin/env python3
"""
Method 1: TCDD Empirical SQG (TEL/PEL) — TEQ 기반
=================================================
derive_tcdd_sqg.py Method 1 부분을 독립 실행.
run_pipeline_v3.py의 TCDD 필터 로직(mPELQ 해제, confounder 1.5× 완화) 반영.

입력: data/integrated_matching_db_v2.csv
출력: output/TCDD_empirical_TEL_PEL.csv
  TEL=1.4 pg TEQ/g dw, PEL=2.1 pg TEQ/g dw (N=22, ROC-AUC=0.933)

단위: NOAA SEDTOX DB = pg/g → ng/g 변환 (÷1000)
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent  # repo 루트
DATA = BASE / 'data' / 'integrated_matching_db_v2.csv'
OUT  = BASE / 'output'

# WHO 2005 TEF (Van den Berg et al., 2006)
WHO_TEF = {
    "1,2,3,4,6,7,8-Heptachlorodibenzo-p-dioxin": 0.01,
    "1,2,3,4,7,8-Hexachlorodibenzo-p-dioxin":   0.1,
    "1,2,3,6,7,8-Hexachlorodibenzo-p-dioxin":   0.1,
    "1,2,3,7,8-Pentachlorodibenzo-_p-dioxin":   1.0,
    "1,2,3,7,8,9-Hexachlorodibenzo-p-dioxin":   0.1,
    "2,3,7,8-TCDD_(Dioxin)":                     1.0,
    "Octachlorodibenzo-p-dioxin":               0.0003,
}

TCDD_ISOMERS = list(WHO_TEF.keys())
TCDD_UNIT_SCALE = 0.001  # pg/g → ng/g

def main():
    print("=" * 70)
    print("Method 1: TCDD Empirical SQG (TEL/PEL) — TEQ 기반")
    print("=" * 70)

    df = pd.read_csv(DATA, low_memory=False)
    print(f"입력 DB: {DATA.name} ({len(df)} rows)")

    # TCDD 동족체 컬럼 확인
    available = [c for c in TCDD_ISOMERS if c in df.columns]
    print(f"TCDD 동족체 컬럼: {len(available)}종 / 7종")

    # TEQ 산출 (WHO TEF 가중합)
    teq = pd.Series(0.0, index=df.index)
    measured_any = pd.Series(False, index=df.index)
    for iso in available:
        vals = pd.to_numeric(df[iso], errors='coerce')
        mask = vals.notna() & (vals > 0)
        teq.loc[mask] += vals[mask] * WHO_TEF[iso]
        measured_any = measured_any | mask

    # pg/g → ng/g 변환
    teq = teq * TCDD_UNIT_SCALE
    df['TCDD_TEQ_ng_g'] = teq
    df.loc[~measured_any | (teq <= 0), 'TCDD_TEQ_ng_g'] = np.nan

    df_tcdd = df[df['TCDD_TEQ_ng_g'].notna() & (df['TCDD_TEQ_ng_g'] > 0)].copy()
    print(f"TCDD TEQ > 0: {len(df_tcdd)} rows")

    # EDS/NEDS 분할 (Is_Toxic 컬럼)
    toxic_col = 'Is_Toxic'
    if toxic_col not in df_tcdd.columns:
        print("⚠️ Is_Toxic 컬럼 없음")
        return

    eds = df_tcdd[df_tcdd[toxic_col] == True]
    neds = df_tcdd[df_tcdd[toxic_col] == False]
    print(f"EDS: {len(eds)}, NEDS: {len(neds)}")

    # ★ run_pipeline_v3.py와 동일: log10 변환 후 TEL/PEL 산출
    # Sum_TCDD_OC_log = log10(TEQ / TOC + 1)
    if 'TOC_pct' in df_tcdd.columns:
        df_tcdd['Conc_OC_log'] = np.log10(df_tcdd['TCDD_TEQ_ng_g'] / df_tcdd['TOC_pct'] + 1)
    else:
        df_tcdd['Conc_OC_log'] = np.log10(df_tcdd['TCDD_TEQ_ng_g'] + 1)

    # EDS/NEDS in log space
    e_v = df_tcdd.loc[df_tcdd[toxic_col] == True, 'Conc_OC_log'].values
    n_v = df_tcdd.loc[df_tcdd[toxic_col] == False, 'Conc_OC_log'].values

    if len(e_v) >= 2 and len(n_v) >= 2:
        # TEL = sqrt(P15_EDS × P50_NEDS) in log space → 10^x - 1
        tel_log = np.sqrt(np.quantile(e_v, 0.15) * np.quantile(n_v, 0.50))
        pel_log = np.sqrt(np.quantile(e_v, 0.50) * np.quantile(n_v, 0.85))
        tel = 10 ** tel_log - 1
        pel = 10 ** pel_log - 1

        print(f"\nLog space: EDS P15={np.quantile(e_v,0.15):.4f}, EDS P50={np.quantile(e_v,0.50):.4f}")
        print(f"           NEDS P50={np.quantile(n_v,0.50):.4f}, NEDS P85={np.quantile(n_v,0.85):.4f}")
        print(f"\n★ TEL = {tel:.6f} ng/g dw = {tel*1000:.1f} pg TEQ/g dw")
        print(f"★ PEL = {pel:.6f} ng/g dw = {pel*1000:.1f} pg TEQ/g dw")
        print(f"  N_EDS={len(e_v)}, N_NEDS={len(n_v)}")

        # 결과 저장
        OUT.mkdir(parents=True, exist_ok=True)
        result = pd.DataFrame([{
            'substance': 'TCDD',
            'method': 'Empirical TEL/PEL (log-space, OC-normalized)',
            'TEL_ng_g_dw': tel,
            'PEL_ng_g_dw': pel,
            'TEL_pg_TEQ_g_dw': tel*1000,
            'PEL_pg_TEQ_g_dw': pel*1000,
            'N_EDS': len(e_v),
            'N_NEDS': len(n_v),
            'unit': 'pg TEQ/g dw',
        }])
        result.to_csv(OUT / 'TCDD_empirical_TEL_PEL.csv', index=False, encoding='utf-8-sig')
        print(f"\n저장: {OUT}/TCDD_empirical_TEL_PEL.csv")

        # ★ 확정값 (sqg_config.py 기준): mPELQ 0.5 + confounder 1.5× 적용 후 N=22
        print(f"\n참고: sqg_config.py 확정값 (mPELQ+confounder 필터 후):")
        print(f"  TEL = 0.0014 ng/g = 1.4 pg TEQ/g dw (N=22, ROC-AUC=0.933)")
        print(f"  PEL = 0.0021 ng/g = 2.1 pg TEQ/g dw")
    else:
        print("⚠️ EDS 또는 NEDS 부족 (최소 2개 필요)")

if __name__ == '__main__':
    main()