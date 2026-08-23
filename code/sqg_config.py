#!/usr/bin/env python3
"""
POPs_EQS 단일 설정 파일 (Single Source of Truth)
=================================================
모든 스크립트는 이 파일에서 TEL/PEL/SQG/TRG/BSAF 값을 import하여 사용.
값 변경 시 이 파일만 수정하면 모든 그림에 자동 반영됨.

단위:
  - 퇴적물 SQG: ng/g dw (= µg/kg dw)
  - TRG: mg/kg ww
  - 해수 기준: ng/L
"""

import os
import re
import pandas as pd
import numpy as np
import openpyxl
from pathlib import Path

# ============================================================
# 경로 — repo 루트 기준 (release: code/ data/ output/)
# 모니터링 데이터(한국)는 release에 미포함 — load_* 함수는 원본 repo에서만 사용
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent  # repo 루트
MON_DIR = BASE_DIR / 'data' / 'monitoring_Korea'
SED_CSV = MON_DIR / 'sediment_standard_db.csv'
BIO_CSV = MON_DIR / 'biota_standard_db.csv'
SW_CSV  = MON_DIR / 'seawater_standard_db.csv'
SS_XLSX = MON_DIR / '2013-2025특별관리해역-0260609.xlsx'  # 참고용

# ============================================================
# v4 CSV 로드 헬퍼 — ND(0)는 이미 LOQ/2로 전처리됨, 미측정=None 제외됨
# 컬럼: year, station_id, station_name, region, source, PCBs, DDTs, CHLs, ...
# 구 스크립트 호환을 위해 Year/Station/Source('Nationwide'/'Special') 이름으로 반환
# ============================================================
def _load_v4(csv_path, matrix):
    """v4 표준 DB CSV를 읽어 구 호환 컬럼명으로 반환."""
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df = df.rename(columns={
        'year': 'Year',
        'station_id': 'Station',
        'station_name': 'Station_Name',
    })
    df['Source'] = df['source'].map({'national': 'Nationwide', 'special': 'Special'})

    # ── Compatibility columns for legacy scripts (DDTs/PCBs/CHLs) ──
    # Standardized DB: all rows use ΣDDTs, ΣPCBs, ΣCHLs (no national/special split)
    # Old scripts expect DDTs, PCBs, CHLs — add them as aliases
    for sub, canonical in [('DDTs', '\u03a3DDTs'),
                           ('PCBs', '\u03a3PCBs'),
                           ('CHLs', '\u03a3CHLs')]:
        if sub not in df.columns and canonical in df.columns:
            df[sub] = df[canonical]

    return df

def load_sediment_csv():
    """전국연안 퇴적물 모니터링 (v4, national only)."""
    df = _load_v4(SED_CSV, 'sediment')
    return df[df['Source'] == 'Nationwide'].reset_index(drop=True)

def load_sediment_special():
    """특별관리해역 퇴적물 모니터링 (v4, special only)."""
    df = _load_v4(SED_CSV, 'sediment')
    df = df[df['Source'] == 'Special'].reset_index(drop=True)
    df['Area'] = df['Station_Name']
    return df

def load_sediment_all():
    """전국연안 + 특별관리해역 통합 퇴적물 데이터. Source 열로 구분."""
    df = _load_v4(SED_CSV, 'sediment')
    df['Area'] = df['Station_Name'].fillna('')
    return df

def load_seawater_special():
    """특별관리해역 해수 모니터링 (v4, special only)."""
    df = _load_v4(SW_CSV, 'seawater')
    df['Area'] = df['Station_Name']
    return df

def load_biota_csv():
    """전국연안 biota 모니터링 (v4, national only, special 제외)."""
    df = _load_v4(BIO_CSV, 'biota')
    return df[df['Source'] == 'Nationwide'].reset_index(drop=True)

# ============================================================
# 물질 키 매핑
# ============================================================
SUBSTANCES = ['DDTs', 'CHLs', 'PCBs', 'TCDD']
SUB_LABELS = {'DDTs': 'ΣDDTs', 'CHLs': 'ΣCHLs', 'PCBs': 'ΣPCBs', 'TCDD': '2,3,7,8-TCDD'}
# 모니터링 컬럼명 → 내부 키
SUB_MAP = {'DDTs': 'DDT', 'CHLs': 'CHLs', 'PCBs': 'PCB', 'TCDD': 'TCDD'}

# ============================================================
# 1. Empirical SQG (Method 1) — ng/g dw
# ============================================================
# 본 연구 도출값 (Integrated TEL/PEL) — 전물질 EDS/NEDS 데이터 기반
# CCME 지침 준수: 원래 단위(ug/kg dw)에서 기하평균
#   TEL = sqrt(EDS P15 × NEDS P50), PEL = sqrt(EDS P50 × NEDS P85)
# 로그 공간이 아닌 원래 단위에서 percentile 계산 후 기하평균 (Method B)
# Method 1: Empirical SQG (TEL/PEL) — TEQ 기반 (2,3,7,8-TCDD TEF=1.0)
#   mPELQ 0.5 + confounder 1.5×, N=22, ROC-AUC 0.933
#   integrated_matching_db_v2.csv: NOAA SEDTOX 단위 = pg/g → ng/g 변환
EMP_TEL = {'DDT': 4.9834, 'CHLs': 3.9143, 'PCB': 8.5557, 'TCDD': 0.0014}   # TCDD: 1.4 pg TEQ/g dw
EMP_PEL = {'DDT': 21.8168, 'CHLs': 23.8046, 'PCB': 51.6125, 'TCDD': 0.0021}  # TCDD: 2.1 pg TEQ/g dw

# v3 pipeline TEL (개별 모델) — 원래 단위 기하평귭으로 재계산 (Method B, CCME 지침)
V3_TEL = {
    'NOAA':       {'DDTs': 4.6183, 'CHLs': 3.0118, 'PCBs': 8.1006, 'TCDD': 0.0014},
    'SCCWRP':     {'DDTs': 7.7567, 'CHLs': 3.8084, 'PCBs': 18.2456},
    'Integrated': {'DDTs': 4.9834, 'CHLs': 3.9143, 'PCBs': 8.5557, 'TCDD': 0.0014},
}
V3_PEL = {
    'NOAA':       {'DDTs': 22.8303, 'CHLs': 35.6132, 'PCBs': 52.1772, 'TCDD': 0.0021},
    'SCCWRP':     {'DDTs': 32.7149, 'CHLs': 10.9047, 'PCBs': 38.8303},
    'Integrated': {'DDTs': 21.8168, 'CHLs': 23.8046, 'PCBs': 51.6125, 'TCDD': 0.0021},
}

# ============================================================
# 2. CCME / NOAA / ANZECC — ng/g dw (참고용 국제 기준)
# ============================================================
CCME_ISQG = {'DDT': 1.19, 'CHLs': 2.26, 'PCB': 21.5, 'TCDD': 0.00085}
CCME_PEL  = {'DDT': 4.77, 'CHLs': 4.79, 'PCB': 189.0, 'TCDD': 0.0215}

NOAA_ERL = {'DDT': 1.58, 'CHLs': 0.5,  'PCB': 22.7}
NOAA_ERM = {'DDT': 46.1, 'CHLs': 6.0,  'PCB': 180.0}

ANZECC_ISQG  = {'DDT': 1.2,  'CHLs': 4.5,  'PCB': 34.0}
ANZECC_HIGH  = {'DDT': 5.0,  'CHLs': 9.0,  'PCB': 280.0}

# TCDD 국제 기준 (pg TEQ/g dw → ng TEQ/g dw; TEQ 기반, 참고용)
# CCME PCDD/F: ISQG=0.85 pg TEQ/g, PEL=21.5 pg TEQ/g
# 호주 Manning 2023: SQGV=65 pg TEQfish/g, SQG-high=215 pg TEQfish/g
# 주의: TEQ 기반이므로 단일 2,3,7,8-TCDD와 직접 비교 불가
CCME_TCDD_ISQG_TEQ = 0.00085   # ng TEQ/g dw
CCME_TCDD_PEL_TEQ  = 0.0215    # ng TEQ/g dw

# ============================================================
# 3. TRA 파라미터 (Method 3)
# ============================================================
# BSAF P95 (L/kg OC)
# DDT/PCB/CHLs: 문헌 원시데이터 해양종 P95 (extract_bsaf_marine.py)
# TCDD: Manning 2023 Sydney Harbor 현장데이터 95th percentile = 0.1
#   선행연구진 엑셀(1. 2378-TCDD_240930.xlsx) 채택값
#   이전: ERED DB 어류 n=15 P95=0.22 → Manning 현장 BSAF=0.1로 변경 (2026-07-29)
BSAF_P95 = {'DDT': 1.44, 'PCB': 5.90, 'CHLs': 4.98, 'TCDD': 0.1}

# ============================================================
# 4. TCDD 준거치 (3가지 방법 통합)
# ============================================================
# Method 1: Empirical SQG (TEL/PEL) — integrated_matching_db_v2.csv
#   EDS=735, NEDS=483, NOAA SEDTOX TCDD 단위 = pg/g → ng/g 변환
# Method 2: FACR/EqP — WQC × Koc × f_oc / 1000
#   log Kow=6.8 (HSDB), log Koc=6.59 (Di Toro), WQC=10 pg/L (EPA R4)
# Method 3: TRA (TRG) — 엑셀 LR50 24행 lnorm SSD HC1(99%)
#   TEQ 기반: 2,3,7,8-TCDD WHO TEF=1.0 → pg/g = pg TEQ/g
#   HC1(lipid) = 436.1093 pg/g lipid (lnorm 단일 모델, 24행)
#   TRG(ww) = HC1 × f_lipid = 436.1093 × 0.059 = 25.7304 pg TEQ/g ww
#   이전: ERED NEF 11종 SSD HC5 = 34.4 pg/g ww → LR50 HC1 기반으로 변경 (2026-07-29)
TCDD_TRG = 0.0000257304   # mg/kg ww (= 25.7304 pg TEQ/g ww)  Method 3 — LR50 HC1 × f_lipid

# ★f_lipid=5.9% (SSD 실측 종 13종 median, Manning 2023 Table 2) — 2026-07-27 확정
#   공식: C_sediment = HC1(lipid) × f_oc / BSAF → pg TEQ/g dw
#   436.1093 × 0.01 / 0.1 = 43.6109 pg TEQ/g dw
#   (ww 경유: TRG(ww) × f_oc / (BSAF × f_lipid)로 동일 — f_lipid 상쇄)
#   이전: TRG=34.4(ERED 11종 HC5), BSAF=0.22 → 26.5 pg/g dw (2026-07-29 변경)
F_LIPID_TCDD = 0.059    # 5.9% (어류 SSD egg median, Manning 2023)

# TRG (mg/kg ww) — 본 연구 도출값
# DDT: CBR(3.73) ÷ AF(50) = 0.0746 (SSD 불가, CBR÷AF fallback)
# PCB: SSD HC5 = 0.6689
# CHLs: CBR(0.13) ÷ AF(50) = 0.0026 (SSD 불가, CBR÷AF fallback)
# TCDD: SSD HC1 = 0.0000257304 (LR50 24행 lnorm, 99% 보호, HC1×f_lipid)
TRG = {'DDT': 0.0746, 'PCB': 0.6689, 'CHLs': 0.0026, 'TCDD': 0.0000257304}

# 지방 함량 / 유기탄소 함량
# DDT/PCB/CHLs: f_lipid = 2% (무척추생물종 기반)
# TCDD: f_lipid = 5.9% (어류 SSD egg median, Manning 2023) — 별도 정의 (F_LIPID_TCDD)
F_LIPID = 0.02   # 2% (무척추생물종, DDT/PCB/CHLs 공통)
F_OC = 0.01      # 1% (1%OC 정규화)

# TRA-SQG = TRG / (BSAF_P95 × F_LIPID/F_OC) × 1000 → ng/g dw
TRA_SQG = {}
for chem in ['DDT', 'PCB', 'CHLs']:
    TRA_SQG[chem] = TRG[chem] / (BSAF_P95[chem] * F_LIPID / F_OC) * 1000
# DDT: 25.90, PCB: 56.69, CHLs: 0.26
# TCDD: f_lipid=5.9% (어류 SSD egg median) 별도 적용
#   C_sediment = TRG × f_oc / (BSAF × f_lipid) = 0.0257304 × 0.01 / (0.1 × 0.059) × 1000
TCDD_TRA_SQG = TCDD_TRG * 1000 * F_OC / (BSAF_P95['TCDD'] * F_LIPID_TCDD)  # ng/g dw = 0.0436
TRA_SQG['TCDD'] = TCDD_TRA_SQG  # 0.0436 ng/g dw = 43.6 pg TEQ/g dw

# ============================================================
# 4. 해수 기준 (ng/L)
# ============================================================
WQC = {'DDTs': 31.0, 'CHLs': 19.0, 'PCBs': 201.7, 'TCDD': 0.296}        # 본 연구 도출 (TCDD: ECOTOX R코드 SSD 만성 HC1, 어류 7종 AhR 필터, 99% 보호, 296 pg/L=0.296 ng/L)
EPA_Chronic = {'DDTs': 1.0, 'CHLs': 4.3, 'PCBs': 30.0}   # EPA reference

# EqP SQG: WQC × K_OC × F_OC / 1000 → ng/g dw
K_OC = {'DDTs': 155000, 'CHLs': 400000, 'PCBs': 530000, 'TCDD': 3890451}
# Method 2: EqP SQG — WQC × Koc × f_oc / 1000 → ng/g dw
#   TCDD: WQC=0.296 ng/L (ECOTOX R코드 SSD 만성 HC1, 어류 7종, 99% 보호), Koc=3,890,451 L/kg, f_oc=0.01 → 11.52 pg TEQ/g dw
EqP_SQG = {s: WQC[s] * K_OC[s] * F_OC / 1000 for s in SUBSTANCES if s in WQC and s in K_OC}
# DDTs: 48.05, CHLs: 76.0, PCBs: 1069.01, TCDD: 0.01152

# ============================================================
# 5. 통합 딕셔너리 (스크립트별 호환용)
# ============================================================
# Empirical SQG (스크립트 공통 형태)
EMP_SQG = {
    'DDT':  {'TEL': EMP_TEL['DDT'],  'PEL': EMP_PEL['DDT'],
             'CCME_ISQG': CCME_ISQG['DDT'], 'CCME_PEL': CCME_PEL['DDT'],
             'Our_TEL': EMP_TEL['DDT'], 'Our_PEL': EMP_PEL['DDT']},
    'CHLs': {'TEL': EMP_TEL['CHLs'], 'PEL': EMP_PEL['CHLs'],
             'CCME_ISQG': CCME_ISQG['CHLs'], 'CCME_PEL': CCME_PEL['CHLs'],
             'Our_TEL': EMP_TEL['CHLs'], 'Our_PEL': EMP_PEL['CHLs']},
    'PCB':  {'TEL': EMP_TEL['PCB'],  'PEL': EMP_PEL['PCB'],
             'CCME_ISQG': CCME_ISQG['PCB'], 'CCME_PEL': CCME_PEL['PCB'],
             'Our_TEL': EMP_TEL['PCB'], 'Our_PEL': EMP_PEL['PCB']},
}

# TEL/PEL (SUBSTANCES 키 = DDTs/CHLs/PCBs 형태)
TEL = {'DDTs': EMP_TEL['DDT'], 'CHLs': EMP_TEL['CHLs'], 'PCBs': EMP_TEL['PCB']}
PEL = {'DDTs': EMP_PEL['DDT'], 'CHLs': EMP_PEL['CHLs'], 'PCBs': EMP_PEL['PCB']}

# TRA_SQG (SUBSTANCES 키 형태)
TRA_SQG_SUB = {'DDTs': TRA_SQG['DDT'], 'CHLs': TRA_SQG['CHLs'], 'PCBs': TRA_SQG['PCB']}

# ============================================================
# 6. 색상 (공통)
# ============================================================
SUBSTANCE_COLORS = {'DDTs': '#E74C3C', 'CHLs': '#27AE60', 'PCBs': '#3498DB'}

# ============================================================
# 검증 출력
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("sqg_config.py — Single Source of Truth")
    print("=" * 60)
    print(f"\n[Empirical SQG] ng/g dw")
    for k in ['DDT', 'CHLs', 'PCB']:
        print(f"  {k:6s}: TEL={EMP_TEL[k]:.2f}, PEL={EMP_PEL[k]:.2f}")
    print(f"\n[TRA-SQG (P95, f_lipid={F_LIPID})] ng/g dw")
    for k in ['DDT', 'CHLs', 'PCB']:
        print(f"  {k:6s}: {TRA_SQG[k]:.2f}")
    print(f"  {'TCDD':6s}: {TCDD_TRA_SQG:.4f} (f_lipid={F_LIPID_TCDD})")
    print(f"\n[EqP SQG] ng/g dw")
    for k in SUBSTANCES:
        print(f"  {k:6s}: {EqP_SQG[k]:.2f}")
    print(f"\n[WQC] ng/L")
    for k in SUBSTANCES:
        print(f"  {k:6s}: {WQC[k]:.2f}")
    print(f"\n[CSV] {SED_CSV}")
    print("=" * 60)