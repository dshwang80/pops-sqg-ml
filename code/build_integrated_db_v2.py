#!/usr/bin/env python3
"""
통합 DB v2: SCCWRP SQO + NOAA SEDDB → NOAA 호환 구조
핵심: SCCWRP를 sample×species 단위로展开 (NOAA와 동일 구조)
→ 동일 3단계 파이프라인이 NOAA / SCCWRP / 통합 모두에 작동
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path

# === Paths (상대경로 — repo 루트 기준) ===
# repo 구조: code/ (스크립트), data/ (입력), output/ (산출물)
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "output"
SQO_BASE = DATA_DIR / "SQO_database" / "SQO_database"
NOAA_PATH = DATA_DIR / "US_Sediment_Risk_Analytical_Set_Mainland.csv"

# === NOAA 호환 컬럼 (파이프라인이 요구하는 것) ===
# 파이프라인이 직접 사용하는 컬럼:
#   Station, Month, Year, Region, Start_Latitude, Start_Longitude,
#   TOC_pct, Mean_Survival, Is_Toxic, Latin_Name_Species,
#   METAL_COLS (9종), TARGET_ISOMERS (DDT 6, CHL 7, PCB 18, PAH 13, Dieldrin 1)

METAL_COLS = ["Arsenic", "Cadmium", "Chromium,_total", "Copper",
              "Lead", "Mercury", "Nickel", "Silver", "Zinc"]

# SCCWRP → NOAA 컬럼명 매핑 (개별 이성질체)
ISOMER_MAP = {
    # DDTs
    "4,4'-DDD": "p,p'-DDD",
    "4,4'-DDE": "p,p'-DDE",
    "4,4'-DDT": "p,p'-DDT",
    "2,4'-DDD": "o,p'-DDD",
    "2,4'-DDE": "o,p'-DDE",
    "2,4'-DDT": "o,p'-DDT",
    # CHLs
    "alpha-Chlordane": "Chlordane,_alpha-",
    "gamma-Chlordane": "Chlordane,_gamma-",
    "Heptachlor": "Heptachlor",
    "Heptachlor epoxide": "Heptachlor_epoxide",
    "cis-Nonachlor": "Nonachlor,_cis-",
    "trans-Nonachlor": "Nonachlor,_trans-",
    "Oxychlordane": "Oxychlordane",
    # PCBs (SCCWRP uses PCB001 → NOAA uses PCB001)
    # PAHs (SCCWRP names → NOAA names)
    "Naphthalene": "Naphthalene",
    "Acenaphthylene": "Acenaphthylene",
    "Acenaphthene": "Acenaphthene",
    "Fluorene": "Fluorene",
    "Phenanthrene": "Phenanthrene",
    "Anthracene": "Anthracene",
    "Fluoranthene": "Fluoranthene",
    "Pyrene": "Pyrene",
    "Benz(a)anthracene": "Benzo(a)anthracene",
    "Chrysene": "Chrysene",
    "Benzo(a)pyrene": "Benzo(a)pyrene",
    "Dibenz(a,h)anthracene": "Dibenzo(a,h)anthracene",
    "Benzo(g,h,i)perylene": "Benzo(g,h,i)perylene",
    # Dieldrin (단일)
    "Dieldrin": "Dieldrin",
}

# NOAA 컬럼명 기준 (파이프라인이 사용)
NOAA_ISOMER_COLS = (
    ["o,p'-DDD", "o,p'-DDE", "o,p'-DDT", "p,p'-DDD", "p,p'-DDE", "p,p'-DDT"]  # DDTs
    + ["Chlordane,_alpha-", "Chlordane,_gamma-", "Heptachlor",
       "Heptachlor_epoxide", "Nonachlor,_cis-", "Nonachlor,_trans-", "Oxychlordane"]  # CHLs
    + ["PCB008","PCB018","PCB028","PCB044","PCB052","PCB066","PCB101",
       "PCB105","PCB118","PCB128","PCB138","PCB153","PCB170","PCB180",
       "PCB187","PCB195","PCB206","PCB209"]  # PCBs
    + ["Naphthalene","Acenaphthylene","Acenaphthene","Fluorene",
       "Phenanthrene","Anthracene","Fluoranthene","Pyrene",
       "Benzo(a)anthracene","Chrysene","Benzo(a)pyrene",
       "Dibenzo(a,h)anthracene","Benzo(g,h,i)perylene"]  # PAHs
    + ["Dieldrin"]
)


def process_sccwrp():
    """SCCWRP SQO → NOAA 호환 wide format (sample × species)"""
    print("=" * 70)
    print("Step 1: Processing SCCWRP SQO → NOAA 호환 구조")
    print("=" * 70)

    # Load
    chem = pd.read_csv(SQO_BASE / "tblChemistryResults.csv.gz", low_memory=False)
    tox_sum = pd.read_csv(SQO_BASE / "tblToxicitySumResults.csv", low_memory=False)
    stations = pd.read_csv(SQO_BASE / "tblStation.csv", low_memory=False)
    samples = pd.read_csv(SQO_BASE / "tblSampleMaster.csv", low_memory=False)
    sp_list = pd.read_csv(SQO_BASE / "luList20_SpeciesList.csv")

    sp_map = dict(zip(sp_list['SpeciesCode'], sp_list['SpeciesName']))

    # ── Chemistry: SD, DW ──
    chem_sd = chem[
        (chem['MaterialCode'] == 'SD') &
        (chem['SampleType'] == 'SD_RESULT') &
        (chem['Measbasis'] == 'DW')
    ].copy()
    chem_sd['sample_key'] = (chem_sd['StudyID'].astype(str) + '|' +
                              chem_sd['StationID'].astype(str) + '|' +
                              chem_sd['SampleID'].astype(str))
    # Replace -99 with NaN
    chem_sd.loc[chem_sd['Result'] == -99, 'Result'] = np.nan

    # ── TOC (PCT units) ──
    toc = chem_sd[(chem_sd['ChemicalName'] == 'TOC') & (chem_sd['Units'] == 'PCT')]
    toc_df = toc.groupby('sample_key')['Result'].mean().reset_index()
    toc_df.columns = ['sample_key', 'TOC_pct']
    print(f"  TOC: {len(toc_df)} samples")

    # ── Metals (MG/KG units, SCCWRP "Chromium" → NOAA "Chromium,_total") ──
    SCCWRP_METAL_NAMES = {"Chromium": "Chromium,_total"}
    metal_data = {}
    for noaa_name in METAL_COLS:
        sccwrp_name = [k for k, v in SCCWRP_METAL_NAMES.items() if v == noaa_name]
        sccwrp_name = sccwrp_name[0] if sccwrp_name else noaa_name
        m = chem_sd[(chem_sd['ChemicalName'] == sccwrp_name) & (chem_sd['Units'] == 'MG/KG')]
        if len(m) > 0:
            m_df = m.groupby('sample_key')['Result'].mean().reset_index()
            m_df.columns = ['sample_key', noaa_name]
            metal_data[noaa_name] = m_df
            print(f"  {noaa_name}: {len(m_df)} samples")

    # ── Individual isomers (UG/KG) → NOAA 컬럼명 ──
    isomer_data = {}
    chem_ugkg = chem_sd[chem_sd['Units'] == 'UG/KG']
    for sccwrp_name, noaa_name in ISOMER_MAP.items():
        m = chem_ugkg[chem_ugkg['ChemicalName'] == sccwrp_name]
        if len(m) > 0:
            m_df = m.groupby('sample_key')['Result'].mean().reset_index()
            m_df.columns = ['sample_key', noaa_name]
            isomer_data[noaa_name] = m_df

    # PCB congeners: SCCWRP PCB001 → NOAA PCB001 (same naming)
    pcb_congeners = chem_ugkg[chem_ugkg['ChemicalName'].str.match(r'^PCB\d{3}$', na=False)]
    for pcb_col in ["PCB008","PCB018","PCB028","PCB044","PCB052","PCB066","PCB101",
                     "PCB105","PCB118","PCB128","PCB138","PCB153","PCB170","PCB180",
                     "PCB187","PCB195","PCB206","PCB209"]:
        m = pcb_congeners[pcb_congeners['ChemicalName'] == pcb_col]
        if len(m) > 0:
            m_df = m.groupby('sample_key')['Result'].mean().reset_index()
            m_df.columns = ['sample_key', pcb_col]
            isomer_data[pcb_col] = m_df

    print(f"  Isomers mapped: {len(isomer_data)} compounds")

    # ── Toxicity: per sample × species ──
    tox_sd = tox_sum[tox_sum['SampleType'] == 'SD_RESULT'].copy()
    toxic_codes = {'SC', 'SCT', 'SR', 'SRM', 'SRT'}
    nontoxic_codes = {'NSC', 'NSR', 'NCT', 'NRT'}
    exclude_codes = {'NR', 'X', 'NA', 'NR ', 'SRT '}

    def classify_toxicity(sig):
        if pd.isna(sig) or str(sig).strip() in exclude_codes:
            return None
        s = str(sig).strip()
        if s in toxic_codes:
            return True
        elif s in nontoxic_codes:
            return False
        return None

    tox_sd['Is_Toxic'] = tox_sd['SigEffect'].apply(classify_toxicity)
    tox_sd = tox_sd[tox_sd['Is_Toxic'].notna()].copy()
    tox_sd['Is_Toxic'] = tox_sd['Is_Toxic'].astype(bool)

    # Per sample × species aggregation
    tox_by_sp = tox_sd.groupby(['StudyID', 'StationID', 'SampleID', 'SpeciesCode']).agg(
        Is_Toxic=('Is_Toxic', 'any'),
        Mean_Survival=('PctControl', 'mean'),
        N_Replicates=('N', 'mean'),
        StdDev=('StdDev', 'mean'),
    ).reset_index()

    # Map SpeciesCode → Latin_Name_Species
    tox_by_sp['Latin_Name_Species'] = tox_by_sp['SpeciesCode'].map(sp_map)
    tox_by_sp['sample_key'] = (tox_by_sp['StudyID'].astype(str) + '|' +
                                tox_by_sp['StationID'].astype(str) + '|' +
                                tox_by_sp['SampleID'].astype(str))

    print(f"  Toxicity (sample×species): {len(tox_by_sp)}")
    print(f"    EDS: {int(tox_by_sp['Is_Toxic'].sum())}")
    print(f"    NEDS: {int((~tox_by_sp['Is_Toxic']).sum())}")
    print(f"    Species: {tox_by_sp['Latin_Name_Species'].nunique()}")

    # ── Station info ──
    stations['station_key'] = stations['StudyID'].astype(str) + '|' + stations['StationID'].astype(str)
    station_info = stations[['station_key', 'Latitude', 'Longitude']].copy()

    # ── Sample info (date) ──
    samples['sample_key'] = (samples['StudyID'].astype(str) + '|' +
                              samples['StationID'].astype(str) + '|' +
                              samples['SampleID'].astype(str))
    sample_info = samples[['sample_key', 'Date']].copy()

    # ── Build wide format: one row per sample × species ──
    print("\n  Building NOAA-compatible wide format...")
    result = tox_by_sp[['sample_key', 'StudyID', 'StationID', 'SampleID',
                         'Latin_Name_Species', 'Is_Toxic', 'Mean_Survival',
                         'N_Replicates', 'StdDev']].copy()

    # Add station key for lat/lon merge
    result['station_key'] = result['StudyID'].astype(str) + '|' + result['StationID'].astype(str)

    # Merge: TOC
    result = result.merge(toc_df, on='sample_key', how='left')

    # Merge: metals (SCCWRP "Chromium" → NOAA "Chromium,_total")
    METAL_NAME_MAP = {"Chromium": "Chromium,_total"}
    for metal in METAL_COLS:
        sccwrp_name = [k for k, v in METAL_NAME_MAP.items() if v == metal]
        sccwrp_name = sccwrp_name[0] if sccwrp_name else metal
        if sccwrp_name in metal_data:
            result = result.merge(metal_data[sccwrp_name], on='sample_key', how='left')
            if sccwrp_name != metal:
                result = result.rename(columns={sccwrp_name: metal})
        else:
            result[metal] = np.nan

    # Merge: isomers
    for col in NOAA_ISOMER_COLS:
        if col in isomer_data:
            result = result.merge(isomer_data[col], on='sample_key', how='left')
        else:
            result[col] = np.nan

    # Merge: lat/lon
    result = result.merge(station_info, on='station_key', how='left')

    # Merge: date
    result = result.merge(sample_info, on='sample_key', how='left')

    # Add NOAA-compatible columns
    result['Station'] = result['StudyID'].astype(str) + '_' + result['StationID'].astype(str)
    result['Region'] = 'SCCWRP'
    result['Start_Latitude'] = result['Latitude']
    result['Start_Longitude'] = result['Longitude']

    # Parse date → Month, Year
    result['Date'] = pd.to_datetime(result['Date'], errors='coerce')
    result['Month'] = result['Date'].dt.month
    result['Year'] = result['Date'].dt.year

    # SD_Survival
    result['SD_Survival'] = result['StdDev']

    # Cap Mean_Survival at 100
    result['Mean_Survival'] = result['Mean_Survival'].clip(upper=100)
    # Replace -99 / negative survival with NaN
    result.loc[result['Mean_Survival'] < 0, 'Mean_Survival'] = np.nan

    print(f"  Final SCCWRP wide format: {len(result)} rows × {len(result.columns)} cols")
    print(f"    With TOC: {result['TOC_pct'].notna().sum()}")
    print(f"    With survival: {result['Mean_Survival'].notna().sum()}")

    return result


def process_noaa():
    """NOAA SEDDB: already in wide format, just load"""
    print("\n" + "=" * 70)
    print("Step 2: Loading NOAA SEDDB")
    print("=" * 70)

    df = pd.read_csv(NOAA_PATH, low_memory=False)
    df['Source_DB'] = 'NOAA_SEDDB'
    print(f"  NOAA rows: {len(df):,}")
    print(f"  EDS: {int(df['Is_Toxic'].sum())}")
    print(f"  NEDS: {int((~df['Is_Toxic'].astype(bool)).sum())}")
    return df


def build_integrated(noaa_df, sccwrp_df):
    """Concatenate NOAA + SCCWRP into unified DB"""
    print("\n" + "=" * 70)
    print("Step 3: Integrating NOAA + SCCWRP")
    print("=" * 70)

    # SCCWRP에 NOAA에 없는 컬럼 추가 / 누락 컬럼 채우기
    noaa_cols = set(noaa_df.columns)
    sccwrp_cols = set(sccwrp_df.columns)

    # NOAA에만 있는 컬럼 → SCCWRP에 NaN으로 추가
    for col in noaa_cols - sccwrp_cols:
        sccwrp_df[col] = np.nan

    # SCCWRP에만 있는 컬럼 → NOAA에 NaN으로 추가
    for col in sccwrp_cols - noaa_cols:
        noaa_df[col] = np.nan

    # Source_DB
    sccwrp_df['Source_DB'] = 'SCCWRP_SQO'

    # Concat
    integrated = pd.concat([noaa_df, sccwrp_df], ignore_index=True)
    print(f"  Integrated: {len(integrated):,} rows")
    print(f"    NOAA: {(integrated['Source_DB']=='NOAA_SEDDB').sum():,}")
    print(f"    SCCWRP: {(integrated['Source_DB']=='SCCWRP_SQO').sum():,}")

    # Save
    out_path = OUT_DIR / "integrated_matching_db_v2.csv"
    integrated.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    # Also save SCCWRP-only
    sccwrp_out = OUT_DIR / "sccwrp_noaa_compatible.csv"
    sccwrp_df.to_csv(sccwrp_out, index=False)
    print(f"  SCCWRP-only: {sccwrp_out}")

    return integrated


if __name__ == "__main__":
    sccwrp = process_sccwrp()
    noaa = process_noaa()
    integrated = build_integrated(noaa, sccwrp)

    # Summary
    print("\n" + "=" * 70)
    print("최종 요약")
    print("=" * 70)
    for source in ['NOAA_SEDDB', 'SCCWRP_SQO']:
        sub = integrated[integrated['Source_DB'] == source]
        has_toc = sub['TOC_pct'].notna() & (sub['TOC_pct'] > 0)
        has_surv = sub['Mean_Survival'].notna() & (sub['Mean_Survival'] >= 0)
        has_both = has_toc & has_surv
        print(f"\n  {source}:")
        print(f"    Total rows: {len(sub):,}")
        print(f"    With TOC>0 & Survival≥0: {has_both.sum():,}")
        print(f"    EDS (toxic): {int(sub['Is_Toxic'].sum()):,}")
        print(f"    NEDS (non-toxic): {int((~sub['Is_Toxic'].astype(bool)).sum()):,}")
        print(f"    Species: {sub['Latin_Name_Species'].nunique()}")