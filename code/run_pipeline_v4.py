#!/usr/bin/env python3
"""
고도화 파이프라인 v3: 신뢰도 기반 TEL/PEL 도출
=============================================

[사용자 설계]
1. TEL/PEL 신뢰도 향상
   - DRC 보이는 생물종 선정 → 독성 인과관계 확인
   - EqP보다 낮은 독성값 → 신뢰도 낮음 → 제거 (Step 1 EqP 필터)
   - 다른 독성기여도 높은 자료 → 기계학습(SHAP) 선별 → 제거 (Step 2.7)

2. 준거치 신뢰도 판단
   - 기 설정 기준(CCME/NOAA)과 비교 검토 (Step 4-1)
   - 독성 예측력(ROC-AUC) 검토 (Step 4-2)

파이프라인 구조:
  Step 1: DB 정제 (EqP 필터 + mPELQ 중금속 사전 필터)
  Step 2: XGBoost SHAP 종 선별 (Spearman ≤ -0.3)
  Step 2.5: DRC 품질 평가 → DRC OK 종만 선별 + 가짜 독성 DROP
  Step 2.7: 다물질 SHAP 타 독성 기여도 평가 → confounder-dominated 샘플 DROP ★
  Step 3: SHAP EDS/NEDS → TEL/PEL
  Step 4: 신뢰도 평가 (기준 비교 + ROC-AUC 예측력) ★
"""
import warnings
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from scipy import stats
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, roc_curve
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)

# =============================================================================
# 0. 경로 및 상수
# =============================================================================
# repo 구조: code/ (스크립트), data/ (입력), output/ (산출물)
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NOAA_DB_PATH = DATA_DIR / "US_Sediment_Risk_Analytical_Set_Mainland.csv"
SCCWRP_DB_PATH = DATA_DIR / "sccwrp_noaa_compatible.csv"
INTEGRATED_DB_PATH = DATA_DIR / "integrated_matching_db_v2.csv"

TARGET_SUBSTANCES = ["DDTs", "CHLs", "PCBs", "TCDD"]

TARGET_ISOMERS = {
    "DDTs":      ["o,p'-DDD", "o,p'-DDE", "o,p'-DDT", "p,p'-DDD", "p,p'-DDE", "p,p'-DDT"],
    "CHLs":      ["Chlordane,_alpha-", "Chlordane,_gamma-", "Heptachlor",
                  "Heptachlor_epoxide", "Nonachlor,_cis-", "Nonachlor,_trans-", "Oxychlordane"],
    "PCBs":      ["PCB008","PCB018","PCB028","PCB044","PCB052","PCB066","PCB101",
                  "PCB105","PCB118","PCB128","PCB138","PCB153","PCB170","PCB180",
                  "PCB187","PCB195","PCB206","PCB209"],
    "TCDD":      ["1,2,3,4,6,7,8-Heptachlorodibenzo-p-dioxin",
                  "1,2,3,4,7,8-Hexachlorodibenzo-p-dioxin",
                  "1,2,3,6,7,8-Hexachlorodibenzo-p-dioxin",
                  "1,2,3,7,8-Pentachlorodibenzo-_p-dioxin",
                  "1,2,3,7,8,9-Hexachlorodibenzo-p-dioxin",
                  "2,3,7,8-TCDD_(Dioxin)",
                  "2,3,7,8-TCDF_(Tetrachlorodibenzofuran)",
                  "Octachlorodibenzo-p-dioxin"],
    "PAHs":      ["Naphthalene","Acenaphthylene","Acenaphthene","Fluorene",
                  "Phenanthrene","Anthracene","Fluoranthene","Pyrene",
                  "Benzo(a)anthracene","Chrysene","Benzo(a)pyrene",
                  "Dibenzo(a,h)anthracene","Benzo(g,h,i)perylene"],
}

METAL_COLS = ["Arsenic","Cadmium","Chromium,_total","Copper",
              "Lead","Mercury","Nickel","Silver","Zinc"]

PEL_METALS = {"Arsenic":41.6, "Cadmium":4.21, "Chromium,_total":160, "Copper":108,
              "Lead":112, "Mercury":0.7, "Nickel":42.8, "Silver":1.77, "Zinc":271}

CCME_ISQG = {"DDTs":1.19, "CHLs":2.26, "PAHs":1684, "Dieldrin":0.71, "PCBs":21.5, "TCDD":0.00085}
CCME_PEL = {"DDTs":4.77, "CHLs":4.79, "PAHs":16770, "Dieldrin":4.30, "PCBs":189, "TCDD":0.0215}
NOAA_TEL_DW = {"DDTs":3.89, "CHLs":2.26, "PAHs":1684, "Dieldrin":0.72, "PCBs":22.7, "TCDD":np.nan}
NOAA_PEL_DW = {"DDTs":51.7, "CHLs":4.79, "PAHs":16770, "Dieldrin":4.30, "PCBs":180, "TCDD":np.nan}

EXCLUDE_PATTERNS = ["Human liver cell line", "species not known", "Photobacterium",
                    "Crassostrea", "Echinoderm"]
RANDOM_SEED = 42

eqp_params = {
    'DDTs':     {'log_kow': 6.91, 'ccc_ugL': 0.001},
    'CHLs':     {'log_kow': 6.00, 'ccc_ugL': 0.004},
    'PCBs':     {'log_kow': 6.80, 'ccc_ugL': 0.03},
    'Dieldrin': {'log_kow': 5.37, 'ccc_ugL': 0.1469},
    'PAHs':     {'log_kow': 5.20, 'ccc_ugL': 2.322},
    # TCDD: EPA 공식 수생태계 WQC 없음 → EqP 필터 skip
}

# mPELQ threshold: 평균 중금속이 PEL의 50% 초과 → 타 독성 기여도 의심
# TCDD는 필터 미적용 (N 부족 → 모든 DB 사용)
MPELQ_METALS_THRESHOLD = 0.5
MPELQ_METALS_THRESHOLD_TCDD = 999.0  # 필터 사실상 해제

# 단위 변환: DB 원본 단위 → ng/g (µg/kg dw)
# TCDD/TEQ는 pg/g로 보고 → ng/g 변환 (÷1000)
UNIT_FACTOR = {
    "DDTs": 1.0, "CHLs": 1.0, "PCBs": 1.0, "PAHs": 1.0, "Dieldrin": 1.0,
    "TCDD": 0.001,  # pg/g → ng/g (TEQ에도 동일 적용)
}

# WHO 2005 TEF (Van den Berg et al., 2006) — TCDD 동족체 TEQ 환산
WHO_TEF = {
    "1,2,3,4,6,7,8-Heptachlorodibenzo-p-dioxin": 0.01,
    "1,2,3,4,7,8-Hexachlorodibenzo-p-dioxin":   0.1,
    "1,2,3,6,7,8-Hexachlorodibenzo-p-dioxin":   0.1,
    "1,2,3,7,8-Pentachlorodibenzo-_p-dioxin":   1.0,
    "1,2,3,7,8,9-Hexachlorodibenzo-p-dioxin":   0.1,
    "2,3,7,8-TCDD_(Dioxin)":                     1.0,
    "2,3,7,8-TCDF_(Tetrachlorodibenzofuran)":    0.1,
    "Octachlorodibenzo-p-dioxin":               0.0003,
}


# =============================================================================
# Step 1: DB 정제 (EqP 필터 + mPELQ 중금속 사전 필터)
# =============================================================================
def step1_db_curation(df_raw, source_name=""):
    print(f"\n{'='*70}")
    print(f"[Step 1] DB 정제 — {source_name}")
    print(f"{'='*70}")
    print(f"[1-1] 원본: {df_raw.shape[0]:,}행 × {df_raw.shape[1]}컬럼")

    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
    df_raw[numeric_cols] = df_raw[numeric_cols].replace(-9, np.nan)

    df_pre = df_raw.copy()
    df_pre = df_pre[df_pre["Mean_Survival"].notna() & (df_pre["Mean_Survival"] >= 0)]
    df_pre["Mean_Survival"] = df_pre["Mean_Survival"].clip(upper=100)
    df_pre = df_pre[df_pre["TOC_pct"].notna() & (df_pre["TOC_pct"] > 0)]
    print(f"[1-2] 기본 필터 후: {df_pre.shape[0]:,}행")

    for pat in EXCLUDE_PATTERNS:
        df_pre = df_pre[~df_pre["Latin_Name_Species"].str.contains(pat, na=False, case=False)]
    df_pre["Latin_Name_Species"] = df_pre["Latin_Name_Species"].apply(
        lambda x: "Chironomus dilutus" if isinstance(x, str) and "Chironomus" in x else x
    )
    print(f"[1-3] 생물종 정제 후: {df_pre.shape[0]:,}행, 종 수: {df_pre['Latin_Name_Species'].nunique()}")

    # 중금속 mPELQ 계산
    for m in METAL_COLS:
        if m not in df_pre.columns:
            df_pre[m] = np.nan
    metal_df = df_pre[METAL_COLS].copy().fillna(0)
    quotients = pd.DataFrame({m: metal_df[m] / PEL_METALS[m] for m in METAL_COLS})
    df_pre["mPELQ_Metals"] = quotients.mean(axis=1)

    # 이성질체 가중 합산
    def process_sum_refined(data, cols):
        complete_mask = data[cols].isna().sum(axis=1) == 0
        complete_df = data.loc[complete_mask, cols]
        if len(complete_df) >= 10:
            ref_means = complete_df.mean(axis=0, skipna=True)
            ref_weights = ref_means / ref_means.sum()
        else:
            ref_weights = pd.Series(1.0 / len(cols), index=cols)
        res = np.full(len(data), np.nan)
        for i in range(len(data)):
            row_vals = data.iloc[i][cols]
            measured_mask = row_vals.notna().values
            measured_cols = [c for c, m in zip(cols, measured_mask) if m]
            if len(measured_cols) < (len(cols) * 0.3):
                continue
            toc = data.iloc[i]["TOC_pct"]
            if pd.isna(toc) or toc <= 0:
                continue
            measured_sum = 0.0
            for col in measured_cols:
                val = row_vals[col]
                if val <= 0:
                    pos_vals = data[col][data[col] > 0]
                    mdl = pos_vals.min() if len(pos_vals) > 0 else 0.01
                    measured_sum += mdl / 2
                else:
                    measured_sum += val
            weight_sum = ref_weights[measured_cols].sum()
            if weight_sum <= 0:
                continue
            estimated_total = measured_sum / weight_sum
            res[i] = np.log10((estimated_total / toc) + 1)
        return res

    df_base = df_pre.copy()
    for substance in TARGET_SUBSTANCES:
        if substance == "Dieldrin":
            if "Dieldrin" not in df_base.columns:
                df_base["Dieldrin"] = np.nan
            df_base["Dieldrin"] = df_base["Dieldrin"] * UNIT_FACTOR.get(substance, 1.0)
            df_base["Sum_Dieldrin_OC_log"] = np.log10(
                (df_base["Dieldrin"] / df_base["TOC_pct"]) + 1)
            df_base.loc[df_base["Dieldrin"].isna(), "Sum_Dieldrin_OC_log"] = np.nan
        elif substance == "TCDD":
            # TCDD: 8종 동족체 WHO TEF 가중합 → TEQ (pg/g)
            # ND(음수)는 0 처리, 측정된 동족체만 가중합
            isomers = TARGET_ISOMERS["TCDD"]
            for iso in isomers:
                if iso not in df_base.columns:
                    df_base[iso] = np.nan
            teq = pd.Series(0.0, index=df_base.index)
            measured_any = pd.Series(False, index=df_base.index)
            for iso in isomers:
                vals = df_base[iso].copy()
                nd_mask = vals.isna() | (vals < 0)
                vals[nd_mask] = 0
                teq += vals * WHO_TEF[iso]
                measured_any |= ~nd_mask
            # pg/g → ng/g 변환
            teq = teq * UNIT_FACTOR["TCDD"]
            df_base["Sum_TCDD_OC_log"] = np.log10(
                (teq / df_base["TOC_pct"]) + 1)
            df_base.loc[~measured_any | (teq <= 0), "Sum_TCDD_OC_log"] = np.nan
        else:
            isomers = TARGET_ISOMERS[substance]
            for iso in isomers:
                if iso not in df_base.columns:
                    df_base[iso] = np.nan
            # 단위 변환 적용
            factor = UNIT_FACTOR.get(substance, 1.0)
            if factor != 1.0:
                for iso in isomers:
                    df_base[iso] = df_base[iso] * factor
            df_base[f"Sum_{substance}_OC_log"] = process_sum_refined(df_base, isomers)
        n_valid = df_base[f"Sum_{substance}_OC_log"].notna().sum()
        print(f"  {substance}: 유효 {n_valid:,}지점")

    # EqP 필터 + mPELQ 사전 필터
    substance_dfs = {}
    for substance in TARGET_SUBSTANCES:
        target_col = f"Sum_{substance}_OC_log"
        df_sub = df_base[df_base[target_col].notna()].copy()
        if len(df_sub) == 0:
            continue

        # ★ mPELQ 중금속 사전 필터: 타 독성 기여도 의심 샘플 제거
        # TCDD는 필터 미적용 (N 부족 → 모든 DB 사용)
        mpelq_thresh = MPELQ_METALS_THRESHOLD_TCDD if substance == "TCDD" else MPELQ_METALS_THRESHOLD
        n_before_mpelq = len(df_sub)
        df_sub = df_sub[df_sub["mPELQ_Metals"] <= mpelq_thresh].copy()
        n_after_mpelq = len(df_sub)
        n_mpelq_removed = n_before_mpelq - n_after_mpelq

        # EqP 필터: EqP 예측 농도보다 낮은 농도에서 독성 → 신뢰도 낮음 → 제거
        if substance in eqp_params:
            p = eqp_params[substance]
            kow = 10 ** p['log_kow']
            koc = 0.41 * kow
            f_oc = 0.02
            c_sed = 10 ** df_sub[target_col].values
            c_pw = c_sed * 1000 / (f_oc * koc)
            implausible = (c_pw < p['ccc_ugL']) & (df_sub["Mean_Survival"].values < 80)
            n_before_eqp = len(df_sub)
            df_sub = df_sub[~implausible].copy()
            n_eqp_removed = n_before_eqp - len(df_sub)
            print(f"  {substance}: mPELQ 제거 {n_mpelq_removed} → EqP 제거 {n_eqp_removed} "
                  f"→ 최종 {len(df_sub):,}지점")
        else:
            print(f"  {substance}: mPELQ 제거 {n_mpelq_removed} → 최종 {len(df_sub):,}지점")

        substance_dfs[substance] = df_sub

    return df_base, substance_dfs


# =============================================================================
# Step 2: XGBoost SHAP 종 선별
# =============================================================================
def step2_species_selection(substance_dfs, source_name=""):
    print(f"\n{'='*70}")
    print(f"[Step 2] XGBoost SHAP 종 선별 — {source_name}")
    print(f"{'='*70}")

    eval_list = []
    for substance in TARGET_SUBSTANCES:
        if substance not in substance_dfs:
            continue
        df_sub = substance_dfs[substance].rename(columns={f"Sum_{substance}_OC_log": "Conc_log"})
        for sp in df_sub["Latin_Name_Species"].unique():
            df_sp = df_sub[df_sub["Latin_Name_Species"] == sp]
            total_n = len(df_sp)
            toxic_n = int((df_sp["Mean_Survival"] < 80).sum())
            if total_n < 30 or toxic_n < 10:
                eval_list.append({"Substance": substance, "Species": sp,
                    "Total_N": total_n, "Toxic_N": toxic_n,
                    "Chem_Importance": np.nan, "TOC_Importance": np.nan,
                    "Chem_Direction": np.nan, "Status": "Excluded (Insufficient)"})
                continue
            X = df_sp[["Conc_log", "TOC_pct"]].values.astype(np.float64)
            y = df_sp["Mean_Survival"].values.astype(np.float64)
            if np.any(np.isnan(X)) or np.any(np.isinf(X)) or np.any(np.isnan(y)):
                eval_list.append({"Substance": substance, "Species": sp,
                    "Total_N": total_n, "Toxic_N": toxic_n,
                    "Chem_Importance": np.nan, "TOC_Importance": np.nan,
                    "Chem_Direction": np.nan, "Status": "Excluded (Data Quality)"})
                continue
            try:
                dtrain = xgb.DMatrix(X, label=y, feature_names=["Conc_log", "TOC_pct"])
                model = xgb.train({"objective": "reg:squarederror", "eta": 0.1,
                    "max_depth": 3, "seed": RANDOM_SEED}, dtrain, num_boost_round=100)
                pred = model.predict(dtrain, pred_contribs=True)
                imp_chem = np.mean(np.abs(pred[:, 0]))
                imp_toc = np.mean(np.abs(pred[:, 1]))
                cor_chem = stats.spearmanr(X[:, 0], pred[:, 0]).correlation
                if np.isnan(cor_chem): cor_chem = 0.0
                status = "Selected (Causal Indicator)"
                if cor_chem > -0.30: status = "Excluded (Non-toxic Trend)"
                elif imp_toc > (imp_chem * 1.5): status = "Excluded (TOC Dominated)"
                eval_list.append({"Substance": substance, "Species": sp,
                    "Total_N": total_n, "Toxic_N": toxic_n,
                    "Chem_Importance": imp_chem, "TOC_Importance": imp_toc,
                    "Chem_Direction": cor_chem, "Status": status})
            except Exception as e:
                eval_list.append({"Substance": substance, "Species": sp,
                    "Total_N": total_n, "Toxic_N": toxic_n,
                    "Chem_Importance": np.nan, "TOC_Importance": np.nan,
                    "Chem_Direction": np.nan, "Status": f"Excluded (Error)"})

    df_eval = pd.DataFrame(eval_list)
    selected = df_eval[df_eval["Status"].str.contains("Selected", na=False)]
    print(f"선택된 종: {len(selected)}종")
    df_eval.to_csv(OUTPUT_DIR / f"Step2_Species_Eval_{source_name}.csv", index=False)
    return df_eval


# =============================================================================
# Step 2.5: DRC 품질 평가 → DRC OK 종만 선별 + 가짜 독성 DROP ★개편
# =============================================================================
def evaluate_drc_quality(conc_log, survival):
    """
    종×물질 조합의 DRC 품질 평가 (3-tier)

    현장 혼합오염 데이터에서 R² ≥ 0.10은 거의 불가능 → R²는 참고용
    대신 독성 인과관계 방향(slope < 0)과 저농도 가짜 독성 비율로 판정

    Tier 1 (DRC OK):    slope < 0 AND 저농도 독성비율 ≤ 0.50 → 그대로 사용
    Tier 2 (DRC Marginal): slope < 0 AND 저농도 독성비율 > 0.50 → 가짜 독성 DROP 후 사용
    Tier 3 (DRC Poor):  slope ≥ 0 → 통째로 제외
    """
    result = {
        "N_total": len(conc_log), "N_toxic": int((survival < 80).sum()),
        "DRC_Tier": "", "DRC_OK": False, "R_squared": np.nan, "Slope": np.nan,
        "Low_Conc_Tox_Ratio": np.nan, "Q1_Threshold": np.nan,
        "N_Low_Tox_Drop": 0, "Reason": ""
    }

    if len(conc_log) < 20:
        result["Reason"] = "Insufficient N for DRC"
        return result

    slope, intercept, r_value, p_value, std_err = stats.linregress(conc_log, survival)
    r_squared = r_value ** 2
    result["R_squared"] = round(r_squared, 4)
    result["Slope"] = round(slope, 4)

    q1 = np.nanpercentile(conc_log, 25)
    result["Q1_Threshold"] = round(q1, 4)
    low_mask = conc_log <= q1
    low_survival = survival[low_mask]
    n_low = len(low_survival)
    n_low_tox = int((low_survival < 80).sum())
    low_tox_ratio = n_low_tox / n_low if n_low > 0 else 0
    result["Low_Conc_Tox_Ratio"] = round(low_tox_ratio, 4)

    # 3-tier 판정
    if slope >= 0:
        result["DRC_Tier"] = "Tier3_Poor"
        result["DRC_OK"] = False
        result["Reason"] = f"slope={slope:.3f} ≥ 0 (no dose-response)"
    elif low_tox_ratio > 0.50:
        result["DRC_Tier"] = "Tier2_Marginal"
        result["DRC_OK"] = True  # 가짜 독성 DROP 후 사용
        result["N_Low_Tox_Drop"] = n_low_tox
        result["Reason"] = f"slope OK but lowTox={low_tox_ratio:.2f} > 0.50 → DROP fake toxicity"
    else:
        result["DRC_Tier"] = "Tier1_OK"
        result["DRC_OK"] = True
        result["Reason"] = "DRC OK (slope < 0, lowTox ≤ 0.50)"

    return result


def step2_5_drc_evaluation(substance_dfs, df_eval, source_name=""):
    """
    DRC 품질 평가 → DRC OK 종만 선별 + 가짜 독성 DROP

    ★ 핵심 변경:
    - DRC 불량 종 → 통째로 제외 (TEL/PEL에 사용하지 않음)
    - DRC OK 종의 저농도 가짜 독성 → DROP (생존 80% 상향 아님)
    """
    print(f"\n{'='*70}")
    print(f"[Step 2.5] DRC 품질 평가 → DRC OK 종만 선별 — {source_name}")
    print(f"{'='*70}")

    # Step 2 선별종 추출
    selected_species_dict = {}
    for sub in TARGET_SUBSTANCES:
        sel = df_eval[(df_eval["Substance"] == sub) &
                       df_eval["Status"].str.contains("Selected", na=False)]
        spp = sel["Species"].unique().tolist()
        if spp:
            selected_species_dict[sub] = spp
    # Leptocheirus plumulosus 하드코딩 추가
    for sub in ["DDTs", "Dieldrin", "PCBs"]:
        if sub in selected_species_dict:
            selected_species_dict[sub] = list(set(selected_species_dict[sub] + ["Leptocheirus plumulosus"]))
        else:
            selected_species_dict[sub] = ["Leptocheirus plumulosus"]

    drc_results = []
    drc_ok_species = {}  # DRC OK 종만 저장
    cleaned_dfs = {}     # 가짜 독성 DROP된 데이터
    total_dropped = 0

    for substance in TARGET_SUBSTANCES:
        if substance not in substance_dfs:
            continue
        accepted = selected_species_dict.get(substance)
        if not accepted:
            continue

        target_col = f"Sum_{substance}_OC_log"
        df_sub = substance_dfs[substance]
        df_pooled = df_sub[df_sub[target_col].notna()].rename(columns={target_col: "Conc_log"})
        df_pooled = df_pooled[df_pooled["Latin_Name_Species"].isin(accepted)].copy()

        if len(df_pooled) < 15:
            cleaned_dfs[substance] = df_pooled
            continue

        ok_species_for_substance = []
        print(f"\n--- {substance} (N={len(df_pooled)}, 후보 {len(accepted)}종) ---")

        for sp in accepted:
            sp_mask = df_pooled["Latin_Name_Species"] == sp
            df_sp = df_pooled[sp_mask]
            if len(df_sp) < 20:
                drc_results.append({
                    "Substance": substance, "Species": sp,
                    "N_total": len(df_sp), "N_toxic": int((df_sp["Mean_Survival"] < 80).sum()),
                    "DRC_OK": False, "R_squared": np.nan, "Slope": np.nan,
                    "Low_Conc_Tox_Ratio": np.nan, "Q1_Threshold": np.nan,
                    "N_Low_Tox_Drop": 0, "Reason": "Insufficient N"
                })
                continue

            conc_log = df_sp["Conc_log"].values
            survival = df_sp["Mean_Survival"].values

            drc = evaluate_drc_quality(conc_log, survival)
            drc["Substance"] = substance
            drc["Species"] = sp
            drc_results.append(drc)

            status_str = "✓" if drc["DRC_OK"] else "✗"
            print(f"  {status_str} {sp[:35]:35s} | N={drc['N_total']:5.0f} "
                  f"R²={drc['R_squared']:.3f} slope={drc['Slope']:.3f} "
                  f"lowTox={drc['Low_Conc_Tox_Ratio']:.2f}")

            if drc["DRC_OK"]:
                ok_species_for_substance.append(sp)
                # DRC OK 종: 저농도 가짜 독성 DROP (생존 80% 상향이 아님)
                if drc["N_Low_Tox_Drop"] > 0:
                    q1 = drc["Q1_Threshold"]
                    low_tox_mask = sp_mask & (df_pooled["Conc_log"] <= q1) & (df_pooled["Mean_Survival"] < 80)
                    df_pooled = df_pooled[~low_tox_mask].copy()
                    total_dropped += drc["N_Low_Tox_Drop"]
            # DRC 불량 종: 해당 종 데이터 전체 DROP
            else:
                df_pooled = df_pooled[~sp_mask].copy()

        if ok_species_for_substance:
            drc_ok_species[substance] = ok_species_for_substance
        cleaned_dfs[substance] = df_pooled

    drc_df = pd.DataFrame(drc_results)
    drc_df.to_csv(OUTPUT_DIR / f"Step2_5_DRC_Quality_{source_name}.csv", index=False)

    ok_count = (drc_df["DRC_OK"] == True).sum()
    bad_count = (drc_df["DRC_OK"] == False).sum()
    total_species = len(drc_df)
    print(f"\n[Step 2.5 요약]")
    print(f"  DRC OK: {ok_count}종 / 불량: {bad_count}종 / 총 {total_species}종 조합")
    print(f"  DRC OK 종만 TEL/PEL 도출에 사용 ★")
    print(f"  가짜 독성 DROP: {total_dropped}개 (생존 80% 상향 아님, 데이터 삭제)")
    print(f"  물질별 DRC OK 종: {drc_ok_species}")

    return cleaned_dfs, drc_df, drc_ok_species


# =============================================================================
# Step 2.7: 다물질 SHAP 타 독성 기여도 평가 ★신규
# =============================================================================
def step2_7_confounder_filtering(cleaned_dfs, drc_ok_species, source_name=""):
    """
    다물질 XGBoost SHAP로 타 독성 기여도 평가

    로직:
    - 각 물질별: [target_conc, mPELQ_Metals, TOC_pct] → XGBoost → 생존율
    - 각 샘플: |SHAP_mPELQ| > |SHAP_target| → 중금속이 타 독성 주원인 → DROP
    - ML(SHAP) 기반 선별로 confounder-dominated 샘플 제거
    """
    print(f"\n{'='*70}")
    print(f"[Step 2.7] 다물질 SHAP 타 독성 기여도 평가 — {source_name}")
    print(f"{'='*70}")

    filtered_dfs = {}
    confounder_stats = []

    for substance in TARGET_SUBSTANCES:
        if substance not in cleaned_dfs or substance not in drc_ok_species:
            continue
        ok_spp = drc_ok_species[substance]
        if not ok_spp:
            continue

        target_col = f"Sum_{substance}_OC_log"
        df_sub = cleaned_dfs[substance]
        df_pooled = df_sub[df_sub["Latin_Name_Species"].isin(ok_spp)].copy()

        if len(df_pooled) < 15:
            filtered_dfs[substance] = df_pooled
            continue

        # Feature: [target_conc, mPELQ_Metals, TOC_pct]
        # Step 2.5에서 rename된 경우 대비: target_col 복원
        for possible_col in [f"Sum_{substance}_OC_log", "Conc_log"]:
            if possible_col in df_pooled.columns:
                if possible_col != f"Sum_{substance}_OC_log":
                    df_pooled = df_pooled.rename(columns={possible_col: f"Sum_{substance}_OC_log"})
                break

        target_col = f"Sum_{substance}_OC_log"
        X = df_pooled[[target_col, "mPELQ_Metals", "TOC_pct"]].values.astype(np.float64)
        y = df_pooled["Mean_Survival"].values.astype(np.float64)

        # NaN check (mPELQ_Metals는 Step 1에서 보장됨)
        valid_mask = ~np.any(np.isnan(X) | np.isinf(X), axis=1)
        if valid_mask.sum() < 15:
            filtered_dfs[substance] = df_pooled
            continue

        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        df_valid = df_pooled[valid_mask].copy()

        feature_names = [f"{substance}_conc", "mPELQ_Metals", "TOC_pct"]

        try:
            dtrain = xgb.DMatrix(X_valid, label=y_valid, feature_names=feature_names)
            model = xgb.train({"objective": "reg:squarederror", "eta": 0.1,
                "max_depth": 4, "seed": RANDOM_SEED}, dtrain, num_boost_round=100)
            pred = model.predict(dtrain, pred_contribs=True)

            shap_target = pred[:, 0]
            shap_mpelq = pred[:, 1]
            shap_toc = pred[:, 2]

            # 타 독성 기여도 평가: |SHAP_mPELQ| > threshold×|SHAP_target| → confounder-dominated
            # TCDD만 1.5× 완화 (N≥20 확보), 다른 물질은 1.0× 유지 (기존 확정값 보존)
            # ★ TCDD no-filter 모드: confounder 필터 완전 해제
            if substance == "TCDD":
                confounder_dominated = np.zeros(len(df_valid), dtype=bool)
            else:
                confounder_ratio = 1.5 if substance == "TCDD" else 1.0
                confounder_dominated = np.abs(shap_mpelq) > confounder_ratio * np.abs(shap_target)
            n_dominated = int(confounder_dominated.sum())
            n_total = len(confounder_dominated)

            # confounder-dominated 샘플 DROP
            df_clean = df_valid[~confounder_dominated].copy()

            print(f"  {substance}: N={n_total} → confounder 제거 {n_dominated} "
                  f"({n_dominated/n_total*100:.1f}%) → 최종 {len(df_clean)}")

            confounder_stats.append({
                "Substance": substance,
                "N_total": n_total,
                "N_confounder_dominated": n_dominated,
                "N_remaining": len(df_clean),
                "Pct_removed": round(n_dominated / n_total * 100, 1) if n_total > 0 else 0,
                "Avg_SHAP_target": round(np.mean(np.abs(shap_target)), 4),
                "Avg_SHAP_mPELQ": round(np.mean(np.abs(shap_mpelq)), 4),
                "Avg_SHAP_TOC": round(np.mean(np.abs(shap_toc)), 4),
            })

            filtered_dfs[substance] = df_clean

        except Exception as e:
            print(f"  {substance}: SHAP 평가 실패 ({e}) → 필터링 없이 통과")
            filtered_dfs[substance] = df_pooled

    conf_df = pd.DataFrame(confounder_stats)
    if len(conf_df) > 0:
        conf_df.to_csv(OUTPUT_DIR / f"Step2_7_Confounder_Stats_{source_name}.csv", index=False)
        total_removed = conf_df["N_confounder_dominated"].sum()
        total_remaining = conf_df["N_remaining"].sum()
        print(f"\n[Step 2.7 요약]")
        print(f"  confounder-dominated 제거: {total_removed:,}개")
        print(f"  잔여: {total_remaining:,}개")

    return filtered_dfs, conf_df


# =============================================================================
# Step 2.7-v4: confounder 필터 재설계 (진단 + Primary/Sensitivity 분리)
# =============================================================================
def _compute_tel_pel_from_df(df, target_col):
    """Primary TEL/PEL 계산 — SHAP-independent (survival만으로 EDS/NEDS 분류).

    df: target_col, TOC_pct, Mean_Survival 컬럼을 가진 DataFrame.
    반환: {"N", "N_EDS", "N_NEDS", "N_Intermediate", "TEL", "PEL"} 또는 None (계산 불가 시).

    Primary 설계 (v4 최종):
      - survival < 80  → EDS (전부)
      - survival >= 80 → NEDS (전부)
      - N_Intermediate = 0 (SHAP 기반 제거 없음)
    Target-SHAP refinement는 sensitivity/diagnostic으로 이동.
    """
    if len(df) < 15:
        return None
    X = df[[target_col, "TOC_pct"]].values.astype(np.float64)
    y = df["Mean_Survival"].values.astype(np.float64)
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        return None

    # Primary: survival만으로 EDS/NEDS 분류 (SHAP 사용 안 함)
    eds_mask = df["Mean_Survival"].values < 80
    neds_mask = df["Mean_Survival"].values >= 80
    n_eds = int(eds_mask.sum())
    n_neds = int(neds_mask.sum())
    n_intermediate = 0  # SHAP 기반 제거 없음 → 중간 구간 없음

    conc_log = df[target_col].values
    e_v = conc_log[eds_mask]
    n_v = conc_log[neds_mask]

    if len(e_v) < 2 or len(n_v) < 2:
        return None

    e_orig = 10 ** e_v - 1
    n_orig = 10 ** n_v - 1
    our_tel = np.sqrt(np.quantile(e_orig, 0.15) * np.quantile(n_orig, 0.50))
    our_pel = np.sqrt(np.quantile(e_orig, 0.50) * np.quantile(n_orig, 0.85))

    return {"N": len(df), "N_EDS": n_eds, "N_NEDS": n_neds,
            "N_Intermediate": n_intermediate, "TEL": our_tel, "PEL": our_pel}


def step2_7_confounder_filtering_v4(cleaned_dfs, drc_ok_species, source_name="",
                                     n_folds=5, n_seeds=10, stability_threshold=0.80):
    """
    v4 confounder 필터 재설계 — 부호 오류 수정 + 공선성 대응.

    분석 구조:
      - Primary:       화학적 mPELQ screen만 (Step 1에서 이미 적용, 추가 SHAP 제거 없음)
      - Sensitivity 1: sign-aware TreeSHAP (OOF CV)
      - Sensitivity 2: sign-aware Interventional SHAP (OOF CV)
      - Ablation:      금속 포함/제외 모델의 OOF prediction difference

    반복 CV SHAP (out-of-fold):
      - 각 fold: training fold로 모델 적합 → held-out fold의 SHAP 계산
      - Interventional background는 해당 fold의 training data에서만 선정
      - 시료별 부호·dominance 출현 빈도 누적
      - 80% 안정성: (negative_sign_frequency >= 0.80) & (dominance_frequency >= 0.80)
    """
    print(f"\n{'='*70}")
    print(f"[Step 2.7-v4] confounder 필터 재설계 (진단) — {source_name}")
    print(f"{'='*70}")

    diag_rows = []
    tel_pel_rows = []

    for substance in TARGET_SUBSTANCES:
        if substance not in cleaned_dfs or substance not in drc_ok_species:
            continue
        ok_spp = drc_ok_species[substance]
        if not ok_spp:
            continue

        target_col = f"Sum_{substance}_OC_log"
        df_sub = cleaned_dfs[substance]
        df_pooled = df_sub[df_sub["Latin_Name_Species"].isin(ok_spp)].copy()

        # target_col 복원 (Step 2.5에서 Conc_log로 rename된 경우)
        if target_col not in df_pooled.columns and "Conc_log" in df_pooled.columns:
            df_pooled = df_pooled.rename(columns={"Conc_log": target_col})

        if target_col not in df_pooled.columns or len(df_pooled) < 15:
            continue

        X = df_pooled[[target_col, "mPELQ_Metals", "TOC_pct"]].values.astype(np.float64)
        y = df_pooled["Mean_Survival"].values.astype(np.float64)
        valid_mask = ~np.any(np.isnan(X) | np.isinf(X), axis=1)
        if valid_mask.sum() < 15:
            continue
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        df_valid = df_pooled[valid_mask].reset_index(drop=True)
        n_total = len(df_valid)

        # 고유 레코드 ID 부여 (행 인덱스 기반, 중복 없음)
        df_valid["record_id"] = [f"r{i:06d}" for i in range(n_total)]
        # 원본 식별 컬럼 보존 (추적용, 중복 가능)
        id_cols = [c for c in ["sample_key", "SampleID", "station_key", "StudyID",
                               "Station", "Latin_Name_Species"] if c in df_valid.columns]
        if id_cols:
            df_valid["sample_id"] = (df_valid[id_cols].fillna("")
                                     .astype(str).agg("|".join, axis=1))
        else:
            df_valid["sample_id"] = df_valid["record_id"]

        feature_names = [f"{substance}_conc", "mPELQ_Metals", "TOC_pct"]

        # ---- Primary: 화학적 mPELQ screen만 (Step 1에서 이미 적용) ----
        # df_valid는 이미 mPELQ <= 0.5가 적용된 상태. 추가 SHAP 제거 없음.
        primary_telpel = _compute_tel_pel_from_df(df_valid, target_col)

        # ---- 반복 CV SHAP (OOF) ----
        # 시료별 누적 카운터
        n_assignments = np.zeros(n_total, dtype=int)          # OOF 예측 횟수
        neg_sign_count = np.zeros(n_total, dtype=int)          # SHAP_mPELQ < 0 횟수
        dom_count_tree = np.zeros(n_total, dtype=int)         # TreeSHAP dominance 횟수
        dom_count_inter = np.zeros(n_total, dtype=int)         # Interventional dominance 횟수
        # ablation: 금속 포함/제외 OOF prediction difference 누적
        pred_with_metal = np.zeros(n_total, dtype=float)
        pred_without_metal = np.zeros(n_total, dtype=float)

        # 시료별 OOF 원자료 수집 (long format)
        raw_rows = []
        bg_sizes = []  # fold별 interventional background 실제 행 수

        for seed in range(n_seeds):
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            for fold_id, (train_idx, test_idx) in enumerate(kf.split(X_valid)):
                X_tr, X_te = X_valid[train_idx], X_valid[test_idx]
                y_tr, y_te = y_valid[train_idx], y_valid[test_idx]
                bg_sizes.append(int(X_tr.shape[0]))

                # 3-feature 모델 (금속 포함)
                dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
                model3 = xgb.train({"objective": "reg:squarederror", "eta": 0.1,
                    "max_depth": 4, "seed": RANDOM_SEED}, dtr, num_boost_round=100)

                # TreeSHAP (path-dependent, 기본)
                dte = xgb.DMatrix(X_te, feature_names=feature_names)
                pred_tree = model3.predict(dte, pred_contribs=True)
                shap_mpelq_tree = pred_tree[:, 1]
                shap_target_tree = pred_tree[:, 0]

                # Interventional SHAP (background = 해당 fold training data)
                try:
                    explainer = shap.TreeExplainer(
                        model3, X_tr, feature_perturbation="interventional")
                    shap_inter = explainer.shap_values(X_te)
                    shap_mpelq_inter = shap_inter[:, 1]
                    shap_target_inter = shap_inter[:, 0]
                except Exception as e:
                    print(f"  {substance} seed={seed}: Interventional SHAP 실패 ({e})")
                    shap_mpelq_inter = shap_mpelq_tree
                    shap_target_inter = shap_target_tree

                # ablation: 금속 제외 2-feature 모델
                X_tr2 = X_tr[:, [0, 2]]
                X_te2 = X_te[:, [0, 2]]
                dtr2 = xgb.DMatrix(X_tr2, label=y_tr,
                                   feature_names=[feature_names[0], feature_names[2]])
                model2 = xgb.train({"objective": "reg:squarederror", "eta": 0.1,
                    "max_depth": 4, "seed": RANDOM_SEED}, dtr2, num_boost_round=100)
                dte2 = xgb.DMatrix(X_te2, feature_names=[feature_names[0], feature_names[2]])
                pred2 = model2.predict(dte2)
                pred3 = model3.predict(dte)

                # 누적 + 시료별 원자료 수집
                for k, idx in enumerate(test_idx):
                    n_assignments[idx] += 1
                    if shap_mpelq_tree[k] < 0:
                        neg_sign_count[idx] += 1
                    if np.abs(shap_mpelq_tree[k]) > np.abs(shap_target_tree[k]):
                        dom_count_tree[idx] += 1
                    if np.abs(shap_mpelq_inter[k]) > np.abs(shap_target_inter[k]):
                        dom_count_inter[idx] += 1
                    pred_with_metal[idx] += pred3[k]
                    pred_without_metal[idx] += pred2[k]

                    # 시료별 원자료 (long format)
                    raw_rows.append({
                        "record_id": df_valid.loc[idx, "record_id"],
                        "sample_id": df_valid.loc[idx, "sample_id"],
                        "substance": substance,
                        "repeat_id": seed,
                        "fold_id": fold_id,
                        "is_oof": True,  # test fold = OOF
                        "shap_tree_mpelq": shap_mpelq_tree[k],
                        "shap_interv_mpelq": shap_mpelq_inter[k],
                        "shap_tree_target": shap_target_tree[k],
                        "shap_interv_target": shap_target_inter[k],
                        "neg_flag": int(shap_mpelq_tree[k] < 0),
                        "dominance_flag_tree": int(
                            np.abs(shap_mpelq_tree[k]) > np.abs(shap_target_tree[k])),
                        "dominance_flag_inter": int(
                            np.abs(shap_mpelq_inter[k]) > np.abs(shap_target_inter[k])),
                        "pred_with_metal": pred3[k],
                        "pred_without_metal": pred2[k],
                        "pred_diff": pred3[k] - pred2[k],
                    })

        # ---- 시료별 빈도 산출 ----
        neg_sign_freq = np.divide(neg_sign_count, n_assignments,
                                  out=np.zeros(n_total), where=n_assignments > 0)
        dom_freq_tree = np.divide(dom_count_tree, n_assignments,
                                  out=np.zeros(n_total), where=n_assignments > 0)
        dom_freq_inter = np.divide(dom_count_inter, n_assignments,
                                   out=np.zeros(n_total), where=n_assignments > 0)

        # 80% 안정성 판정 (부호와 dominance를 별도 조건으로)
        stable_tree = (neg_sign_freq >= stability_threshold) & (dom_freq_tree >= stability_threshold)
        stable_inter = (neg_sign_freq >= stability_threshold) & (dom_freq_inter >= stability_threshold)

        # ---- Sensitivity별 포함 시료 수 + 잠정 TEL/PEL ----
        sens1_df = df_valid[~stable_tree].reset_index(drop=True)
        sens2_df = df_valid[~stable_inter].reset_index(drop=True)
        sens1_telpel = _compute_tel_pel_from_df(sens1_df, target_col)
        sens2_telpel = _compute_tel_pel_from_df(sens2_df, target_col)

        # ablation: 금속 포함에 따른 OOF prediction difference (평균)
        pred_diff = np.divide(pred_with_metal - pred_without_metal, n_assignments,
                              out=np.zeros(n_total), where=n_assignments > 0)
        mean_ablation = float(np.mean(pred_diff))
        n_ablation_negative = int((pred_diff < 0).sum())  # 금속 포함 시 예측 감소(독성 방향)

        # ablation 통계 보강 (median, IQR, 5th-95th percentile)
        med_ablation = float(np.median(pred_diff))
        q25_ablation = float(np.quantile(pred_diff, 0.25))
        q75_ablation = float(np.quantile(pred_diff, 0.75))
        p5_ablation = float(np.quantile(pred_diff, 0.05))
        p95_ablation = float(np.quantile(pred_diff, 0.95))

        # EDS/NEDS별 ablation 분포 (Primary 분류 기준)
        primary_eds_mask = (df_valid["Mean_Survival"].values < 80)
        primary_neds_mask = df_valid["Mean_Survival"].values >= 80
        eds_ablation = pred_diff[primary_eds_mask]
        neds_ablation = pred_diff[primary_neds_mask]
        mean_ablation_eds = float(np.mean(eds_ablation)) if len(eds_ablation) > 0 else np.nan
        mean_ablation_neds = float(np.mean(neds_ablation)) if len(neds_ablation) > 0 else np.nan

        # ---- 결과 기록 ----
        n_removed_tree = int(stable_tree.sum())
        n_removed_inter = int(stable_inter.sum())
        n_neg_majority = int((neg_sign_freq >= stability_threshold).sum())

        print(f"\n  {substance}: N={n_total}")
        print(f"    Primary (mPELQ only): N={n_total} → "
              f"TEL={primary_telpel['TEL']:.4f} PEL={primary_telpel['PEL']:.4f}" if primary_telpel else
              f"    Primary: TEL/PEL 계산 불가")
        print(f"    Sens1 (TreeSHAP sign-aware): 제거 {n_removed_tree} → "
              f"TEL={sens1_telpel['TEL']:.4f} PEL={sens1_telpel['PEL']:.4f}" if sens1_telpel else
              f"    Sens1: TEL/PEL 계산 불가")
        print(f"    Sens2 (Interventional sign-aware): 제거 {n_removed_inter} → "
              f"TEL={sens2_telpel['TEL']:.4f} PEL={sens2_telpel['PEL']:.4f}" if sens2_telpel else
              f"    Sens2: TEL/PEL 계산 불가")
        print(f"    부호 안정(SHAP_mPELQ<0 ≥{stability_threshold:.0%}): {n_neg_majority} 시료")
        print(f"    Ablation: 금속 포함 시 OOF 예측 평균 변화 {mean_ablation:+.4f} "
              f"(음수 {n_ablation_negative} 시료)")

        diag_rows.append({
            "Source": source_name, "Substance": substance, "N_total": n_total,
            "N_neg_sign_freq_ge_threshold": n_neg_majority,
            "N_removed_TreeSHAP": n_removed_tree,
            "N_removed_Interventional": n_removed_inter,
            "n_oof_per_sample_min": int(n_assignments.min()) if n_total > 0 else 0,
            "n_oof_per_sample_max": int(n_assignments.max()) if n_total > 0 else 0,
            "interventional_background_min": int(min(bg_sizes)) if bg_sizes else 0,
            "interventional_background_max": int(max(bg_sizes)) if bg_sizes else 0,
            "Mean_ablation_pred_diff": round(mean_ablation, 4),
            "Median_ablation_pred_diff": round(med_ablation, 4),
            "IQR_ablation_pred_diff": round(q75_ablation - q25_ablation, 4),
            "P5_ablation_pred_diff": round(p5_ablation, 4),
            "P95_ablation_pred_diff": round(p95_ablation, 4),
            "Mean_ablation_EDS": round(mean_ablation_eds, 4),
            "Mean_ablation_NEDS": round(mean_ablation_neds, 4),
            "N_ablation_negative": n_ablation_negative,
            "N_ablation_negative_pct": round(100.0 * n_ablation_negative / n_total, 2),
        })

        for label, telpel in [("Primary", primary_telpel),
                              ("Sensitivity1_TreeSHAP", sens1_telpel),
                              ("Sensitivity2_Interventional", sens2_telpel)]:
            if telpel:
                tel_pel_rows.append({
                    "Source": source_name, "Substance": substance, "Analysis": label,
                    "N": telpel["N"], "N_EDS": telpel["N_EDS"], "N_NEDS": telpel["N_NEDS"],
                    "N_Intermediate": telpel["N_Intermediate"],
                    "TEL": round(telpel["TEL"], 4), "PEL": round(telpel["PEL"], 4),
                })

        # 시료별 OOF 원자료 CSV 저장 (long format)
        if raw_rows:
            raw_df = pd.DataFrame(raw_rows)
            raw_df.to_csv(
                OUTPUT_DIR / f"Step2_7v4_OOF_Raw_{source_name}_{substance}.csv",
                index=False)

    diag_df = pd.DataFrame(diag_rows)
    telpel_df = pd.DataFrame(tel_pel_rows)

    # ---- OOF 반복 구조 메타데이터 (재현성·검증용) ----
    # dominance_ratio: |SHAP_metal| > ratio * |SHAP_target| (현재 1.0 = 단순 크기 비교)
    dominance_ratio = 1.0
    # interventional background = 각 fold의 training data 전체 (shap.sample 없음)
    # fold별 실제 행 수는 substance별로 diag_rows에 이미 기록됨 (interventional_background_min/max)

    if len(diag_df) > 0:
        diag_df.to_csv(OUTPUT_DIR / f"Step2_7v4_Diagnosis_{source_name}.csv", index=False)
    if len(telpel_df) > 0:
        telpel_df.to_csv(OUTPUT_DIR / f"Step2_7v4_TELPEL_Comparison_{source_name}.csv", index=False)

    # 메타데이터 별도 CSV (substance 단위 — background size가 물질별로 다름)
    meta_rows = []
    for _, drow in diag_df.iterrows():
        meta_rows.append({
            "Source": drow["Source"],
            "Substance": drow["Substance"],
            "N_total": drow["N_total"],
            "n_splits": n_folds,
            "n_repeats": n_seeds,
            "n_oof_per_sample_min": drow["n_oof_per_sample_min"],
            "n_oof_per_sample_max": drow["n_oof_per_sample_max"],
            "stability_threshold": stability_threshold,
            "dominance_ratio": dominance_ratio,
            "interventional_background_min": drow["interventional_background_min"],
            "interventional_background_max": drow["interventional_background_max"],
            "interventional_background_sampling": "full_training_fold",
        })
    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(OUTPUT_DIR / f"Step2_7v4_Metadata_{source_name}.csv", index=False)

    return diag_df, telpel_df


# =============================================================================
# Step 3: SHAP EDS/NEDS → TEL/PEL (DRC OK 종만 사용)
# =============================================================================
def step3_tel_pel(filtered_dfs, drc_ok_species, source_name=""):
    print(f"\n{'='*70}")
    print(f"[Step 3] SHAP EDS/NEDS → TEL/PEL — {source_name}")
    print(f"{'='*70}")

    sqg_list = []
    edsneds_data = {}  # Step 4용 저장

    for substance in TARGET_SUBSTANCES:
        if substance not in filtered_dfs or substance not in drc_ok_species:
            continue
        ok_spp = drc_ok_species[substance]
        if not ok_spp:
            continue

        df_pooled = filtered_dfs[substance]
        df_pooled = df_pooled[df_pooled["Latin_Name_Species"].isin(ok_spp)].copy()

        # target_col 복원
        target_col = f"Sum_{substance}_OC_log"
        if target_col not in df_pooled.columns:
            # step2_7에서 rename된 경우
            if "Conc_log" in df_pooled.columns:
                df_pooled = df_pooled.rename(columns={"Conc_log": target_col})

        if target_col not in df_pooled.columns or len(df_pooled) < 15:
            continue

        print(f"\n▶ {substance} (N={len(df_pooled)}, DRC OK {len(ok_spp)}종: {ok_spp})")

        # Primary: survival만으로 EDS/NEDS 분류 (SHAP 사용 안 함)
        df_pooled = df_pooled.copy()
        df_pooled["Conc_log"] = df_pooled[target_col]

        eds_mask = df_pooled["Mean_Survival"] < 80
        df_pooled["Data_Type"] = "NEDS"
        df_pooled.loc[eds_mask, "Data_Type"] = "EDS"

        n_eds = int(eds_mask.sum())
        n_neds = int((~eds_mask).sum())
        print(f"  EDS: {n_eds}, NEDS: {n_neds}")

        e_v = df_pooled.loc[df_pooled["Data_Type"] == "EDS", "Conc_log"].values
        n_v = df_pooled.loc[df_pooled["Data_Type"] == "NEDS", "Conc_log"].values

        if len(e_v) >= 2 and len(n_v) >= 2:
            # Method B (CCME 지침): 원래 단위(µg/kg dw)에서 percentile 계산 후 기하평균.
            # Conc_log = log10(원래농도 + 1) → 원래농도 = 10^Conc_log - 1
            e_orig = 10 ** e_v - 1
            n_orig = 10 ** n_v - 1
            our_tel = np.sqrt(np.quantile(e_orig, 0.15) * np.quantile(n_orig, 0.50))
            our_pel = np.sqrt(np.quantile(e_orig, 0.50) * np.quantile(n_orig, 0.85))

            sqg_list.append({
                "Source": source_name, "Substance": substance,
                "Total_N": len(df_pooled), "N_EDS": n_eds, "N_NEDS": n_neds,
                "N_DRC_OK_Species": len(ok_spp),
                "DRC_OK_Species": "; ".join(ok_spp),
                "Our_TEL_dw": round(our_tel, 4), "Our_PEL_dw": round(our_pel, 4),
                "CCME_ISQG": CCME_ISQG.get(substance, np.nan),
                "CCME_PEL": CCME_PEL.get(substance, np.nan),
                "NOAA_TEL": NOAA_TEL_DW.get(substance, np.nan),
                "NOAA_PEL": NOAA_PEL_DW.get(substance, np.nan),
            })
            print(f"  TEL = {our_tel:.4f}, PEL = {our_pel:.4f}  "
                  f"(CCME: {CCME_ISQG.get(substance)}/{CCME_PEL.get(substance)})")

            # Step 4용 데이터 저장
            edsneds_data[substance] = df_pooled[["Conc_log", "Mean_Survival",
                "Data_Type", "Latin_Name_Species", "TOC_pct"]].copy()

    final = pd.DataFrame(sqg_list)
    final.to_csv(OUTPUT_DIR / f"Step3_TEL_PEL_{source_name}.csv", index=False)
    return final, edsneds_data


# =============================================================================
# Step 4: 신뢰도 평가 (기준 비교 + ROC-AUC 예측력) ★신규
# =============================================================================
def step4_reliability_assessment(final_df, edsneds_data, source_name=""):
    """
    준거치 신뢰도 판단:
    4-1. 기준 비교: TEL/PEL vs CCME ISQG/PEL, NOAA TEL/PEL
    4-2. 독성 예측력: ROC-AUC + TEL 기준 민감도/특이도
    """
    print(f"\n{'='*70}")
    print(f"[Step 4] 신뢰도 평가 — {source_name}")
    print(f"{'='*70}")

    reliability_list = []

    for _, row in final_df.iterrows():
        substance = row["Substance"]
        if substance not in edsneds_data:
            continue

        df_data = edsneds_data[substance]
        our_tel = row["Our_TEL_dw"]
        our_pel = row["Our_PEL_dw"]

        # --- 4-1. 기준 비교 ---
        ccme_isqg = row["CCME_ISQG"]
        ccme_pel = row["CCME_PEL"]
        noaa_tel = row["NOAA_TEL"]
        noaa_pel = row["NOAA_PEL"]

        tel_ratio_ccme = our_tel / ccme_isqg if ccme_isqg > 0 else np.nan
        pel_ratio_ccme = our_pel / ccme_pel if ccme_pel > 0 else np.nan
        tel_ratio_noaa = our_tel / noaa_tel if noaa_tel > 0 else np.nan
        pel_ratio_noaa = our_pel / noaa_pel if noaa_pel > 0 else np.nan

        # --- 4-2. 독성 예측력 (ROC-AUC) ---
        # EDS=1 (toxic), NEDS=0 (non-toxic) binary classification
        df_eval_data = df_data[df_data["Data_Type"].isin(["EDS", "NEDS"])].copy()
        y_true = (df_eval_data["Data_Type"] == "EDS").astype(int).values
        conc_log = df_eval_data["Conc_log"].values

        auc_score = np.nan
        sensitivity = np.nan
        specificity = np.nan
        n_pos = int(y_true.sum())
        n_neg = int((1 - y_true).sum())

        if n_pos >= 5 and n_neg >= 5:
            # ROC-AUC: 농도로 EDS/NEDS 분류 능력
            try:
                auc_score = roc_auc_score(y_true, conc_log)
                # AUC < 0.5면 방향 반전 (농도↑ → 독성↑이 정상)
                if auc_score < 0.5:
                    auc_score = 1 - auc_score
            except Exception:
                auc_score = np.nan

            # TEL 기준 민감도/특이도
            tel_log = np.log10(our_tel + 1) if our_tel > 0 else np.nan
            if not np.isnan(tel_log):
                predict_toxic = conc_log > tel_log  # conc > TEL → toxic 예측
                tp = int((predict_toxic & (y_true == 1)).sum())
                fn = int((~predict_toxic & (y_true == 1)).sum())
                tn = int((~predict_toxic & (y_true == 0)).sum())
                fp = int((predict_toxic & (y_true == 0)).sum())
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        # --- 5-fold CV AUC ---
        cv_auc_mean = np.nan
        cv_auc_std = np.nan
        if n_pos >= 10 and n_neg >= 10:
            try:
                cv_scores = []
                cv = min(5, n_pos, n_neg)
                if cv >= 2:
                    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
                    for train_idx, test_idx in skf.split(conc_log, y_true):
                        c_train, c_test = conc_log[train_idx], conc_log[test_idx]
                        y_train, y_test = y_true[train_idx], y_true[test_idx]
                        if y_train.sum() < 2 or (1 - y_train).sum() < 2:
                            continue
                        # XGBoost binary classification
                        dtrain = xgb.DMatrix(c_train.reshape(-1, 1), label=y_train,
                                             feature_names=["Conc_log"])
                        model = xgb.train({"objective": "binary:logistic", "eta": 0.1,
                            "max_depth": 3, "seed": RANDOM_SEED}, dtrain, num_boost_round=50)
                        dtest = xgb.DMatrix(c_test.reshape(-1, 1), feature_names=["Conc_log"])
                        y_pred = model.predict(dtest)
                        if len(np.unique(y_test)) == 2:
                            fold_auc = roc_auc_score(y_test, y_pred)
                            if fold_auc < 0.5:
                                fold_auc = 1 - fold_auc
                            cv_scores.append(fold_auc)
                    if cv_scores:
                        cv_auc_mean = np.mean(cv_scores)
                        cv_auc_std = np.std(cv_scores)
            except Exception:
                pass

        reliability_list.append({
            "Source": source_name,
            "Substance": substance,
            "Our_TEL": our_tel,
            "Our_PEL": our_pel,
            "CCME_ISQG": ccme_isqg,
            "TEL/CCME_ISQG": round(tel_ratio_ccme, 3) if not np.isnan(tel_ratio_ccme) else np.nan,
            "PEL/CCME_PEL": round(pel_ratio_ccme, 3) if not np.isnan(pel_ratio_ccme) else np.nan,
            "TEL/NOAA_TEL": round(tel_ratio_noaa, 3) if not np.isnan(tel_ratio_noaa) else np.nan,
            "PEL/NOAA_PEL": round(pel_ratio_noaa, 3) if not np.isnan(pel_ratio_noaa) else np.nan,
            "N_EDS": n_pos,
            "N_NEDS": n_neg,
            "ROC_AUC": round(auc_score, 4) if not np.isnan(auc_score) else np.nan,
            "CV_AUC_mean": round(cv_auc_mean, 4) if not np.isnan(cv_auc_mean) else np.nan,
            "CV_AUC_std": round(cv_auc_std, 4) if not np.isnan(cv_auc_std) else np.nan,
            "TEL_Sensitivity": round(sensitivity, 4) if not np.isnan(sensitivity) else np.nan,
            "TEL_Specificity": round(specificity, 4) if not np.isnan(specificity) else np.nan,
        })

        print(f"\n  {substance}:")
        print(f"    TEL={our_tel:.2f} (CCME {ccme_isqg}, 비율 {tel_ratio_ccme:.2f}x)")
        print(f"    PEL={our_pel:.2f} (CCME {ccme_pel}, 비율 {pel_ratio_ccme:.2f}x)")
        print(f"    ROC-AUC={auc_score:.3f}, CV-AUC={cv_auc_mean:.3f}±{cv_auc_std:.3f}" if not np.isnan(auc_score) else "    ROC-AUC=N/A")
        print(f"    TEL 민감도={sensitivity:.2f}, 특이도={specificity:.2f}" if not np.isnan(sensitivity) else "    TEL 민감도/특이도=N/A")

    reliability_df = pd.DataFrame(reliability_list)
    reliability_df.to_csv(OUTPUT_DIR / f"Step4_Reliability_{source_name}.csv", index=False)

    # ★ EDS/NEDS 데이터 저장 (논문용 figure 생성)
    edsneds_dir = OUTPUT_DIR / "edsneds_data"
    edsneds_dir.mkdir(exist_ok=True)
    for substance, df_data in edsneds_data.items():
        df_data.to_csv(edsneds_dir / f"{source_name}_{substance}_edsneds.csv", index=False)

    return reliability_df, edsneds_data


# =============================================================================
# Main
# =============================================================================
def main():
    diagnose_mode = "--diagnose" in sys.argv
    print("=" * 70)
    print("고도화 파이프라인 v4: 신뢰도 기반 TEL/PEL 도출")
    print("  Step 1: DB 정제 (EqP + mPELQ 필터)")
    print("  Step 2: XGBoost SHAP 종 선별")
    print("  Step 2.5: DRC 평가 → DRC OK 종만 선별 + 가짜 독성 DROP")
    print("  Step 2.7: 다물질 SHAP confounder 필터링")
    print("  Step 3: SHAP EDS/NEDS → TEL/PEL")
    print("  Step 4: 신뢰도 평가 (기준 비교 + ROC-AUC)")
    if diagnose_mode:
        print("  ★ 진단 모드: Step 2.7-v4 (Primary/Sensitivity/Ablation 비교)")
    print("=" * 70)

    all_sqg = []
    all_reliability = []
    all_diag = []
    all_telpel = []
    sources = [
        ("NOAA", NOAA_DB_PATH),
        ("SCCWRP", SCCWRP_DB_PATH),
        ("Integrated", INTEGRATED_DB_PATH),
    ]

    for source_name, db_path in sources:
        print(f"\n{'#'*70}")
        print(f"# 소스: {source_name} — {db_path.name}")
        print(f"{'#'*70}")

        df_raw = pd.read_csv(db_path, low_memory=False)
        print(f"  로딩: {df_raw.shape[0]:,}행")

        df_base, substance_dfs = step1_db_curation(df_raw, source_name)
        df_eval = step2_species_selection(substance_dfs, source_name)
        cleaned_dfs, drc_df, drc_ok_species = step2_5_drc_evaluation(
            substance_dfs, df_eval, source_name)

        if diagnose_mode:
            # 진단 모드: Step 2.7-v4 (Primary/Sensitivity/Ablation) 실행 후 종료
            diag_df, telpel_df = step2_7_confounder_filtering_v4(
                cleaned_dfs, drc_ok_species, source_name)
            if len(diag_df) > 0:
                all_diag.append(diag_df)
            if len(telpel_df) > 0:
                all_telpel.append(telpel_df)
            continue

        # ★ v4 통합: Primary = 화학적 mPELQ screen만 (SHAP confounder 필터 제거)
        #   Sensitivity(TreeSHAP/Interventional)는 진단 모듈로 별도 산출
        final, edsneds_data = step3_tel_pel(cleaned_dfs, drc_ok_species, source_name)
        reliability, edsneds_data = step4_reliability_assessment(final, edsneds_data, source_name)

        # Sensitivity 분석 (Supplementary용): Step 2.7-v4 진단 모듈 실행
        diag_df, telpel_df = step2_7_confounder_filtering_v4(
            cleaned_dfs, drc_ok_species, source_name)
        if len(diag_df) > 0:
            all_diag.append(diag_df)
        if len(telpel_df) > 0:
            all_telpel.append(telpel_df)

        if len(final) > 0:
            all_sqg.append(final)
        if len(reliability) > 0:
            all_reliability.append(reliability)

    if diagnose_mode:
        # 진단 모드: Sensitivity 결과 저장 후 종료
        print(f"\n{'='*70}")
        print("진단 모드 결과 요약 (Step 2.7-v4)")
        print(f"{'='*70}")
        if all_diag:
            diag_all = pd.concat(all_diag, ignore_index=True)
            diag_all.to_csv(OUTPUT_DIR / "Step2_7v4_Diagnosis_All.csv", index=False)
            print("\n--- 진단 요약 (Source × Substance) ---")
            print(diag_all.to_string(index=False))
        if all_telpel:
            telpel_all = pd.concat(all_telpel, ignore_index=True)
            telpel_all.to_csv(OUTPUT_DIR / "Step2_7v4_TELPEL_Comparison_All.csv", index=False)
            print("\n--- Primary vs Sensitivity TEL/PEL 비교 ---")
            print(telpel_all.to_string(index=False))
        print(f"\n진단 결과 저장: {OUTPUT_DIR}")
        print("\n진단 완료 ✅")
        return

    # ★ v4 통합 모드: Sensitivity 결과 저장 (Supplementary용)
    if all_diag:
        diag_all = pd.concat(all_diag, ignore_index=True)
        diag_all.to_csv(OUTPUT_DIR / "Step2_7v4_Diagnosis_All.csv", index=False)
    if all_telpel:
        telpel_all = pd.concat(all_telpel, ignore_index=True)
        telpel_all.to_csv(OUTPUT_DIR / "Step2_7v4_TELPEL_Comparison_All.csv", index=False)

    # 통합 비교표
    print(f"\n{'='*70}")
    print("최종 소스별 TEL/PEL 비교표 (v4 고도화 파이프라인)")
    print(f"{'='*70}")

    if all_sqg:
        comparison = pd.concat(all_sqg, ignore_index=True)
        comparison.to_csv(OUTPUT_DIR / "TEL_PEL_v4_comparison.csv", index=False)

        print("\n--- TEL 비교 ---")
        tel_pivot = comparison.pivot(index="Substance", columns="Source", values="Our_TEL_dw")
        tel_pivot["CCME_ISQG"] = comparison.groupby("Substance")["CCME_ISQG"].first()
        tel_pivot["NOAA_TEL"] = comparison.groupby("Substance")["NOAA_TEL"].first()
        print(tel_pivot.to_string(float_format="%.2f"))

        print("\n--- PEL 비교 ---")
        pel_pivot = comparison.pivot(index="Substance", columns="Source", values="Our_PEL_dw")
        pel_pivot["CCME_PEL"] = comparison.groupby("Substance")["CCME_PEL"].first()
        pel_pivot["NOAA_PEL"] = comparison.groupby("Substance")["NOAA_PEL"].first()
        print(pel_pivot.to_string(float_format="%.2f"))

        print("\n--- TEL / CCME ISQG 비율 ---")
        for src in ["NOAA", "SCCWRP", "Integrated"]:
            if src in tel_pivot.columns:
                ratio = (tel_pivot[src] / tel_pivot["CCME_ISQG"]).round(2)
                print(f"  {src}: {dict(ratio)}")

        tel_pivot.to_csv(OUTPUT_DIR / "TEL_comparison_v4.csv")
        pel_pivot.to_csv(OUTPUT_DIR / "PEL_comparison_v4.csv")

    if all_reliability:
        all_rel = pd.concat(all_reliability, ignore_index=True)
        all_rel.to_csv(OUTPUT_DIR / "Reliability_v4_summary.csv", index=False)

        print(f"\n{'='*70}")
        print("신뢰도 평가 요약 (ROC-AUC)")
        print(f"{'='*70}")
        print("\n--- ROC-AUC by Source × Substance ---")
        auc_pivot = all_rel.pivot(index="Substance", columns="Source", values="ROC_AUC")
        print(auc_pivot.to_string(float_format="%.3f"))

        print("\n--- CV-AUC (5-fold) by Source × Substance ---")
        cv_pivot = all_rel.pivot(index="Substance", columns="Source", values="CV_AUC_mean")
        print(cv_pivot.to_string(float_format="%.3f"))

        print("\n--- TEL 민감도/특이도 ---")
        for _, r in all_rel.iterrows():
            print(f"  {r['Source']:10s} {r['Substance']:10s} "
                  f"민감도={r['TEL_Sensitivity']:.2f} 특이도={r['TEL_Specificity']:.2f} "
                  f"AUC={r['ROC_AUC']:.3f}")

    print(f"\n결과 저장: {OUTPUT_DIR}")
    print("\n완료 ✅")


if __name__ == "__main__":
    main()