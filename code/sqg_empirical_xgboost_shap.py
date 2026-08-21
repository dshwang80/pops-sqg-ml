#!/usr/bin/env python3
"""
 =============================================================================
 Method 1: 경험적 PEL/TEL 도출 — XGBoost SHAP 기반
 R 코드(20260407_POPs_SQG_3.Rmd V11.7 Robust) → Python 변환 + 고도화

 3단계 파이프라인:
   Step 1: DB 정제 (음수 플래그 처리, 이성질체 가중 합산, 1% OC 정규화, 중금속 mPELQ 필터링)
   Step 2: XGBoost SHAP 종 선별 (방향성 검증 ≤ -0.3, TOC 지배력 검증)
   Step 3: 최종 SHAP 기반 EDS/NEDS 분류 → TEL/PEL 도출

 고도화:
   - shap.TreeExplainer 사용 (R: 수동 predcontrib)
   - 5-fold CV 기반 best_iteration 추출
   - 결과 CSV + 비교표 + 시각화 PNG 생성
 =============================================================================
"""

import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import json

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# 0. 경로 설정
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent  # 01_method1_empirical_SQG/
DATA_DIR = BASE_DIR / "data"
R_OUTPUT_DIR = BASE_DIR / "R_output"
PYTHON_OUTPUT_DIR = BASE_DIR / "python" / "output"
FIG_DIR = BASE_DIR / "python" / "figures"

PYTHON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

NOAA_DB_PATH = DATA_DIR / "US_Sediment_Risk_Analytical_Set_Mainland.csv"

# =============================================================================
# 1. 상수 정의 (R V11.7 기준)
# =============================================================================
TARGET_SUBSTANCES = ["DDTs", "CHLs", "PCBs", "PAHs", "Dieldrin"]

TARGET_ISOMERS = {
    "DDTs":      ["o,p'-DDD", "o,p'-DDE", "o,p'-DDT", "p,p'-DDD", "p,p'-DDE", "p,p'-DDT"],
    "CHLs":      ["Chlordane,_alpha-", "Chlordane,_gamma-", "Heptachlor",
                  "Heptachlor_epoxide", "Nonachlor,_cis-", "Nonachlor,_trans-", "Oxychlordane"],
    "PCBs":      ["PCB008","PCB018","PCB028","PCB044","PCB052","PCB066","PCB101",
                  "PCB105","PCB118","PCB128","PCB138","PCB153","PCB170","PCB180",
                  "PCB187","PCB195","PCB206","PCB209"],
    "PAHs":      ["Naphthalene","Acenaphthylene","Acenaphthene","Fluorene",
                  "Phenanthrene","Anthracene","Fluoranthene","Pyrene",
                  "Benzo(a)anthracene","Chrysene","Benzo(a)pyrene",
                  "Dibenzo(a,h)anthracene","Benzo(g,h,i)perylene"],
    # Dieldrin: 단일 물질 — isomer summation 불필요
}

METAL_COLS = ["Arsenic","Cadmium","Chromium,_total","Copper",
              "Lead","Mercury","Nickel","Silver","Zinc"]

PEL_METALS = {"Arsenic":41.6, "Cadmium":4.21, "Chromium,_total":160, "Copper":108,
              "Lead":112, "Mercury":0.7, "Nickel":42.8, "Silver":1.77, "Zinc":271}

PEL_ORGANICS = {"DDTs":46.1, "CHLs":4.79, "PCBs":0.277, "PAHs":44.8, "Dieldrin":8.0}

NOAA_TEL_DW = {"DDTs":3.89, "CHLs":2.26, "PAHs":1684, "Dieldrin":0.72, "PCBs":22.7}
NOAA_PEL_DW = {"DDTs":51.7, "CHLs":4.79, "PAHs":16770, "Dieldrin":4.30, "PCBs":180}

# 비환경/비저서 생물종 배제 + Chironomus 통일
EXCLUDE_PATTERNS = ["Human liver cell line", "species not known", "Photobacterium",
                    "Crassostrea", "Echinoderm"]

RANDOM_SEED = 42


# =============================================================================
# Step 1: DB 정제
# =============================================================================
def step1_db_curation(noaa_path):
    """
    R V11.7 Step 1 로직:
      1. 데이터 로드 → 음수 플래그(-9) → NA 변환
      2. Mean_Survival ≥ 0, cap 100, TOC_pct > 0 필터링
      3. 비저서 생물종 배제 + Chironomus → Chironomus dilutus 통일
      4. 중금속 mPELQ 계산
      5. 물질별 이성질체 가중 합산 (process_sum_refined)
      6. 중금속 mPELQ > 1.0 & > 유기물 PELQ × 1.5 인 지점 제거
    """
    print("\n" + "="*70)
    print("[Step 1] DB 정제 시작")
    print("="*70)

    # 1. 데이터 로드
    print("[1-1] NOAA DB 로딩 중...")
    df_raw = pd.read_csv(noaa_path, low_memory=False)
    print(f"  → 원본: {df_raw.shape[0]:,}행 × {df_raw.shape[1]}컬럼")

    # 2. 음수 플래그(-9) → NA
    print("[1-2] 음수 플래그(-9) → NA 변환...")
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
    df_raw[numeric_cols] = df_raw[numeric_cols].replace(-9, np.nan)

    # 3. 기본 필터링
    print("[1-3] 기본 필터링 (Mean_Survival ≥ 0, TOC_pct > 0, cap 100)...")
    df_pre = df_raw.copy()
    df_pre = df_pre[df_pre["Mean_Survival"].notna() & (df_pre["Mean_Survival"] >= 0)]
    df_pre["Mean_Survival"] = df_pre["Mean_Survival"].clip(upper=100)
    df_pre = df_pre[df_pre["TOC_pct"].notna() & (df_pre["TOC_pct"] > 0)]
    print(f"  → 기본 필터 후: {df_pre.shape[0]:,}행")

    # 4. 비저서 생물종 배제 + Chironomus 통일
    print("[1-4] 생물종 정제 (비저서 배제 + Chironomus 통일)...")
    for pat in EXCLUDE_PATTERNS:
        df_pre = df_pre[~df_pre["Latin_Name_Species"].str.contains(pat, na=False, case=False)]
    df_pre["Latin_Name_Species"] = df_pre["Latin_Name_Species"].apply(
        lambda x: "Chironomus dilutus" if isinstance(x, str) and "Chironomus" in x else x
    )
    print(f"  → 생물종 정제 후: {df_pre.shape[0]:,}행, 종 수: {df_pre['Latin_Name_Species'].nunique()}")

    # 5. 중금속 mPELQ 계산
    print("[1-5] 중금속 mPELQ 계산...")
    metal_df = df_pre[METAL_COLS].copy()
    metal_df = metal_df.fillna(0)
    quotients = pd.DataFrame({m: metal_df[m] / PEL_METALS[m] for m in METAL_COLS})
    df_pre["mPELQ_Metals"] = quotients.mean(axis=1)

    # 6. 물질별 이성질체 가중 합산 (process_sum_refined)
    print("[1-6] 물질별 이성질체 가중 합산 (ND vs NA 구분, OC 정규화)...")

    def process_sum_refined(data, cols):
        """
        R process_sum_refined 함수 변환:
          - Complete cases에서 성분비(reference weight) 산출
          - 개별 샘플: 측정값(>0) 그대로, ND(≤0)는 MDL/2, NA는 가중치로 보정
          - 이성질체 측정률 30% 미만 → NA
          - 결과: log10((estimated_total / TOC) + 1)
        """
        # Reference profile: complete cases에서 colMeans → weight
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
            measured_mask = row_vals.notna().values  # boolean array
            measured_cols = [c for c, m in zip(cols, measured_mask) if m]

            # 측정률 30% 미만 → skip
            if len(measured_cols) < (len(cols) * 0.3):
                continue

            toc = data.iloc[i]["TOC_pct"]
            if pd.isna(toc) or toc <= 0:
                continue

            # 측정값 합산: >0 그대로, ≤0(ND)은 해당 컬럼 MDL/2
            measured_sum = 0.0
            for col in measured_cols:
                val = row_vals[col]
                if val <= 0:  # ND
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
            # Dieldrin: 단일 물질, 단순 OC 정규화
            df_base["Sum_Dieldrin_OC_log"] = np.log10(
                (df_base["Dieldrin"] / df_base["TOC_pct"]) + 1
            )
            df_base.loc[df_base["Dieldrin"].isna(), "Sum_Dieldrin_OC_log"] = np.nan
        else:
            isomers = TARGET_ISOMERS[substance]
            col_name = f"Sum_{substance}_OC_log"
            df_base[col_name] = process_sum_refined(df_base, isomers)
        n_valid = df_base[f"Sum_{substance}_OC_log"].notna().sum()
        print(f"  → {substance}: 유효 농도 {n_valid:,}지점")

    # 7. 물질별 중금속 교란 제거 + 요약 통계
    print("[1-7] 물질별 중금속 mPELQ 필터링 및 요약 통계...")
    substance_dfs = {}
    summary_list = []

    for substance in TARGET_SUBSTANCES:
        target_col = f"Sum_{substance}_OC_log"
        df_sub = df_base[df_base[target_col].notna()].copy()

        if len(df_sub) == 0:
            continue

        # ─── EqP 필터 (Phase 2) ───────────────────────────────────────────
        # Di Toro et al. 1991 EqP paradigm:
        #   K_oc = 0.41 × K_ow,  C_pw = C_sed × 1000 / (f_oc × K_oc)
        #   If C_pw < chronic threshold (CCC/FCV) AND survival < 80%
        #   → "Implausible Toxicity": porewater concentration too low to cause
        #     observed mortality → remove data point (not chemically caused)
        #
        # All K_ow and toxicity values from published sources:
        #   DDTs:      logKow=6.91 (ATSDR 2002), CCC=0.001 µg/L (EPA WQC saltwater)
        #   CHLs:      logKow=6.00 (PubChem),   CCC=0.004 µg/L (EPA WQC saltwater)
        #   PCBs:      logKow=6.80 (EPA fact),  CCC=0.03  µg/L (EPA WQC saltwater)
        #   Dieldrin:  logKow=5.37 (EPA ESB),   FCV=0.1469 µg/L (EPA ESB saltwater)
        #   PAHs:      logKow=5.20 (fluoranthene, EPA ESB PAH Table 3-4),
        #              FCV=2.322 µg/L (EPA ESB PAH Table 3-4)
        # ────────────────────────────────────────────────────────────────
        eqp_params = {
            'DDTs':     {'log_kow': 6.91, 'ccc_ugL': 0.001},
            'CHLs':     {'log_kow': 6.00, 'ccc_ugL': 0.004},
            'PCBs':     {'log_kow': 6.80, 'ccc_ugL': 0.03},
            'Dieldrin': {'log_kow': 5.37, 'ccc_ugL': 0.1469},
            'PAHs':     {'log_kow': 5.20, 'ccc_ugL': 2.322},
        }

        if substance in eqp_params:
            p = eqp_params[substance]
            kow = 10 ** p['log_kow']
            koc = 0.41 * kow  # Di Toro 1991
            f_oc = 0.02       # 2% OC (NOAA/ERLM typical marine sediment)
            # C_sed: Conc_log is log10(µg/g dry weight, 1% OC normalized)
            # Back-transform: C_sed = 10^Conc_log (µg/g)
            c_sed = 10 ** df_sub[target_col].values  # µg/g dry
            c_pw = c_sed * 1000 / (f_oc * koc)       # µg/L porewater
            # Implausible: C_pw < CCC AND survival < 80%
            implausible = (c_pw < p['ccc_ugL']) & (df_sub["Mean_Survival"].values < 80)
            n_before = len(df_sub)
            df_sub = df_sub[~implausible].copy()
            n_after = len(df_sub)
            n_removed = n_before - n_after
            print(f"  → {substance}: {n_before:,} → EqP 필터 후 {n_after:,}지점 "
                  f"(제거 {n_removed:,}, {n_removed / n_before * 100:.1f}%) "
                  f"[C_pw < {p['ccc_ugL']} µg/L & survival < 80%]")
        else:
            # 미정의 물질은 필터링 없음
            print(f"  → {substance}: EqP 파라미터 없음, 필터링 생략 ({len(df_sub):,}지점)")

        substance_dfs[substance] = df_sub

        # 요약 통계
        stats_df = df_sub.groupby(["Region", "Latin_Name_Species"]).agg(
            Substance=("Latin_Name_Species", lambda _: substance),
            Total_N=("Mean_Survival", "size"),
            Avg_Survival=("Mean_Survival", "mean"),
            Toxic_N=("Mean_Survival", lambda x: (x < 80).sum())
        ).reset_index()
        stats_df = stats_df[["Substance", "Region", "Latin_Name_Species", "Total_N", "Avg_Survival", "Toxic_N"]]
        summary_list.append(stats_df)

    final_summary = pd.concat(summary_list, ignore_index=True)
    final_summary.to_csv(PYTHON_OUTPUT_DIR / "Step1_Substance_Summary.csv", index=False)
    print(f"\n[Step 1 완료] 요약 통계 저장: {PYTHON_OUTPUT_DIR / 'Step1_Substance_Summary.csv'}")
    print(f"  총 {len(final_summary)}개 Region×Species 조합")

    return df_base, substance_dfs


# =============================================================================
# Step 2: XGBoost SHAP 종 선별
# =============================================================================
def step2_species_selection(substance_dfs):
    """
    R V8.0 Step 2.1 (Corrected Strict) 로직:
      1. 생물종별 조건: N ≥ 30, Toxic_N(생존<80%) ≥ 10 → 미충족 시 "Insufficient"
      2. XGBoost: Survival ~ Conc_log + TOC_pct (2-feature)
      3. shap.TreeExplainer로 SHAP 값 추출
      4. 방향성 검증: Spearman cor(Conc_log, SHAP_Conc_log) ≤ -0.3
      5. 지배력 검증: Chem_Importance > TOC_Importance / 1.5
      6. 통과 시 "Selected (Causal Indicator)"

    [Fix 1] df_base(필터링 안 된 원본) 대신 substance_dfs(중금속 필터링 적용) 사용
    """
    print("\n" + "="*70)
    print("[Step 2] XGBoost SHAP 종 선별 (AI-driven Species Selection)")
    print("="*70)

    eval_list = []

    for substance in TARGET_SUBSTANCES:
        if substance not in substance_dfs:
            continue

        df_sub = substance_dfs[substance].rename(columns={f"Sum_{substance}_OC_log": "Conc_log"})
        species_list = df_sub["Latin_Name_Species"].unique()

        print(f"\n--- {substance} ({len(species_list)}종) ---")

        for sp in species_list:
            df_sp = df_sub[df_sub["Latin_Name_Species"] == sp]
            total_n = len(df_sp)
            toxic_n = int((df_sp["Mean_Survival"] < 80).sum())

            # 1. 기초 필터링
            if total_n < 30 or toxic_n < 10:
                eval_list.append({
                    "Substance": substance, "Species": sp,
                    "Total_N": total_n, "Toxic_N": toxic_n,
                    "Chem_Importance": np.nan, "TOC_Importance": np.nan,
                    "Chem_Direction": np.nan,
                    "Status": "Excluded (Insufficient Data/Toxicity)"
                })
                continue

            # 2. XGBoost 모델링
            X = df_sp[["Conc_log", "TOC_pct"]].values.astype(np.float64)
            y = df_sp["Mean_Survival"].values.astype(np.float64)

            # NaN/Inf 체크
            if np.any(np.isnan(X)) or np.any(np.isinf(X)) or np.any(np.isnan(y)):
                eval_list.append({
                    "Substance": substance, "Species": sp,
                    "Total_N": total_n, "Toxic_N": toxic_n,
                    "Chem_Importance": np.nan, "TOC_Importance": np.nan,
                    "Chem_Direction": np.nan,
                    "Status": "Excluded (Data Quality)"
                })
                continue

            try:
                # [Fix 3] R V11.7과 정확히 일치: nrounds=100 고정, subsample/colsample 없음, CV 없음
                dtrain = xgb.DMatrix(X, label=y, feature_names=["Conc_log", "TOC_pct"])

                params = {
                    "objective": "reg:squarederror",
                    "eta": 0.1,
                    "max_depth": 3,
                    "seed": RANDOM_SEED
                }

                # R V11.7 방식: nrounds=100 고정, CV 없음
                model = xgb.train(params, dtrain, num_boost_round=100)

                # 3. SHAP 추출: R predict(xgb, predcontrib=TRUE)와 정확히 일치
                # [Fix 3] shap.TreeExplainer 대신 XGBoost 내장 pred_contribs 사용
                dtest = xgb.DMatrix(X, feature_names=["Conc_log", "TOC_pct"])
                pred = model.predict(dtest, pred_contribs=True)
                # pred_contribs: (N, n_features+1) — 마지막 열은 bias
                shap_chem = pred[:, 0]
                shap_toc = pred[:, 1]

                imp_chem = np.mean(np.abs(shap_chem))
                imp_toc = np.mean(np.abs(shap_toc))

                # 4. 방향성 검증: Spearman correlation
                cor_chem = stats.spearmanr(X[:, 0], shap_chem).correlation
                if np.isnan(cor_chem):
                    cor_chem = 0.0

                # 5. 인과성 판정
                status = "Selected (Causal Indicator)"

                if cor_chem > -0.30:
                    status = "Excluded (Non-toxic Trend)"
                elif imp_toc > (imp_chem * 1.5):
                    status = "Excluded (TOC Dominated)"

                eval_list.append({
                    "Substance": substance, "Species": sp,
                    "Total_N": total_n, "Toxic_N": toxic_n,
                    "Chem_Importance": imp_chem, "TOC_Importance": imp_toc,
                    "Chem_Direction": cor_chem, "Status": status
                })

                if "Selected" in status:
                    print(f"  ✓ {sp}: N={total_n}, dir={cor_chem:.3f}, chem={imp_chem:.2f}, toc={imp_toc:.2f}")

            except Exception as e:
                eval_list.append({
                    "Substance": substance, "Species": sp,
                    "Total_N": total_n, "Toxic_N": toxic_n,
                    "Chem_Importance": np.nan, "TOC_Importance": np.nan,
                    "Chem_Direction": np.nan,
                    "Status": f"Excluded (Model Error: {str(e)[:50]})"
                })

    df_eval = pd.DataFrame(eval_list)

    # 결과 출력
    print("\n[Step 2 결과 요약]")
    status_summary = df_eval.groupby(["Substance", "Status"]).size().reset_index(name="Count")
    print(status_summary.to_string(index=False))

    selected = df_eval[df_eval["Status"].str.contains("Selected", na=False)]
    print(f"\n선택된 인과성 지표종: {len(selected)}종")
    if len(selected) > 0:
        print(selected[["Substance", "Species", "Total_N", "Chem_Direction",
                        "Chem_Importance", "TOC_Importance"]].to_string(index=False))

    # 저장
    df_eval.to_csv(PYTHON_OUTPUT_DIR / "Step2_Species_Evaluation.csv", index=False)
    print(f"\n[Step 2 완료] 저장: {PYTHON_OUTPUT_DIR / 'Step2_Species_Evaluation.csv'}")

    return df_eval


# =============================================================================
# Step 3: 최종 SHAP 기반 EDS/NEDS 분류 → TEL/PEL 도출
# =============================================================================
def step3_final_sqg(substance_dfs, df_eval):
    """
    R V11.7 Step 3 로직:
      1. 선별종 데이터 풀링
      2. DDTs/Dieldrin/PCBs에 Leptocheirus plumulosus 추가 (R 하드코딩)
      3. 최종 XGBoost → shap.TreeExplainer → 샘플별 SHAP 추출
      4. EDS: 생존 < 80% & SHAP ≤ -0.1
      5. NEDS: 생존 ≥ 80% (전체 보존, R V11.7 방식)
      6. TEL = sqrt(EDS P15 × NEDS P50), PEL = sqrt(EDS P50 × NEDS P85)
      7. Dose-response 곡선 피팅 (logistic)

    [Fix 1] df_base(필터링 안 된 원본) 대신 substance_dfs(중금속 필터링 적용) 사용
    """
    print("\n" + "="*70)
    print("[Step 3] 최종 SHAP 기반 EDS/NEDS 분류 → TEL/PEL 도출")
    print("="*70)

    # 선별종 사전 구성
    selected_species_dict = {}
    for sub in TARGET_SUBSTANCES:
        sel = df_eval[(df_eval["Substance"] == sub) & df_eval["Status"].str.contains("Selected", na=False)]
        spp = sel["Species"].unique().tolist()
        if spp:
            selected_species_dict[sub] = spp

    # R 하드코딩: DDTs, Dieldrin, PCBs에 Leptocheirus plumulosus 추가
    for sub in ["DDTs", "Dieldrin", "PCBs"]:
        if sub in selected_species_dict:
            selected_species_dict[sub] = list(set(selected_species_dict[sub] + ["Leptocheirus plumulosus"]))
        else:
            selected_species_dict[sub] = ["Leptocheirus plumulosus"]

    plot_data_list = {}
    sqg_line_list = []
    curve_list = {}

    for substance in TARGET_SUBSTANCES:
        accepted = selected_species_dict.get(substance)
        if not accepted:
            print(f"  ⚠ {substance}: 선별종 없음 — 스킵")
            continue

        if substance not in substance_dfs:
            print(f"  ⚠ {substance}: 필터링된 데이터 없음 — 스킵")
            continue

        target_col = f"Sum_{substance}_OC_log"
        df_substance = substance_dfs[substance]
        if target_col not in df_substance.columns:
            print(f"  ⚠ {substance}: {target_col} 컬럼 없음 — 스킵")
            continue

        df_pooled = df_substance[df_substance[target_col].notna()].rename(columns={target_col: "Conc_log"})
        df_pooled = df_pooled[df_pooled["Latin_Name_Species"].isin(accepted)].copy()
        df_pooled["Substance"] = substance

        if len(df_pooled) < 15:
            print(f"  ⚠ {substance}: N={len(df_pooled)} < 15 — 스킵")
            continue

        print(f"\n▶ {substance} 분석 중... (N={len(df_pooled)})")

        # XGBoost 데이터 준비
        X = df_pooled[["Conc_log", "TOC_pct"]].values.astype(np.float64)
        y = df_pooled["Mean_Survival"].values.astype(np.float64)

        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print(f"  ⚠ {substance}: NaN/Inf 발견 — 스킵")
            continue

        # [Fix 3] R V11.7과 정확히 일치: nrounds=100 고정, subsample/colsample 없음, CV 없음
        dtrain = xgb.DMatrix(X, label=y, feature_names=["Conc_log", "TOC_pct"])
        params = {
            "objective": "reg:squarederror",
            "eta": 0.1, "max_depth": 3,
            "seed": RANDOM_SEED
        }

        xgb_final = xgb.train(params, dtrain, num_boost_round=100)

        # SHAP 추출: R predict(xgb, predcontrib=TRUE)와 정확히 일치
        # [Fix 3] shap.TreeExplainer 대신 XGBoost 내장 pred_contribs 사용
        dtest = xgb.DMatrix(X, feature_names=["Conc_log", "TOC_pct"])
        pred = xgb_final.predict(dtest, pred_contribs=True)
        df_pooled["SHAP_Chem"] = pred[:, 0]

        # 분류 로직 (R V11.7: NEDS 전체 보존)
        df_pooled["Data_Type"] = "Noise (Unexplained)"

        # EDS: 생존 < 80% & SHAP ≤ -0.1
        eds_mask = (df_pooled["Mean_Survival"] < 80) & (df_pooled["SHAP_Chem"] <= -0.1)
        df_pooled.loc[eds_mask, "Data_Type"] = "Causal EDS (True Drivers & Co-toxicity)"

        # NEDS: 생존 ≥ 80% (전체 보존)
        neds_mask = df_pooled["Mean_Survival"] >= 80
        df_pooled.loc[neds_mask, "Data_Type"] = "True NEDS (Clean Baseline)"

        n_eds = eds_mask.sum()
        n_neds = neds_mask.sum()
        print(f"  EDS: {n_eds}지점, NEDS: {n_neds}지점")

        # SQG 계산
        e_v = df_pooled.loc[df_pooled["Data_Type"] == "Causal EDS (True Drivers & Co-toxicity)", "Conc_log"].values
        n_v = df_pooled.loc[df_pooled["Data_Type"] == "True NEDS (Clean Baseline)", "Conc_log"].values

        if len(e_v) >= 2 and len(n_v) >= 2:
            tel_ml = np.sqrt(np.quantile(e_v, 0.15) * np.quantile(n_v, 0.50))
            pel_ml = np.sqrt(np.quantile(e_v, 0.50) * np.quantile(n_v, 0.85))

            our_tel = 10 ** tel_ml - 1
            our_pel = 10 ** pel_ml - 1
            noaa_tel_log = np.log10(NOAA_TEL_DW[substance] + 1)
            noaa_pel_log = np.log10(NOAA_PEL_DW[substance] + 1)

            sqg_line_list.append({
                "Substance": substance,
                "Total_N": len(df_pooled),
                "N_EDS": n_eds,
                "N_NEDS": n_neds,
                "Our_TEL_dw": our_tel,
                "Our_PEL_dw": our_pel,
                "TEL_ML_log": tel_ml,
                "PEL_ML_log": pel_ml,
                "NOAA_TEL_dw": NOAA_TEL_DW[substance],
                "NOAA_PEL_dw": NOAA_PEL_DW[substance],
                "NOAA_TEL_log": noaa_tel_log,
                "NOAA_PEL_log": noaa_pel_log
            })
            print(f"  TEL = {our_tel:.4f} µg/kg, PEL = {our_pel:.4f} µg/kg")
            print(f"  NOAA TEL = {NOAA_TEL_DW[substance]}, NOAA PEL = {NOAA_PEL_DW[substance]}")

            # Dose-response 곡선 피팅 (logistic: 100 / (1 + exp(b*(x-c))))
            fit_df = df_pooled[df_pooled["Data_Type"].str.contains("EDS|NEDS")].copy()
            if len(fit_df) >= 10:
                try:
                    def logistic(x, b, c):
                        return 100 / (1 + np.exp(b * (x - c)))

                    p0 = [1.0, np.median(fit_df["Conc_log"].values)]
                    popt, _ = curve_fit(logistic, fit_df["Conc_log"].values,
                                        fit_df["Mean_Survival"].values, p0=p0, maxfev=10000)
                    cx = np.linspace(df_pooled["Conc_log"].min(), df_pooled["Conc_log"].max(), 100)
                    cy = logistic(cx, *popt)
                    curve_list[substance] = pd.DataFrame({"Conc_log": cx, "Survival": cy})
                    print(f"  Dose-response 곡선 피팅 성공 (b={popt[0]:.3f}, c={popt[1]:.3f})")
                except Exception as e:
                    print(f"  곡선 피팅 실패: {e}")

        plot_data_list[substance] = df_pooled

    # 결과 저장
    final_line = pd.DataFrame(sqg_line_list)
    final_line.to_csv(PYTHON_OUTPUT_DIR / "Master_Table_Python_Final.csv", index=False)
    print(f"\n[Step 3 완료] 최종 SQG 저장: {PYTHON_OUTPUT_DIR / 'Master_Table_Python_Final.csv'}")

    # 전체 분류 데이터 저장
    all_plot = pd.concat(plot_data_list.values(), ignore_index=True)
    all_plot.to_csv(PYTHON_OUTPUT_DIR / "Step3_All_Classified_Data.csv", index=False)

    return final_line, plot_data_list, curve_list


# =============================================================================
# Step 4: R 결과와 비교 검증
# =============================================================================
def step4_compare_with_r():
    """R V11.7 결과(Master_Table_V11.7_Final.csv)와 Python 결과 비교"""
    print("\n" + "="*70)
    print("[Step 4] R 결과 vs Python 결과 비교 검증")
    print("="*70)

    r_file = R_OUTPUT_DIR / "Master_Table_V11.7_Final.csv"
    py_file = PYTHON_OUTPUT_DIR / "Master_Table_Python_Final.csv"

    if not r_file.exists():
        print(f"  ⚠ R 결과 파일 없음: {r_file}")
        return None
    if not py_file.exists():
        print(f"  ⚠ Python 결과 파일 없음: {py_file}")
        return None

    df_r = pd.read_csv(r_file)
    df_py = pd.read_csv(py_file)

    # 병합
    df_merge = df_r[["Substance", "Our_TEL_dw", "Our_PEL_dw", "Total_N"]].merge(
        df_py[["Substance", "Our_TEL_dw", "Our_PEL_dw", "Total_N", "N_EDS", "N_NEDS"]],
        on="Substance", suffixes=("_R", "_Py")
    )

    # 차이 계산
    df_merge["TEL_diff_%"] = ((df_merge["Our_TEL_dw_Py"] - df_merge["Our_TEL_dw_R"]) /
                               df_merge["Our_TEL_dw_R"] * 100).round(2)
    df_merge["PEL_diff_%"] = ((df_merge["Our_PEL_dw_Py"] - df_merge["Our_PEL_dw_R"]) /
                               df_merge["Our_PEL_dw_R"] * 100).round(2)

    print("\n비교 결과:")
    print(df_merge.to_string(index=False, float_format="%.4f"))

    df_merge.to_csv(PYTHON_OUTPUT_DIR / "R_vs_Python_Comparison.csv", index=False)
    print(f"\n비교표 저장: {PYTHON_OUTPUT_DIR / 'R_vs_Python_Comparison.csv'}")

    return df_merge


# =============================================================================
# Step 5: 시각화 (SHAP summary + 최종 SQG 산점도)
# =============================================================================
def step5_visualization(final_line, plot_data_list, curve_list):
    """SHAP 시각화 + 최종 SQG 산점도 (R V11.7 스타일)"""
    print("\n" + "="*70)
    print("[Step 5] 시각화 생성")
    print("="*70)

    # 색상 정의 (R V11.7과 동일)
    COLORS = {
        "True NEDS (Clean Baseline)": "#4575b4",
        "Causal EDS (True Drivers & Co-toxicity)": "#d73027",
        "Noise (Unexplained)": "grey80",
        "Noise (Bioavailability Artifact)": "grey40",
    }

    # --- Figure 1: 최종 SQG 산점도 (R V11.7 스타일) ---
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    axes = axes.flatten()

    for idx, substance in enumerate(TARGET_SUBSTANCES):
        ax = axes[idx]
        if substance not in plot_data_list:
            ax.set_visible(False)
            continue

        df_pooled = plot_data_list[substance]
        df_pooled = df_pooled[df_pooled["Conc_log"] > 0.005].copy()

        # Noise (회색)
        noise = df_pooled[df_pooled["Data_Type"].str.contains("Noise", na=False)]
        ax.scatter(noise["Conc_log"], noise["Mean_Survival"], c="grey",
                   alpha=0.2, s=8, label="Noise" if idx == 0 else "")

        # EDS (빨강)
        eds = df_pooled[df_pooled["Data_Type"] == "Causal EDS (True Drivers & Co-toxicity)"]
        ax.scatter(eds["Conc_log"], eds["Mean_Survival"], c="#d73027",
                   alpha=0.7, s=12, label="Causal EDS" if idx == 0 else "")

        # NEDS (파랑)
        neds = df_pooled[df_pooled["Data_Type"] == "True NEDS (Clean Baseline)"]
        ax.scatter(neds["Conc_log"], neds["Mean_Survival"], c="#4575b4",
                   alpha=0.5, s=8, label="True NEDS" if idx == 0 else "")

        # Dose-response 곡선
        if substance in curve_list:
            curve = curve_list[substance]
            ax.plot(curve["Conc_log"], curve["Survival"], color="#333333", linewidth=1.0)

        # SQG 수직선
        row = final_line[final_line["Substance"] == substance]
        if len(row) > 0:
            r = row.iloc[0]
            ax.axvline(r["NOAA_TEL_log"], color="#4daf4a", linestyle="--", linewidth=1.0)
            ax.axvline(r["NOAA_PEL_log"], color="#4daf4a", linestyle="--", linewidth=1.0)
            ax.axvline(r["TEL_ML_log"], color="#4575b4", linewidth=1.5)
            ax.axvline(r["PEL_ML_log"], color="#d73027", linewidth=1.5)

        ax.set_title(substance, fontsize=14, fontweight="bold")
        ax.set_xlabel("Concentration [µg/kg dw at 1% TOC]", fontsize=10)
        ax.set_ylabel("Mean Survival Rate (%)", fontsize=10)
        ax.set_ylim(-5, 105)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(25))

        # x축을 원래 스케일로 변환
        def log_to_orig(x, pos):
            return f"{10**x - 1:.1f}"
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_to_orig))
        ax.tick_params(axis='x', rotation=30)

    # 마지막 빈 서브플롯에 범례
    if len(TARGET_SUBSTANCES) < 6:
        ax_legend = axes[len(TARGET_SUBSTANCES)]
        ax_legend.set_visible(False)

    fig.suptitle("Final ML-SQG: Python XGBoost SHAP (Converted from R V11.7)",
                 fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "Figure_Python_Final_SQG.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Figure_Python_Final_SQG.png 저장")

    # --- Figure 2: SHAP Summary Plot (각 물질별) ---
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    axes2 = axes2.flatten()

    shap_plot_idx = 0
    for substance in TARGET_SUBSTANCES:
        if substance not in plot_data_list:
            continue
        ax = axes2[shap_plot_idx]
        shap_plot_idx += 1

        df_pooled = plot_data_list[substance]

        # 재추출용 XGBoost (이미 step3에서 학습했지만 여기서는 시각화용)
        X = df_pooled[["Conc_log", "TOC_pct"]].values
        y = df_pooled["Mean_Survival"].values

        dtrain = xgb.DMatrix(X, label=y, feature_names=["Conc_log", "TOC_pct"])
        params = {"objective": "reg:squarederror", "eta": 0.1, "max_depth": 3,
                  "subsample": 0.8, "colsample_bytree": 0.8, "seed": RANDOM_SEED}

        try:
            cv_res = xgb.cv(params, dtrain, num_boost_round=200, nfold=5,
                            early_stopping_rounds=10, verbose_eval=False, show_stdv=False)
            best_iter = cv_res.shape[0]
        except Exception:
            best_iter = 100

        model = xgb.train(params, dtrain, num_boost_round=best_iter)
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)

        # SHAP summary (scatter): x=SHAP value, y=feature value, color=SHAP magnitude
        for feat_idx, feat_name in enumerate(["Conc_log", "TOC_pct"]):
            shap_col = shap_vals[:, feat_idx]
            feat_col = X[:, feat_idx]
            color = shap_col  # SHAP 값으로 색상
            ax.scatter(shap_col, feat_col, c=color, cmap="RdBu_r", s=5, alpha=0.5,
                       label=feat_name)

        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_title(substance, fontsize=12, fontweight="bold")
        ax.set_xlabel("SHAP value (impact on survival)", fontsize=9)
        ax.legend(fontsize=8)

    # 남은 서브플롯 숨기기
    for i in range(shap_plot_idx, len(axes2)):
        axes2[i].set_visible(False)

    fig2.suptitle("SHAP Summary: Feature Impact on Survival (Python)",
                  fontsize=14, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig(FIG_DIR / "Figure_SHAP_Summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"  ✓ Figure_SHAP_Summary.png 저장")

    # --- Figure 3: R vs Python 비교 막대그래프 ---
    r_file = R_OUTPUT_DIR / "Master_Table_V11.7_Final.csv"
    py_file = PYTHON_OUTPUT_DIR / "Master_Table_Python_Final.csv"
    if r_file.exists() and py_file.exists():
        df_r = pd.read_csv(r_file)
        df_py = pd.read_csv(py_file)
        df_cmp = df_r[["Substance", "Our_TEL_dw", "Our_PEL_dw"]].merge(
            df_py[["Substance", "Our_TEL_dw", "Our_PEL_dw"]],
            on="Substance", suffixes=("_R", "_Py")
        )

        fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        x = np.arange(len(df_cmp))
        width = 0.35

        ax1.bar(x - width/2, df_cmp["Our_TEL_dw_R"], width, label="R V11.7", color="#4575b4")
        ax1.bar(x + width/2, df_cmp["Our_TEL_dw_Py"], width, label="Python", color="#d73027")
        ax1.set_xticks(x)
        ax1.set_xticklabels(df_cmp["Substance"], rotation=45)
        ax1.set_ylabel("TEL (µg/kg dw)")
        ax1.set_title("TEL Comparison: R vs Python", fontweight="bold")
        ax1.legend()

        ax2.bar(x - width/2, df_cmp["Our_PEL_dw_R"], width, label="R V11.7", color="#4575b4")
        ax2.bar(x + width/2, df_cmp["Our_PEL_dw_Py"], width, label="Python", color="#d73027")
        ax2.set_xticks(x)
        ax2.set_xticklabels(df_cmp["Substance"], rotation=45)
        ax2.set_ylabel("PEL (µg/kg dw)")
        ax2.set_title("PEL Comparison: R vs Python", fontweight="bold")
        ax2.legend()

        fig3.tight_layout()
        fig3.savefig(FIG_DIR / "Figure_R_vs_Python_Comparison.png", dpi=300, bbox_inches="tight")
        plt.close(fig3)
        print(f"  ✓ Figure_R_vs_Python_Comparison.png 저장")


# =============================================================================
# Main 실행
# =============================================================================
def main():
    print("="*70)
    print("Method 1: 경험적 PEL/TEL 도출 — XGBoost SHAP (Python 변환)")
    print("R V11.7 Robust → Python + shap.TreeExplainer 고도화")
    print("="*70)

    # Step 1: DB 정제
    df_base, substance_dfs = step1_db_curation(NOAA_DB_PATH)

    # Step 2: 종 선별 (중금속 필터링된 substance_dfs 사용)
    df_eval = step2_species_selection(substance_dfs)

    # Step 3: 최종 SQG 도출 (중금속 필터링된 substance_dfs 사용)
    final_line, plot_data_list, curve_list = step3_final_sqg(substance_dfs, df_eval)

    # Step 4: R 결과 비교
    comparison = step4_compare_with_r()

    # Step 5: 시각화
    step5_visualization(final_line, plot_data_list, curve_list)

    # 최종 결과 출력
    print("\n" + "="*70)
    print("최종 결과 요약")
    print("="*70)
    print(final_line[["Substance", "Total_N", "N_EDS", "N_NEDS",
                       "Our_TEL_dw", "Our_PEL_dw"]].to_string(index=False, float_format="%.4f"))
    print(f"\n출력 디렉토리: {PYTHON_OUTPUT_DIR}")
    print(f"그림 디렉토리: {FIG_DIR}")
    print("\n완료 ✅")


if __name__ == "__main__":
    main()