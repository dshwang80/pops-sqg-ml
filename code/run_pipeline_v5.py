#!/usr/bin/env python3
"""
고도화 파이프라인 v5.3: OOF-SHAP 합의 Primary + R_NT 기여도 tier + 다중 POPs feature
=====================================================

[사용자 설계 (2026-08-28 확정)]
1. TEL/PEL 신뢰도 향상
   - OOF-SHAP 합의 기반 비대상 화학물질 기여 판정 → 비대상 영향 EDS 제거 (Step 2.7)
   - DRC는 보조 진단으로만 사용 (Primary 종 선별에 관여 안 함; 불량 종도 Primary 유지)

2. 준거치 신뢰도 판단
   - 기 설정 기준(CCME/NOAA)과 비교 검토 (Step 4-1)
   - 독성 예측력(ROC-AUC, GroupKFold) 검토 (Step 4-2)

파이프라인 구조:
  Step 1: DB 정제 (기본 QA + PAHs Sum 계산; mPELQ_Metals는 ML 입력변수)
  Step 2: XGBoost SHAP 종 선별 (Spearman ≤ -0.3)
  Step 2.5: DRC 품질 평가 (진단 전용 — Primary 종 선별에 관여 안 함)
  Step 2.7: OOF-SHAP 합의 Primary (R_NT 기여도 기반) + DRC 보조진단 ★
  Step 3: EDS/NEDS → TEL/PEL
  Step 4: 신뢰도 평가 (기준 비교 + ROC-AUC, GroupKFold) ★

[Primary 필터 (Step 2.7) — v5.3 확정]
  - R_NT = (pred_without_non_target − pred_full) / (80 − pred_full) ≥ 0.5
    ("비대상물질이 예측 독성결손의 과반을 설명", 사전 확정값)
  - 판정 조건 (모두 충족한 EDS만 제외):
      ① 관측 생존율 < 80  ② full model 예측 < 80 (deficit > 0)
      ③ target SHAP ≥ 0  ④ measured non-target contaminant SHAP 합 < 0
      ⑤ R_NT ≥ 0.5
      ⑥ TreeSHAP·Interventional 각각 10회 중 8회 이상 동일 판정 (fold 합의)
  - 결측 PAH/POPs의 SHAP은 non_target 판정에서 제외 (observed_mask)
  - mPELQ ≤ 0.5 하드 제외는 Primary에서 제거됨 (v5.3) —
    mPELQ_Metals는 ML 입력변수로 유지되며, 화학 스크린은
    Chemical-screen sensitivity (Sensitivity_ChemScreen_mPELQ0.5)로만 존재.
  - 농도 < TEL은 제외 조건으로 절대 사용하지 않음 (순환성 방지; TEL 이하 EDS는
    별도 audit로 SHAP 판정 비율만 표시)
  - DRC(C80 gate, 잔차)는 보조 진단으로만 유지 (Primary 제거 아님)
  - 안전장치: 제거율 > max_removal_rate(15%) 시 applicability_pass=False 플래그 (all-or-none 폐지)

[Sensitivity — v5.3 재구성 (5 arms)]
  1. Strict ≥ 1.0:       R_NT ≥ 1.0 — 비대상물질이 독성 결손을 100% 이상 설명
                          (구 c5 pred_wo_nt≥80과 수학적 동치)
  2. Primary ≥ 0.5:      위 Primary 필터 그 자체 (R_NT ≥ 0.5 합의)
  3. Exploratory ≥ 0.25: R_NT ≥ 0.25 — 탐색적 (동일 합의 절차)
  4. Chemical-screen:    mPELQ ≤ 0.5 통과 자료만 유지 (구 Step 1 화학 필터의 재현;
                          SHAP 제외 미적용 — 독립 arm)
  5. Unfiltered:         기본 QA·종 선별만 적용 (SHAP·화학 스크린 모두 미적용, 기준선)

  중첩성(사전 확정 원칙): 제외 집합은 반드시 Strict ⊆ Primary ⊆ Exploratory.
  런타임 단언으로 강제하며 위반 시 RuntimeError (집계 로직 오류).

[진단 vs 합의 (v5.3 명명)]
  - 제외 건수는 fold별 TreeSHAP ∩ Interventional 합의(consensus)로만 확정:
    N_excluded_primary_consensus 등
  - N_mean_TreeRNT_ge*_diagnostic은 TreeSHAP 평균 R_NT 기준의 표본 수준 진단으로
    consensus 제외 건수와 다를 수 있음 — 논문 표에는 consensus 수치만 사용.
"""
import warnings
import sys
import math
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from scipy import stats
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
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

TARGET_SUBSTANCES = ["DDTs", "CHLs", "PCBs"]

# ★ v5: PAHs는 TARGET_SUBSTANCES에 포함하지 않되(파이프라인 대상 아님),
#   Sum_PAHs_OC_log를 feature로 사용하기 위해 별도 계산이 필요.
#   step1_db_curation에서 TARGET_SUBSTANCES + ["PAHs"]에 대해 Sum을 계산한다.
FEATURE_SUBSTANCES = TARGET_SUBSTANCES + ["PAHs"]

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
}

METAL_COLS = ["Arsenic","Cadmium","Chromium,_total","Copper",
              "Lead","Mercury","Nickel","Silver","Zinc"]

PEL_METALS = {"Arsenic":41.6, "Cadmium":4.21, "Chromium,_total":160, "Copper":108,
              "Lead":112, "Mercury":0.7, "Nickel":42.8, "Silver":1.77, "Zinc":271}

CCME_ISQG = {"DDTs":1.19, "CHLs":2.26, "PAHs":1684, "Dieldrin":0.71, "PCBs":21.5}
CCME_PEL = {"DDTs":4.77, "CHLs":4.79, "PAHs":16770, "Dieldrin":4.30, "PCBs":189}
NOAA_TEL_DW = {"DDTs":3.89, "CHLs":2.26, "PAHs":1684, "Dieldrin":0.72, "PCBs":22.7}
NOAA_PEL_DW = {"DDTs":51.7, "CHLs":4.79, "PAHs":16770, "Dieldrin":4.30, "PCBs":180}

EXCLUDE_PATTERNS = ["Human liver cell line", "species not known", "Photobacterium",
                    "Crassostrea", "Echinoderm"]
RANDOM_SEED = 42

# ---- 재현성 config 로딩 (pipeline_v5_config.json) ----
CONFIG_PATH = REPO_ROOT / "code" / "pipeline_v5_config.json"


def load_config():
    """pipeline_v5_config.json을 읽어 재현성 파라미터를 반환한다.

    ★ v5.3: 배포 안전장치 — config 누락 시 {} 반환 대신 즉시 중단
    (사용자 지적: 묵시적 기본값 진행은 재현성 리스크).
    """
    import json
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(CONFIG_PATH)
    with open(CONFIG_PATH) as f:
        return json.load(f)

# mPELQ threshold: 평균 중금속이 PEL의 50% 초과 → 타 독성 기여도 의심
MPELQ_METALS_THRESHOLD = 0.5

# 단위 변환: DB 원본 단위 → ng/g (µg/kg dw)
UNIT_FACTOR = {
    "DDTs": 1.0, "CHLs": 1.0, "PCBs": 1.0, "PAHs": 1.0, "Dieldrin": 1.0,
}


# =============================================================================
# Step 1: DB 정제 (금속 측정 QA; mPELQ≤0.5는 Chemical-screen sensitivity용 플래그로만 기록)
# =============================================================================
def step1_db_curation(df_raw, source_name="", mpelq_threshold=None):
    cfg = load_config()
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
    # ★ 미측정 금속을 0으로 위장하지 않는다: fillna(0) 제거.
    #   데이터에는 불검출(0) 코딩이 없고, -9/NaN은 모두 "미측정"이다.
    #   측정된 금속만으로 평균을 내어 mPELQ를 산출한다(mean의 skipna 기본 동작).
    for m in METAL_COLS:
        if m not in df_pre.columns:
            df_pre[m] = np.nan
    metal_df = df_pre[METAL_COLS].copy()
    quotients = pd.DataFrame({m: metal_df[m] / PEL_METALS[m] for m in METAL_COLS})
    # 측정된 금속 수 기록 (미측정 금속을 0으로 위장하지 않음)
    df_pre["N_metals_measured"] = metal_df.notna().sum(axis=1)
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
    # ★ v5: FEATURE_SUBSTANCES (TARGET_SUBSTANCES + PAHs)에 대해 Sum 계산
    for substance in FEATURE_SUBSTANCES:
        if substance == "Dieldrin":
            if "Dieldrin" not in df_base.columns:
                df_base["Dieldrin"] = np.nan
            df_base["Dieldrin"] = df_base["Dieldrin"] * UNIT_FACTOR.get(substance, 1.0)
            df_base["Sum_Dieldrin_OC_log"] = np.log10(
                (df_base["Dieldrin"] / df_base["TOC_pct"]) + 1)
            df_base.loc[df_base["Dieldrin"].isna(), "Sum_Dieldrin_OC_log"] = np.nan
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

    # mPELQ 플래그 생성 (Primary 제외 아님 — Chemical-screen sensitivity용)
    substance_dfs = {}
    for substance in TARGET_SUBSTANCES:
        target_col = f"Sum_{substance}_OC_log"
        df_sub = df_base[df_base[target_col].notna()].copy()
        if len(df_sub) == 0:
            continue

        # ★ v5.3 설계 변경 (2026-08-28 사용자 확정):
        #   mPELQ ≤ 0.5 하드 제외는 Primary에서 제거하고 Chemical-screen sensitivity로 이동.
        #   Primary는 최소 측정 금속 수 QA만 적용 — mPELQ_Metals는 ML 입력변수로 유지되어
        #   모델이 혼합오염 기여도를 직접 판별하고, 제외 판정은 SHAP R_NT(과반 기여)가 담당.
        #   (mPELQ 하드 선별을 민감도로 이동 — 예전 필터로의 복귀 아님)
        # mpelq_threshold 파라미터가 주어지면 민감도 분석용으로 override
        if mpelq_threshold is not None:
            mpelq_thresh = mpelq_threshold
        else:
            mpelq_thresh = float(
                cfg.get("mpelq_metals_threshold", MPELQ_METALS_THRESHOLD)
            )
        n_before_mpelq = len(df_sub)
        # QA(하드 제외 유지): 측정 금속이 min_metals_measured 미만이면 mPELQ 신뢰 불가 → 제거
        min_metals = int(cfg.get("min_metals_measured", 3))
        df_sub = df_sub[df_sub["N_metals_measured"] >= min_metals].copy()
        n_qa_removed = n_before_mpelq - len(df_sub)
        # mPELQ ≤ threshold 통과 여부는 제외 조건이 아니라 sensitivity arm 정의용 플래그로 기록
        df_sub["screen_pass_mpelq"] = df_sub["mPELQ_Metals"] <= mpelq_thresh
        n_screen_excluded = int((~df_sub["screen_pass_mpelq"]).sum())
        print(f"  {substance}: 금속 QA 제거 {n_qa_removed} → 최종 {len(df_sub):,}지점 "
              f"(mPELQ<={mpelq_thresh} 초과 {n_screen_excluded}지점은 Primary 유지, "
              f"Chemical-screen sensitivity에서 분리)")

        substance_dfs[substance] = df_sub

    return df_base, substance_dfs


# =============================================================================
# Step 2: XGBoost OOF SHAP 종 선별
# =============================================================================
# OOF 종 선별 파라미터 (사용자 확정 2026-08-26)
#   - 종별 5-fold × 10회 GroupKFold OOF SHAP 계산
#   - test fold만 SHAP 계산, KFold 대체 금지
#   - 반복별 조건: Chem_Direction ≤ −0.30 AND TOC importance ≤ 1.5 × chemical importance
#   - 10회 중 8회 이상 충족 시 OOF-supported species
#   - in-sample SHAP은 비교용으로만 유지, 최종 선별은 OOF 기준
N_OOF_REPEATS = 10
N_OOF_FOLDS = 5
OOF_MIN_SUCCESS = 8
CHEM_DIRECTION_THRESHOLD = -0.30
TOC_IMPORTANCE_RATIO = 1.5


def _build_group_ids(df):
    """GroupKFold용 group_id를 행별 우선순위 결합으로 생성한다.

    사용자 확정 우선순위 (2026-08-26):
      1. station_key
      2. StudyID + Station
      3. sample_key
      4. SampleID
      5. 좌표 (Start_Latitude + Start_Longitude) — NOAA 전용
      6. record_id (모든 식별자가 결측일 때만)

    각 행은 위 우선순위에서 첫 번째로 존재하는(비결측) 식별자를 사용한다.
    결측 식별자는 건너뛰고, 모든 식별자가 결측이면 record_id(고유)를 사용한다.

    ★ NOAA의 Station은 고유 정점 식별자가 아니다 (2026-08-26 발견):
      - Station "3"이 48행·12좌표·2지역·18연도에 걸쳐 존재
      - 좌표가 여러 개인 Station이 739개
      - NOAA에는 StudyID가 없어 Station은 "연구 내부 임의 라벨"일 뿐
      → NOAA 행은 좌표(round4)로 그룹핑해야 정점을 고유하게 식별할 수 있다.
      SCCWRP 행은 station_key 등이 존재하므로 좌표까지 내려가지 않는다.

    반환: str 배열 (len(df),) — 각 행의 group_id.
    """
    n = len(df)
    if "record_id" not in df.columns:
        df = df.copy()
        df["record_id"] = [f"r{i:06d}" for i in range(n)]
    rec_ids = df["record_id"].astype(str).values

    # 결과 배열을 record_id로 초기화 (모든 식별자 결측 시 고유 그룹)
    groups = rec_ids.copy()

    # 우선순위가 낮은 것부터 덮어쓰고, 높은 우선순위가 마지막에 덮어쓴다.
    # (같은 행에 여러 식별자가 있어도 최종적으로 가장 높은 우선순위가 남는다)

    # 5. 좌표 (NOAA 전용 — Station이 고유 정점이 아니므로)
    if "Start_Latitude" in df.columns and "Start_Longitude" in df.columns:
        lat = df["Start_Latitude"]
        lon = df["Start_Longitude"]
        mask = lat.notna().values & lon.notna().values
        if mask.any():
            groups[mask] = ("coord:" + lat[mask].round(4).astype(str).values
                            + "|" + lon[mask].round(4).astype(str).values)

    # 4. SampleID
    if "SampleID" in df.columns:
        vals = df["SampleID"]
        mask = vals.notna().values
        if mask.any():
            groups[mask] = "sid:" + vals[mask].astype(str).values

    # 3. sample_key
    if "sample_key" in df.columns:
        vals = df["sample_key"]
        mask = vals.notna().values
        if mask.any():
            groups[mask] = "sak:" + vals[mask].astype(str).values

    # 2. StudyID + Station
    if "StudyID" in df.columns and "Station" in df.columns:
        sid = df["StudyID"]
        st = df["Station"]
        mask = sid.notna().values & st.notna().values
        if mask.any():
            groups[mask] = ("ss:" + sid[mask].astype(str).values
                            + "|" + st[mask].astype(str).values)

    # 1. station_key (최고 우선순위)
    if "station_key" in df.columns:
        vals = df["station_key"]
        mask = vals.notna().values
        if mask.any():
            groups[mask] = "sk:" + vals[mask].astype(str).values

    return groups


def _oof_shap_for_species(X, y, groups, random_seed):
    """
    종별 5-fold × N_OOF_REPEATS회 GroupKFold OOF SHAP 계산.

    각 반복에서 GroupKFold로 분할(KFold 대체 금지), train fold로 학습 후
    test fold만 SHAP(pred_contribs) 계산. test fold SHAP을 모아 종 전체의
    Chem_Direction / TOC importance 산출.

    반환: (success_count, oof_imp_chem_mean, oof_imp_toc_mean, oof_cor_chem_mean,
           n_unique_fold_splits)
    """
    success_count = 0
    oof_imp_chem_list = []
    oof_imp_toc_list = []
    oof_cor_chem_list = []
    fold_split_signatures = set()

    for rep in range(N_OOF_REPEATS):
        # ★ shuffle + rep별 random_state로 서로 다른 분할 생성 (동일 분할 반복 방지)
        gkf = GroupKFold(n_splits=N_OOF_FOLDS, shuffle=True,
                         random_state=random_seed + rep)
        oof_shap_chem = np.full(len(X), np.nan)
        oof_shap_toc = np.full(len(X), np.nan)
        rep_folds = []
        try:
            for train_idx, test_idx in gkf.split(X, y, groups):
                # 이 반복의 각 fold test 인덱스 집합을 수집
                rep_folds.append(tuple(sorted(test_idx.tolist())))
                X_tr, y_tr = X[train_idx], y[train_idx]
                X_te = X[test_idx]
                dtr = xgb.DMatrix(X_tr, label=y_tr,
                                  feature_names=["Conc_log", "TOC_pct"])
                dte = xgb.DMatrix(X_te, feature_names=["Conc_log", "TOC_pct"])
                m = xgb.train(
                    {"objective": "reg:squarederror", "eta": 0.1,
                     "max_depth": 3, "seed": random_seed + rep},
                    dtr, num_boost_round=100,
                )
                contrib = m.predict(dte, pred_contribs=True)
                oof_shap_chem[test_idx] = contrib[:, 0]
                oof_shap_toc[test_idx] = contrib[:, 1]
        except Exception:
            # GroupKFold 실패 시 KFold 대체 금지 → 이 반복은 실패 처리
            continue

        # 반복별 5개 fold를 하나의 signature로 묶어 저장 (반복 분할 단위 고유성)
        if rep_folds:
            rep_signature = tuple(sorted(rep_folds))
            fold_split_signatures.add(rep_signature)

        valid = ~np.isnan(oof_shap_chem)
        if valid.sum() == 0:
            continue

        oof_imp_chem = np.mean(np.abs(oof_shap_chem[valid]))
        oof_imp_toc = np.mean(np.abs(oof_shap_toc[valid]))
        oof_cor_chem = stats.spearmanr(X[valid, 0], oof_shap_chem[valid]).correlation
        if np.isnan(oof_cor_chem):
            oof_cor_chem = 0.0

        oof_imp_chem_list.append(oof_imp_chem)
        oof_imp_toc_list.append(oof_imp_toc)
        oof_cor_chem_list.append(oof_cor_chem)

        # 반복별 조건 적용 (기존 조건과 동일)
        cond_direction = oof_cor_chem <= CHEM_DIRECTION_THRESHOLD
        cond_toc = oof_imp_toc <= (oof_imp_chem * TOC_IMPORTANCE_RATIO)
        if cond_direction and cond_toc:
            success_count += 1

    oof_imp_chem_mean = float(np.mean(oof_imp_chem_list)) if oof_imp_chem_list else np.nan
    oof_imp_toc_mean = float(np.mean(oof_imp_toc_list)) if oof_imp_toc_list else np.nan
    oof_cor_chem_mean = float(np.mean(oof_cor_chem_list)) if oof_cor_chem_list else np.nan
    n_unique_fold_splits = len(fold_split_signatures)

    return (success_count, oof_imp_chem_mean, oof_imp_toc_mean, oof_cor_chem_mean,
            n_unique_fold_splits)


def step2_species_selection(substance_dfs, source_name=""):
    print(f"\n{'='*70}")
    print(f"[Step 2] XGBoost OOF SHAP 종 선별 — {source_name}")
    print(f"  (종별 5-fold × {N_OOF_REPEATS}회 GroupKFold OOF SHAP, "
          f"{OOF_MIN_SUCCESS}/{N_OOF_REPEATS}회 충족 시 OOF-supported)")
    print(f"{'='*70}")

    cfg = load_config()
    random_seed = int(cfg.get("random_seed", RANDOM_SEED))

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
                    "Chem_Direction": np.nan,
                    "OOF_Chem_Importance": np.nan, "OOF_TOC_Importance": np.nan,
                    "OOF_Chem_Direction": np.nan, "OOF_Success_Count": 0,
                    "OOF_Supported": False, "Status": "Excluded (Insufficient)"})
                continue
            X = df_sp[["Conc_log", "TOC_pct"]].values.astype(np.float64)
            y = df_sp["Mean_Survival"].values.astype(np.float64)
            if np.any(np.isnan(X)) or np.any(np.isinf(X)) or np.any(np.isnan(y)):
                eval_list.append({"Substance": substance, "Species": sp,
                    "Total_N": total_n, "Toxic_N": toxic_n,
                    "Chem_Importance": np.nan, "TOC_Importance": np.nan,
                    "Chem_Direction": np.nan,
                    "OOF_Chem_Importance": np.nan, "OOF_TOC_Importance": np.nan,
                    "OOF_Chem_Direction": np.nan, "OOF_Success_Count": 0,
                    "OOF_Supported": False, "Status": "Excluded (Data Quality)"})
                continue

            # ---- in-sample SHAP (비교용) ----
            try:
                dtrain = xgb.DMatrix(X, label=y, feature_names=["Conc_log", "TOC_pct"])
                model = xgb.train({"objective": "reg:squarederror", "eta": 0.1,
                    "max_depth": 3, "seed": random_seed}, dtrain, num_boost_round=100)
                pred = model.predict(dtrain, pred_contribs=True)
                imp_chem = np.mean(np.abs(pred[:, 0]))
                imp_toc = np.mean(np.abs(pred[:, 1]))
                cor_chem = stats.spearmanr(X[:, 0], pred[:, 0]).correlation
                if np.isnan(cor_chem): cor_chem = 0.0
            except Exception:
                imp_chem = imp_toc = cor_chem = np.nan

            # ---- OOF SHAP (5-fold × 10회 GroupKFold) ----
            # ★ 행별 우선순위 결합으로 group_id 생성 (station_key → StudyID+Station →
            #   sample_key → SampleID → 좌표 → record_id).
            #   NOAA의 Station은 고유 정점이 아니므로 좌표로 그룹핑한다.
            groups = _build_group_ids(df_sp)
            n_groups = len(np.unique(groups))

            if n_groups < N_OOF_FOLDS:
                # GroupKFold 불가 → OOF 결측 처리 (KFold 대체 금지)
                eval_list.append({"Substance": substance, "Species": sp,
                    "Total_N": total_n, "Toxic_N": toxic_n,
                    "Chem_Importance": imp_chem, "TOC_Importance": imp_toc,
                    "Chem_Direction": cor_chem,
                    "OOF_Chem_Importance": np.nan, "OOF_TOC_Importance": np.nan,
                    "OOF_Chem_Direction": np.nan, "OOF_Success_Count": 0,
                    "OOF_Supported": False,
                    "Status": "Excluded (OOF Infeasible: groups < folds)"})
                continue

            success_count, oof_imp_chem, oof_imp_toc, oof_cor_chem, n_unique_splits = \
                _oof_shap_for_species(X, y, groups, random_seed)

            oof_supported = success_count >= OOF_MIN_SUCCESS

            # 최종 Status는 OOF 기준으로 결정
            if oof_supported:
                status = "Selected (Concentration-response-supported)"
            elif np.isnan(oof_cor_chem):
                status = "Excluded (OOF Infeasible)"
            elif oof_cor_chem > CHEM_DIRECTION_THRESHOLD:
                status = "Excluded (Non-toxic Trend)"
            elif oof_imp_toc > (oof_imp_chem * TOC_IMPORTANCE_RATIO):
                status = "Excluded (TOC Dominated)"
            else:
                status = "Excluded (OOF Unstable)"

            eval_list.append({"Substance": substance, "Species": sp,
                "Total_N": total_n, "Toxic_N": toxic_n,
                "Chem_Importance": imp_chem, "TOC_Importance": imp_toc,
                "Chem_Direction": cor_chem,
                "OOF_Chem_Importance": oof_imp_chem, "OOF_TOC_Importance": oof_imp_toc,
                "OOF_Chem_Direction": oof_cor_chem, "OOF_Success_Count": success_count,
                "N_Unique_Fold_Splits": n_unique_splits,
                "OOF_Supported": oof_supported, "Status": status})

    df_eval = pd.DataFrame(eval_list)
    selected = df_eval[df_eval["Status"].str.contains("Selected", na=False)]
    print(f"선택된 종 (OOF-supported): {len(selected)}종")
    df_eval.to_csv(OUTPUT_DIR / f"Step2_Species_Eval_{source_name}.csv", index=False)
    return df_eval


# =============================================================================
# Step 2.5: DRC 품질 평가 → 종 선별 (DRC는 진단 전용, 선별=OOF-SHAP 합의)
# =============================================================================
def evaluate_drc_quality(conc_log, survival):
    """
    종×물질 조합의 DRC 품질 평가 (3-tier)

    현장 혼합오염 데이터에서 R² ≥ 0.10은 거의 불가능 → R²는 참고용
    대신 독성 인과관계 방향(slope < 0)과 저농도 가짜 독성 비율로 판정

    Tier 1 (DRC OK):    slope < 0 AND 저농도 독성비율 ≤ 0.50 → 그대로 사용
    Tier 2 (DRC Marginal): slope < 0 AND 저농도 독성비율 > 0.50 → 사용 (가짜 독성 DROP 없음)
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
        result["DRC_OK"] = True  # 가짜 독성 DROP 없이 사용
        result["N_Low_Tox_Drop"] = 0
        result["Reason"] = f"slope OK but lowTox={low_tox_ratio:.2f} > 0.50 (kept, no fake-toxicity drop)"
    else:
        result["DRC_Tier"] = "Tier1_OK"
        result["DRC_OK"] = True
        result["Reason"] = "DRC OK (slope < 0, lowTox ≤ 0.50)"

    return result


def step2_5_drc_evaluation(substance_dfs, df_eval, source_name="",
                           anchor_species=None):
    """
    DRC 품질 평가 → 종 선별 (DRC는 진단 전용)

    ★ v5.3 설계 (사용자 확정):
    - DRC 불량 종도 Primary에 유지 (DRC는 보조 진단 지표로만 기록)
    - 종 선별은 OOF-SHAP 합의 기준이며, 선별 종은 selected_species로 보고한다
      (DRC 통과 여부를 선별 명칭에 혼입하지 않는다).
    - DRC OK 종의 저농도 가짜 독성 → DROP 없음 (Primary는 survival 기준 EDS 전부 포함)

    anchor_species: dict {substance: [species, ...]} — Sensitivity 분석에서만
    표준 시험종을 anchor로 강제 추가할 때 사용. Primary에서는 None (강제 추가 없음).
    """
    print(f"\n{'='*70}")
    print(f"[Step 2.5] DRC 품질 평가 → 종 선별 (DRC 진단 전용) — {source_name}")
    print(f"{'='*70}")

    # Step 2 선별종 추출
    selected_species_dict = {}
    for sub in TARGET_SUBSTANCES:
        sel = df_eval[(df_eval["Substance"] == sub) &
                       df_eval["Status"].str.contains("Selected", na=False)]
        spp = sel["Species"].unique().tolist()
        if spp:
            selected_species_dict[sub] = spp

    # Sensitivity 전용: anchor species 강제 추가 (Primary에서는 실행 안 됨)
    if anchor_species:
        for sub, anchor_spp in anchor_species.items():
            if sub in selected_species_dict:
                selected_species_dict[sub] = list(
                    set(selected_species_dict[sub] + anchor_spp))
            else:
                selected_species_dict[sub] = list(anchor_spp)

    drc_results = []
    selected_species = {}  # ★ v5.3: DRC는 진단 전용 — 불량 종도 Primary에 유지된다.
    #   'selected' = OOF-SHAP 선별 종 (DRC_OK 여부와 무관하게 Primary 유지).
    cleaned_dfs = {}     # DRC 불량 종 포함 전체 데이터 (가짜 독성 DROP 없음)

    for substance in TARGET_SUBSTANCES:
        if substance not in substance_dfs:
            continue
        accepted = selected_species_dict.get(substance)
        if not accepted:
            continue

        # anchor species 집합 (Sensitivity 전용): DRC 불량이어도 포함
        anchor_set = set(anchor_species.get(substance, [])) if anchor_species else set()

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
                # ★ 가짜 독성 DROP 중단: Primary에서는 survival 기준 EDS를 모두 포함.
                #   저농도 가짜 독성 제거는 순환성·선택편향 우려 → 민감도 분석으로 이동.
                #   N_Low_Tox_Drop은 진단 정보로만 기록(삭제하지 않음).
            # DRC 불량 종: 해당 종 데이터 전체 DROP
            #   단, anchor species(Sensitivity 전용)는 DRC 불량이어도 유지
            elif sp in anchor_set:
                ok_species_for_substance.append(sp)
                print(f"  ⚓ {sp[:35]:35s} | anchor species — DRC 불량이지만 유지 (Sensitivity)")
            else:
                # ★ v5.2 fix: DRC 불량 종이라도 Primary에서는 삭제하지 않음.
                # DRC는 보조 진단(진단·Sensitivity)으로만 사용하고,
                # Primary 종 선별은 OOF-SHAP 기준만 사용.
                # (기존: df_pooled = df_pooled[~sp_mask].copy() — DRC가 Primary 종 선별에 관여)
                ok_species_for_substance.append(sp)
                print(f"  ◇ {sp[:35]:35s} | DRC 불량 — Primary 유지 (DRC는 진단/Sensitivity 전용)")

        if ok_species_for_substance:
            selected_species[substance] = ok_species_for_substance
        cleaned_dfs[substance] = df_pooled

    drc_df = pd.DataFrame(drc_results)
    if drc_df.empty:
        drc_df = pd.DataFrame(columns=["Substance", "Species", "N_total", "N_toxic",
                                       "DRC_OK", "R_squared", "Slope",
                                       "Low_Conc_Tox_Ratio", "Q1_Threshold",
                                       "N_Low_Tox_Drop", "Reason"])
    drc_df.to_csv(OUTPUT_DIR / f"Step2_5_DRC_Quality_{source_name}.csv", index=False)

    ok_count = int((drc_df["DRC_OK"] == True).sum()) if not drc_df.empty else 0
    bad_count = int((drc_df["DRC_OK"] == False).sum()) if not drc_df.empty else 0
    total_species = len(drc_df)
    print(f"\n[Step 2.5 요약]")
    print(f"  DRC OK: {ok_count}종 / 불량: {bad_count}종 / 총 {total_species}종 조합")
    print(f"  종 선별: OOF-SHAP 합의 기준 (DRC는 진단 전용 — 불량 종도 Primary 유지)")
    print(f"  가짜 독성 DROP: 중단됨 (Primary는 survival 기준 EDS 전부 포함, 민감도 분석으로 이동)")
    print(f"  물질별 선별 종: {selected_species}")

    return cleaned_dfs, drc_df, selected_species


# =============================================================================
# Step 2.7: 다물질 SHAP 타 독성 기여도 평가 ★신규
# =============================================================================
def step2_7_confounder_filtering(cleaned_dfs, selected_species, source_name=""):
    """[DEPRECATED] 구버전 confounder 필터 — 부호 오류(절대값 비교)로 재사용 금지.

    이 함수는 |SHAP_mPELQ| > |SHAP_target| 절대값 비교로 confounder를 판정해
    부호(방향) 정보를 무시하는 오류가 있었다. v4에서 sign-aware 방식으로
    재설계되어 step2_7_confounder_filtering_v5로 대체되었다.
    재사용을 막기 위해 호출 시 예외를 발생시킨다.
    """
    raise NotImplementedError(
        "step2_7_confounder_filtering은 부호 오류로 폐기되었습니다. "
        "step2_7_confounder_filtering_v5를 사용하세요."
    )


# =============================================================================
# Step 2.7-v5.3: confounder 필터 재설계 (진단 + Primary/Sensitivity 분리)
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


def step2_7_confounder_filtering_v5(cleaned_dfs, selected_species, source_name="",
                                     n_folds=None, n_seeds=None, stability_threshold=None,
                                     use_group_kfold=False):
    """
    v5.3 Primary 필터 — OOF-SHAP 기반 비대상 화학물질 기여 판정 + DRC 보조진단.

    분석 구조:
      - Primary:       OOF-SHAP sign-aware joint 카운터 (TreeSHAP ∩ Interventional 8/10 합의)
                      R_NT = (pred_without_non_target − pred_full) / (80 − pred_full) ≥ 0.5
                      fold gate: observed_survival<80 AND pred_full<80 AND shap_target≥0
                                  AND negative_non_target<0
      - Sensitivity 1: Strict (R_NT ≥ 1.0, 완전 회복 — 구 c5)
      - Sensitivity 2: Exploratory (R_NT ≥ 0.25)
      - Sensitivity 3: ChemScreen (chemistry screen only, ML 기록 필터 미적용)
      - Sensitivity 4: Unfiltered (기본 QA·종 선별만 적용, 비교 기준선)
      - Ablation:      금속-only 제거 모델의 OOF prediction difference (보조 진단)

    OOF-SHAP 판정:
      - 각 fold: training fold로 다중 feature XGBoost 적합 → held-out fold SHAP 계산
      - TreeSHAP + Interventional SHAP 양쪽에서 8/10회 이상 만족 시 Primary 제외
      - 결측 PAH/POPs의 SHAP은 observed_mask로 non_target 판정에서 제외
      - DRC(C80 gate, 잔차)는 보조 진단 (Primary 제거 아님)

    재현성: n_folds/n_seeds/stability_threshold가 None이면 pipeline_v5_config.json에서 읽는다.
      - Strict/Exploratory/ChemScreen/Unfiltered 민감도를 Supplementary에 함께 제시.
      - 제거율이 max_removal_rate 초과 시 applicability_pass=False 플래그 (all-or-none 폐지).

    use_group_kfold=True: record-level KFold 대신 GroupKFold(그룹=station/sample cluster)를
      사용해 동일 station의 train/test 중복(군집 누출)을 방지. 각 seed마다 그룹을
      shuffle하여 서로 다른 분할을 생성한다(repeated group split).
    """
    # ---- 재현성 config 로딩 (None이면 config 값 사용) ----
    cfg = load_config()
    if n_folds is None:
        n_folds = int(cfg.get("n_folds", 5))
    if n_seeds is None:
        n_seeds = int(cfg.get("n_repeats", 10))
    if stability_threshold is None:
        stability_threshold = float(cfg.get("stability_threshold", 0.80))
    dominance_ratio = float(cfg.get("dominance_ratio", 1.0))
    residual_cutoff = float(cfg.get("residual_cutoff", 0.05))
    residual_cutoffs = cfg.get("residual_cutoffs", [0.025, 0.05, 0.10])
    max_removal_rate = float(cfg.get("max_removal_rate", 0.15))
    # ★ v5.3: R_NT tier (사전 확정 — 결과 건수를 보고 정한 값이 아님)
    #   Strict=1.0(완전 회복, 구 c5) / Primary=0.5(과반 기여) / Exploratory=0.25(탐색적)
    ratio_tier_strict = float(cfg.get("ratio_tier_strict", 1.0))
    ratio_tier_primary = float(cfg.get("ratio_tier_primary", 0.5))
    ratio_tier_exploratory = float(cfg.get("ratio_tier_exploratory", 0.25))
    random_seed = int(cfg.get("random_seed", 42))
    # repeat_seeds 목록을 config에서 읽음 (없으면 range(n_seeds)로 대체)
    repeat_seeds = cfg.get("repeat_seeds", None)
    if repeat_seeds is None or len(repeat_seeds) != n_seeds:
        repeat_seeds = list(range(n_seeds))
    else:
        repeat_seeds = [int(s) for s in repeat_seeds]
    print(f"\n{'='*70}")
    print(f"[Step 2.7-v5.3] SHAP Primary + DRC 보조진단 — {source_name}")
    print(f"{'='*70}")

    diag_rows = []
    tel_pel_rows = []
    primary_filtered_dfs = {}  # Primary로 선별된 DataFrame (Step 3 입력)

    for substance in TARGET_SUBSTANCES:
        if substance not in cleaned_dfs or substance not in selected_species:
            continue
        ok_spp = selected_species[substance]
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

        # ★ v5: 확장 feature — [target_conc, other_POP1, other_POP2, PAHs, mPELQ_Metals, TOC_pct, species_onehot...]
        # PAHs 결측은 NaN으로 두고 XGBoost가 자체 처리 (사용자 확정 option b).
        # 다른 POPs 농도를 추가 feature로 사용하여 혼합독성 기여도를 SHAP이 동시 평가.
        # species one-hot: 시험종별 생존율 차이를 모델이 학습.
        all_pops = ["DDTs", "CHLs", "PCBs", "PAHs"]
        other_pops = [p for p in all_pops if p != substance]
        pop_cols = {p: f"Sum_{p}_OC_log" for p in all_pops}

        # species one-hot
        species_col = "Latin_Name_Species"
        all_species = sorted(df_pooled[species_col].dropna().unique().tolist())
        # one-hot 생성 (df_pooled에 추가)
        for sp in all_species:
            df_pooled[f"_sp_{sp}"] = (df_pooled[species_col] == sp).astype(float)

        # feature 컬럼 순서: [target, otherPOP1, otherPOP2, PAHs, mPELQ_Metals, TOC_pct, sp_onehot...]
        # ★ 주의: target_col은 이미 df_pooled에 존재. other POPs/PAHs 컬럼이 없으면 NaN.
        feat_base_cols = [target_col] + [pop_cols[p] for p in other_pops] + ["mPELQ_Metals", "TOC_pct"]
        sp_cols = [f"_sp_{sp}" for sp in all_species]
        all_feat_cols = feat_base_cols + sp_cols

        # feature_names: 모델용 이름
        feat_names_base = [f"{substance}_conc"] + [f"{p}_conc" for p in other_pops] + ["mPELQ_Metals", "TOC_pct"]
        feature_names = feat_names_base + [f"sp_{sp}" for sp in all_species]

        # X 구성 — PAHs 결측은 NaN으로 유지 (XGBoost 자체 처리)
        for col in all_feat_cols:
            if col not in df_pooled.columns:
                df_pooled[col] = np.nan

        X = df_pooled[all_feat_cols].values.astype(np.float64)
        y = df_pooled["Mean_Survival"].values.astype(np.float64)

        # ★ v5: valid_mask는 target_col + mPELQ_Metals + TOC_pct만 검사 (other POPs/PAHs는 NaN 허용)
        core_mask_cols = [target_col, "mPELQ_Metals", "TOC_pct"]
        core_X = df_pooled[core_mask_cols].values.astype(np.float64)
        valid_mask = ~np.any(np.isnan(core_X) | np.isinf(core_X), axis=1)
        # species one-hot도 결측이면 안 됨 (하지만 이미 float으로 변환됨)
        sp_X = df_pooled[sp_cols].values.astype(np.float64) if sp_cols else np.ones((len(df_pooled), 0))
        valid_mask &= ~np.any(np.isnan(sp_X) | np.isinf(sp_X), axis=1) if sp_cols else True

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

        # GroupKFold용 group_id: 행별 우선순위 결합 (station_key → StudyID+Station →
        # sample_key → SampleID → 좌표 → record_id).
        # NOAA의 Station은 고유 정점이 아니므로 좌표로 그룹핑한다.
        groups = _build_group_ids(df_valid)
        group_col = "group_id"  # 진단 기록용 (실제 컬럼은 아님)

        # ---- Primary: 화학적 mPELQ screen만 (Step 1에서 이미 적용) ----
        # df_valid는 이미 mPELQ <= 0.5가 적용된 상태. 추가 SHAP 제거 없음.
        primary_telpel = _compute_tel_pel_from_df(df_valid, target_col)

        # ---- v5: feature index 정의 ----
        # feature_names = [target, otherPOP1, otherPOP2, PAHs, mPELQ_Metals, TOC_pct, sp...]
        n_base_feats = len(feat_names_base)  # 6 (target + 3 other POPs + metal + TOC)
        n_sp_feats = len(sp_cols)
        idx_target = 0
        idx_other_pops = [feat_names_base.index(f"{p}_conc") for p in other_pops]
        idx_metal = feat_names_base.index("mPELQ_Metals")
        idx_toc = feat_names_base.index("TOC_pct")
        # ★ v5.1: PAHs 인덱스 (결측 마스킹용)
        idx_pah = feat_names_base.index("PAHs_conc") if "PAHs_conc" in feat_names_base else None
        # ★ v5.2 fix: PAH 중복 제거 — other_pops에 PAHs가 이미 포함되어 있으므로
        #   idx_pah를 별도로 append하면 SHAP이 2회 합산되는 버그 발생.
        #   dict.fromkeys로 순서를 유지하며 중복 제거.
        non_target_chem_idx = list(dict.fromkeys(
            idx_other_pops + [idx_metal]
        ))  # other POPs (PAHs 포함) + mPELQ_Metals, 중복 없음

        # ---- 반복 CV SHAP (OOF) ----
        # ★ v5: SHAP Primary 필터 (sign-aware joint 카운터)
        # target_toxic = shap_target < 0 (대상물질이 독성 방향으로 기여)
        # non_target_toxic = 다른 feature(metal, other POPs)가 독성 방향으로 기여
        # fold_target_unsupported = (survival<80) & (~target_toxic) & non_target_toxic
        #   → 대상물질이 기여하지 않는데 다른 물질이 기여 → 노이즈 후보
        # joint 카운트: 동일 fold에서 target_unsupported 조건을 동시 만족해야 1회 카운트
        n_assignments = np.zeros(n_total, dtype=int)          # OOF 예측 횟수
        # TreeSHAP joint 카운터
        target_unsupported_tree = np.zeros(n_total, dtype=int)  # fold_target_unsupported (tree)
        target_toxic_tree = np.zeros(n_total, dtype=int)        # shap_target < 0 (tree)
        non_target_toxic_tree = np.zeros(n_total, dtype=int)     # 다른 물질 독성 기여 (tree)
        # Interventional joint 카운터
        target_unsupported_inter = np.zeros(n_total, dtype=int)
        target_toxic_inter = np.zeros(n_total, dtype=int)
        non_target_toxic_inter = np.zeros(n_total, dtype=int)
        # dominance (sensitivity 분석용)
        non_target_dom_tree = np.zeros(n_total, dtype=int)     # |non_target|>|target| (tree)
        non_target_dom_inter = np.zeros(n_total, dtype=int)
        n_inter_assignments = np.zeros(n_total, dtype=int)     # Interventional OOF 성공 횟수
        # ★ v5.3: R_NT(기여도 비율) joint 카운터 — Primary=≥0.5, Exploratory=≥0.25
        target_unsup_ratio_05_tree = np.zeros(n_total, dtype=int)
        target_unsup_ratio_05_inter = np.zeros(n_total, dtype=int)
        target_unsup_ratio_025_tree = np.zeros(n_total, dtype=int)
        target_unsup_ratio_025_inter = np.zeros(n_total, dtype=int)
        # ablation: 금속 포함/제외 OOF prediction difference 누적
        pred_with_metal = np.zeros(n_total, dtype=float)
        pred_without_metal = np.zeros(n_total, dtype=float)
        # ★ v5.2: sensitivity ratio (contribution_ratio) 계산용 누적
        #   delta_nt = -ntt_sum (비대상물질 음의 SHAP 합의 절대값 = 비대상 기여)
        #   toxicity_deficit = 80 - pred_full (대상물질만으로 채우지 못한 독성 결손)
        #   contribution_ratio = delta_nt / toxicity_deficit
        pred_full_sum = np.zeros(n_total, dtype=float)     # OOF pred_full (TreeSHAP) 평균용 누적
        ntt_sum_accum = np.zeros(n_total, dtype=float)     # OOF ntt_sum_tree 평균용 누적
        # OOF DRC 잔차 기반 — 보조 진단 (Primary 아님, v5에서 격하)
        residual_sum = np.zeros(n_total, dtype=float)
        residual_anomaly_counts = {c: np.zeros(n_total, dtype=int) for c in residual_cutoffs}
        n_c80_valid = np.zeros(n_total, dtype=int)
        c80_sum = np.zeros(n_total, dtype=float)
        n_blocked_high_conc = np.zeros(n_total, dtype=int)
        n_below_c80 = np.zeros(n_total, dtype=int)
        n_blocked_high_conc = np.zeros(n_total, dtype=int) # 농도≥C80 으로 gate에 차단된 이상 후보 횟수
        n_below_c80 = np.zeros(n_total, dtype=int)         # fold별 below_c80=True 횟수 (fold별 판정 검증용)

        # fold signature 저장: 각 seed의 fold 배정을 기록해 10개 반복의 분할 차이 검증
        fold_signatures = []  # [{seed, fold_id, test_record_ids}]

        # 시료별 OOF 원자료 수집 (long format)
        raw_rows = []
        bg_sizes = []  # fold별 interventional background 실제 행 수
        group_overlap_count = 0  # GroupKFold train/test 그룹 교집합 검사 (0이어야 정상)
        n_inter_attempts = 0  # Interventional SHAP 시도 fold 수
        n_inter_success = 0   # Interventional SHAP 성공 fold 수

        for seed in repeat_seeds:
            if use_group_kfold:
                # GroupKFold: 동일 station/sample cluster가 train/test에 중복되지 않도록 분할.
                # sklearn 1.9.0의 GroupKFold는 shuffle=True + random_state를 지원하므로,
                # 각 seed마다 서로 다른 분할을 생성한다(repeated group split).
                if len(np.unique(groups)) < n_folds:
                    raise ValueError(
                        f"Number of unique groups ({len(np.unique(groups))}) "
                        f"is smaller than n_folds ({n_folds})")
                gkf = GroupKFold(n_splits=n_folds, shuffle=True,
                                 random_state=random_seed + seed)
                splits = list(gkf.split(X_valid, groups=groups))
                # 교집합 검사: 각 fold에서 train/test 그룹이 겹치면 카운트
                for train_idx, test_idx in splits:
                    tr_g = set(groups[train_idx])
                    te_g = set(groups[test_idx])
                    group_overlap_count += len(tr_g & te_g)
            else:
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
                splits = list(kf.split(X_valid))
            for fold_id, (train_idx, test_idx) in enumerate(splits):
                X_tr, X_te = X_valid[train_idx], X_valid[test_idx]
                y_tr, y_te = y_valid[train_idx], y_valid[test_idx]
                bg_sizes.append(int(X_tr.shape[0]))

                # fold signature 기록 (분할 차이 검증용)
                fold_signatures.append({
                    "seed": int(seed),
                    "fold_id": int(fold_id),
                    "test_record_ids": tuple(sorted(
                        df_valid.iloc[test_idx]["record_id"].tolist())),
                })

                # ★ v5: 확장 feature 모델 (monotone_constraints 적용)
                # 화학물질 농도(0~3) = 단조감소(-1), mPELQ_Metals = -1, TOC = 0, species = 0
                n_feat = len(feature_names)
                # ★ v5.2 fix: TOC에 -1(단조감소) 잘못 적용 수정
                # feat_names_base = [target, otherPOP1, otherPOP2, PAHs, metal, TOC]
                # target·otherPOPs·PAHs·metal은 -1(농도↑→생존율↓), TOC는 0(제약없음)
                # species one-hot도 0
                mono_list = [-1] * (n_base_feats - 1) + [0] + [0] * n_sp_feats
                mono_str = "(" + ",".join(str(x) for x in mono_list) + ")"

                dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
                model3 = xgb.train({"objective": "reg:squarederror", "eta": 0.1,
                    "max_depth": 4, "seed": random_seed,
                    "monotone_constraints": mono_str}, dtr, num_boost_round=100)

                # TreeSHAP (path-dependent, 기본) — 모든 feature의 SHAP 값 추출
                dte = xgb.DMatrix(X_te, feature_names=feature_names)
                pred_contribs_raw = model3.predict(dte, pred_contribs=True)
                # pred_contribs: (n_test, n_feat + 1) — 마지막 열은 base value
                shap_all_tree = pred_contribs_raw[:, :n_feat]  # (n_test, n_feat)
                base_tree = pred_contribs_raw[:, n_feat]       # ★ v5.2 fix: base value를 원본에서 직접 추출
                shap_target_tree = shap_all_tree[:, idx_target]
                shap_metal_tree = shap_all_tree[:, idx_metal]
                shap_other_pops_tree = {p: shap_all_tree[:, idx_other_pops[i]] for i, p in enumerate(other_pops)}

                # ★ v5.1: 관측 마스크 — 결측 feature의 SHAP은 non_target 판정에서 제외
                # X_te가 NaN이면 해당 feature는 미측정 → SHAP 값은 "미측정 패턴"의 영향이지 농도 영향 아님
                observed_mask = ~np.isnan(X_te)  # (n_test, n_feat)

                # Interventional SHAP (background = 해당 fold training data)
                # 실패 시 TreeSHAP으로 대체하지 않고 결측 처리한다(성공률 별도 기록).
                n_inter_attempts += 1
                inter_ok = False
                try:
                    explainer = shap.TreeExplainer(
                        model3, X_tr, feature_perturbation="interventional")
                    shap_inter = explainer.shap_values(X_te)
                    shap_all_inter = shap_inter[:, :n_feat]
                    base_inter_vals = np.full(len(test_idx), float(explainer.expected_value))  # ★ v5.2 fix: expected_value 직접 추출
                    shap_target_inter = shap_all_inter[:, idx_target]
                    shap_metal_inter = shap_all_inter[:, idx_metal]
                    shap_other_pops_inter = {p: shap_all_inter[:, idx_other_pops[i]] for i, p in enumerate(other_pops)}
                    inter_ok = True
                    n_inter_success += 1
                except Exception as e:
                    print(f"  {substance} seed={seed} fold={fold_id}: "
                          f"Interventional SHAP 실패 ({e}) → 결측 처리")
                    shap_all_inter = np.full((len(test_idx), n_feat + 1), np.nan)
                    base_inter_vals = np.zeros(len(test_idx))  # ★ v5.2 fix: fallback base
                    shap_target_inter = np.full(len(test_idx), np.nan)
                    shap_metal_inter = np.full(len(test_idx), np.nan)
                    shap_other_pops_inter = {p: np.full(len(test_idx), np.nan) for p in other_pops}

                # ablation: 금속 제외 모델 (target + otherPOPs + PAHs + TOC, metal만 제외)
                # ★ v5.2 fix: 금속 ablation은 전체 feature에서 metal만 제외하고 재학습.
                #   기존: model2 = [target, TOC] 2-feature → pred_diff에 otherPOPs+species 효과 포함 (버그)
                #   수정: model4 = 전체 feature - metal → pred_diff = pred3 - pred4 (순수 금속 효과)
                # DRC 보조진단용 2-feature 모델: [target_conc, TOC_pct] (기존 유지, DRC 전용)
                X_tr2 = X_tr[:, [idx_target, idx_toc]]
                X_te2 = X_te[:, [idx_target, idx_toc]]
                dtr2 = xgb.DMatrix(X_tr2, label=y_tr,
                                   feature_names=[feature_names[idx_target], feature_names[idx_toc]])
                model2 = xgb.train({"objective": "reg:squarederror", "eta": 0.1,
                    "max_depth": 4, "seed": random_seed,
                    "monotone_constraints": "(-1,0)"}, dtr2, num_boost_round=100)
                dte2 = xgb.DMatrix(X_te2, feature_names=[feature_names[idx_target], feature_names[idx_toc]])
                pred2 = model2.predict(dte2)

                # ★ v5.2: 금속-only ablation 모델 (metal feature만 제외, 나머지 전부 포함)
                all_idx = list(range(n_feat))
                no_metal_idx = [i for i in all_idx if i != idx_metal]
                feat_names_no_metal = [feature_names[i] for i in no_metal_idx]
                X_tr4 = X_tr[:, no_metal_idx]
                X_te4 = X_te[:, no_metal_idx]
                mono_no_metal = [mono_list[i] for i in no_metal_idx]
                mono_str4 = "(" + ",".join(str(x) for x in mono_no_metal) + ")"
                dtr4 = xgb.DMatrix(X_tr4, label=y_tr, feature_names=feat_names_no_metal)
                model4 = xgb.train({"objective": "reg:squarederror", "eta": 0.1,
                    "max_depth": 4, "seed": random_seed,
                    "monotone_constraints": mono_str4}, dtr4, num_boost_round=100)
                dte4 = xgb.DMatrix(X_te4, feature_names=feat_names_no_metal)
                pred4 = model4.predict(dte4)
                pred3 = model3.predict(dte)

                # ---- OOF DRC 잔차 기반 이상자료 판정 ----
                # DRC = 금속 제외 2-feature 모델(model2)의 농도-생존율 관계.
                # cutoff는 in-sample 잔차가 아니라 inner-OOF 잔차로 산출한다.
                # (in-sample 잔차는 과적합으로 과소추정되어 test가 과도하게 이상치 판정됨)
                # outer training fold 안에서 inner GroupKFold OOF 예측 → inner-OOF 잔차
                # → 그 잔차 분포의 하위 cutoff 분위수를 임계값으로,
                # outer test(OOF) 잔차(y_te - pred2)가 그 임계값보다 낮으면(더 음수) 이상 판정.
                resid_te = y_te - pred2                  # outer test(OOF) 잔차

                # inner GroupKFold OOF 예측 (training fold 내부)
                inner_groups = groups[train_idx]
                inner_pred = np.full(len(train_idx), np.nan, dtype=float)
                if len(np.unique(inner_groups)) >= 2:
                    try:
                        inner_gkf = GroupKFold(n_splits=min(n_folds, len(np.unique(inner_groups))))
                        for itr_idx, ite_idx in inner_gkf.split(X_tr2, groups=inner_groups):
                            dtr2_in = xgb.DMatrix(X_tr2[itr_idx], label=y_tr[itr_idx],
                                                  feature_names=[feature_names[idx_target], feature_names[idx_toc]])
                            m2_in = xgb.train({"objective": "reg:squarederror", "eta": 0.1,
                                "max_depth": 4, "seed": random_seed,
                                "monotone_constraints": "(-1,0)"}, dtr2_in, num_boost_round=100)
                            dte2_in = xgb.DMatrix(X_tr2[ite_idx],
                                                  feature_names=[feature_names[idx_target], feature_names[idx_toc]])
                            inner_pred[ite_idx] = m2_in.predict(dte2_in)
                    except Exception as e:
                        # GroupKFold 실패 시 KFold로 대체하지 않는다.
                        # (KFold 대체 시 동일 station이 train/test에 섞여 leakage 발생 가능)
                        # → 해당 fold의 inner-OOF 잔차를 결측 처리하여 이상 판정에서 제외.
                        print(f"  {substance} seed={seed} fold={fold_id}: "
                              f"inner GroupKFold 실패 ({e}) → 결측 처리 (KFold 대체 금지)")
                        inner_pred = np.full(len(train_idx), np.nan, dtype=float)
                else:
                    # 그룹이 1개뿐이면 inner GroupKFold 불가 → 결측 처리 (KFold 대체 금지)
                    print(f"  {substance} seed={seed} fold={fold_id}: "
                          f"inner 그룹 1개뿐 → 결측 처리 (KFold 대체 금지)")
                    inner_pred = np.full(len(train_idx), np.nan, dtype=float)

                resid_inner_oof = y_tr - inner_pred       # inner-OOF 잔차
                # ---- 저농도 DRC gate (C80) — 표준화된 단일 C80 (TOC=1% 고정) ----
                # C80 = monotonic 2-feature DRC가 80% survival로 하강하는 농도 (로그 스케일).
                # 사용자 확정 (2026-08-27): DRC 예측은 2-feature [conc, TOC] (실제 TOC 사용),
                #   C80 산출은 TOC=1.0 고정으로 fold당 단일값.
                #   → 잔차 예측: 실제 TOC / 저농도 판정: 동일 C80_fold / Fig.2 X축과 일치.
                # 노이즈 제거 목적 = "DRC가 무독성(생존>=80)이라 예측하는 저농도 영역인데
                #   실제로는 독성(생존<80)을 보이는" 독성 기여도가 낮은 자료만 제외.
                # 고농도 강독성 자료는 진짜 독성 근거이므로 제외하지 않는다 → conc < C80 gate 보호.
                feature_names_2 = [feature_names[idx_target], feature_names[idx_toc]]
                conc_min_tr = float(X_tr2[:, 0].min())
                conc_max_tr = float(X_tr2[:, 0].max())
                conc_grid = np.linspace(conc_min_tr, conc_max_tr, 200)
                # 표준화된 단일 C80: TOC=1.0 고정 (시료별 TOC 미사용)
                X_grid_std = np.column_stack([conc_grid, np.ones(len(conc_grid))])
                pred_grid_std = model2.predict(
                    xgb.DMatrix(X_grid_std, feature_names=feature_names_2))
                cross_idx = np.where(pred_grid_std < 80)[0]
                if len(cross_idx) == 0 or cross_idx[0] == 0:
                    # crossing 없음 OR 저농도부터 이미 80% 미만(유효 no-effect 영역 없음)
                    # → 해당 fold에 대해 필터 미적용 (gate 통과 불가)
                    c80_fold = None
                else:
                    c80_fold = float(conc_grid[cross_idx[0]])

                for k, idx in enumerate(test_idx):
                    residual_sum[idx] += resid_te[k]
                    surv_k = float(y_te[k])           # 관측 생존율
                    conc_k = float(X_te2[k, 0])       # 관측 농도 (로그, 1% TOC 보정)

                    if c80_fold is not None:
                        n_c80_valid[idx] += 1
                        c80_sum[idx] += c80_fold
                        if conc_k < c80_fold:
                            n_below_c80[idx] += 1

                    for c in residual_cutoffs:
                        resid_cut = np.quantile(resid_inner_oof, c)
                        # 저농도 DRC gate 적용: 관측생존<80 AND 관측농도<C80_fold AND 잔차<cutoff
                        if (c80_fold is not None) and (surv_k < 80) and (conc_k < c80_fold) \
                                and (resid_te[k] < resid_cut):
                            residual_anomaly_counts[c][idx] += 1
                        elif (c80_fold is not None) and (surv_k < 80) and (conc_k >= c80_fold) \
                                and (resid_te[k] < resid_cut):
                            # 고농도(농도>=C80_fold) 강독성 자료 → gate에 차단되어 이상 판정에서 제외
                            # (진단 카운터: 기존 코드라면 노이즈로 오판됐을 후보)
                            n_blocked_high_conc[idx] += 1


                # ★ v5.1: SHAP Primary joint 카운터 누적 (sign-aware + 관측 마스킹 + pred 검증)
                # 핵심 변경:
                # 1. 결측 PAH SHAP은 non_target 판정에서 제외 (~np.isnan(X_te[:, idx]))
                # 2. non_target_toxic_sum = 관측된 비대상 화학물질 SHAP의 음수 합
                # 3. fold_target_unsupported 강화:
                #    (survival < 80) & (pred_full < 80) & (shap_target >= 0)
                #    & (non_target_toxic_sum < 0) & (pred_without_non_target >= 80)
                # → 비대상물질 기여가 실제 독성 예측을 결정했는지 확인
                surv_te = y_te  # (n_test,)
                # pred3 = model3 (전체 feature) OOF 예측
                pred_full_k = pred3  # (n_test,)
                # ★ v5.2 fix: base_tree는 이미 위에서 pred_contribs_raw[:, n_feat]로 추출됨

                for k, idx in enumerate(test_idx):
                    n_assignments[idx] += 1

                    # ---- TreeSHAP joint 판정 ----
                    tt = shap_target_tree[k] < 0  # target_toxic

                    # non_target_toxic_sum: 관측된 비대상 화학물질만 음수 SHAP 합산
                    ntt_sum_tree = 0.0
                    for j in non_target_chem_idx:
                        if observed_mask[k, j]:  # 결측이 아닐 때만
                            ntt_sum_tree += min(float(shap_all_tree[k, j]), 0.0)

                    # pred_without_non_target = pred_full - non_target_toxic_sum
                    pred_full_tree = base_tree[k] + float(shap_all_tree[k, :n_feat].sum())
                    pred_wo_nt_tree = pred_full_tree - ntt_sum_tree
                    # ★ v5.3: R_NT = delta_nt / deficit — tier별 fold 카운트
                    #   판정 gate (사용자 확정 6조건 중 fold 수준 4조건):
                    #     c1 관측 생존율 <80, c2 pred_full <80 (deficit>0로 내포),
                    #     c3 target SHAP ≥0, c4 관측 비대상 음의 SHAP 합 <0 (delta_nt>0로 내포)
                    #   → gate 통과 & R_NT ≥ tier 인 fold만 카운트
                    delta_nt_tree = -ntt_sum_tree
                    deficit_tree = 80.0 - pred_full_tree
                    gate_r_tree = ((surv_te[k] < 80) and (deficit_tree > 1e-6)
                                   and (shap_target_tree[k] >= 0)
                                   and (ntt_sum_tree < 0))
                    if gate_r_tree:
                        r_tree_k = delta_nt_tree / deficit_tree
                        if r_tree_k >= ratio_tier_primary:
                            target_unsup_ratio_05_tree[idx] += 1
                        if r_tree_k >= ratio_tier_exploratory:
                            target_unsup_ratio_025_tree[idx] += 1

                    # 강화된 fold_target_unsupported
                    fold_unsup_tree = (
                        (surv_te[k] < 80)
                        & (pred_full_tree < 80)
                        & (shap_target_tree[k] >= 0)
                        & (ntt_sum_tree < 0)
                        & (pred_wo_nt_tree >= 80)
                    )
                    if fold_unsup_tree:
                        target_unsupported_tree[idx] += 1
                    if tt:
                        target_toxic_tree[idx] += 1
                    if ntt_sum_tree < 0:
                        non_target_toxic_tree[idx] += 1
                    # dominance: 음수 부분만 비교 (sensitivity)
                    # ★ v5.2 fix: 기존 (not tt) & abs(non_target) > abs(target)는
                    #   target_toxic>=8 과 모순 → 항상 0이 됨.
                    #   수정: 동일 fold에서 target_negative = min(shap_target,0),
                    #   non_target_negative = sum(min(observed_shap, 0)),
                    #   mixed_dominant = target_negative < 0 & |non_target_negative| > |target_negative|
                    target_neg_tree = min(float(shap_target_tree[k]), 0.0)
                    non_target_neg_tree = 0.0
                    for j in non_target_chem_idx:
                        if observed_mask[k, j]:
                            non_target_neg_tree += min(float(shap_all_tree[k, j]), 0.0)
                    if (target_neg_tree < 0) and (abs(non_target_neg_tree) > abs(target_neg_tree)):
                        non_target_dom_tree[idx] += 1

                    # ---- Interventional joint 판정 (성공한 fold만) ----
                    tt_i = False
                    ntt_i_inter = False
                    fold_unsup_inter = False
                    ntt_sum_inter = 0.0
                    pred_full_inter = np.nan
                    pred_wo_nt_inter = np.nan
                    if inter_ok:
                        n_inter_assignments[idx] += 1
                        tt_i = shap_target_inter[k] < 0

                        # non_target_toxic_sum (Interventional)
                        ntt_sum_inter = 0.0
                        for j in non_target_chem_idx:
                            if observed_mask[k, j]:
                                ntt_sum_inter += min(float(shap_all_inter[k, j]), 0.0)

                        pred_full_inter = base_inter_vals[k] + float(shap_all_inter[k, :n_feat].sum())
                        pred_wo_nt_inter = pred_full_inter - ntt_sum_inter
                        # ★ v5.3: R_NT Interventional 카운터 (성공 fold만)
                        #   TreeSHAP과 동일 gate 적용: c1 관측<80, c3 target SHAP≥0
                        #   (c2=deficit>0, c4=delta_nt>0은 분모·분자 조건으로 내포)
                        delta_nt_inter = -ntt_sum_inter
                        deficit_inter = 80.0 - pred_full_inter
                        gate_r_inter = ((surv_te[k] < 80) and (deficit_inter > 1e-6)
                                        and (shap_target_inter[k] >= 0)
                                        and (ntt_sum_inter < 0))
                        if gate_r_inter:
                            r_inter_k = delta_nt_inter / deficit_inter
                            if r_inter_k >= ratio_tier_primary:
                                target_unsup_ratio_05_inter[idx] += 1
                            if r_inter_k >= ratio_tier_exploratory:
                                target_unsup_ratio_025_inter[idx] += 1

                        fold_unsup_inter = (
                            (surv_te[k] < 80)
                            & (pred_full_inter < 80)
                            & (shap_target_inter[k] >= 0)
                            & (ntt_sum_inter < 0)
                            & (pred_wo_nt_inter >= 80)
                        )
                        if fold_unsup_inter:
                            target_unsupported_inter[idx] += 1
                        if tt_i:
                            target_toxic_inter[idx] += 1
                        if ntt_sum_inter < 0:
                            non_target_toxic_inter[idx] += 1
                        # ★ v5.2: dominance 음수 부분만 비교 (TreeSHAP과 동일 로직)
                        target_neg_inter = min(float(shap_target_inter[k]), 0.0)
                        non_target_neg_inter = 0.0
                        for j in non_target_chem_idx:
                            if observed_mask[k, j]:
                                non_target_neg_inter += min(float(shap_all_inter[k, j]), 0.0)
                        if (target_neg_inter < 0) and (abs(non_target_neg_inter) > abs(target_neg_inter)):
                            non_target_dom_inter[idx] += 1

                    # ★ v5.2: 금속 ablation = 전체 모델(pred3) - 금속제외 모델(pred4)
                    #   (기존: pred3 - pred2 → otherPOPs+species 효과 포함 버그)
                    pred_with_metal[idx] += pred3[k]
                    pred_without_metal[idx] += pred4[k]
                    # ★ v5.2: sensitivity ratio 누적 (TreeSHAP 기준)
                    pred_full_sum[idx] += pred_full_tree
                    ntt_sum_accum[idx] += ntt_sum_tree

                    # ---- DRC 잔차 (보조 진단) ----
                    # C80 gate 원자료 (fold별 표준화 단일 C80, TOC=1% 고정)
                    conc_k = float(X_te2[k, 0])
                    surv_k = float(y_te[k])
                    c80_k = c80_fold
                    below_c80_k = (c80_k is not None) and (conc_k < c80_k)
                    fold_inconsistent_row = {}
                    for c in residual_cutoffs:
                        resid_cut = np.quantile(resid_inner_oof, c)
                        fold_inconsistent_row[c] = int(
                            (c80_k is not None) and (surv_k < 80) and (conc_k < c80_k)
                            and (resid_te[k] < resid_cut))

                    # 시료별 원자료 (long format)
                    row = {
                        "record_id": df_valid.loc[idx, "record_id"],
                        "sample_id": df_valid.loc[idx, "sample_id"],
                        "substance": substance,
                        "repeat_id": seed,
                        "fold_id": fold_id,
                        "is_oof": True,
                        # v5: SHAP 값 (다중 feature)
                        "shap_tree_target": shap_target_tree[k],
                        "shap_tree_metal": shap_metal_tree[k],
                        **{f"shap_tree_{p}": shap_other_pops_tree[p][k] for p in other_pops},
                        "shap_interv_target": (shap_target_inter[k] if inter_ok else np.nan),
                        "shap_interv_metal": (shap_metal_inter[k] if inter_ok else np.nan),
                        **{f"shap_interv_{p}": (shap_other_pops_inter[p][k] if inter_ok else np.nan)
                           for p in other_pops},
                        # v5.1: joint 판정 flags
                        "target_toxic_tree": int(tt),
                        "non_target_toxic_tree": int(ntt_sum_tree < 0),
                        "non_target_toxic_sum_tree": float(ntt_sum_tree),
                        "pred_full_tree": float(pred_full_tree),
                        "pred_without_non_target_tree": float(pred_wo_nt_tree),
                        "fold_target_unsupported_tree": int(fold_unsup_tree),
                        "target_toxic_inter": (int(tt_i) if inter_ok else np.nan),
                        "non_target_toxic_inter": (int(ntt_sum_inter < 0) if inter_ok else np.nan),
                        "non_target_toxic_sum_inter": (float(ntt_sum_inter) if inter_ok else np.nan),
                        "pred_full_inter": (float(pred_full_inter) if inter_ok else np.nan),
                        "pred_without_non_target_inter": (float(pred_wo_nt_inter) if inter_ok else np.nan),
                        "fold_target_unsupported_inter": (int(fold_unsup_inter) if inter_ok else np.nan),
                        # 기존 DRC 보조진단
                        "pred_with_metal": pred3[k],
                        "pred_without_metal": pred4[k],  # ★ v5.2: 금속-only ablation (기존 pred2)
                        "pred_diff": pred3[k] - pred4[k],  # ★ v5.2: 순수 금속 효과
                        "c80": c80_k if c80_k is not None else np.nan,
                        "c80_valid": int(c80_k is not None),
                        "below_c80": int(below_c80_k),
                        "pred_drc": pred2[k],
                        "residual_oof": resid_te[k],
                    }
                    for c in residual_cutoffs:
                        row[f"fold_inconsistent_{c}"] = fold_inconsistent_row[c]
                    raw_rows.append(row)

        # ---- GroupKFold leakage 검사 (train/test 그룹 교집합 0이어야 정상) ----
        if use_group_kfold and group_overlap_count != 0:
            raise RuntimeError(
                f"Group leakage detected: {group_overlap_count} overlapping "
                f"group(s) between train/test folds for {substance}")

        # ---- fold signature 검증: 10개 반복의 분할이 실제로 다른지 확인 ----
        # 각 seed의 fold 배정(record_id 집합)을 비교해 동일 분할 반복 여부를 검증
        n_unique_splits = 0
        if fold_signatures:
            seed_sigs = {}
            for fs in fold_signatures:
                seed_sigs.setdefault(fs["seed"], []).append(fs["test_record_ids"])
            # seed별로 fold 배정 집합을 정렬해 비교
            seed_keys = sorted(seed_sigs.keys())
            unique_seed_splits = set()
            for s in seed_keys:
                # 해당 seed의 fold 배정을 정규화(정렬된 tuple의 tuple)
                normalized = tuple(sorted(seed_sigs[s]))
                unique_seed_splits.add(normalized)
            n_unique_splits = len(unique_seed_splits)

        # ---- v5: 시료별 빈도 산출 (SHAP Primary 기반) ----
        target_unsup_freq_tree = np.divide(target_unsupported_tree, n_assignments,
                                           out=np.zeros(n_total), where=n_assignments > 0)
        target_toxic_freq_tree = np.divide(target_toxic_tree, n_assignments,
                                           out=np.zeros(n_total), where=n_assignments > 0)
        non_target_toxic_freq_tree = np.divide(non_target_toxic_tree, n_assignments,
                                                out=np.zeros(n_total), where=n_assignments > 0)
        # Interventional: 성공한 fold만 분모로 사용
        target_unsup_freq_inter = np.divide(target_unsupported_inter, n_inter_assignments,
                                             out=np.zeros(n_total), where=n_inter_assignments > 0)
        target_toxic_freq_inter = np.divide(target_toxic_inter, n_inter_assignments,
                                            out=np.zeros(n_total), where=n_inter_assignments > 0)
        non_target_toxic_freq_inter = np.divide(non_target_toxic_inter, n_inter_assignments,
                                                 out=np.zeros(n_total), where=n_inter_assignments > 0)
        non_target_dom_freq_tree = np.divide(non_target_dom_tree, n_assignments,
                                              out=np.zeros(n_total), where=n_assignments > 0)
        non_target_dom_freq_inter = np.divide(non_target_dom_inter, n_inter_assignments,
                                               out=np.zeros(n_total), where=n_inter_assignments > 0)
        inter_success_rate = (n_inter_success / n_inter_attempts) if n_inter_attempts > 0 else 0.0

        # ---- v5: 최종 분류 (SHAP Primary) ----
        # survival < 80 → EDS 후보, survival >= 80 → NEDS
        # Primary 제외 = 대상물질이 독성 방향으로 기여하지 않는데 다른 물질이 독성 기여
        #   (fold_target_unsupported)가 TreeSHAP·Interventional 양쪽 모두 min_success회 이상
        #   AND Interventional 성공 횟수도 min_success회 이상
        # → 대상물질 기여 없는 생존율 저하 = 노이즈 → 제외
        min_success = math.ceil(stability_threshold * n_seeds)
        survival = df_valid["Mean_Survival"].values
        is_eds_raw = survival < 80
        is_neds = survival >= 80

        # ★ v5.3 Primary 제외: R_NT 기여도 합의 (TreeSHAP ∩ Interventional)
        #   R_NT = (pred_wo_nt − pred_full) / (80 − pred_full) ≥ 0.5 (과반 기여, 사전 확정)
        #   구 c5(완전 회복, ≥1.0)는 Strict sensitivity로 이동 — Primary 정의 아님
        primary_excluded = (
            is_eds_raw
            & (target_unsup_ratio_05_tree >= min_success)
            & (target_unsup_ratio_05_inter >= min_success)
            & (n_inter_assignments >= min_success)
        )
        primary_retained_eds = is_eds_raw & ~primary_excluded

        # ---- 안전장치: 제거율이 max_removal_rate 초과 시 (v5.2: all-or-none → 플래그) ----
        # ★ v5.2 fix: 기존은 초과 시 primary_excluded 전면 해제(all-or-none).
        #   수정: 초과해도 SHAP strong signal(8/10 fold)은 유지,
        #         applicability_pass=False 플래그로 표시만.
        # ★ v5.2 fix2: 변수명 교정 — filter_applied=False이면 필터가 적용되지 않은 것이므로
        #   primary_excluded도 무효화해야 하지만, v5.2 설계는 primary_excluded를 유지.
        #   따라서 변수명을 applicability_pass로 변경하여 의미를 명확히 함.
        #   applicability_pass=True: SHAP 필터 정상 적용 (제거율 ≤ max_removal_rate)
        #   applicability_pass=False indicates an applicability warning; consensus exclusions remain applied.
        removal_rate = (primary_excluded.sum() / (primary_retained_eds.sum() + primary_excluded.sum())) \
            if (primary_retained_eds.sum() + primary_excluded.sum()) > 0 else 0.0
        applicability_pass = removal_rate <= max_removal_rate

        # ---- Sensitivity: non_target_dominant (혼합독성 우세) ----
        # Primary에서 바로 제거 금지, sensitivity로 분리
        # non_target_dominant = target_toxic & (non_target 독성 기여 > target 독성 기여)
        # → 대상물질도 기여하지만 다른 물질 기여가 더 큰 경우
        non_target_dom_mask = (
            is_eds_raw
            & ~primary_excluded
            & (target_toxic_tree >= min_success)
            & (non_target_dom_tree >= min_success)
            & (non_target_dom_inter >= min_success)
            & (n_inter_assignments >= min_success)
        )

        # ---- DRC 보조 진단 (Primary 제거 아님, v5에서 격하) ----
        drc_inconsistent = residual_anomaly_counts[residual_cutoff] >= min_success

        # ---- 저농도 DRC gate (C80) 검증 (fold별 판정 직접 집계) ----
        # 제외된 모든 기록이 fold별로 below_c80=True(관측농도 < fold-specific C80)였는지 확인.
        # drc_inconsistent는 이미 각 fold에서 (surv<80 AND conc<C80 AND resid<cutoff)를
        # 만족한 fold만 카운트하므로, 제외 기록은 n_below_c80 >= min_success여야 정상.
        # C80_above_count는 반드시 0이어야 정상 (고농도 강독성 자료가 노이즈로 빠지면 위반).
        conc_obs = df_valid[target_col].values  # 관측 농도 (로그 스케일, Conc_log)
        # fold별 below_c80 횟수가 min_success 미만인데 제외된 기록 = 게이트 위반 (반드시 0)
        c80_above_count = int(np.sum(drc_inconsistent & (n_below_c80 < min_success)))
        c80_below_count = int(np.sum(drc_inconsistent & (n_below_c80 >= min_success)))
        n_valid_c80 = int(np.sum(n_c80_valid > 0))
        n_drc_inconsistent = int(drc_inconsistent.sum())
        # 제외·유지 EDS 농도 중앙값 (로그 스케일 → 원래 µg/kg dw 복원)
        # ★ v5.3 통계 분리 (사용자 지적): C80/DRC 통계는 DRC 후보(drc_inconsistent)에만,
        #   SHAP 제외 농도 통계는 Primary 제외자료(primary_excluded)에만 연결.
        #   (기존: 후보자료·제외자료가 동일 마스크로 묶여 두 중앙값이 동일해지는 버그)
        primary_retained_eds = is_eds_raw & ~primary_excluded
        shap_excluded_conc_log = conc_obs[primary_excluded]   # SHAP Primary 제외자료
        drc_candidate_conc_log = conc_obs[drc_inconsistent]   # DRC/C80 후보자료 (제외 여부 무관)
        retained_conc_log = conc_obs[primary_retained_eds]    # 실제 Primary 유지 EDS
        excl_med_orig = float(np.median(10 ** shap_excluded_conc_log - 1)) if len(shap_excluded_conc_log) > 0 else np.nan
        cand_med_orig = float(np.median(10 ** drc_candidate_conc_log - 1)) if len(drc_candidate_conc_log) > 0 else np.nan
        ret_med_orig = float(np.median(10 ** retained_conc_log - 1)) if len(retained_conc_log) > 0 else np.nan

        # ---- Primary 선별 DataFrame (Step 3 입력용) ----
        # Primary = NEDS 전체 + (EDS 중 primary_excluded가 아닌 것)
        # primary_excluded는 applicability_pass=False 시에도 유지됨 (v5.2 설계)
        # → primary_keep은 primary_excluded를 그대로 사용하여 EDS에서 제외
        #   applicability_pass=False 시에도 primary_excluded가 실제 동작함
        primary_keep = is_neds | (is_eds_raw & ~primary_excluded)
        primary_filtered_dfs[substance] = df_valid[primary_keep].copy()

        # ---- Sensitivity 구성 (v5.3: tier + chemical screen + unfiltered) ----
        # Strict       = R_NT ≥ 1.0 (완전 회복, 구 c5) — TreeSHAP ∩ Interventional 합의
        # Exploratory  = R_NT ≥ 0.25 — 동일 합의
        # ChemScreen   = mPELQ ≤ 0.5 하드 제외(구 Step 1 필터)만 적용 (SHAP 미적용 arm)
        # Unfiltered   = 기본 QA·종 선별만 적용 (SHAP 제외 없음, 비교 기준선)
        strict_excluded = (
            is_eds_raw
            & (target_unsupported_tree >= min_success)
            & (target_unsupported_inter >= min_success)
            & (n_inter_assignments >= min_success)
        )
        exploratory_excluded = (
            is_eds_raw
            & (target_unsup_ratio_025_tree >= min_success)
            & (target_unsup_ratio_025_inter >= min_success)
            & (n_inter_assignments >= min_success)
        )
        screen_fail = ~df_valid["screen_pass_mpelq"].astype(bool)
        sens1_eds = is_eds_raw & ~strict_excluded
        sens2_eds = is_eds_raw & ~exploratory_excluded
        # ChemScreen sensitivity = 구 Step 1 화학 필터의 재현: mPELQ≤0.5 '통과' 자료만 유지.
        #   SHAP Primary 제외를 중복 적용하지 않는다 (SHAP 미적용 arm).
        #   mPELQ 초과 EDS·NEDS 모두 제외되는 것이 구 Step 1과 동일한 동작이다.
        sens3_df = df_valid[df_valid["screen_pass_mpelq"].astype(bool)].reset_index(drop=True)
        sens4_eds = is_eds_raw  # Unfiltered: EDS 전부 유지 (SHAP 제외 없음)
        # ★ 중첩성 검증 (사전 확정 원칙): 제외 집합은 Strict ⊆ Primary ⊆ Exploratory여야 함
        n_nesting_violation = int(
            (strict_excluded & ~primary_excluded).sum()
            + (primary_excluded & ~exploratory_excluded).sum())
        if n_nesting_violation != 0:
            raise RuntimeError(
                f"중첩성 위반: Strict⊆Primary⊆Exploratory 성립 실패 "
                f"({n_nesting_violation}건) — 집계 로직 오류")

        # ---- Primary TEL/PEL (SHAP Primary 기반, primary_excluded 기준) ----
        primary_df = df_valid[primary_keep].reset_index(drop=True)
        primary_telpel = _compute_tel_pel_from_df(primary_df, target_col)
        # Sens1/Sens2 TEL/PEL (sens3_df는 위 ChemScreen 정의 사용 — EDS/NEDS 통합)
        sens1_df = df_valid[sens1_eds | is_neds].reset_index(drop=True)
        sens2_df = df_valid[sens2_eds | is_neds].reset_index(drop=True)
        sens1_telpel = _compute_tel_pel_from_df(sens1_df, target_col)
        sens2_telpel = _compute_tel_pel_from_df(sens2_df, target_col)
        sens3_telpel = _compute_tel_pel_from_df(sens3_df, target_col)
        # Unfiltered comparator: SHAP 제외 없음 (기본 QA·종 선별만 적용, 비교 기준선)
        sens4_df = df_valid[sens4_eds | is_neds].reset_index(drop=True)
        sens4_telpel = _compute_tel_pel_from_df(sens4_df, target_col)

        # ablation: 금속 포함에 따른 OOF prediction difference (평균)
        pred_diff = np.divide(pred_with_metal - pred_without_metal, n_assignments,
                              out=np.zeros(n_total), where=n_assignments > 0)
        mean_ablation = float(np.mean(pred_diff))

        # ★ v5.2: sensitivity ratio (contribution_ratio) — 비대상물질 기여도 정량화
        #   pred_full_mean: OOF 평균 pred_full (TreeSHAP)
        #   ntt_sum_mean: OOF 평균 non_target_toxic_sum (음수 = 비대상물질이 독성 방향)
        #   delta_nt = -ntt_sum_mean (비대상물질 음의 SHAP 합 절대값 = 비대상 기여 크기)
        #   toxicity_deficit = 80 - pred_full_mean (pred_full이 80 미만일 때의 독성 결손)
        #   contribution_ratio = delta_nt / toxicity_deficit
        #     ≥ 1.0: Strict — 완전 회복(구 c5), sensitivity
        #     ≥ 0.5: Primary — 과반 기여(사전 확정)
        #     ≥ 0.25: 비대상물질이 독성 결손의 25% 이상 설명 (Exploratory)
        pred_full_mean = np.divide(pred_full_sum, n_assignments,
                                    out=np.full(n_total, np.nan), where=n_assignments > 0)
        ntt_sum_mean = np.divide(ntt_sum_accum, n_assignments,
                                 out=np.full(n_total, np.nan), where=n_assignments > 0)
        delta_nt = -ntt_sum_mean  # 음수 SHAP 합의 절대값
        toxicity_deficit = 80.0 - pred_full_mean  # pred_full < 80일 때 양수
        # contribution_ratio: toxicity_deficit > 0일 때만 계산 (0으로 나누기 방지)
        contribution_ratio = np.where(
            (toxicity_deficit > 1e-6) & (delta_nt > 1e-6),
            delta_nt / toxicity_deficit,
            0.0
        )
        # sensitivity 등급
        sens_strict = contribution_ratio >= ratio_tier_strict        # 완전 회복(구 c5)
        sens_moderate = contribution_ratio >= ratio_tier_primary     # 과반 기여 (Primary 동치, 표본 수준)
        sens_exploratory = contribution_ratio >= 0.25  # 탐색적 (25%+)
        n_ablation_negative = int((pred_diff < 0).sum())  # 금속 포함 시 예측 감소(독성 방향)

        # ablation 통계 보강 (median, IQR, 5th-95th percentile)
        med_ablation = float(np.median(pred_diff))
        q25_ablation = float(np.quantile(pred_diff, 0.25))
        q75_ablation = float(np.quantile(pred_diff, 0.75))
        p5_ablation = float(np.quantile(pred_diff, 0.05))
        p95_ablation = float(np.quantile(pred_diff, 0.95))

        # EDS/NEDS별 ablation 분포 (실제 Primary EDS 기준: primary_retained_eds vs NEDS)
        eds_ablation = pred_diff[primary_retained_eds]
        neds_ablation = pred_diff[is_neds]
        mean_ablation_eds = float(np.mean(eds_ablation)) if len(eds_ablation) > 0 else np.nan
        mean_ablation_neds = float(np.mean(neds_ablation)) if len(neds_ablation) > 0 else np.nan

        # ---- 결과 기록 ----
        n_eds_retained_primary = int(primary_retained_eds.sum())
        n_target_attribution_unsupported_eds = int(primary_excluded.sum())
        n_neds = int(is_neds.sum())
        n_neg_majority = int((target_toxic_freq_tree >= stability_threshold).sum())
        # 실제 Primary EDS 수 (primary_excluded 기준)
        n_eds_primary = int((is_eds_raw & ~primary_excluded).sum())
        # ★ v5.3 사용자 지정 명명: 합의(consensus) 제외 = 실제 제외 건수 (논문 표용)
        #   진단(diagnostic) = TreeSHAP 평균 ratio 표본 수준 카운트 (합의와 다를 수 있음)
        n_excluded_primary_consensus = int(primary_excluded.sum())
        n_excluded_strict_consensus = int(strict_excluded.sum())
        n_excluded_exploratory_consensus = int(exploratory_excluded.sum())
        n_mean_TreeRNT_ge05_diagnostic = int(sens_moderate.sum())
        n_mean_TreeRNT_ge10_diagnostic = int(sens_strict.sum())
        n_mean_TreeRNT_ge025_diagnostic = int(sens_exploratory.sum())
        # ChemScreen arm 건수 (mPELQ≤0.5 미통과 = 화학 필터 단독 제외 대상)
        n_ChemScreen_excluded_total = int(screen_fail.sum())
        n_ChemScreen_excluded_EDS = int((screen_fail & is_eds_raw).sum())
        n_ChemScreen_excluded_NEDS = int((screen_fail & is_neds).sum())
        # (removal_rate / applicability_pass / primary_excluded는 위에서 이미 정의됨 — 중복 정의 제거)
        print(f"\n  {substance}: N={n_total}")
        print(f"    Primary (R_NT≥0.5 과반 기여, TreeSHAP∩Interventional ≥{min_success}/{n_seeds}): "
              f"EDS={n_eds_primary} NEDS={n_neds} "
              f"제외(합의)={n_excluded_primary_consensus} "
              f"(제거율 {removal_rate:.1%}) → "
              f"TEL={primary_telpel['TEL']:.4f} PEL={primary_telpel['PEL']:.4f}" if primary_telpel else
              f"    Primary: TEL/PEL 계산 불가")
        if not applicability_pass:
            print(f"    ⚠️ 제거율 {removal_rate:.1%} > {max_removal_rate:.0%} → "
                  f"applicability_pass=False (적용성 한계 표시). "
                  f"primary_excluded는 설계대로 유지 적용됨 — 보고 시 명시할 것")
        n_non_target_dom = int(non_target_dom_mask.sum())
        print(f"    Sensitivity (non_target dominant): {n_non_target_dom} 시료 (혼합독성 우세, 제외 아님)")
        print(f"    DRC 보조진단: drc_inconsistent 후보 {n_drc_inconsistent}건 "
              f"(Primary 제거 아님, 참고용)")
        print(f"    대상물질 독성 기여 안정(SHAP_target<0 ≥{stability_threshold:.0%}): {n_neg_majority} 시료")
        print(f"    Ablation: 금속 포함 시 OOF 예측 평균 변화 {mean_ablation:+.4f} "
              f"(음수 {n_ablation_negative} 시료)")
        # ★ v5.2: sensitivity ratio 출력
        print(f"    Sensitivity Ratio: Strict/R_NT≥1.0={int(sens_strict.sum())} "
              f"Primary/R_NT≥0.5={int(sens_moderate.sum())} "
              f"Exploratory/R_NT≥0.25={int(sens_exploratory.sum())} "
              f"(평균 ratio={np.nanmean(contribution_ratio):.3f})")
        print(f"    C80 gate 검증(DRC 보조): 후보 {n_drc_inconsistent}건 중 fold별 below_c80≥{min_success}회 {c80_below_count} / "
              f"미달 {c80_above_count} (유효 C80 기록 {n_valid_c80}건)")
        # ★ v5.3: DRC(C80) 진단과 SHAP 제외 통계는 독립 분석 — 출력 분리 (사용자 지적)
        print(f"    DRC 후보 농도 중앙값 {cand_med_orig:.4f} µg/kg (1% TOC 정규화)")
        print(f"    SHAP Primary 제외자료: 제외 EDS 농도 중앙값 {excl_med_orig:.4f} vs "
              f"유지 EDS {ret_med_orig:.4f} µg/kg (1% TOC 정규화)")

        diag_rows.append({
            "Source": source_name, "Substance": substance, "N_total": n_total,
            "group_col": group_col,
            "n_groups": int(len(np.unique(groups))) if groups is not None else 0,
            "group_kfold_train_test_overlap": int(group_overlap_count),
            "n_unique_fold_splits": int(n_unique_splits),
            "N_target_toxic_freq_ge_threshold": n_neg_majority,
            "N_EDS_retained_primary": n_eds_retained_primary,
            "N_target_attribution_unsupported_eds": n_target_attribution_unsupported_eds,
            "N_EDS_primary": n_eds_primary,
            "N_excluded_primary_consensus": n_excluded_primary_consensus,
            # ★ v5.3: tier별 제외 건수(합의) + diagnostic(평균 TreeSHAP ratio) 분리
            "N_excluded_strict_consensus": n_excluded_strict_consensus,
            "N_excluded_exploratory_consensus": n_excluded_exploratory_consensus,
            "N_mean_TreeRNT_ge1.0_diagnostic": n_mean_TreeRNT_ge10_diagnostic,
            "N_mean_TreeRNT_ge0.5_diagnostic": n_mean_TreeRNT_ge05_diagnostic,
            "N_mean_TreeRNT_ge0.25_diagnostic": n_mean_TreeRNT_ge025_diagnostic,
            # ChemScreen arm: mPELQ≤0.5 미통과 (화학 필터 단독 — SHAP 무관)
            "N_ChemScreen_excluded_total": n_ChemScreen_excluded_total,
            "N_ChemScreen_excluded_EDS": n_ChemScreen_excluded_EDS,
            "N_ChemScreen_excluded_NEDS": n_ChemScreen_excluded_NEDS,
            "N_screen_fail_mPELQ0.5": n_ChemScreen_excluded_total,  # 호환 유지(동일 정의)
            "nesting_violations": n_nesting_violation,
            "N_non_target_dominant": n_non_target_dom,
            "N_NEDS": n_neds,
            "residual_cutoff": residual_cutoff,
            "removal_rate": round(removal_rate, 4),
            "applicability_pass": applicability_pass,
            "n_oof_per_sample_min": int(n_assignments.min()) if n_total > 0 else 0,
            "n_oof_per_sample_max": int(n_assignments.max()) if n_total > 0 else 0,
            "interventional_background_min": int(min(bg_sizes)) if bg_sizes else 0,
            "interventional_background_max": int(max(bg_sizes)) if bg_sizes else 0,
            "interventional_success_rate": round(inter_success_rate, 4),
            "interventional_attempts": n_inter_attempts,
            "interventional_successes": n_inter_success,
            "Mean_ablation_pred_diff": round(mean_ablation, 4),
            "Median_ablation_pred_diff": round(med_ablation, 4),
            "IQR_ablation_pred_diff": round(q75_ablation - q25_ablation, 4),
            "P5_ablation_pred_diff": round(p5_ablation, 4),
            "P95_ablation_pred_diff": round(p95_ablation, 4),
            "Mean_ablation_EDS": round(mean_ablation_eds, 4),
            "Mean_ablation_NEDS": round(mean_ablation_neds, 4),
            "N_ablation_negative": n_ablation_negative,
            "N_ablation_negative_pct": round(100.0 * n_ablation_negative / n_total, 2),
            "C80_below_count": c80_below_count,
            "C80_above_count": c80_above_count,
            "N_records_with_valid_C80": n_valid_c80,
            "N_DRC_inconsistent": n_drc_inconsistent,
            "SHAP_Excluded_EDS_Conc_Median_OC1pct": (round(excl_med_orig, 4) if not np.isnan(excl_med_orig) else None),
            "DRC_Candidate_EDS_Conc_Median_OC1pct": (round(cand_med_orig, 4) if not np.isnan(cand_med_orig) else None),
            "Primary_Retained_EDS_Conc_Median_OC1pct": (round(ret_med_orig, 4) if not np.isnan(ret_med_orig) else None),
            # ★ v5.2: sensitivity ratio (contribution_ratio) 통계 — 표본 수준 진단치
            #   (N_sens_* 명칭은 폐기: N_mean_TreeRNT_ge*_diagnostic로 통일, 상단 참조)
            "Mean_contribution_ratio": round(float(np.nanmean(contribution_ratio)), 4),
            "Median_contribution_ratio": round(float(np.nanmedian(contribution_ratio)), 4),
            "Mean_pred_full_OOF": round(float(np.nanmean(pred_full_mean)), 4),
            "Mean_delta_nt": round(float(np.nanmean(delta_nt)), 4),
            "Mean_toxicity_deficit": round(float(np.nanmean(toxicity_deficit)), 4),
        })

        for label, telpel in [("Primary", primary_telpel),
                              ("Sensitivity_Strict_RNT1.0", sens1_telpel),
                              ("Sensitivity_Exploratory_RNT0.25", sens2_telpel),
                              ("Sensitivity_ChemScreen_mPELQ0.5", sens3_telpel),
                              ("Sensitivity_Unfiltered", sens4_telpel)]:
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
                OUTPUT_DIR / f"Step2_7v5_OOF_Raw_{source_name}_{substance}.csv",
                index=False)

        # Fig. 2 재설계용: OOF DRC 잔차 + primary_excluded flag 저장 (필터 전 전체 기록)
        # residual_mean = OOF 잔차 평균 (실제 생존율 - DRC 예측 생존율)
        # drc_inconsistent = Primary cutoff 기준 안정적 이상 판정 (제외 "후보")
        # primary_excluded = 실제 제외자료 (applicability_pass=True일 때만 drc_inconsistent와 동일,
        #   applicability_pass=False(안전장치 발동) 시에도 primary_excluded는 유지됨 — 후보로 표시)
        residual_mean = np.divide(residual_sum, n_assignments,
                                  out=np.full(n_total, np.nan), where=n_assignments > 0)
        drc_resid_df = df_valid.copy()
        drc_resid_df["residual_mean"] = residual_mean
        drc_resid_df["drc_inconsistent"] = drc_inconsistent.astype(int)
        drc_resid_df["primary_excluded"] = primary_excluded.astype(int)
        drc_resid_df["applicability_pass"] = int(applicability_pass)
        drc_resid_df["is_eds_raw"] = is_eds_raw.astype(int)
        # ★ v5.2: sensitivity ratio (contribution_ratio) per-sample
        drc_resid_df["pred_full_OOF_mean"] = pred_full_mean
        drc_resid_df["ntt_sum_OOF_mean"] = ntt_sum_mean
        drc_resid_df["delta_nt"] = delta_nt
        drc_resid_df["toxicity_deficit"] = toxicity_deficit
        drc_resid_df["contribution_ratio"] = contribution_ratio
        drc_resid_df["sens_strict_ge1.0"] = sens_strict.astype(int)
        drc_resid_df["sens_moderate_ge0.5"] = sens_moderate.astype(int)
        drc_resid_df["sens_exploratory_ge0.25"] = sens_exploratory.astype(int)
        # Data_Type은 반드시 primary_excluded 기준 (실제 분석과 일치)
        drc_resid_df["Data_Type"] = np.where(
            is_neds, "NEDS",
            np.where(primary_excluded, "SHAP-excluded", "EDS"))
        drc_resid_df.to_csv(
            OUTPUT_DIR / f"Step2_7v5_DRC_Residual_{source_name}_{substance}.csv",
            index=False)

        # fold signature CSV 저장 (10개 반복의 분할 차이 검증용)
        if fold_signatures:
            fs_df = pd.DataFrame(fold_signatures)
            fs_df.to_csv(
                OUTPUT_DIR / f"Step2_7v5_FoldSignatures_{source_name}_{substance}.csv",
                index=False)

    diag_df = pd.DataFrame(diag_rows)
    telpel_df = pd.DataFrame(tel_pel_rows)

    # ---- OOF 반복 구조 메타데이터 (재현성·검증용) ----
    # dominance_ratio: |SHAP_metal| > ratio * |SHAP_target| (config에서 읽음, 기본 1.0)
    # interventional background = 각 fold의 training data 전체 (shap.sample 없음)
    # fold별 실제 행 수는 substance별로 diag_rows에 이미 기록됨 (interventional_background_min/max)

    if len(diag_df) > 0:
        diag_df.to_csv(OUTPUT_DIR / f"Step2_7v5_Diagnosis_{source_name}.csv", index=False)
    if len(telpel_df) > 0:
        telpel_df.to_csv(OUTPUT_DIR / f"Step2_7v5_TELPEL_Comparison_{source_name}.csv", index=False)

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
    meta_df.to_csv(OUTPUT_DIR / f"Step2_7v5_Metadata_{source_name}.csv", index=False)

    return primary_filtered_dfs, diag_df, telpel_df


# =============================================================================
# Step 3: EDS/NEDS → TEL/PEL (OOF-SHAP 선별 종 사용; DRC는 진단 전용)
# =============================================================================
def step3_tel_pel(filtered_dfs, selected_species, source_name=""):
    """EDS/NEDS → TEL/PEL. 종 집합은 OOF-SHAP 선별 종(selected_species)이며
    DRC는 진단 전용으로 평가된다 (DRC 불량 종도 Primary에 유지됨)."""
    print(f"\n{'='*70}")
    print(f"[Step 3] EDS/NEDS → TEL/PEL — {source_name}")
    print(f"{'='*70}")

    sqg_list = []
    edsneds_data = {}  # Step 4용 저장

    for substance in TARGET_SUBSTANCES:
        if substance not in filtered_dfs or substance not in selected_species:
            continue
        ok_spp = selected_species[substance]
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

        print(f"\n▶ {substance} (N={len(df_pooled)}, 선별 {len(ok_spp)}종: {ok_spp})")

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
            # Method B (CCME 지침): 원래 단위에서 percentile 계산 후 기하평균.
            # Conc_log = log10(estimated_total/TOC_pct + 1) → 1% TOC 정규화 농도.
            # 역변환 10^Conc_log - 1 = estimated_total/TOC_pct (µg/kg per 1% OC),
            # 일반 µg/kg dw가 아님에 유의. CCME/NOAA 기준과 비교 시 단위 기준 일치 필요.
            e_orig = 10 ** e_v - 1
            n_orig = 10 ** n_v - 1
            our_tel = np.sqrt(np.quantile(e_orig, 0.15) * np.quantile(n_orig, 0.50))
            our_pel = np.sqrt(np.quantile(e_orig, 0.50) * np.quantile(n_orig, 0.85))

            sqg_list.append({
                "Source": source_name, "Substance": substance,
                "Total_N": len(df_pooled), "N_EDS": n_eds, "N_NEDS": n_neds,
                "N_Selected_Species": len(ok_spp),
                "Selected_Species": "; ".join(ok_spp),
                "Our_TEL_OC1pct": round(our_tel, 4), "Our_PEL_OC1pct": round(our_pel, 4),
                "CCME_ISQG": CCME_ISQG.get(substance, np.nan),
                "CCME_PEL": CCME_PEL.get(substance, np.nan),
                "NOAA_TEL": NOAA_TEL_DW.get(substance, np.nan),
                "NOAA_PEL": NOAA_PEL_DW.get(substance, np.nan),
            })
            print(f"  TEL = {our_tel:.4f}, PEL = {our_pel:.4f}  "
                  f"(CCME: {CCME_ISQG.get(substance)}/{CCME_PEL.get(substance)})")

            # Step 4용 데이터 저장 (식별자 컬럼 유지 — bootstrap cluster용)
            id_cols = [c for c in ["Station", "sample_key", "SampleID", "StationID",
                                    "station_key", "StudyID", "Date", "Region",
                                    "Month", "Year", "Start_Latitude", "Start_Longitude"] if c in df_pooled.columns]
            keep_cols = ["Conc_log", "Mean_Survival", "Data_Type",
                         "Latin_Name_Species", "TOC_pct"] + id_cols
            edsneds_data[substance] = df_pooled[keep_cols].copy()

    final = pd.DataFrame(sqg_list)
    final.to_csv(OUTPUT_DIR / f"Step3_TEL_PEL_{source_name}.csv", index=False)

    # ★ 추가 분석용: EDS/NEDS 원자료 저장 (bootstrap CI / mPELQ 민감도 / GroupKFold)
    for substance, edf in edsneds_data.items():
        edf.to_csv(OUTPUT_DIR / f"Step3_EDSNEDS_{source_name}_{substance}.csv", index=False)

    return final, edsneds_data


# =============================================================================
# Step 4: 신뢰도 평가 (기준 비교 + ROC-AUC 예측력) ★신규
# =============================================================================
def step4_reliability_assessment(final_df, edsneds_data, source_name="", random_seed=42):
    """
    준거치 신뢰도 판단:
    4-1. 기준 비교: TEL/PEL vs CCME ISQG/PEL, NOAA TEL/PEL
    4-2. 독성 예측력: ROC-AUC + TEL 기준 민감도/특이도
    ★ v5.2 fix: random_seed 명시적 전달, GroupKFold 적용, except:pass 제거
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
        our_tel = row["Our_TEL_OC1pct"]
        our_pel = row["Our_PEL_OC1pct"]

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
                # 원래 방향 그대로 보고 (뒤집지 않음)
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

        # --- 5-fold CV AUC (★ v5.2: GroupKFold + 예외 상세 로깅) ---
        cv_auc_mean = np.nan
        cv_auc_std = np.nan
        if n_pos >= 10 and n_neg >= 10:
            cv_scores = []
            cv = min(5, n_pos, n_neg)
            if cv >= 2:
                # ★ v5.2: group leakage 방지 — _build_group_ids로 group_id 생성
                df_eval_grp = df_eval_data.reset_index(drop=True).copy()
                groups_eval = _build_group_ids(df_eval_grp)
                n_groups = len(np.unique(groups_eval))
                cv_groups = min(cv, n_groups)
                if cv_groups >= 2:
                    gkf = GroupKFold(n_splits=cv_groups)
                    for train_idx, test_idx in gkf.split(conc_log, y_true, groups=groups_eval):
                        c_train, c_test = conc_log[train_idx], conc_log[test_idx]
                        y_train, y_test = y_true[train_idx], y_true[test_idx]
                        if y_train.sum() < 2 or (1 - y_train).sum() < 2:
                            continue
                        # XGBoost binary classification
                        dtrain = xgb.DMatrix(c_train.reshape(-1, 1), label=y_train,
                                             feature_names=["Conc_log"])
                        model = xgb.train({"objective": "binary:logistic", "eta": 0.1,
                            "max_depth": 3, "seed": random_seed}, dtrain, num_boost_round=50)
                        dtest = xgb.DMatrix(c_test.reshape(-1, 1), feature_names=["Conc_log"])
                        y_pred = model.predict(dtest)
                        if len(np.unique(y_test)) == 2:
                            fold_auc = roc_auc_score(y_test, y_pred)
                            cv_scores.append(fold_auc)
                    if cv_scores:
                        cv_auc_mean = np.mean(cv_scores)
                        cv_auc_std = np.std(cv_scores)
                    else:
                        print(f"    [경고] {substance}: CV fold 생성 실패 (그룹 수={n_groups})")
                else:
                    print(f"    [경고] {substance}: GroupKFold 불가 (유효 그룹 수={n_groups} < 2)")

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
    cfg = load_config()
    random_seed = int(cfg.get("random_seed", 42))  # ★ v5.2: main scope에서 정의
    group_cfg = cfg.get("group_kfold", {})
    group_kfold_mode = (
        "--group-kfold" in sys.argv
        or group_cfg.get("enabled", True)  # 기본값 True: config가 없어도 GroupKFold 사용
    )
    print("=" * 70)
    print("고도화 파이프라인 v5.3: R_NT 기여도 Primary (mPELQ 하드 제외 → sensitivity)")
    print("  Step 1: DB 정제 (금속 QA; mPELQ≤0.5는 Chemical-screen sensitivity로 이동) + PAHs Sum 계산")
    print("  Step 2: XGBoost SHAP 종 선별")
    print("  Step 2.5: DRC 평가 → 종 선별 (DRC는 진단 전용, Primary 선별에 관여 안 함)")
    print("  Step 2.7: SHAP Primary (R_NT≥0.5 과반 기여, Tree∩Inter 합의) + DRC 보조진단")
    print("  Step 3: EDS/NEDS → TEL/PEL")
    print("  Step 4: 신뢰도 평가 (기준 비교 + ROC-AUC, GroupKFold)")
    if diagnose_mode:
        print("  ★ 진단 모드: Step 2.7-v5.3 (SHAP Primary/Sensitivity/DRC 보조진단)")
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
        cleaned_dfs, drc_df, selected_species = step2_5_drc_evaluation(
            substance_dfs, df_eval, source_name)

        if diagnose_mode:
            # 진단 모드: Step 2.7-v5.3 (Primary/Sensitivity/Ablation) 실행 후 종료
            primary_filtered_dfs, diag_df, telpel_df = step2_7_confounder_filtering_v5(
                cleaned_dfs, selected_species, source_name,
                use_group_kfold=group_kfold_mode)
            if len(diag_df) > 0:
                all_diag.append(diag_df)
            if len(telpel_df) > 0:
                all_telpel.append(telpel_df)
            continue

        # ★ v5.2 통합: Step 2.7 SHAP Primary → Step 3 empirical TEL/PEL
        #   Primary = NEDS 전체 + (EDS 중 primary_excluded가 아닌 것)
        primary_filtered_dfs, diag_df, telpel_df = step2_7_confounder_filtering_v5(
            cleaned_dfs, selected_species, source_name,
            use_group_kfold=group_kfold_mode)
        if len(diag_df) > 0:
            all_diag.append(diag_df)
        if len(telpel_df) > 0:
            all_telpel.append(telpel_df)

        final, edsneds_data = step3_tel_pel(primary_filtered_dfs, selected_species, source_name)
        reliability, edsneds_data = step4_reliability_assessment(final, edsneds_data, source_name,
                                                                 random_seed=random_seed)

        if len(final) > 0:
            all_sqg.append(final)
        if len(reliability) > 0:
            all_reliability.append(reliability)

    if diagnose_mode:
        # 진단 모드: Sensitivity 결과 저장 후 종료
        print(f"\n{'='*70}")
        print("진단 모드 결과 요약 (Step 2.7-v5.3)")
        print(f"{'='*70}")
        if all_diag:
            diag_all = pd.concat(all_diag, ignore_index=True)
            diag_all.to_csv(OUTPUT_DIR / "Step2_7v5_Diagnosis_All.csv", index=False)
            print("\n--- 진단 요약 (Source × Substance) ---")
            print(diag_all.to_string(index=False))
        if all_telpel:
            telpel_all = pd.concat(all_telpel, ignore_index=True)
            telpel_all.to_csv(OUTPUT_DIR / "Step2_7v5_TELPEL_Comparison_All.csv", index=False)
            print("\n--- Primary vs Sensitivity TEL/PEL 비교 ---")
            print(telpel_all.to_string(index=False))
        print(f"\n진단 결과 저장: {OUTPUT_DIR}")
        print("\n진단 완료 ✅")
        return

    # ★ v5: Sensitivity 결과 저장 (Supplementary용)
    if all_diag:
        diag_all = pd.concat(all_diag, ignore_index=True)
        diag_all.to_csv(OUTPUT_DIR / "Step2_7v5_Diagnosis_All.csv", index=False)
    if all_telpel:
        telpel_all = pd.concat(all_telpel, ignore_index=True)
        telpel_all.to_csv(OUTPUT_DIR / "Step2_7v5_TELPEL_Comparison_All.csv", index=False)

    # 통합 비교표
    print(f"\n{'='*70}")
    print("최종 소스별 TEL/PEL 비교표 (v5 SHAP Primary 파이프라인)")
    print(f"{'='*70}")

    if all_sqg:
        comparison = pd.concat(all_sqg, ignore_index=True)
        comparison.to_csv(OUTPUT_DIR / "TEL_PEL_v5_comparison.csv", index=False)

        print("\n--- TEL 비교 ---")
        tel_pivot = comparison.pivot(index="Substance", columns="Source", values="Our_TEL_OC1pct")
        tel_pivot["CCME_ISQG"] = comparison.groupby("Substance")["CCME_ISQG"].first()
        tel_pivot["NOAA_TEL"] = comparison.groupby("Substance")["NOAA_TEL"].first()
        print(tel_pivot.to_string(float_format="%.2f"))

        print("\n--- PEL 비교 ---")
        pel_pivot = comparison.pivot(index="Substance", columns="Source", values="Our_PEL_OC1pct")
        pel_pivot["CCME_PEL"] = comparison.groupby("Substance")["CCME_PEL"].first()
        pel_pivot["NOAA_PEL"] = comparison.groupby("Substance")["NOAA_PEL"].first()
        print(pel_pivot.to_string(float_format="%.2f"))

        print("\n--- TEL / CCME ISQG 비율 ---")
        for src in ["NOAA", "SCCWRP", "Integrated"]:
            if src in tel_pivot.columns:
                ratio = (tel_pivot[src] / tel_pivot["CCME_ISQG"]).round(2)
                print(f"  {src}: {dict(ratio)}")

        tel_pivot.to_csv(OUTPUT_DIR / "TEL_comparison_v5.csv")
        pel_pivot.to_csv(OUTPUT_DIR / "PEL_comparison_v5.csv")

    if all_reliability:
        all_rel = pd.concat(all_reliability, ignore_index=True)
        all_rel.to_csv(OUTPUT_DIR / "Reliability_v5_summary.csv", index=False)

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