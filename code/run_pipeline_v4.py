#!/usr/bin/env python3
"""
고도화 파이프라인 v4: 신뢰도 기반 TEL/PEL 도출
=============================================

[사용자 설계]
1. TEL/PEL 신뢰도 향상
   - DRC 보이는 생물종 선정 → 독성 인과관계 확인
   - DRC-inconsistent toxicity(저농도·생존율<80·하위 OOF 잔차) 자료 → 제거 (Step 2.7)

2. 준거치 신뢰도 판단
   - 기 설정 기준(CCME/NOAA)과 비교 검토 (Step 4-1)
   - 독성 예측력(ROC-AUC) 검토 (Step 4-2)

파이프라인 구조:
  Step 1: DB 정제 (mPELQ 중금속 사전 필터)
  Step 2: XGBoost SHAP 종 선별 (Spearman ≤ -0.3)
  Step 2.5: DRC 품질 평가 → DRC OK 종만 선별
  Step 2.7: nested OOF monotonic DRC + 저농도 C80 gate → DRC-inconsistent 기록 제외 ★
  Step 3: EDS/NEDS → TEL/PEL
  Step 4: 신뢰도 평가 (기준 비교 + ROC-AUC 예측력) ★

[Primary 필터 (Step 2.7) — 최종 확정]
  - 반복 GroupKFold OOF로 대상물질 농도–생존율 monotonic DRC 학습 (금속 제외 2-feature)
  - inner-OOF 잔차 분포의 하위 cutoff 분위수를 임계값으로, outer OOF 잔차가 그보다 낮으면 이상 판정
  - 저농도 C80 gate: 관측생존<80 AND 관측농도<C80 AND 잔차<cutoff (fold별) — 10회 중 8회 이상
  - SHAP(부호·dominance)은 보조 진단일 뿐, Primary 필터가 아님
  - 안전장치: 제거율 > max_removal_rate(15%) 시 필터 미적용(chemistry screen only)
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

# ---- 재현성 config 로딩 (pipeline_v4_config.json) ----
CONFIG_PATH = REPO_ROOT / "code" / "pipeline_v4_config.json"


def load_config():
    """pipeline_v4_config.json을 읽어 재현성 파라미터를 반환한다."""
    import json
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}

# mPELQ threshold: 평균 중금속이 PEL의 50% 초과 → 타 독성 기여도 의심
MPELQ_METALS_THRESHOLD = 0.5

# 단위 변환: DB 원본 단위 → ng/g (µg/kg dw)
UNIT_FACTOR = {
    "DDTs": 1.0, "CHLs": 1.0, "PCBs": 1.0, "PAHs": 1.0, "Dieldrin": 1.0,
}


# =============================================================================
# Step 1: DB 정제 (mPELQ 중금속 사전 필터)
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
    for substance in TARGET_SUBSTANCES:
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

    # mPELQ 사전 필터
    substance_dfs = {}
    for substance in TARGET_SUBSTANCES:
        target_col = f"Sum_{substance}_OC_log"
        df_sub = df_base[df_base[target_col].notna()].copy()
        if len(df_sub) == 0:
            continue

        # ★ mPELQ 중금속 사전 필터: 타 독성 기여도 의심 샘플 제거
        # mpelq_threshold 파라미터가 주어지면 민감도 분석용으로 override
        if mpelq_threshold is not None:
            mpelq_thresh = mpelq_threshold
        else:
            mpelq_thresh = float(
                cfg.get("mpelq_metals_threshold", MPELQ_METALS_THRESHOLD)
            )
        n_before_mpelq = len(df_sub)
        # 최소 측정 금속 수 기준: 측정 금속이 min_metals_measured 미만이면 mPELQ 신뢰 불가 → 제거
        min_metals = int(cfg.get("min_metals_measured", 3))
        df_sub = df_sub[
            (df_sub["N_metals_measured"] >= min_metals) &
            (df_sub["mPELQ_Metals"] <= mpelq_thresh)
        ].copy()
        n_after_mpelq = len(df_sub)
        n_mpelq_removed = n_before_mpelq - n_after_mpelq
        print(f"  {substance}: mPELQ 제거 {n_mpelq_removed} → 최종 {len(df_sub):,}지점")

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
# Step 2.5: DRC 품질 평가 → DRC OK 종만 선별
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
    DRC 품질 평가 → DRC OK 종만 선별

    ★ 핵심 변경:
    - DRC 불량 종 → 통째로 제외 (TEL/PEL에 사용하지 않음)
    - DRC OK 종의 저농도 가짜 독성 → DROP 없음 (Primary는 survival 기준 EDS 전부 포함)

    anchor_species: dict {substance: [species, ...]} — Sensitivity 분석에서만
    표준 시험종을 anchor로 강제 추가할 때 사용. Primary에서는 None (강제 추가 없음).
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

    # Sensitivity 전용: anchor species 강제 추가 (Primary에서는 실행 안 됨)
    if anchor_species:
        for sub, anchor_spp in anchor_species.items():
            if sub in selected_species_dict:
                selected_species_dict[sub] = list(
                    set(selected_species_dict[sub] + anchor_spp))
            else:
                selected_species_dict[sub] = list(anchor_spp)

    drc_results = []
    drc_ok_species = {}  # DRC OK 종만 저장
    cleaned_dfs = {}     # DRC OK 종 데이터 (가짜 독성 DROP 없음)

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
                df_pooled = df_pooled[~sp_mask].copy()

        if ok_species_for_substance:
            drc_ok_species[substance] = ok_species_for_substance
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
    print(f"  DRC OK 종만 TEL/PEL 도출에 사용 ★")
    print(f"  가짜 독성 DROP: 중단됨 (Primary는 survival 기준 EDS 전부 포함, 민감도 분석으로 이동)")
    print(f"  물질별 DRC OK 종: {drc_ok_species}")

    return cleaned_dfs, drc_df, drc_ok_species


# =============================================================================
# Step 2.7: 다물질 SHAP 타 독성 기여도 평가 ★신규
# =============================================================================
def step2_7_confounder_filtering(cleaned_dfs, drc_ok_species, source_name=""):
    """[DEPRECATED] 구버전 confounder 필터 — 부호 오류(절대값 비교)로 재사용 금지.

    이 함수는 |SHAP_mPELQ| > |SHAP_target| 절대값 비교로 confounder를 판정해
    부호(방향) 정보를 무시하는 오류가 있었다. v4에서 sign-aware 방식으로
    재설계되어 step2_7_confounder_filtering_v4로 대체되었다.
    재사용을 막기 위해 호출 시 예외를 발생시킨다.
    """
    raise NotImplementedError(
        "step2_7_confounder_filtering은 부호 오류로 폐기되었습니다. "
        "step2_7_confounder_filtering_v4를 사용하세요."
    )


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
                                     n_folds=None, n_seeds=None, stability_threshold=None,
                                     use_group_kfold=False):
    """
    v4 Primary 필터 — nested OOF monotonic DRC + 저농도 C80 gate 기반 이상자료 판정.

    분석 구조:
      - Primary:       nested OOF DRC 잔차 기반 필터 (survival<80 AND NOT stable DRC-inconsistent)
      - Sensitivity 1: residual cutoff 2.5% (더 보수적)
      - Sensitivity 2: residual cutoff 10% (더 관대)
      - Sensitivity 3: chemistry screen only (ML 기록 필터 미적용)
      - Ablation:      금속 포함/제외 모델의 OOF prediction difference (보조 진단)

    반복 CV OOF DRC 잔차:
      - 각 fold: training fold로 금속 제외 2-feature monotonic DRC 모델 적합 → held-out fold 예측
      - 잔차 = 실제 생존율 - DRC 예측 생존율
      - inner-OOF 잔차 분포의 하위 cutoff 분위수를 임계값으로,
        test(OOF) 잔차가 그 임계값보다 낮으면(더 음수) 이상 판정
      - 저농도 C80 gate: 관측생존<80 AND 관측농도<C80 AND 잔차<cutoff (fold별)
      - 80% 안정성: 이상 판정이 n_seeds 중 min_success회 이상 반복되는 기록만 제외
      - 금속·TOC·SHAP은 이상 독성이 금속과 연관되는지 설명하는 보조 진단일 뿐

    재현성: n_folds/n_seeds/stability_threshold가 None이면 pipeline_v4_config.json에서 읽는다.
      - residual cutoff 민감도(2.5%/5%/10%)를 Supplementary에 함께 제시.
      - 제거율이 max_removal_rate 초과 시 DRC 모델 부적합으로 판정, Primary 필터 미적용.

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
    random_seed = int(cfg.get("random_seed", 42))
    # repeat_seeds 목록을 config에서 읽음 (없으면 range(n_seeds)로 대체)
    repeat_seeds = cfg.get("repeat_seeds", None)
    if repeat_seeds is None or len(repeat_seeds) != n_seeds:
        repeat_seeds = list(range(n_seeds))
    else:
        repeat_seeds = [int(s) for s in repeat_seeds]
    print(f"\n{'='*70}")
    print(f"[Step 2.7-v4] nested OOF DRC + C80 gate (진단) — {source_name}")
    print(f"{'='*70}")

    diag_rows = []
    tel_pel_rows = []
    primary_filtered_dfs = {}  # Primary로 선별된 DataFrame (Step 3 입력)

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

        # GroupKFold용 group_id: 행별 우선순위 결합 (station_key → StudyID+Station →
        # sample_key → SampleID → 좌표 → record_id).
        # NOAA의 Station은 고유 정점이 아니므로 좌표로 그룹핑한다.
        groups = _build_group_ids(df_valid)
        group_col = "group_id"  # 진단 기록용 (실제 컬럼은 아님)

        feature_names = [f"{substance}_conc", "mPELQ_Metals", "TOC_pct"]

        # ---- Primary: 화학적 mPELQ screen만 (Step 1에서 이미 적용) ----
        # df_valid는 이미 mPELQ <= 0.5가 적용된 상태. 추가 SHAP 제거 없음.
        primary_telpel = _compute_tel_pel_from_df(df_valid, target_col)

        # ---- 반복 CV SHAP (OOF) ----
        # 시료별 누적 카운터 (대상물질 기준: SHAP_target < 0 = 독성 방향,
        # |SHAP_target| > |SHAP_metal| = 대상물질 dominant)
        n_assignments = np.zeros(n_total, dtype=int)          # OOF 예측 횟수
        target_neg_count_tree = np.zeros(n_total, dtype=int)  # TreeSHAP SHAP_target < 0 횟수
        target_neg_count_inter = np.zeros(n_total, dtype=int) # Interventional SHAP_target < 0 횟수
        target_dom_count_tree = np.zeros(n_total, dtype=int)  # TreeSHAP |target|>|metal| 횟수
        target_dom_count_inter = np.zeros(n_total, dtype=int) # Interventional |target|>|metal| 횟수
        n_inter_assignments = np.zeros(n_total, dtype=int)     # Interventional OOF 성공 횟수
        # ablation: 금속 포함/제외 OOF prediction difference 누적
        pred_with_metal = np.zeros(n_total, dtype=float)
        pred_without_metal = np.zeros(n_total, dtype=float)
        # OOF DRC 잔차 기반 이상자료 판정 (핵심 Primary 필터)
        # residual = 실제 생존율 - DRC(금속 제외 2-feature) 예측 생존율
        # 잔차 < 0 = 실제 생존율이 DRC 예측보다 낮음 = 비정상적으로 낮은 생존율
        residual_sum = np.zeros(n_total, dtype=float)          # OOF 잔차 누적
        # cutoff별 이상 판정 횟수 (민감도 분석: 2.5%/5%/10%)
        residual_anomaly_counts = {c: np.zeros(n_total, dtype=int) for c in residual_cutoffs}
        # 저농도 DRC gate (C80) 보조 카운터 — 고농도 강독성 자료를 노이즈로 오판하는 것 방지
        n_c80_valid = np.zeros(n_total, dtype=int)         # 유효 C80 할당 횟수 (gate 적용 가능 fold)
        c80_sum = np.zeros(n_total, dtype=float)           # C80 누적 (record별 중앙 C80 산출용)
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

                # 3-feature 모델 (금속 포함)
                dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
                model3 = xgb.train({"objective": "reg:squarederror", "eta": 0.1,
                    "max_depth": 4, "seed": random_seed}, dtr, num_boost_round=100)

                # TreeSHAP (path-dependent, 기본)
                dte = xgb.DMatrix(X_te, feature_names=feature_names)
                pred_tree = model3.predict(dte, pred_contribs=True)
                shap_mpelq_tree = pred_tree[:, 1]
                shap_target_tree = pred_tree[:, 0]

                # Interventional SHAP (background = 해당 fold training data)
                # 실패 시 TreeSHAP으로 대체하지 않고 결측 처리한다(성공률 별도 기록).
                n_inter_attempts += 1
                inter_ok = False
                try:
                    explainer = shap.TreeExplainer(
                        model3, X_tr, feature_perturbation="interventional")
                    shap_inter = explainer.shap_values(X_te)
                    shap_mpelq_inter = shap_inter[:, 1]
                    shap_target_inter = shap_inter[:, 0]
                    inter_ok = True
                    n_inter_success += 1
                except Exception as e:
                    print(f"  {substance} seed={seed} fold={fold_id}: "
                          f"Interventional SHAP 실패 ({e}) → 결측 처리")
                    shap_mpelq_inter = np.full(len(test_idx), np.nan)
                    shap_target_inter = np.full(len(test_idx), np.nan)

                # ablation: 금속 제외 2-feature 모델
                X_tr2 = X_tr[:, [0, 2]]
                X_te2 = X_te[:, [0, 2]]
                dtr2 = xgb.DMatrix(X_tr2, label=y_tr,
                                   feature_names=[feature_names[0], feature_names[2]])
                model2 = xgb.train({"objective": "reg:squarederror", "eta": 0.1,
                    "max_depth": 4, "seed": random_seed,
                    "monotone_constraints": "(-1,0)"}, dtr2, num_boost_round=100)
                dte2 = xgb.DMatrix(X_te2, feature_names=[feature_names[0], feature_names[2]])
                pred2 = model2.predict(dte2)
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
                                                  feature_names=[feature_names[0], feature_names[2]])
                            m2_in = xgb.train({"objective": "reg:squarederror", "eta": 0.1,
                                "max_depth": 4, "seed": random_seed,
                                "monotone_constraints": "(-1,0)"}, dtr2_in, num_boost_round=100)
                            dte2_in = xgb.DMatrix(X_tr2[ite_idx],
                                                  feature_names=[feature_names[0], feature_names[2]])
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
                # ---- 저농도 DRC gate (C80) ----
                # C80 = monotonic DRC가 80% survival로 하강하는 농도 (로그 스케일).
                # 노이즈 제거의 목적 = "DRC가 무독성(생존>=80)이라 예측하는 저농도 영역인데
                #   실제로는 독성(생존<80)을 보이는" 독성 기여도가 낮은 자료만 제외.
                # 고농도에서 DRC가 예측한 것보다 생존율이 더 낮은 강독성 자료는
                #   진짜 독성 근거이므로 제외하지 않는다 → conc < C80 gate가 이를 보호.
                # 각 test 기록의 TOC를 고정한 채 training 농도범위에서 예측해 C80 산출.
                feature_names_2 = [feature_names[0], feature_names[2]]
                conc_min_tr = float(X_tr2[:, 0].min())
                conc_max_tr = float(X_tr2[:, 0].max())
                conc_grid = np.linspace(conc_min_tr, conc_max_tr, 200)
                # TOC 값별 C80 캐시 (성능 최적화: 동일 TOC 기록 다수 존재)
                c80_cache = {}
                for k, idx in enumerate(test_idx):
                    residual_sum[idx] += resid_te[k]
                    surv_k = float(y_te[k])           # 관측 생존율
                    conc_k = float(X_te2[k, 0])       # 관측 농도 (로그)
                    toc_k = float(X_te2[k, 1])        # TOC (고정)

                    # C80 산출 (TOC 고정)
                    c80 = c80_cache.get(toc_k)
                    if c80 is None:
                        X_grid = np.column_stack([conc_grid, np.full(len(conc_grid), toc_k)])
                        pred_grid = model2.predict(
                            xgb.DMatrix(X_grid, feature_names=feature_names_2))
                        cross_idx = np.where(pred_grid < 80)[0]
                        if len(cross_idx) == 0 or cross_idx[0] == 0:
                            # crossing 없음 OR 저농도부터 이미 80% 미만(유효 no-effect 영역 없음)
                            # → 해당 TOC에 대해 필터 미적용 (gate 통과 불가)
                            c80 = None
                        else:
                            c80 = float(conc_grid[cross_idx[0]])
                        c80_cache[toc_k] = c80

                    if c80 is not None:
                        n_c80_valid[idx] += 1
                        c80_sum[idx] += c80
                        if conc_k < c80:
                            n_below_c80[idx] += 1

                    for c in residual_cutoffs:
                        resid_cut = np.quantile(resid_inner_oof, c)
                        # 저농도 DRC gate 적용: 관측생존<80 AND 관측농도<C80 AND 잔차<cutoff
                        if (c80 is not None) and (surv_k < 80) and (conc_k < c80) \
                                and (resid_te[k] < resid_cut):
                            residual_anomaly_counts[c][idx] += 1
                        elif (c80 is not None) and (surv_k < 80) and (conc_k >= c80) \
                                and (resid_te[k] < resid_cut):
                            # 고농도(농도>=C80) 강독성 자료 → gate에 차단되어 이상 판정에서 제외
                            # (진단 카운터: 기존 코드라면 노이즈로 오판됐을 후보)
                            n_blocked_high_conc[idx] += 1


                # 누적 + 시료별 원자료 수집
                for k, idx in enumerate(test_idx):
                    n_assignments[idx] += 1
                    if shap_target_tree[k] < 0:
                        target_neg_count_tree[idx] += 1
                    if np.abs(shap_target_tree[k]) > np.abs(shap_mpelq_tree[k]):
                        target_dom_count_tree[idx] += 1
                    # Interventional: 성공한 fold만 누적 (실패 fold는 결측)
                    if inter_ok:
                        n_inter_assignments[idx] += 1
                        if shap_target_inter[k] < 0:
                            target_neg_count_inter[idx] += 1
                        if np.abs(shap_target_inter[k]) > np.abs(shap_mpelq_inter[k]):
                            target_dom_count_inter[idx] += 1
                    pred_with_metal[idx] += pred3[k]
                    pred_without_metal[idx] += pred2[k]

                    # ---- C80 gate 원자료 (fold별 판정 검증용) ----
                    # c80 = 해당 fold·TOC의 C80 (없으면 NaN)
                    # below_c80 = 관측농도 < C80 여부 (fold별)
                    # fold_inconsistent_{cutoff} = 해당 fold에서 이상 판정 여부 (cutoff별)
                    toc_k = float(X_te2[k, 1])
                    c80_k = c80_cache.get(toc_k)  # 첫 루프에서 채워짐
                    conc_k = float(X_te2[k, 0])
                    surv_k = float(y_te[k])
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
                        "is_oof": True,  # test fold = OOF
                        "shap_tree_mpelq": shap_mpelq_tree[k],
                        "shap_interv_mpelq": (shap_mpelq_inter[k] if inter_ok else np.nan),
                        "shap_tree_target": shap_target_tree[k],
                        "shap_interv_target": (shap_target_inter[k] if inter_ok else np.nan),
                        "target_neg_flag": int(shap_target_tree[k] < 0),
                        "target_neg_flag_inter": (int(shap_target_inter[k] < 0) if inter_ok else np.nan),
                        "target_dominance_flag_tree": int(
                            np.abs(shap_target_tree[k]) > np.abs(shap_mpelq_tree[k])),
                        "target_dominance_flag_inter": (
                            int(np.abs(shap_target_inter[k]) > np.abs(shap_mpelq_inter[k]))
                            if inter_ok else np.nan),
                        "pred_with_metal": pred3[k],
                        "pred_without_metal": pred2[k],
                        "pred_diff": pred3[k] - pred2[k],
                        # C80 gate 원자료
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

        # ---- 시료별 빈도 산출 (대상물질 기준) ----
        target_neg_freq_tree = np.divide(target_neg_count_tree, n_assignments,
                                         out=np.zeros(n_total), where=n_assignments > 0)
        target_dom_freq_tree = np.divide(target_dom_count_tree, n_assignments,
                                         out=np.zeros(n_total), where=n_assignments > 0)
        # Interventional: 성공한 fold만 분모로 사용 (실패 fold는 결측)
        target_neg_freq_inter = np.divide(target_neg_count_inter, n_inter_assignments,
                                          out=np.zeros(n_total), where=n_inter_assignments > 0)
        target_dom_freq_inter = np.divide(target_dom_count_inter, n_inter_assignments,
                                          out=np.zeros(n_total), where=n_inter_assignments > 0)
        inter_success_rate = (n_inter_success / n_inter_attempts) if n_inter_attempts > 0 else 0.0

        # 80% 안정성 판정 (대상물질 부호·dominance를 별도 조건으로, attribution 방법별 독립)
        # 절대 카운트 기준: n_seeds 중 최소 ceil(threshold * n_seeds)회 조건을 만족해야 안정.
        # (성공 fold 비율로 나누면 2/2 성공 시 100%로 오판 → 절대 횟수 기준 필수)
        min_success = math.ceil(stability_threshold * n_seeds)
        # 대상물질 기여 안정성 (각 attribution 방법별 독립) — 보조 진단으로만 사용
        stable_target_tree = (
            (target_neg_count_tree >= min_success)
            & (target_dom_count_tree >= min_success)
        )
        stable_target_inter = (
            (n_inter_assignments >= min_success)
            & (target_neg_count_inter >= min_success)
            & (target_dom_count_inter >= min_success)
        )

        # ---- 최종 분류 (사용자 확정 구조: OOF DRC 잔차 기반) ----
        # survival < 80 → EDS 후보
        # survival >= 80 → NEDS
        # DRC-inconsistent toxicity = 실제 생존율이 DRC 예측보다 비정상적으로 낮은
        #   (잔차 < training 하위 cutoff) 판정이 10회 중 8회 이상 반복되는 기록 → 제외
        # EDS = survival < 80 AND NOT stable DRC-inconsistent toxicity
        survival = df_valid["Mean_Survival"].values
        is_eds_raw = survival < 80
        is_neds = survival >= 80
        # DRC-inconsistent toxicity (안정적 이상 판정: min_success회 이상)
        drc_inconsistent = residual_anomaly_counts[residual_cutoff] >= min_success
        # Primary: DRC 잔차 기반 필터 (SHAP 부호/dominance는 보조 진단일 뿐)
        target_supported_eds = is_eds_raw & ~drc_inconsistent
        # DRC-inconsistent toxicity (제외 대상, Noise로 단정하지 않음)
        uncertain_eds = is_eds_raw & drc_inconsistent

        # ---- 안전장치 + Primary 제외 마스크 (여기서 한 번만 정의) ----
        # 제거율 = EDS 중 DRC-inconsistent로 제외된 비율
        removal_rate = (uncertain_eds.sum() / (target_supported_eds.sum() + uncertain_eds.sum())) \
            if (target_supported_eds.sum() + uncertain_eds.sum()) > 0 else 0.0
        # 안전장치: 제거율이 max_removal_rate 초과 시 DRC 모델 부적합으로 판정
        filter_applied = removal_rate <= max_removal_rate
        # Primary 제외 마스크: 필터가 실제 적용된 경우에만 drc_inconsistent를 제외자료로 취급.
        # filter_applied=False(안전장치 발동) 시에는 drc_inconsistent가 "후보"일 뿐
        # 실제 제외자료가 아니므로, 최종 EDS 수·Data_Type·Fig.2 회색 ×는 반드시
        # primary_excluded를 사용해야 한다. drc_inconsistent는 후보 수로 별도 저장.
        primary_excluded = drc_inconsistent if filter_applied \
            else np.zeros(n_total, dtype=bool)

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
        # ★ 실제 제외 = primary_excluded, 후보 = drc_inconsistent (uncertain_eds)로 분리
        excluded_conc_log = conc_obs[primary_excluded]      # 실제 제외자료
        candidate_conc_log = conc_obs[uncertain_eds]        # 후보자료 (제외 여부 무관)
        retained_conc_log = conc_obs[target_supported_eds]  # 유지 EDS
        excl_med_orig = float(np.median(10 ** excluded_conc_log - 1)) if len(excluded_conc_log) > 0 else np.nan
        cand_med_orig = float(np.median(10 ** candidate_conc_log - 1)) if len(candidate_conc_log) > 0 else np.nan
        ret_med_orig = float(np.median(10 ** retained_conc_log - 1)) if len(retained_conc_log) > 0 else np.nan

        # ---- Primary 선별 DataFrame (Step 3 입력용) ----
        # Primary = NEDS 전체 + (EDS 중 primary_excluded가 아닌 것)
        # primary_excluded는 filter_applied=False 시 전부 False이므로
        # 자동으로 chemistry screen only(is_neds | is_eds_raw)로 복귀한다.
        primary_keep = is_neds | (is_eds_raw & ~primary_excluded)
        primary_filtered_dfs[substance] = df_valid[primary_keep].copy()

        # ---- Sensitivity 구성 (residual cutoff 민감도) ----
        # Sens1: cutoff 2.5% (더 보수적, 이상 판정 더 적음)
        # Sens2: cutoff 10% (더 관대, 이상 판정 더 많음)
        # Sens3: ML 기록 필터 미적용, chemistry screen only (is_eds_raw)
        sens1_eds = is_eds_raw & ~(residual_anomaly_counts[0.025] >= min_success)
        sens2_eds = is_eds_raw & ~(residual_anomaly_counts[0.10] >= min_success)
        sens3_eds = is_eds_raw  # chemistry screen only

        # ---- Primary TEL/PEL (DRC 잔차 기반 필터, primary_excluded 기준) ----
        primary_df = df_valid[primary_keep].reset_index(drop=True)
        primary_telpel = _compute_tel_pel_from_df(primary_df, target_col)
        # Sens1/Sens2/Sens3 TEL/PEL
        sens1_df = df_valid[sens1_eds | is_neds].reset_index(drop=True)
        sens2_df = df_valid[sens2_eds | is_neds].reset_index(drop=True)
        sens3_df = df_valid[sens3_eds | is_neds].reset_index(drop=True)
        sens1_telpel = _compute_tel_pel_from_df(sens1_df, target_col)
        sens2_telpel = _compute_tel_pel_from_df(sens2_df, target_col)
        sens3_telpel = _compute_tel_pel_from_df(sens3_df, target_col)

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

        # EDS/NEDS별 ablation 분포 (Primary 분류 기준: target-supported EDS vs NEDS)
        eds_ablation = pred_diff[target_supported_eds]
        neds_ablation = pred_diff[is_neds]
        mean_ablation_eds = float(np.mean(eds_ablation)) if len(eds_ablation) > 0 else np.nan
        mean_ablation_neds = float(np.mean(neds_ablation)) if len(neds_ablation) > 0 else np.nan

        # ---- 결과 기록 ----
        n_eds_supported = int(target_supported_eds.sum())
        n_eds_uncertain = int(uncertain_eds.sum())
        n_neds = int(is_neds.sum())
        n_neg_majority = int((target_neg_freq_tree >= stability_threshold).sum())
        # 실제 Primary EDS 수 (primary_excluded 기준 — filter_applied=False 시 chemistry screen only)
        n_eds_primary = int((is_eds_raw & ~primary_excluded).sum())
        n_excluded_primary = int(primary_excluded.sum())
        # (removal_rate / filter_applied / primary_excluded는 위에서 이미 정의됨 — 중복 정의 제거)
        print(f"\n  {substance}: N={n_total}")
        print(f"    Primary (DRC 잔차 필터, cutoff={residual_cutoff:.0%}): "
              f"EDS={n_eds_primary} NEDS={n_neds} "
              f"제외(primary_excluded)={n_excluded_primary} "
              f"(제거율 {removal_rate:.1%}) → "
              f"TEL={primary_telpel['TEL']:.4f} PEL={primary_telpel['PEL']:.4f}" if primary_telpel else
              f"    Primary: TEL/PEL 계산 불가")
        if not filter_applied:
            print(f"    ⚠️ 제거율 {removal_rate:.1%} > {max_removal_rate:.0%} → "
                  f"DRC 모델 부적합 판정, Primary 필터 미적용(chemistry screen only) "
                  f"[drc_inconsistent 후보 {n_eds_uncertain}건은 제외하지 않음]")
        print(f"    Sens1 (cutoff 2.5%): EDS={int(sens1_eds.sum())} → "
              f"TEL={sens1_telpel['TEL']:.4f} PEL={sens1_telpel['PEL']:.4f}" if sens1_telpel else
              f"    Sens1: TEL/PEL 계산 불가")
        print(f"    Sens2 (cutoff 10%): EDS={int(sens2_eds.sum())} → "
              f"TEL={sens2_telpel['TEL']:.4f} PEL={sens2_telpel['PEL']:.4f}" if sens2_telpel else
              f"    Sens2: TEL/PEL 계산 불가")
        print(f"    Sens3 (chemistry screen only): EDS={int(sens3_eds.sum())} → "
              f"TEL={sens3_telpel['TEL']:.4f} PEL={sens3_telpel['PEL']:.4f}" if sens3_telpel else
              f"    Sens3: TEL/PEL 계산 불가")
        print(f"    대상물질 부호 안정(SHAP_target<0 ≥{stability_threshold:.0%}): {n_neg_majority} 시료 (보조 진단)")
        print(f"    Ablation: 금속 포함 시 OOF 예측 평균 변화 {mean_ablation:+.4f} "
              f"(음수 {n_ablation_negative} 시료)")
        print(f"    C80 gate 검증: 후보 {n_drc_inconsistent}건 중 fold별 below_c80≥{min_success}회 {c80_below_count} / "
              f"미달 {c80_above_count} (유효 C80 {n_valid_c80}건) "
              f"→ 실제 제외 EDS 농도 중앙값 {excl_med_orig:.4f} vs 유지 EDS {ret_med_orig:.4f} µg/kg (1% TOC 정규화)")

        diag_rows.append({
            "Source": source_name, "Substance": substance, "N_total": n_total,
            "group_col": group_col,
            "n_groups": int(len(np.unique(groups))) if groups is not None else 0,
            "group_kfold_train_test_overlap": int(group_overlap_count),
            "n_unique_fold_splits": int(n_unique_splits),
            "N_target_neg_freq_ge_threshold": n_neg_majority,
            "N_EDS_supported": n_eds_supported,
            "N_EDS_uncertain": n_eds_uncertain,
            "N_EDS_primary": n_eds_primary,
            "N_excluded_primary": n_excluded_primary,
            "N_NEDS": n_neds,
            "residual_cutoff": residual_cutoff,
            "removal_rate": round(removal_rate, 4),
            "filter_applied": filter_applied,
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
            "N_valid_C80_assignments": n_valid_c80,
            "N_DRC_inconsistent": n_drc_inconsistent,
            "Excluded_EDS_Conc_Median_OC1pct": (round(excl_med_orig, 4) if not np.isnan(excl_med_orig) else None),
            "Candidate_EDS_Conc_Median_OC1pct": (round(cand_med_orig, 4) if not np.isnan(cand_med_orig) else None),
            "Retained_EDS_Conc_Median_OC1pct": (round(ret_med_orig, 4) if not np.isnan(ret_med_orig) else None),
        })

        for label, telpel in [("Primary", primary_telpel),
                              ("Sensitivity1_ResidCut2.5pct", sens1_telpel),
                              ("Sensitivity2_ResidCut10pct", sens2_telpel),
                              ("Sensitivity3_ChemOnly", sens3_telpel)]:
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

        # Fig. 2 재설계용: OOF DRC 잔차 + primary_excluded flag 저장 (필터 전 전체 기록)
        # residual_mean = OOF 잔차 평균 (실제 생존율 - DRC 예측 생존율)
        # drc_inconsistent = Primary cutoff 기준 안정적 이상 판정 (제외 "후보")
        # primary_excluded = 실제 제외자료 (filter_applied=True일 때만 drc_inconsistent와 동일,
        #   filter_applied=False(안전장치 발동) 시에는 전부 False → 제외자료 없음)
        residual_mean = np.divide(residual_sum, n_assignments,
                                  out=np.full(n_total, np.nan), where=n_assignments > 0)
        drc_resid_df = df_valid.copy()
        drc_resid_df["residual_mean"] = residual_mean
        drc_resid_df["drc_inconsistent"] = drc_inconsistent.astype(int)
        drc_resid_df["primary_excluded"] = primary_excluded.astype(int)
        drc_resid_df["filter_applied"] = int(filter_applied)
        drc_resid_df["is_eds_raw"] = is_eds_raw.astype(int)
        # Data_Type은 반드시 primary_excluded 기준 (실제 분석과 일치)
        drc_resid_df["Data_Type"] = np.where(
            is_neds, "NEDS",
            np.where(primary_excluded, "DRC-inconsistent", "EDS"))
        drc_resid_df.to_csv(
            OUTPUT_DIR / f"Step2_7v4_DRC_Residual_{source_name}_{substance}.csv",
            index=False)

        # fold signature CSV 저장 (10개 반복의 분할 차이 검증용)
        if fold_signatures:
            fs_df = pd.DataFrame(fold_signatures)
            fs_df.to_csv(
                OUTPUT_DIR / f"Step2_7v4_FoldSignatures_{source_name}_{substance}.csv",
                index=False)

    diag_df = pd.DataFrame(diag_rows)
    telpel_df = pd.DataFrame(tel_pel_rows)

    # ---- OOF 반복 구조 메타데이터 (재현성·검증용) ----
    # dominance_ratio: |SHAP_metal| > ratio * |SHAP_target| (config에서 읽음, 기본 1.0)
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

    return primary_filtered_dfs, diag_df, telpel_df


# =============================================================================
# Step 3: EDS/NEDS → TEL/PEL (DRC OK 종만 사용)
# =============================================================================
def step3_tel_pel(filtered_dfs, drc_ok_species, source_name=""):
    print(f"\n{'='*70}")
    print(f"[Step 3] EDS/NEDS → TEL/PEL — {source_name}")
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
                "N_DRC_OK_Species": len(ok_spp),
                "DRC_OK_Species": "; ".join(ok_spp),
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
                                    "Month", "Year"] if c in df_pooled.columns]
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

        # --- 5-fold CV AUC ---
        cv_auc_mean = np.nan
        cv_auc_std = np.nan
        if n_pos >= 10 and n_neg >= 10:
            try:
                cv_scores = []
                cv = min(5, n_pos, n_neg)
                if cv >= 2:
                    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_seed)
                    for train_idx, test_idx in skf.split(conc_log, y_true):
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
    cfg = load_config()
    group_cfg = cfg.get("group_kfold", {})
    group_kfold_mode = (
        "--group-kfold" in sys.argv
        or group_cfg.get("enabled", True)  # 기본값 True: config가 없어도 GroupKFold 사용
    )
    print("=" * 70)
    print("고도화 파이프라인 v4: 신뢰도 기반 TEL/PEL 도출")
    print("  Step 1: DB 정제 (mPELQ 필터)")
    print("  Step 2: XGBoost SHAP 종 선별")
    print("  Step 2.5: DRC 평가 → DRC OK 종만 선별")
    print("  Step 2.7: nested OOF monotonic DRC + C80 gate (진단/Sensitivity)")
    print("  Step 3: EDS/NEDS → TEL/PEL")
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
            primary_filtered_dfs, diag_df, telpel_df = step2_7_confounder_filtering_v4(
                cleaned_dfs, drc_ok_species, source_name,
                use_group_kfold=group_kfold_mode)
            if len(diag_df) > 0:
                all_diag.append(diag_df)
            if len(telpel_df) > 0:
                all_telpel.append(telpel_df)
            continue

        # ★ v4 통합: Step 2.7 nested OOF DRC + C80 gate → Step 3 empirical TEL/PEL
        #   Primary = NEDS 전체 + (EDS 중 primary_excluded가 아닌 것)
        primary_filtered_dfs, diag_df, telpel_df = step2_7_confounder_filtering_v4(
            cleaned_dfs, drc_ok_species, source_name,
            use_group_kfold=group_kfold_mode)
        if len(diag_df) > 0:
            all_diag.append(diag_df)
        if len(telpel_df) > 0:
            all_telpel.append(telpel_df)

        final, edsneds_data = step3_tel_pel(primary_filtered_dfs, drc_ok_species, source_name)
        reliability, edsneds_data = step4_reliability_assessment(final, edsneds_data, source_name)

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