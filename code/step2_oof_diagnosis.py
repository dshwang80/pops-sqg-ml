#!/usr/bin/env python3
"""
Step 2 OOF 진단 스크립트 (별도 실행 — run_pipeline_v4.py 수정 없음)
====================================================================

목적:
  Step 2 종 선별은 현재 in-sample SHAP으로 수행된다. 이는 최종 포함 생물종을
  결정하는 선택 단계이므로, in-sample SHAP이 최종 TEL/PEL에 영향을 주는
  순환성(circularity) 문제를 리뷰어가 지적할 가능성이 있다.

  본 스크립트는 Step 2 종 선별을 OOF(out-of-fold) SHAP으로 재계산하여,
  기존 in-sample 결과와 비교한다. 결과가 거의 같으면 OOF 종 선별을 Primary에
  통합하고, 차이가 크면 원인 확인 후 sensitivity로 둘지 결정한다.

방법 (사용자 지시 verbatim):
  - 종별 5-fold × 10회 GroupKFold OOF SHAP 계산
  - test fold만 SHAP 계산, KFold 대체 금지
  - 반복별 기존 조건 적용:
      Chem_Direction ≤ −0.30
      TOC importance ≤ 1.5 × chemical importance
  - 10회 중 8회 이상 충족 시 OOF-supported species
  - 기존 종 선별과 비교: 포함·제외 종 목록, 선택 안정성, EDS/NEDS 수, TEL/PEL 변화

주의:
  - run_pipeline_v4.py는 수정하지 않는다. (cf5d2e8 상태 보존)
  - Primary 교체는 하지 않는다. 진단 결과 검토 후 통합 여부 결정.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# run_pipeline_v4.py의 상수·함수 재사용 (Step 1 DB 정제, config, 경로)
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

import run_pipeline_v4 as P  # noqa: E402

OUTPUT_DIR = P.OUTPUT_DIR / "step2_oof_diagnosis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_REPEATS = 10
N_FOLDS = 5
MIN_SUCCESS = 8  # 10회 중 8회 이상 충족 시 OOF-supported

# 기존 Step 2 조건 (run_pipeline_v4.py와 동일)
CHEM_DIRECTION_THRESHOLD = -0.30
TOC_IMPORTANCE_RATIO = 1.5


def get_group_col(df):
    """GroupKFold용 group_id를 행별 우선순위 결합으로 생성한다 (run_pipeline_v4.py와 동일).

    우선순위: station_key → sample_key → StudyID+Station → SampleID → 좌표 → record_id.
    NOAA의 Station은 고유 정점이 아니므로 좌표(round4)로 그룹핑한다.
    """
    n = len(df)
    if "record_id" not in df.columns:
        df = df.copy()
        df["record_id"] = [f"r{i:06d}" for i in range(n)]
    rec_ids = df["record_id"].astype(str).values
    groups = rec_ids.copy()

    if "Start_Latitude" in df.columns and "Start_Longitude" in df.columns:
        lat = df["Start_Latitude"]
        lon = df["Start_Longitude"]
        mask = lat.notna().values & lon.notna().values
        if mask.any():
            groups[mask] = ("coord:" + lat[mask].round(4).astype(str).values
                            + "|" + lon[mask].round(4).astype(str).values)
    if "SampleID" in df.columns:
        vals = df["SampleID"]
        mask = vals.notna().values
        if mask.any():
            groups[mask] = "sid:" + vals[mask].astype(str).values
    if "StudyID" in df.columns and "Station" in df.columns:
        sid = df["StudyID"]
        st = df["Station"]
        mask = sid.notna().values & st.notna().values
        if mask.any():
            groups[mask] = ("ss:" + sid[mask].astype(str).values
                            + "|" + st[mask].astype(str).values)
    if "sample_key" in df.columns:
        vals = df["sample_key"]
        mask = vals.notna().values
        if mask.any():
            groups[mask] = "sak:" + vals[mask].astype(str).values
    if "station_key" in df.columns:
        vals = df["station_key"]
        mask = vals.notna().values
        if mask.any():
            groups[mask] = "sk:" + vals[mask].astype(str).values
    return groups


def oof_shap_species_selection(substance_dfs, source_name=""):
    """
    종별 5-fold × 10회 GroupKFold OOF SHAP 계산.

    각 반복에서:
      - GroupKFold로 5-fold 분할 (KFold 대체 금지)
      - train fold로 XGBoost 학습, test fold만 SHAP(pred_contribs) 계산
      - test fold SHAP을 모아 종 전체의 Chem_Direction / TOC importance 산출
      - 조건 충족 여부 판정
    10회 중 8회 이상 충족 시 OOF-supported species.

    반환: df_eval_oof (기존 df_eval과 동일 스키마 + OOF 통계 컬럼)
    """
    print(f"\n{'='*70}")
    print(f"[Step 2 OOF 진단] 종별 5-fold × {N_REPEATS}회 GroupKFold OOF SHAP — {source_name}")
    print(f"{'='*70}")

    cfg = P.load_config()
    random_seed = int(cfg.get("random_seed", P.RANDOM_SEED))

    eval_list = []
    for substance in P.TARGET_SUBSTANCES:
        if substance not in substance_dfs:
            continue
        df_sub = substance_dfs[substance].rename(
            columns={f"Sum_{substance}_OC_log": "Conc_log"}
        )
        for sp in df_sub["Latin_Name_Species"].unique():
            df_sp = df_sub[df_sub["Latin_Name_Species"] == sp]
            total_n = len(df_sp)
            toxic_n = int((df_sp["Mean_Survival"] < 80).sum())

            # 기존 Step 2와 동일한 최소 표본 조건
            if total_n < 30 or toxic_n < 10:
                eval_list.append({
                    "Substance": substance, "Species": sp,
                    "Total_N": total_n, "Toxic_N": toxic_n,
                    "Chem_Importance": np.nan, "TOC_Importance": np.nan,
                    "Chem_Direction": np.nan,
                    "OOF_Chem_Importance": np.nan, "OOF_TOC_Importance": np.nan,
                    "OOF_Chem_Direction": np.nan,
                    "OOF_Success_Count": 0, "OOF_Supported": False,
                    "Status": "Excluded (Insufficient)",
                })
                continue

            X = df_sp[["Conc_log", "TOC_pct"]].values.astype(np.float64)
            y = df_sp["Mean_Survival"].values.astype(np.float64)
            if np.any(np.isnan(X)) or np.any(np.isinf(X)) or np.any(np.isnan(y)):
                eval_list.append({
                    "Substance": substance, "Species": sp,
                    "Total_N": total_n, "Toxic_N": toxic_n,
                    "Chem_Importance": np.nan, "TOC_Importance": np.nan,
                    "Chem_Direction": np.nan,
                    "OOF_Chem_Importance": np.nan, "OOF_TOC_Importance": np.nan,
                    "OOF_Chem_Direction": np.nan,
                    "OOF_Success_Count": 0, "OOF_Supported": False,
                    "Status": "Excluded (Data Quality)",
                })
                continue

            # ---- 기존 in-sample SHAP (비교용) ----
            try:
                dtrain = xgb.DMatrix(X, label=y, feature_names=["Conc_log", "TOC_pct"])
                model = xgb.train(
                    {"objective": "reg:squarederror", "eta": 0.1,
                     "max_depth": 3, "seed": random_seed},
                    dtrain, num_boost_round=100,
                )
                pred = model.predict(dtrain, pred_contribs=True)
                imp_chem = np.mean(np.abs(pred[:, 0]))
                imp_toc = np.mean(np.abs(pred[:, 1]))
                cor_chem = stats.spearmanr(X[:, 0], pred[:, 0]).correlation
                if np.isnan(cor_chem):
                    cor_chem = 0.0
            except Exception:
                imp_chem = imp_toc = cor_chem = np.nan

            # ---- OOF SHAP (5-fold × 10회 GroupKFold) ----
            # 행별 우선순위 결합으로 group_id 생성 (station_key → sample_key →
            # StudyID+Station → SampleID → 좌표 → record_id).
            groups = get_group_col(df_sp)
            n_groups = len(np.unique(groups))

            # GroupKFold는 n_splits <= n_groups 필요. 불가능하면 OOF 불가(결측 처리).
            if n_groups < N_FOLDS:
                eval_list.append({
                    "Substance": substance, "Species": sp,
                    "Total_N": total_n, "Toxic_N": toxic_n,
                    "Chem_Importance": imp_chem, "TOC_Importance": imp_toc,
                    "Chem_Direction": cor_chem,
                    "OOF_Chem_Importance": np.nan, "OOF_TOC_Importance": np.nan,
                    "OOF_Chem_Direction": np.nan,
                    "OOF_Success_Count": 0, "OOF_Supported": False,
                    "Status": "Excluded (OOF Infeasible: groups < folds)",
                })
                continue

            success_count = 0
            oof_imp_chem_list = []
            oof_imp_toc_list = []
            oof_cor_chem_list = []

            for rep in range(N_REPEATS):
                gkf = GroupKFold(n_splits=N_FOLDS)
                # KFold 대체 금지: GroupKFold만 사용
                oof_shap_chem = np.full(len(X), np.nan)
                oof_shap_toc = np.full(len(X), np.nan)
                try:
                    for train_idx, test_idx in gkf.split(X, y, groups):
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

                valid = ~np.isnan(oof_shap_chem)
                if valid.sum() == 0:
                    continue

                oof_imp_chem = np.mean(np.abs(oof_shap_chem[valid]))
                oof_imp_toc = np.mean(np.abs(oof_shap_toc[valid]))
                oof_cor_chem = stats.spearmanr(
                    X[valid, 0], oof_shap_chem[valid]
                ).correlation
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

            oof_imp_chem_mean = (
                float(np.mean(oof_imp_chem_list)) if oof_imp_chem_list else np.nan
            )
            oof_imp_toc_mean = (
                float(np.mean(oof_imp_toc_list)) if oof_imp_toc_list else np.nan
            )
            oof_cor_chem_mean = (
                float(np.mean(oof_cor_chem_list)) if oof_cor_chem_list else np.nan
            )

            oof_supported = success_count >= MIN_SUCCESS

            # 기존 in-sample Status (비교용)
            status_insample = "Selected (Concentration-response-supported)"
            if np.isnan(cor_chem):
                status_insample = "Excluded (Error)"
            elif cor_chem > CHEM_DIRECTION_THRESHOLD:
                status_insample = "Excluded (Non-toxic Trend)"
            elif imp_toc > (imp_chem * TOC_IMPORTANCE_RATIO):
                status_insample = "Excluded (TOC Dominated)"

            eval_list.append({
                "Substance": substance, "Species": sp,
                "Total_N": total_n, "Toxic_N": toxic_n,
                "Chem_Importance": imp_chem, "TOC_Importance": imp_toc,
                "Chem_Direction": cor_chem,
                "OOF_Chem_Importance": oof_imp_chem_mean,
                "OOF_TOC_Importance": oof_imp_toc_mean,
                "OOF_Chem_Direction": oof_cor_chem_mean,
                "OOF_Success_Count": success_count,
                "OOF_Supported": oof_supported,
                "Status": status_insample,
            })

    df_eval = pd.DataFrame(eval_list)
    return df_eval


def compare_selection(df_eval_insample, df_eval_oof, source_name):
    """기존 in-sample 종 선별 vs OOF 종 선별 비교 요약."""
    print(f"\n{'='*70}")
    print(f"[비교] in-sample vs OOF 종 선별 — {source_name}")
    print(f"{'='*70}")

    insample_selected = set(
        df_eval_insample.loc[
            df_eval_insample["Status"].str.contains("Selected", na=False),
            ["Substance", "Species"],
        ].itertuples(index=False, name=None)
    )
    oof_selected = set(
        df_eval_oof.loc[
            df_eval_oof["OOF_Supported"] == True,  # noqa: E712
            ["Substance", "Species"],
        ].itertuples(index=False, name=None)
    )

    both = insample_selected & oof_selected
    only_insample = insample_selected - oof_selected
    only_oof = oof_selected - insample_selected

    print(f"\n  in-sample Selected: {len(insample_selected)}종")
    print(f"  OOF-supported:      {len(oof_selected)}종")
    print(f"  공통(둘 다 포함):   {len(both)}종")
    print(f"  in-sample만 포함:   {len(only_insample)}종")
    print(f"  OOF만 포함:         {len(only_oof)}종")

    if only_insample:
        print(f"\n  [in-sample만 포함 (OOF에서 탈락)]:")
        for sub, sp in sorted(only_insample):
            row = df_eval_oof[
                (df_eval_oof["Substance"] == sub) & (df_eval_oof["Species"] == sp)
            ].iloc[0]
            print(f"    {sub:8s} {sp:40s} "
                  f"OOF_Success={row['OOF_Success_Count']}/10 "
                  f"OOF_dir={row['OOF_Chem_Direction']:.3f} "
                  f"OOF_impTOC/impChem={row['OOF_TOC_Importance']/row['OOF_Chem_Importance']:.2f}")
    if only_oof:
        print(f"\n  [OOF만 포함 (in-sample에서 탈락)]:")
        for sub, sp in sorted(only_oof):
            row = df_eval_oof[
                (df_eval_oof["Substance"] == sub) & (df_eval_oof["Species"] == sp)
            ].iloc[0]
            print(f"    {sub:8s} {sp:40s} "
                  f"OOF_Success={row['OOF_Success_Count']}/10 "
                  f"OOF_dir={row['OOF_Chem_Direction']:.3f}")

    return {
        "Source": source_name,
        "N_insample_selected": len(insample_selected),
        "N_oof_supported": len(oof_selected),
        "N_both": len(both),
        "N_only_insample": len(only_insample),
        "N_only_oof": len(only_oof),
    }


def main():
    sources = [
        ("NOAA", P.NOAA_DB_PATH),
        ("SCCWRP", P.SCCWRP_DB_PATH),
        ("Integrated", P.INTEGRATED_DB_PATH),
    ]

    all_compare = []
    all_eval_oof = []

    for source_name, db_path in sources:
        print(f"\n{'#'*70}")
        print(f"# 소스: {source_name} — {db_path.name}")
        print(f"{'#'*70}")

        df_raw = pd.read_csv(db_path, low_memory=False)
        df_base, substance_dfs = P.step1_db_curation(df_raw, source_name)

        # 기존 in-sample 종 선별 (run_pipeline_v4.py의 함수 재사용)
        df_eval_insample = P.step2_species_selection(substance_dfs, source_name)

        # OOF 종 선별 (본 스크립트)
        df_eval_oof = oof_shap_species_selection(substance_dfs, source_name)

        # 저장
        df_eval_oof.to_csv(
            OUTPUT_DIR / f"Step2_OOF_Species_Eval_{source_name}.csv", index=False
        )

        # 비교
        cmp = compare_selection(df_eval_insample, df_eval_oof, source_name)
        all_compare.append(cmp)
        all_eval_oof.append(df_eval_oof)

    # 전체 비교 요약 저장
    df_compare = pd.DataFrame(all_compare)
    df_compare.to_csv(OUTPUT_DIR / "Step2_OOF_Comparison_Summary.csv", index=False)
    print(f"\n{'='*70}")
    print("[전체 비교 요약]")
    print(f"{'='*70}")
    print(df_compare.to_string(index=False))

    # OOF-supported 종 목록 저장 (Primary 통합 검토용)
    all_oof = pd.concat(all_eval_oof, ignore_index=True)
    oof_supported = all_oof[all_oof["OOF_Supported"] == True]  # noqa: E712
    oof_supported.to_csv(
        OUTPUT_DIR / "Step2_OOF_Supported_Species.csv", index=False
    )
    print(f"\nOOF-supported 종 수: {len(oof_supported)}")
    print(f"결과 저장: {OUTPUT_DIR}")
    print("\n완료 ✅")


if __name__ == "__main__":
    main()
