#!/usr/bin/env python3
"""
진단 스크립트: Metal-SHAP 부호 오류 실증 검증
================================================

목적: run_pipeline_v3.py Step 2.7의 confounder 필터에서
      `np.abs(shap_mpelq) > ratio * np.abs(shap_target)` (절댓값 비교)가
      실제로 SHAP_mPELQ > 0 (금속이 생존율을 *높이는* 방향)인 샘플을
      얼마나 제거하는지 실증적으로 확인.

★ 파이프라인 재실행 아님: step1 → step2 → step2.5까지만 실행해
  cleaned_dfs / drc_ok_species를 얻고, step2.7의 SHAP 계산만 재현.
  step3(TEL/PEL) / step4(신뢰도)는 실행하지 않음 → 기존 결과 불변.

산출:
  - 각 물질별 SHAP_mPELQ 부호 분포 (양수/음수/0 개수)
  - 현재 필터(절댓값) 제거 수 vs 부호 조건 추가 시 제거 수
  - SHAP_mPELQ > 0 이면서 confounder-dominated로 제거되는 샘플 수
"""

import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

# run_pipeline_v3.py import (같은 디렉토리)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_pipeline_v3 as P

RANDOM_SEED = P.RANDOM_SEED
TARGET_SUBSTANCES = P.TARGET_SUBSTANCES

sources = [
    ("NOAA", P.NOAA_DB_PATH),
    ("SCCWRP", P.SCCWRP_DB_PATH),
    ("Integrated", P.INTEGRATED_DB_PATH),
]

print("=" * 80)
print("Metal-SHAP 부호 오류 실증 진단")
print("=" * 80)

for source_name, db_path in sources:
    print(f"\n{'#'*80}")
    print(f"# 소스: {source_name}")
    print(f"{'#'*80}")

    df_raw = pd.read_csv(db_path, low_memory=False)
    df_base, substance_dfs = P.step1_db_curation(df_raw, source_name)
    df_eval = P.step2_species_selection(substance_dfs, source_name)
    cleaned_dfs, drc_df, drc_ok_species = P.step2_5_drc_evaluation(
        substance_dfs, df_eval, source_name)

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
            continue

        for possible_col in [f"Sum_{substance}_OC_log", "Conc_log"]:
            if possible_col in df_pooled.columns:
                if possible_col != f"Sum_{substance}_OC_log":
                    df_pooled = df_pooled.rename(
                        columns={possible_col: f"Sum_{substance}_OC_log"})
                break

        target_col = f"Sum_{substance}_OC_log"
        X = df_pooled[[target_col, "mPELQ_Metals", "TOC_pct"]].values.astype(np.float64)
        y = df_pooled["Mean_Survival"].values.astype(np.float64)

        valid_mask = ~np.any(np.isnan(X) | np.isinf(X), axis=1)
        if valid_mask.sum() < 15:
            continue

        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        df_valid = df_pooled[valid_mask].copy()

        feature_names = [f"{substance}_conc", "mPELQ_Metals", "TOC_pct"]

        dtrain = xgb.DMatrix(X_valid, label=y_valid, feature_names=feature_names)
        model = xgb.train({"objective": "reg:squarederror", "eta": 0.1,
                           "max_depth": 4, "seed": RANDOM_SEED},
                          dtrain, num_boost_round=100)
        pred = model.predict(dtrain, pred_contribs=True)

        shap_target = pred[:, 0]
        shap_mpelq = pred[:, 1]

        # --- 부호 분포 ---
        n_pos = int((shap_mpelq > 0).sum())   # 금속이 생존율을 높이는 방향
        n_neg = int((shap_mpelq < 0).sum())   # 금속이 생존율을 낮추는 방향
        n_zero = int((shap_mpelq == 0).sum())
        n_total = len(shap_mpelq)

        # --- 현재 필터 (절댓값) ---
        confounder_ratio = 1.0
        cur_dominated = np.abs(shap_mpelq) > confounder_ratio * np.abs(shap_target)

        # --- 부호 조건 추가 필터 (리뷰어 제안) ---
        # SHAP_mPELQ < 0 AND |SHAP_mPELQ| > ratio*|SHAP_target|
        sign_dominated = (shap_mpelq < 0) & \
                         (np.abs(shap_mpelq) > confounder_ratio * np.abs(shap_target))

        n_cur = int(cur_dominated.sum())
        n_sign = int(sign_dominated.sum())

        # --- SHAP_mPELQ > 0 이면서 현재 필터로 제거되는 샘플 (부호 오류의 실질 피해) ---
        false_removed = cur_dominated & (shap_mpelq > 0)
        n_false = int(false_removed.sum())

        # --- SHAP_mPELQ > 0 이면서 |SHAP_mPELQ| > |SHAP_target| 인 샘플 (방향 무관 지배) ---
        pos_dominated = (shap_mpelq > 0) & \
                        (np.abs(shap_mpelq) > confounder_ratio * np.abs(shap_target))
        n_pos_dominated = int(pos_dominated.sum())

        print(f"\n  [{substance}]  N_total={n_total}")
        print(f"    SHAP_mPELQ 부호 분포:  양수(+)={n_pos} ({n_pos/n_total*100:.1f}%)  "
              f"음수(-)={n_neg} ({n_neg/n_total*100:.1f}%)  영(0)={n_zero}")
        print(f"    현재 필터(절댓값) 제거: {n_cur} ({n_cur/n_total*100:.1f}%)")
        print(f"    부호 조건 추가 제거:    {n_sign} ({n_sign/n_total*100:.1f}%)")
        print(f"    ★ SHAP_mPELQ>0 인데 제거되는 샘플(부호 오류 피해): {n_false}")
        print(f"    ★ SHAP_mPELQ>0 & |SHAP|>|target| (방향 무관 지배): {n_pos_dominated}")

print("\n" + "=" * 80)
print("진단 완료")
print("=" * 80)
