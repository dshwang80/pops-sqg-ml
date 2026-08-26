#!/usr/bin/env python3
"""
자동 검증 스크립트 — SHAP-independent Primary SQG 파이프라인 검증 체크포인트
============================================================================

리뷰어가 요구한 8개 검증 항목을 자동으로 확인한다:

  1. n_unique_fold_splits = 10
  2. group_kfold_train_test_overlap = 0
  3. OOF 횟수 = 10/10
  4. Interventional 최소 8/10 기준 적용
  5. config와 실제 실행값 일치
  6. Primary / anchor-species / mPELQ sensitivity 출력 경로 분리
  7. bootstrap·mPELQ sensitivity 결과 재현 (파일 존재 + 비어있지 않음)
  8. TEL/PEL 회귀검증 통과 (Step3 저장값 vs edsneds_data Method B 재계산)

실행:
    python code/verify_pipeline.py

종료 코드: 0 = 전부 통과, 1 = 하나 이상 실패
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "output"
CODE = BASE / "code"

TARGET_SUBSTANCES = ["DDTs", "CHLs", "PCBs"]
SOURCES = ["Integrated", "NOAA", "SCCWRP"]

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f"  — {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# 1-4. Diagnosis 파일에서 fold/OOF/Interventional 지표 확인
# ---------------------------------------------------------------------------
diag = pd.read_csv(OUT / "Step2_7v4_Diagnosis_All.csv")

# 1. n_unique_fold_splits = 10
if "n_unique_fold_splits" in diag.columns:
    vals = diag["n_unique_fold_splits"].dropna().astype(int)
    check("1. n_unique_fold_splits = 10", (vals == 10).all(),
          f"min={vals.min()}, max={vals.max()}, n={len(vals)}")
else:
    check("1. n_unique_fold_splits = 10", False, "컬럼 없음")

# 2. group overlap = 0
ov_col = None
for c in diag.columns:
    if "overlap" in c.lower():
        ov_col = c
        break
if ov_col:
    vals = diag[ov_col].dropna().astype(int)
    check("2. group_kfold_train_test_overlap = 0", (vals == 0).all(),
          f"컬럼={ov_col}, max={vals.max()}")
else:
    check("2. group_kfold_train_test_overlap = 0", False, "overlap 컬럼 없음")

# 3. OOF 횟수 = 10/10
oof_min_col = next((c for c in diag.columns if "oof" in c.lower() and "min" in c.lower()), None)
oof_max_col = next((c for c in diag.columns if "oof" in c.lower() and "max" in c.lower()), None)
if oof_min_col and oof_max_col:
    mn = diag[oof_min_col].dropna().astype(int)
    mx = diag[oof_max_col].dropna().astype(int)
    check("3. OOF 횟수 = 10/10", (mn == 10).all() and (mx == 10).all(),
          f"min={mn.min()}, max={mx.max()}")
else:
    check("3. OOF 횟수 = 10/10", False, f"컬럼 {oof_min_col}/{oof_max_col}")

# 4. Interventional 최소 8/10 (절대 횟수 기준)
#    interventional_successes = 성공 횟수, interventional_attempts = 시도 횟수
succ_col = next((c for c in diag.columns if c == "interventional_successes"), None)
att_col = next((c for c in diag.columns if c == "interventional_attempts"), None)
if succ_col and att_col:
    succ = diag[succ_col].dropna().astype(int)
    att = diag[att_col].dropna().astype(int)
    # 절대 횟수 >= 8 AND 성공률 >= 0.8 (둘 다 확인)
    ok_abs = (succ >= 8).all()
    ok_rate = ((succ / att.replace(0, np.nan)) >= 0.8).all()
    check("4. Interventional >= 8/10", ok_abs and ok_rate,
          f"successes min={succ.min()}, max={succ.max()}, attempts min={att.min()}, max={att.max()}")
else:
    check("4. Interventional >= 8/10", False, f"컬럼 {succ_col}/{att_col} 없음")

# ---------------------------------------------------------------------------
# 5. config와 실제 실행값 일치
# ---------------------------------------------------------------------------
cfg = json.load(open(CODE / "pipeline_v4_config.json"))
gk_enabled = cfg.get("group_kfold", {}).get("enabled", False)
mpelq_thresh = cfg.get("mpelq_metals_threshold", None)
random_seed = cfg.get("random_seed", None)

# 코드에서 config를 읽는지 확인 (정적 검사)
src = (CODE / "run_pipeline_v4.py").read_text()
cfg_connected = (
    'group_cfg.get("enabled"' in src
    and 'cfg.get("mpelq_metals_threshold"' in src
    and 'cfg.get("random_seed"' in src
)
check("5. config와 실제 실행값 일치",
      gk_enabled and mpelq_thresh is not None and random_seed is not None and cfg_connected,
      f"group_kfold.enabled={gk_enabled}, mpelq_thresh={mpelq_thresh}, seed={random_seed}, code_connected={cfg_connected}")

# ---------------------------------------------------------------------------
# 6. 출력 경로 분리
# ---------------------------------------------------------------------------
anchor_primary = (OUT / "anchor_sensitivity" / "primary").is_dir()
anchor_with = (OUT / "anchor_sensitivity" / "with_anchor").is_dir()
mpelq_dirs = list((OUT / "mpelq_sensitivity").glob("cutoff_*")) if (OUT / "mpelq_sensitivity").is_dir() else []
check("6. 출력 경로 분리",
      anchor_primary and anchor_with and len(mpelq_dirs) >= 1,
      f"anchor_primary={anchor_primary}, anchor_with={anchor_with}, mpelq_cutoffs={len(mpelq_dirs)}")

# ---------------------------------------------------------------------------
# 7. bootstrap·mPELQ sensitivity 결과 재현
# ---------------------------------------------------------------------------
boot_ok = (OUT / "TEL_PEL_bootstrap_CI.csv").exists() and (OUT / "TEL_PEL_bootstrap_CI.csv").stat().st_size > 0
mpelq_ok = (OUT / "mPELQ_cutoff_sensitivity_summary.csv").exists() and (OUT / "mPELQ_cutoff_sensitivity_summary.csv").stat().st_size > 0
check("7. bootstrap·mPELQ sensitivity 재현", boot_ok and mpelq_ok,
      f"bootstrap={boot_ok}, mpelq={mpelq_ok}")

# ---------------------------------------------------------------------------
# 8. TEL/PEL 회귀검증 (Step3 저장값 vs edsneds_data Method B 재계산)
# ---------------------------------------------------------------------------
def method_b_tel_pel(edsneds_df):
    """Method B (CCME 지침): 원래 단위(µg/kg dw)에서 percentile 후 기하평균.

    EDS = Mean_Survival < 80, NEDS = Mean_Survival >= 80
    e_orig = 10^Conc_log - 1
    TEL = sqrt(P15(e_orig) * P50(n_orig))
    PEL = sqrt(P50(e_orig) * P85(n_orig))
    """
    df = edsneds_df.copy()
    e_v = df.loc[df["Data_Type"] == "EDS", "Conc_log"].values
    n_v = df.loc[df["Data_Type"] == "NEDS", "Conc_log"].values
    e_orig = 10 ** e_v - 1
    n_orig = 10 ** n_v - 1
    tel = np.sqrt(np.quantile(e_orig, 0.15) * np.quantile(n_orig, 0.50))
    pel = np.sqrt(np.quantile(e_orig, 0.50) * np.quantile(n_orig, 0.85))
    return tel, pel


reg_ok = True
reg_detail = []
for src_name in SOURCES:
    step3 = pd.read_csv(OUT / f"Step3_TEL_PEL_{src_name}.csv")
    for sub in TARGET_SUBSTANCES:
        row = step3[step3["Substance"] == sub]
        if row.empty:
            reg_ok = False
            reg_detail.append(f"{src_name}/{sub}: MISSING")
            continue
        tel_stored = row["Our_TEL_dw"].values[0]
        pel_stored = row["Our_PEL_dw"].values[0]
        n_stored = row["Total_N"].values[0]
        # edsneds_data에서 Method B 재계산
        eds = pd.read_csv(OUT / f"Step3_EDSNEDS_{src_name}_{sub}.csv")
        n_eds = len(eds)
        tel_calc, pel_calc = method_b_tel_pel(eds)
        tel_match = np.isclose(tel_stored, round(tel_calc, 4), atol=1e-3)
        pel_match = np.isclose(pel_stored, round(pel_calc, 4), atol=1e-3)
        n_match = (n_stored == n_eds)
        ok = tel_match and pel_match and n_match
        if not ok:
            reg_ok = False
        reg_detail.append(
            f"{src_name}/{sub}: TEL={tel_stored:.4f}(calc {tel_calc:.4f}) "
            f"PEL={pel_stored:.4f}(calc {pel_calc:.4f}) N={n_stored}/{n_eds} "
            f"{'OK' if ok else 'MISMATCH'}"
        )

check("8. TEL/PEL 회귀검증 통과", reg_ok, "; ".join(reg_detail))

# ---------------------------------------------------------------------------
# 최종 판정
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
n_pass = sum(1 for _, ok, _ in results if ok)
n_total = len(results)
print(f"결과: {n_pass}/{n_total} 통과")
if n_pass == n_total:
    print("★ 모든 검증 항목 통과 — ML 필터 코드 검증 종료 가능")
    sys.exit(0)
else:
    print("✗ 일부 항목 실패 — 위 FAIL 항목 확인 필요")
    sys.exit(1)
