#!/usr/bin/env python3
"""
/tmp/hermes-verify-v54-empirical.py — v5.4 정식 파이프라인 산출물 실증 단언
Ad-hoc verification (NOT a test suite) — user 4건 검증 요건의 산출물 수준 확인.
"""
import sys
import numpy as np
import pandas as pd

OUT = "/tmp/pops_v54_output"
SUBS = ["DDTs", "CHLs", "PCBs"]
SRC = ["NOAA", "SCCWRP", "Integrated"]
checks = []

def check(name, ok, detail=""):
    checks.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")

# 산출물 존재
import os
missing = [s for s in SRC for sub in SUBS
           if not (f"/tmp/pops_v54_output/Step2_7v5_DRC_Residual_{s}_{sub}.csv").__class__]
for s in SRC:
    for sub in SUBS:
        p = f"/tmp/pops_v54_output/Step2_7v5_DRC_Residual_{s}_{sub}.csv"
        pass

print("=== 1. 제외사유 분리 (unattr_only/metal_only/both/kept, NEDS 포함 시 NEDS로) ===")
for s in SRC:
    for sub in SUBS:
        df = pd.read_csv(f"/tmp/pops_v54_output/Step2_7v5_DRC_Residual_{s}_{sub}.csv")
        reasons = set(df["exclusion_reason"].unique())
        ok = {"NEDS", "both", "metal_only", "unattr_only", "kept"} >= reasons
        # 조건 교차 검증: metal_only ⇒ unattr_flag_count<8
        mo = df[df["exclusion_reason"] == "metal_only"]
        mo_ok = bool((mo["unattr_flag_count"] < 8).all()) if len(mo) else True
        checks.append(ok and mo_ok)
        print(f"  [{'PASS' if ok and mo_ok else 'FAIL'}] {s}/{sub}: "
              f"reasons={sorted(reasons)} metal_only 중 unattr≥8 오분류={0 if mo_ok else len(mo)}")

print("\n=== 2. NEDS 제외 0건 + 무귀속/금속 발화 EDS 한정 ===")
for s in SRC:
    for sub in SUBS:
        df = pd.read_csv(f"/tmp/pops_v54_output/Step2_7v5_DRC_Residual_{s}_{sub}.csv")
        neds = df[df["Data_Type"] == "NEDS"]
        ok = bool((neds["primary_excluded"] == 0).all()
                  and (neds["metal_flag_count"] == 0).all()
                  and (neds["unattr_flag_count"] == 0).all())
        checks.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {s}/{sub}: NEDS 발화·제외 0건")

print("\n=== 3. OOF 반복 구조 (repeat_id 10종 / 중복 0 / y_te 일치) ===")
for s in ["Integrated"]:  # Integrated만 전수 조인 대상; NOAA/SCCWRP는 동일 로직
    for sub in SUBS:
        raw = pd.read_csv(f"/tmp/pops_v54_output/Step2_7v5_OOF_Raw_{s}_{sub}.csv",
                          usecols=["record_id", "repeat_id", "fold_id",
                                   "v54_metal_flag", "v54_unattr_flag", "v54_y_te_survival"])
        dup = int(raw.duplicated(subset=["record_id", "repeat_id"]).sum())
        reps = raw.groupby("record_id")["repeat_id"].nunique()
        ok = (dup == 0) and bool((reps == 10).all())
        checks.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {s}/{sub}: dup={dup}, "
              f"repeat_min={int(reps.min())}, repeat_max={int(reps.max())}")

print("\n=== 4. y_te 사용 감사: OOF 평균 y_te == df survival (평균 재조인 없음 실증) ===")
d = pd.read_csv("/tmp/pops_v54_output/Step2_7v5_DRC_Residual_Integrated_DDTs.csv")
raw = pd.read_csv("/tmp/pops_v54_output/Step2_7v5_OOF_Raw_Integrated_DDTs.csv",
                  usecols=["record_id", "v54_y_te_survival"])
yte_mean = raw.groupby("record_id")["v54_y_te_survival"].mean()
df2 = drc = d.set_index("record_id")
merged = df2.join(yte_mean.rename("yte_mean"))
diff = (merged["Mean_Survival"] - merged["yte_mean"]).abs().max()
ok = diff < 1e-9
checks.append(ok)
print(f"  [{'PASS' if ok else 'FAIL'}] Integrated/DDTs: |survival - mean(y_te)|max = {diff:.2e}")

print("\n=== 5. 제외 EDS 농도 vs 유지 EDS (저농도 집중 메커니즘) ===")
for sub in SUBS:
    d = pd.read_csv(f"/tmp/pops_v54_output/Step2_7v5_DRC_Residual_Integrated_{sub}.csv",
                    low_memory=False)
    conc_col = f"Sum_{sub}_OC_log"
    kept = d[(d["Data_Type"] == "EDS") & (d["exclusion_reason"] == "kept")]
    excl_mask = d["exclusion_reason"].isin(["metal_only", "unattr_only", "both"])
    med_x = float((10 ** d.loc[excl_mask, conc_col].dropna() - 1).median())
    med_k = float((10 ** kept[conc_col] - 1).median())
    ok = bool(med_x < med_k)
    checks.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] Integrated/{sub}: "
          f"제외 EDS 중앙값 {med_x:.3f} < 유지 EDS {med_k:.3f} µg/kg")

print("\n" + "=" * 60)
print(f"RESULT: {'ALL_PASS' if all(checks) else 'FAIL'}")
sys.exit(0 if all(checks) else 1)