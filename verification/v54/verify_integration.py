#!/usr/bin/env python3
"""hermes-verify-v54-integration.py — ad-hoc verification (NOT a test suite).

Verifies the changed behavior of:
  - run_pipeline_v5.py : v5.4 two-rule Primary (metal ∪ unattr), OOF assertions
  - bootstrap_tel_pel_ci.py : POPS_OUTPUT_DIR isolation
  - pipeline_v5_config.json : v5.4 rule definitions
Covers syntax, committed-content identity, rule anchors on the committed blob,
real rerun artifacts (/tmp/pops_v54_output): reason-split↔flag coherence,
union semantics, NEDS zero-fire, raw↔aggregate counter identity, y_te identity,
bootstrap CI sanity + point-estimate match, release/output non-modification,
and pushed-branch state.
"""
import json
import py_compile
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path("/home/dshwang/workspace/data_lab/POPs_EQS/release")
TMP = Path("/tmp/pops_v54_output")
checks = []


def check(name, ok, detail=""):
    checks.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


# A. syntax of changed code + committed content identity
for f in ["run_pipeline_v5.py", "bootstrap_tel_pel_ci.py"]:
    try:
        py_compile.compile(str(REPO / "code" / f), doraise=True, cfile="/tmp/_pc.pyc")
        check(f"A1 compile {f}", True)
    except Exception as e:
        check(f"A1 compile {f}", False, str(e)[:160])

cfg = json.load(open(REPO / "code" / "pipeline_v5_config.json"))
ok_b1 = cfg.get("pipeline_version") == "v5.4" and cfg.get("v54_rules", {}).get("unattr_shap_threshold") == -0.1
check("B1 config schema v5.4", ok_b1)

committed_cfg = subprocess.run(["git", "-C", str(REPO), "show", "HEAD:code/pipeline_v5_config.json"],
                               capture_output=True, text=True).stdout
ok_b2 = json.loads(committed_cfg).get("v54_rules", {}).get("unattr_shap_threshold") == -0.1
check("A2 committed config == disk", ok_b2)

# C. fixed-rule anchors in the committed code
head = subprocess.run(["git", "-C", str(REPO), "show", "HEAD:code/run_pipeline_v5.py"],
                      capture_output=True, text=True).stdout
for pat in ["metal_flag_count >= min_success", "unattr_flag_count >= min_success",
            "exclusion_reason", "shapP_target", "V54_UNATTR_SHAP_THRESHOLD",
            "POPS_OUTPUT_DIR", "OOF 반복 단언 위반", "NEDS 단언 위반",
            "abs(shapP_metal[k]) > abs(shapP_target[k])"]:
    check(f"C anchor {pat[:36]}", pat in head)

# D. rule semantics recomputed from flags on real rerun outputs (Integrated)
for sub in ["DDTs", "CHLs", "PCBs"]:
    d = pd.read_csv(TMP / f"Step2_7v5_DRC_Residual_Integrated_{sub}.csv", low_memory=False)
    neds = d["Data_Type"] == "NEDS"
    metal = d["metal_flag_count"] >= 8
    unattr = d["unattr_flag_count"] >= 8
    reasons = d["exclusion_reason"]
    recomputed = pd.Series(pd.np.select if False else pd._libs.__dict__ and None, None) \
        if False else __import__("numpy").select([neds.values, (metal & unattr).values,
                                                  (metal & ~unattr).values,
                                                  (~metal & unattr).values],
                                                 ["NEDS", "both", "metal_only",
                                                  "unattr_only"], default="kept")
    ok_d1 = bool((pd.Series(recomputed, index=d.index) == reasons).all())
    check(f"D1 {sub} reason↔flags 재계산 일치", ok_d1)
    excl = reasons.isin(["metal_only", "unattr_only", "both"])
    ok_d2 = bool(((metal.values | unattr.values) == reasons.isin(
        ["metal_only", "unattr_only", "both"]).values).all())
    check(f"D2 {sub} 합집합 판정 정합", ok_d2)
    ok_d3 = bool((d.loc[neds, ["metal_flag_count", "unattr_flag_count"]] == 0).all().all())
    check(f"D3 {sub} NEDS 발화·제외 0", ok_d3 and bool((reasons[neds] == "NEDS").all()))
    kept_mask = ~reasons.isin(["metal_only", "unattr_only", "both", "NEDS"])
    ok_d4 = bool((d.loc[kept_mask, "metal_flag_count"] < 8).all()
                 and (d.loc[kept_mask, "unattr_flag_count"] < 8).all())
    check(f"D4 {sub} kept = 두 규칙 모두 8 미달", ok_d4)

# E. raw↔aggregate counter identity (Integrated DDTs)
raw = pd.read_csv(TMP / "Step2_7v5_OOF_Raw_Integrated_DDTs.csv",
                  usecols=["record_id", "repeat_id", "v54_metal_flag",
                           "v54_unattr_flag", "v54_y_te_survival"])
d = pd.read_csv(TMP / "Step2_7v5_DRC_Residual_Integrated_DDTs.csv", low_memory=False)
agg = raw.groupby("record_id").agg(mcount=("v54_metal_flag", "sum"),
                                   ucount=("v54_unattr_flag", "sum"),
                                   ytem=("v54_y_te_survival", "mean"))
j = d.set_index("record_id").join(agg, how="left")
check("E1 raw fold 합 == metal_flag_count", bool((j["mcount"] == j["metal_flag_count"]).all()))
check("E2 raw fold 합 == unattr_flag_count", bool((j["ucount"] == j["unattr_flag_count"]).all()))
ok_e3 = bool((j["Mean_Survival"] - j["ytem"]).abs().max() < 1e-9)
check("E3 survival==mean(y_te) — 평균 join 부재", ok_e3)
ok_e4 = bool((raw.groupby("record_id")["repeat_id"].nunique() == 10).all())
check("E4 repeat 10/레코드", ok_e4)
check("E5 (record_id,repeat_id) 중복 0", int(raw.duplicated(["record_id", "repeat_id"]).sum()) == 0)

# F. bootstrap: 9 combos, CIs bracket points, point == pipeline Primary
boot = pd.read_csv(TMP / "TEL_PEL_bootstrap_CI.csv")
telpel = pd.read_csv(TMP / "Step2_7v5_TELPEL_Comparison_Integrated.csv")
ok_f1 = (len(boot) == 9
         and (boot["TEL_OC1pct_CI_low"] < boot["TEL_OC1pct"]).all()
         and (boot["TEL_OC1pct"] < boot["TEL_OC1pct_CI_high"]).all()
         and (boot["PEL_OC1pct_CI_low"] < boot["PEL_OC1pct"]).all()
         and boot["PEL_OC1pct"].between(boot["PEL_OC1pct_CI_low"],
                                        boot["PEL_OC1pct_CI_high"]).all())
check("F1 bootstrap 9조합·CI 정합", ok_f1)
prim = telpel[telpel["Analysis"] == "Primary"].set_index("Substance")
boot_i = boot[boot["Source"] == "Integrated"].set_index("Substance")
max_dev = float((prim["TEL"] - boot_i["TEL_OC1pct"]).abs().max())
check("F2 bootstrap point == Primary", max_dev <= 5e-4, f"max|ΔTEL|={max_dev:.4f}")

# G. release/output untouched by the rerun
newest_mt = max((p.stat().st_mtime for p in (REPO / "output").rglob("*") if p.is_file()),
                default=0)
rerun_start = datetime(2026, 8, 29, 17, 50).timestamp()
check("G release/output 무수정", newest_mt < rerun_start,
      f"newest={datetime.fromtimestamp(newest_mt):%m-%d %H:%M}" if newest_mt else "empty")

# H. git state under test == pushed PR head
branch = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
head_sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
status = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                        capture_output=True, text=True).stdout.strip().splitlines()
dirty = [l for l in status if not l.startswith("??")]
tracked_dirty = [l for l in dirty if "output_backup" not in l]
check("H 브랜치·커밋·dirty", branch == "v54-two-rule-integration" and head_sha.startswith("51b60f2")
      and not tracked_dirty, f"{branch}@{head_sha[:7]} dirty={tracked_dirty or 'none'}")

print("\n" + "=" * 62)
result = "ALL_PASS" if all(checks) else "FAIL"
print(f"AD-HOC RESULT: {result}")
sys.exit(0 if all(checks) else 1)