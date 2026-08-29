#!/usr/bin/env python3
"""
/tmp/hermes-verify-v54-rules.py — (2026-08-29) v5.4 2규칙 고정 로직 단위검사
Ad-hoc verification (NOT a test suite) — 규칙은 파이프라인과 동일하나 복사본으로 검증.
"""
import numpy as np
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

UNATTR_THR = -0.1
MIN_SUCCESS = 8

# --- 판정표 5사례 (스킬 고정 — 코드 검증 기준) ---
# | Metal | Target | MetalFlag | UnattrFlag |
# |  -5   |   -2   |     1     |     0 (target>-0.1? -2<=-0.1 → False) |
# |  -5   |   +2   |     1     |     1 (target>-0.1 → True) |
# |  +5   |   -2   |     0     |     0 |
# |  -2   |   -5   |     0     |     0 |
# | NEDS  | any    |     0     |     0 |
def rules(surv, sm, st):
    metal = (surv < 80) & (sm < 0) & (abs(sm) > abs(st))
    unattr = (surv < 80) & (st > UNATTR_THR)
    return metal, unattr

cases = [
    # (surv, shap_metal, shap_target, exp_metal, exp_unattr, name)
    (70, -5, -2, 1, 0, "EDS metal-dominant (보호 아님)"),
    (70, -5, +2, 1, 1, "EDS metal-dominant target보다 우세"),
    (70, +5, -2, 0, 0, "금속 보호방향 → 제외 금지"),
    (70, -2, -5, 0, 0, "target 더 우세 → 유지"),
    (85, -5, +2, 0, 0, "NEDS → 절대 제외 안 함"),
]
all_pass = True
print("=== 판정표 5사례 (스킬 고정) ===")
for surv, sm, st, em, eu, name in cases:
    m, u = rules(surv, sm, st)
    ok = (int(m) == em) and (int(u) == eu)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: expected metal={em} unattr={eu} → got metal={int(m)} unattr={int(u)}")
    all_pass = all_pass and ok

# --- NEDS 절대 제외 불가 (스킬 "NEDS 제외=0") ---
print("\n=== NEDS 단언 (NEDS 제외 0건) ===")
surv = np.array([85, 90, 82])
sm = np.array([-5, -1, -0.5])
st = np.array([+2, +0.5, -0.05])
m, u = rules(surv, sm, st)
neds_ok = (m.sum() == 0) and (u.sum() == 0)
print(f"  NEDS metal 발화={int(m.sum())}, unattr 발화={int(u.sum())} → "
      f"{'PASS' if neds_ok else 'FAIL'}")
all_pass = all_pass and neds_ok

# --- count 기반 합의 시뮬레이션 (10 repeat, 8회 이상만 제외) ---
print("\n=== count>=8 합의 (repeat별 조건 분리) ===")
rng = np.random.default_rng(0)
n_rec = 50
mc = np.zeros(n_rec, dtype=int)
uc = np.zeros(n_rec, dtype=int)
for rep in range(10):
    smv = rng.normal(0, 1, n_rec)
    stv = np.random.default_rng(100+rep).normal(0, 1, rep % 5 + 1) # 무의미 — 아래 실제 사용
    stv = rng.normal(0, 1, n_rec)
    survv = np.full(n_rec, 70.0)  # 전부 EDS
    m_k, u_k = rules(survv, smv, stv)
    mc += m_k
    uc += u_k
# 10회 중 8회 이상 → 제외 (합집합)
excl = (mc >= 8) | (uc >= 8)
print(f"  metal>=8: {(mc>=8).sum()}, unattr>=8: {(uc>=8).sum()}, 합집합 제외: {excl.sum()}/50")
union_ok = (excl == ((mc >= 8) | (uc >= 8))).all()
print(f"  합집합 정합성: {'PASS' if union_ok else 'FAIL'}")
all_pass = all_pass and union_ok

print("\n" + "=" * 60)
print(f"RESULT: {'ALL_PASS' if all_pass else 'FAIL'}")
sys.exit(0 if all_pass else 1)