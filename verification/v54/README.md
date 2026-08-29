# v5.4 Two-Rule Primary — 병합 전 검증 증적 (2026-08-29)

PR #4 (`v54-two-rule-integration`) 병합 전 증적. **알고리즘 동결** — 이후 갱신은 결과·그림·논문만.

## 기본 정보
- **검증 커밋**: `51b60f2eab967dd386e039fc28dbf239e073d57e` (PR #4 head, base=origin/main 4a9e8f5)
- **실행 배경**: `POPS_OUTPUT_DIR` 격리 재실행 (release/output 미수정, G 검증 PASS)
- 환경 상세: `environment.json` / 사용 config: `pipeline_v5_config_used.json`

## 검증 결과 요약
| 스크립트 | 검증 대상 | 결과 |
|---|---|---|
| `verify_rules.py` (+`verify_rules.log`) | 고정 2규칙 판정표 5사례 (부호·\|SHAP\| 우세·NEDS 보호·count≥8/10) | **5 PASS / 0 FAIL** |
| `verify_empirical.py` (+log) | 재실행 산출물 실증 (사유분리 9·NEDS 0발화 9·OOF 반복구조·y_te 일치·농도 메커니즘 3) | **25 PASS / 0 FAIL** |
| `verify_integration.py` (+log) | 변경 3파일 컴파일·커밋 blob 일치·커밋 앵커 9·flag↔reason 재계산·bootstrap CI·output 무수정·PR head | **34 PASS / 0 FAIL** |

합계 **64건 ALL_PASS** (ad-hoc 검증, 정식 테스트 스위트 아님).

## 주요 결과 (Integrated, Primary v5.4 2-규칙)
### 제외 사유분리 (`exclusion_reason_breakdown.csv`)
| 물질 | 고려 EDS | 제외(metal_only/unattr_only/both) | 제외율 | kept | NEDS 위반 |
|---|---|---|---|---|---|
| DDTs | 1820 | 781 (166/479/136) | 42.9% | 1039 | 0 |
| CHLs | 1695 | 991 (371/329/291) | 58.5% | 704 | 0 |
| PCBs | 1663 | 824 (272/390/162) | 49.5% | 839 | 0 |

사유 3분해 합 == 총 제외 (스크립트 검증 True). N_EDS_primary=kept(최종 유지).

### OOF 단언 (`oof_assertions.json`)
- 180,700행 = 18,070 records × 10 repeats (repeat_id 0–9), 중복(substance, record_id, repeat_id) **0건**
- NEDS(y≥80) 제외 발화 **0건**
- 생존율: fold y_te 직접 사용 (`v54_survival_source`="fold y_te (record-level, no averaging join)")
- Primary SHAP: 3-feature [target, mPELQ_Metals, TOC] OOF TreeSHAP

### Primary TEL/PEL + bootstrap CI 95% (`telpel_primary_integrated.csv`, `bootstrap_ci.csv`)
| 물질 | TEL | PEL | TEL CI | PEL CI |
|---|---|---|---|---|
| ΣDDTs | 5.7861 | 20.8765 | 5.5665–6.0451 | 19.555–22.421 |
| ΣCHLs | 4.3342 | 16.9249 | 4.1381–4.5578 | 15.751–18.189 |
| ΣPCBs | 14.7016 | 56.6121 | 14.171–15.496 | 53.644–59.560 |

bootstrap point == 파이프라인 Primary (Δ=0.0000, F2 검증 PASS). 2,000 iter × seed 42, cluster bootstrap.

## 환경
python 3.11.15 / numpy 2.4.6 / xgboost 3.2.0 / shap 0.51.0 / pandas 3.0.5 / WSL2 (Linux-6.6.87.2)

## 대용량 원자료 안내
OOF raw(Step2_7v5_OOF_Raw_Integrated_*.csv, 9파일)·전체 산출물은 GitHub 미포함.
재실행으로 재현 가능: `POPS_OUTPUT_DIR=<dir> python3 code/run_pipeline_v5.py && POPS_OUTPUT_DIR=<dir> python3 code/bootstrap_tel_pel_ci.py`