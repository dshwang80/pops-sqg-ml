#!/usr/bin/env python3
"""
TEL/PEL 95% 신뢰구간 (cluster bootstrap)
=========================================

Step3에서 저장된 EDS/NEDS 원자료(Step3_EDSNEDS_*.csv)를 읽어,
cluster 단위로 재표집(bootstrap)하여 TEL/PEL의 95% CI를 산출한다.

방법:
- cluster = 파이프라인 GroupKFold와 동일한 group_id (run_pipeline_v5._build_group_ids 재사용)
  ★ v5.3 (2026-08-28): 구 우선순위 [station_key > Station > sample_key > SampleID]의
    NOAA Station 단독 사용 문제 수정 — Station은 고유 정점 식별자가 아님(739개 Station이
    복수 좌표 보유). 파이프라인과 동일한 6단계 우선순위로 통일:
    station_key > StudyID+Station > sample_key > SampleID > 좌표(round4) > record_id
- 각 bootstrap 반복에서 cluster를 복원추출(replace=True)
  ★ v5.3 (2026-08-28): 복원추출에서 중복 선택된 cluster의 자료를 빈도만큼 반영.
    구 구현은 isin()으로 중복을 1회만 반영(무작위 cluster 부분집합과 동일) — 폐기.
    동일 cluster가 k회 추출되면 해당 cluster의 record가 k배 중복 포함되어야 함.
- EDS/NEDS 분류는 원자료의 Data_Type 컬럼을 그대로 사용 (Primary survival-only 분류)
- TEL = sqrt(q15(EDS_orig) * q50(NEDS_orig)), PEL = sqrt(q50(EDS_orig) * q85(NEDS_orig))
- orig = 10^Conc_log - 1 (1% TOC 정규화 농도, Conc/TOC_pct)
- 95% CI = 2.5th / 97.5th percentile (percentile method)
- 유효 반복 수 검증: 유한값이 95% 미만이면 RuntimeError (조용한 편향 방지)
- 입력 파일 결측 시 skip 없이 즉시 중단 (9개 조합 전부 필수)

재현성: seed 고정(42), 반복 2000회 — pipeline_v5_config.json에서 로드.
config 누락 시 즉시 중단 (run_pipeline_v5.load_config fail-fast 재사용).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "output"
sys.path.insert(0, str(REPO_ROOT / "code"))

from run_pipeline_v5 import _build_group_ids, load_config  # noqa: E402

TARGET_SUBSTANCES = ["DDTs", "CHLs", "PCBs"]
SOURCES = ["NOAA", "SCCWRP", "Integrated"]
MIN_VALID_FRACTION = 0.95

GROUP_PRIORITY_DOC = ("station_key > StudyID+Station > sample_key > SampleID "
                      "> coord(round4) > record_id")


def compute_tel_pel(e_orig, n_orig):
    """Method B (CCME): TEL/PEL 계산. e_orig/n_orig는 1% TOC 정규화 농도."""
    if len(e_orig) < 2 or len(n_orig) < 2:
        return np.nan, np.nan
    tel = np.sqrt(np.quantile(e_orig, 0.15) * np.quantile(n_orig, 0.50))
    pel = np.sqrt(np.quantile(e_orig, 0.50) * np.quantile(n_orig, 0.85))
    return tel, pel


def cluster_bootstrap(df, n_iter, rng):
    """cluster 단위 복원추출 bootstrap (중복 cluster 빈도가중). df는 _cluster 컬럼 포함."""
    clusters = df["_cluster"].unique()
    n_clusters = len(clusters)

    # cluster별 record 행번호 사전 — 중복 추출 시 같은 행들이 k배 중복 포함됨
    cluster_rows = {
        c: np.flatnonzero(df["_cluster"].to_numpy() == c)
        for c in clusters
    }

    tel_vals = np.full(n_iter, np.nan)
    pel_vals = np.full(n_iter, np.nan)

    for i in range(n_iter):
        sampled_clusters = rng.choice(clusters, size=n_clusters, replace=True)
        sampled_idx = np.concatenate([
            cluster_rows[c] for c in sampled_clusters
        ])
        sub = df.iloc[sampled_idx]

        e_orig = 10 ** sub.loc[
            sub["Data_Type"] == "EDS", "Conc_log"
        ].to_numpy() - 1

        n_orig = 10 ** sub.loc[
            sub["Data_Type"] == "NEDS", "Conc_log"
        ].to_numpy() - 1

        tel_vals[i], pel_vals[i] = compute_tel_pel(e_orig, n_orig)

    return tel_vals, pel_vals


def main():
    cfg = load_config()  # ★ v5 config — 누락 시 FileNotFoundError
    n_iter = cfg.get("bootstrap_n_iterations", 2000)
    seed = cfg.get("bootstrap_random_seed", 42)
    ci_level = cfg.get("bootstrap_ci_level", 0.95)
    rng = np.random.default_rng(seed)

    lo = (1 - ci_level) / 2 * 100
    hi = (1 + ci_level) / 2 * 100

    rows = []
    for source in SOURCES:
        for substance in TARGET_SUBSTANCES:
            path = OUTPUT_DIR / f"Step3_EDSNEDS_{source}_{substance}.csv"
            if not path.exists():
                # 9개 조합 전부 필수 — skip 없이 중단
                raise FileNotFoundError(path)
            df = pd.read_csv(path)

            # ★ v5.3: 파이프라인과 동일한 group_id 사용 (사용자 검토 지적 반영)
            #   구 방식(Station 단독 포함 4단계 우선순위)은 NOAA 잘못된 그룹화였음.
            df["_cluster"] = _build_group_ids(df)

            n_clusters = df["_cluster"].nunique()
            n_eds = int((df["Data_Type"] == "EDS").sum())
            n_neds = int((df["Data_Type"] == "NEDS").sum())

            # point estimate (원자료 그대로)
            e_orig = 10 ** df.loc[df["Data_Type"] == "EDS", "Conc_log"].values - 1
            n_orig = 10 ** df.loc[df["Data_Type"] == "NEDS", "Conc_log"].values - 1
            tel_pt, pel_pt = compute_tel_pel(e_orig, n_orig)

            tel_vals, pel_vals = cluster_bootstrap(df, n_iter, rng)

            # 유효 반복 수 검증 — 유한값 95% 미만이면 중단 (조용한 편향 방지)
            n_valid_tel = int(np.isfinite(tel_vals).sum())
            n_valid_pel = int(np.isfinite(pel_vals).sum())
            if (n_valid_tel < MIN_VALID_FRACTION * n_iter
                    or n_valid_pel < MIN_VALID_FRACTION * n_iter):
                raise RuntimeError(
                    f"Valid bootstrap replicates insufficient: "
                    f"TEL={n_valid_tel}/{n_iter}, PEL={n_valid_pel}/{n_iter}"
                )

            tel_lo = np.nanpercentile(tel_vals, lo)
            tel_hi = np.nanpercentile(tel_vals, hi)
            pel_lo = np.nanpercentile(pel_vals, lo)
            pel_hi = np.nanpercentile(pel_vals, hi)

            rows.append({
                "Source": source, "Substance": substance,
                "N_total": len(df), "N_EDS": n_eds, "N_NEDS": n_neds,
                "N_clusters": n_clusters,
                "cluster_method": "run_pipeline_v5._build_group_ids",
                "group_priority": GROUP_PRIORITY_DOC,
                "TEL_OC1pct": round(tel_pt, 4),
                "TEL_OC1pct_CI_low": round(tel_lo, 4),
                "TEL_OC1pct_CI_high": round(tel_hi, 4),
                "PEL_OC1pct": round(pel_pt, 4),
                "PEL_OC1pct_CI_low": round(pel_lo, 4),
                "PEL_OC1pct_CI_high": round(pel_hi, 4),
                "Concentration_basis": "1% TOC-normalized",
                "n_valid_TEL": n_valid_tel,
                "n_valid_PEL": n_valid_pel,
                "n_iter": n_iter, "seed": seed,
            })
            print(f"  {source:10s} {substance:6s} N={len(df):5d} "
                  f"TEL={tel_pt:.4f} [{tel_lo:.4f}, {tel_hi:.4f}] "
                  f"PEL={pel_pt:.4f} [{pel_lo:.4f}, {pel_hi:.4f}] "
                  f"valid={n_valid_tel}/{n_iter}")

    out = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "TEL_PEL_bootstrap_CI.csv"
    out.to_csv(out_path, index=False)
    print(f"\n저장: {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()