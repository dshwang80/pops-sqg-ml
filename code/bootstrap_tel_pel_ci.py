#!/usr/bin/env python3
"""
TEL/PEL 95% 신뢰구간 (cluster bootstrap)
=========================================

Step3에서 저장된 EDS/NEDS 원자료(Step3_EDSNEDS_*.csv)를 읽어,
station/sample cluster 단위로 재표집(bootstrap)하여 TEL/PEL의 95% CI를 산출한다.

방법:
- cluster = station/sample 식별자 (우선순위: station_key > Station > sample_key > SampleID)
- 각 bootstrap 반복에서 cluster를 복원추출 → 해당 cluster의 모든 record 포함
- EDS/NEDS 분류는 원자료의 Data_Type 컬럼을 그대로 사용 (Primary survival-only 분류)
- TEL = sqrt(q15(EDS_orig) * q50(NEDS_orig)), PEL = sqrt(q50(EDS_orig) * q85(NEDS_orig))
- orig = 10^Conc_log - 1 (µg/kg dw)
- 95% CI = 2.5th / 97.5th percentile (percentile method)

재현성: seed 고정(42), 반복 2000회.
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "output"
CONFIG_PATH = REPO_ROOT / "code" / "pipeline_v4_config.json"

TARGET_SUBSTANCES = ["DDTs", "CHLs", "PCBs"]
SOURCES = ["NOAA", "SCCWRP", "Integrated"]

CLUSTER_COL_PRIORITY = ["station_key", "Station", "sample_key", "SampleID"]


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def compute_tel_pel(e_orig, n_orig):
    """Method B (CCME): TEL/PEL 계산. e_orig/n_orig는 µg/kg dw 원래 단위."""
    if len(e_orig) < 2 or len(n_orig) < 2:
        return np.nan, np.nan
    tel = np.sqrt(np.quantile(e_orig, 0.15) * np.quantile(n_orig, 0.50))
    pel = np.sqrt(np.quantile(e_orig, 0.50) * np.quantile(n_orig, 0.85))
    return tel, pel


def cluster_bootstrap(df, n_iter, rng):
    """cluster 단위 복원추출 bootstrap. df는 cluster_col 컬럼 포함."""
    clusters = df["_cluster"].unique()
    n_clusters = len(clusters)
    tel_vals = np.full(n_iter, np.nan)
    pel_vals = np.full(n_iter, np.nan)

    for i in range(n_iter):
        # cluster 복원추출
        sampled_clusters = rng.choice(clusters, size=n_clusters, replace=True)
        # 해당 cluster의 모든 record 수집
        mask = df["_cluster"].isin(sampled_clusters)
        sub = df[mask]
        e_orig = 10 ** sub.loc[sub["Data_Type"] == "EDS", "Conc_log"].values - 1
        n_orig = 10 ** sub.loc[sub["Data_Type"] == "NEDS", "Conc_log"].values - 1
        tel, pel = compute_tel_pel(e_orig, n_orig)
        tel_vals[i] = tel
        pel_vals[i] = pel

    return tel_vals, pel_vals


def main():
    cfg = load_config()
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
                print(f"  [skip] {path.name} 없음")
                continue
            df = pd.read_csv(path)
            if len(df) < 15:
                print(f"  [skip] {source}/{substance}: N={len(df)} < 15")
                continue

            # cluster 컬럼 결정
            cluster_col = None
            for cand in CLUSTER_COL_PRIORITY:
                if cand in df.columns and df[cand].notna().sum() > 1:
                    cluster_col = cand
                    break
            if cluster_col is None:
                cluster_col = "record_id"
                df["record_id"] = np.arange(len(df))
            df["_cluster"] = df[cluster_col].astype(str)

            n_clusters = df["_cluster"].nunique()
            n_eds = int((df["Data_Type"] == "EDS").sum())
            n_neds = int((df["Data_Type"] == "NEDS").sum())

            # point estimate (원자료 그대로)
            e_orig = 10 ** df.loc[df["Data_Type"] == "EDS", "Conc_log"].values - 1
            n_orig = 10 ** df.loc[df["Data_Type"] == "NEDS", "Conc_log"].values - 1
            tel_pt, pel_pt = compute_tel_pel(e_orig, n_orig)

            tel_vals, pel_vals = cluster_bootstrap(df, n_iter, rng)

            tel_lo = np.nanpercentile(tel_vals, lo)
            tel_hi = np.nanpercentile(tel_vals, hi)
            pel_lo = np.nanpercentile(pel_vals, lo)
            pel_hi = np.nanpercentile(pel_vals, hi)

            rows.append({
                "Source": source, "Substance": substance,
                "N_total": len(df), "N_EDS": n_eds, "N_NEDS": n_neds,
                "N_clusters": n_clusters, "cluster_col": cluster_col,
                "TEL": round(tel_pt, 4),
                "TEL_CI_low": round(tel_lo, 4), "TEL_CI_high": round(tel_hi, 4),
                "PEL": round(pel_pt, 4),
                "PEL_CI_low": round(pel_lo, 4), "PEL_CI_high": round(pel_hi, 4),
                "n_iter": n_iter, "seed": seed,
            })
            print(f"  {source:10s} {substance:6s} N={len(df):5d} "
                  f"TEL={tel_pt:.4f} [{tel_lo:.4f}, {tel_hi:.4f}] "
                  f"PEL={pel_pt:.4f} [{pel_lo:.4f}, {pel_hi:.4f}]")

    out = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "TEL_PEL_bootstrap_CI.csv"
    out.to_csv(out_path, index=False)
    print(f"\n저장: {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
