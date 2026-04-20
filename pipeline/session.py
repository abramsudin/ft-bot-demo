# ============================================================
# pipeline/session.py
#
# The startup orchestrator. Runs exactly once per conversation.
# Ties all four pipeline functions together and returns the
# single session dict that the agent reads.
#
# Public API:
#   build_session(train_path, labels_path, output_dir) -> dict
#
# Session dict schema:
#   df               pd.DataFrame   Full merged dataset (includes churn)
#   target           pd.Series      Churn column alone (0/1)
#   feature_cols     list[str]      All column names except churn
#   num_cols         list[str]      Numeric feature names
#   cat_cols         list[str]      Categorical feature names
#   null_group_map   dict[str,str]  column_name -> 'G1'/'G2'/etc
#   null_scan_df     pd.DataFrame   Output of run_null_signal_scan()
#   verdict_df       pd.DataFrame   Full M3 results per column
#   pairs_df         pd.DataFrame   Redundant pairs among KEEP columns
#   redundancy_drop  list[str]      Columns recommended to drop (redundancy)
#
# Cache behaviour:
#   If scan_results.pkl already exists in output_dir it is loaded
#   directly — the expensive M3 stat tests are NOT re-run.
#   Delete the .pkl to force a fresh scan.
# ============================================================

import os
import numpy as np
import pandas as pd

from pipeline.loader         import load_data
from pipeline.validator      import validate
from pipeline.m2_eda_scanner import find_null_groups, run_null_signal_scan
from pipeline.m3_scanner     import run_m3_scan


def build_session(train_path: str,
                  labels_path: str,
                  output_dir: str = "./outputs") -> dict:
    """
    Load data, validate, run EDA + M3 scan, return session dict.

    Parameters
    ----------
    train_path  : path to the feature CSV
    labels_path : path to the churn labels CSV
    output_dir  : directory for scan_results.pkl (created if missing)

    Returns
    -------
    dict — see module docstring for full schema.

    Raises
    ------
    FileNotFoundError  if either CSV path does not exist
    ValueError         if validate() finds a structural problem
    """

    # ── STEP 1: Load ──────────────────────────────────────────
    print("build_session | step 1/6  loading data...")
    df = load_data(train_path, labels_path)

    # ── STEP 2: Validate ──────────────────────────────────────
    print("build_session | step 2/6  validating...")
    validate(df)

    # ── STEP 3: Split cols / target ───────────────────────────
    print("build_session | step 3/6  splitting columns...")
    feature_cols = [c for c in df.columns if c != "churn"]
    num_cols     = df[feature_cols].select_dtypes(
                       include=[np.number]).columns.tolist()
    cat_cols     = df[feature_cols].select_dtypes(
                       exclude=[np.number]).columns.tolist()
    target       = df["churn"]

    # ── STEP 4: Null groups ───────────────────────────────────
    print("build_session | step 4/6  finding null groups...")
    null_group_map = find_null_groups(df)

    # ── STEP 5: Null signal scan ──────────────────────────────
    print("build_session | step 5/6  running null signal scan...")
    null_scan_df = run_null_signal_scan(df, target)

    # ── STEP 6: M3 scan (cached) ──────────────────────────────
    pkl_path = os.path.join(output_dir, "scan_results.pkl")

    if os.path.exists(pkl_path):
        print(f"build_session | step 6/6  loading cached scan from {pkl_path}")
        m3_results = pd.read_pickle(pkl_path)
    else:
        print("build_session | step 6/6  running M3 scan (first run — may take 1-2 mins)...")
        m3_results = run_m3_scan(df, target, null_group_map, output_dir)

    # ── STEP 7: Assemble session dict ─────────────────────────
    session = {
        "df"             : df,
        "target"         : target,
        "feature_cols"   : feature_cols,
        "num_cols"       : num_cols,
        "cat_cols"       : cat_cols,
        "null_group_map" : null_group_map,
        "null_scan_df"   : null_scan_df,
        "verdict_df"     : m3_results["verdict_df"],
        "pairs_df"       : m3_results["pairs_df"],
        "redundancy_drop": m3_results["redundancy_drop"],
    }

    vdf = session["verdict_df"]
    print(
        f"\nbuild_session | complete\n"
        f"  rows={len(df):,}  features={len(feature_cols)}\n"
        f"  KEEP={( vdf.verdict=='KEEP').sum()}  "
        f"FLAG={(vdf.verdict=='FLAG').sum()}  "
        f"DROP={(vdf.verdict=='DROP').sum()}  "
        f"DROP-NULL={(vdf.verdict=='DROP-NULL').sum()}\n"
        f"  null_groups={len(set(null_group_map.values()))}  "
        f"redundant_pairs={len(session['pairs_df'])}"
    )

    return session
