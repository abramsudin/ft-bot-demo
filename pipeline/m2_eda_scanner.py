# ============================================================
# pipeline/m2_eda_scanner.py
#
# Merged from milestone2/m2_02_null_analysis.py
#          and milestone2/m2_03_churn_signal.py
#
# Provides two pure-logic functions for the pipeline:
#
#   find_null_groups(df)
#       Detects columns that always go null together.
#       Returns dict[column_name -> group_label]  e.g. {"Var3": "G1"}
#       Columns not in any group are absent from the dict.
#
#   run_null_signal_scan(df, target)
#       Scans every partially-null column for churn-rate gap
#       between null rows and present rows.
#       Returns a DataFrame sorted by gap_pp descending.
#
# What was deleted vs the originals:
#   - All matplotlib / seaborn / plt code
#   - All file saves (.png, .csv)
#   - All print statements
#   - os.makedirs / OUTPUT_DIR / SCRIPT_DIR scaffolding
#   - pd.read_csv / hardcoded data paths
#   - The duplicate find_null_groups_fast() from m2_03
#   - The EDA summary text writer
#   - near-zero variance block (EDA-only, not needed at runtime)
# ============================================================

import numpy as np
import pandas as pd
from collections import defaultdict


# ── find_null_groups ──────────────────────────────────────────
def find_null_groups(df: pd.DataFrame) -> dict:
    """
    Find columns that always go null together (perfect null
    co-occurrence, correlation == 1.0).

    Algorithm: builds a null indicator matrix, computes pairwise
    null correlation, then union-finds perfectly correlated pairs
    into named groups (G1, G2, …).

    Parameters
    ----------
    df : DataFrame — must include a "churn" column (it is excluded
         from the scan automatically).

    Returns
    -------
    dict[str, str]
        Maps each grouped column name to its group label.
        e.g. {"Var3": "G1", "Var7": "G1", "Var12": "G2", ...}
        Columns not in any group are absent from the dict.
    """
    feature_cols      = [c for c in df.columns if c != "churn"]
    null_rates        = df[feature_cols].isnull().mean() * 100
    partial_null_cols = null_rates[
        (null_rates > 0) & (null_rates < 100)
    ].index.tolist()

    if len(partial_null_cols) < 2:
        return {}

    null_matrix = df[partial_null_cols].isnull().astype(int)
    null_corr   = null_matrix.corr()

    # ── union-find over perfectly correlated pairs ────────────
    col_to_group = {}
    groups       = defaultdict(set)
    group_id     = 0
    cols         = null_corr.columns.tolist()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if null_corr.iloc[i, j] >= 1.0:
                a, b = cols[i], cols[j]
                if a in col_to_group and b in col_to_group:
                    ga, gb = col_to_group[a], col_to_group[b]
                    if ga != gb:
                        groups[ga].update(groups[gb])
                        for c in groups[gb]:
                            col_to_group[c] = ga
                        del groups[gb]
                elif a in col_to_group:
                    g = col_to_group[a]
                    groups[g].add(b)
                    col_to_group[b] = g
                elif b in col_to_group:
                    g = col_to_group[b]
                    groups[g].add(a)
                    col_to_group[a] = g
                else:
                    groups[group_id].update([a, b])
                    col_to_group[a] = col_to_group[b] = group_id
                    group_id += 1

    # ── build column → label mapping ─────────────────────────
    # Sort groups by size descending so the largest group is G1
    sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]))
    group_membership: dict[str, str] = {}
    for rank, (gid, members) in enumerate(sorted_groups, start=1):
        label = f"G{rank}"
        for col in members:
            group_membership[col] = label

    return group_membership


# ── run_null_signal_scan ──────────────────────────────────────
def run_null_signal_scan(df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """
    Scan every partially-null column for a churn-rate gap between
    rows where the column is null vs rows where it has a value.

    A large gap means: being null is itself predictive of churn,
    so those columns need an is_null indicator feature.

    Parameters
    ----------
    df     : DataFrame with feature columns (churn column may be present
             but is excluded from scanning automatically).
    target : Binary churn Series (0/1), same index as df.

    Returns
    -------
    pd.DataFrame with columns:
        col            — feature column name
        null_rate      — null rate (%)
        churn_null     — churn rate when column is null (%)
        churn_present  — churn rate when column has a value (%)
        gap_pp         — absolute gap in percentage points
        direction      — "null=more_churn" or "null=less_churn"

    Sorted by gap_pp descending.
    Only columns with >= 100 null rows AND >= 100 present rows are
    included (same threshold as test_null_signal).
    """
    feature_cols = [c for c in df.columns if c != "churn"]
    null_rates   = df[feature_cols].isnull().mean() * 100

    rows = []
    for col in feature_cols:
        nr = null_rates[col]
        if nr < 1 or nr > 99:          # skip fully complete or fully dead
            continue

        is_null   = df[col].isnull()
        n_null    = int(is_null.sum())
        n_present = int((~is_null).sum())

        if n_null < 100 or n_present < 100:
            continue

        churn_null    = float(target[is_null].mean() * 100)
        churn_present = float(target[~is_null].mean() * 100)
        gap           = abs(churn_null - churn_present)
        direction     = (
            "null=more_churn" if (churn_null - churn_present) > 0
            else "null=less_churn"
        )

        rows.append({
            "col"          : col,
            "null_rate"    : round(nr, 1),
            "churn_null"   : round(churn_null, 2),
            "churn_present": round(churn_present, 2),
            "gap_pp"       : round(gap, 2),
            "direction"    : direction,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("gap_pp", ascending=False)
        .reset_index(drop=True)
    )
