# ============================================================
# pipeline/validator.py
#
# Validates the DataFrame produced by loader.load_data()
# before anything else in the pipeline touches it.
#
# Public API:
#   validate(df) -> None   (raises ValueError on failure)
#
# Checks performed (in order):
#   1. "churn" column exists
#   2. No duplicate column names
#   3. At least 100 rows
#   4. "churn" is binary (only 0 and 1)
#   5. At least one numeric feature and one categorical feature
# ============================================================

import numpy as np
import pandas as pd


def validate(df: pd.DataFrame) -> None:
    """
    Run five structural checks on the loaded DataFrame.
    Raises ValueError with a clear message on the first failure.

    Parameters
    ----------
    df : DataFrame produced by pipeline.loader.load_data()

    Returns
    -------
    None — silent success means all checks passed.
    """

    # ── CHECK 1: churn column exists ─────────────────────────
    if "churn" not in df.columns:
        raise ValueError(
            "Validation failed: 'churn' column is missing. "
            "loader.load_data() should have added it — check labels CSV."
        )

    # ── CHECK 2: no duplicate column names ───────────────────
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        raise ValueError(
            f"Validation failed: duplicate column names found: {duplicate_cols}. "
            "Each column must have a unique name."
        )

    # ── CHECK 3: at least 100 rows ───────────────────────────
    if len(df) < 100:
        raise ValueError(
            f"Validation failed: only {len(df):,} rows found — need at least 100. "
            "Check that the correct train CSV was loaded."
        )

    # ── CHECK 4: churn is binary ─────────────────────────────
    churn_values = set(df["churn"].dropna().unique())
    if not churn_values.issubset({0, 1}):
        unexpected = churn_values - {0, 1}
        raise ValueError(
            f"Validation failed: 'churn' column contains non-binary values: {unexpected}. "
            "Expected only 0 and 1."
        )

    # ── CHECK 5: at least one numeric and one categorical feature ──
    feature_cols = [c for c in df.columns if c != "churn"]
    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()

    if len(num_cols) == 0:
        raise ValueError(
            "Validation failed: no numeric feature columns found. "
            "The pipeline requires at least one numeric feature."
        )
    if len(cat_cols) == 0:
        raise ValueError(
            "Validation failed: no categorical feature columns found. "
            "The pipeline requires at least one categorical feature."
        )
