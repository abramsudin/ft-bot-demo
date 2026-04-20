# ============================================================
# stats/null_signal.py
#
# Null-pattern signal tests (Tests 4, 8):
#   test_null_signal             — churn gap null vs present
#   test_categorical_null_signal — wrapper for categorical cols
#
# Every function:
#   - Takes (column, target)
#   - Returns a dict
#   - Always includes a "meaningful" flag
# ============================================================

import pandas as pd


# ── TEST 4: Null Signal ───────────────────────────────────────
def test_null_signal(column, target):
    """
    Question: Do customers where this column IS null churn at a
    different rate than customers where it HAS a value?

    If yes → being null is itself predicting churn.
    The column's actual values might still be useless.
    Gap > 3 percentage points = meaningful.
    Need at least 100 rows on each side to be reliable.
    """
    is_null   = column.isnull()
    n_null    = int(is_null.sum())
    n_present = int((~is_null).sum())

    if n_null < 100 or n_present < 100:
        return {"test": "Null Signal",
                "error": f"Not enough rows (null={n_null}, present={n_present}, need 100 each)"}

    churn_null    = float(target[is_null].mean() * 100)
    churn_present = float(target[~is_null].mean() * 100)
    gap           = churn_null - churn_present

    return {
        "test"             : "Null Signal",
        "n_null"           : n_null,
        "n_present"        : n_present,
        "churn_null_pct"   : round(churn_null, 2),
        "churn_present_pct": round(churn_present, 2),
        "gap_pp"           : round(gap, 2),
        "meaningful"       : bool(abs(gap) > 3),
        "threshold"        : "|gap| > 3 percentage points"
    }


# ── TEST 8: Categorical Null Signal ──────────────────────────
def test_categorical_null_signal(column, target):
    """
    Question: For a categorical column — is the signal coming
    from the category VALUES, or just from whether it's null?

    Run this before Chi-Square. If the null gap is large,
    the is_null flag is what matters, not the categories.
    This prevents us from misattributing null-driven signal
    to the category values themselves.
    """
    result         = test_null_signal(column, target)
    result["test"] = "Categorical Null Signal"
    result["note"] = (
        "SIGNAL IS FROM NULLNESS — create is_null indicator"
        if result.get("meaningful")
        else "Signal comes from category values, not from being null"
        if "error" not in result else ""
    )
    return result
