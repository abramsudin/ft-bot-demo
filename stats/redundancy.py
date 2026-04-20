# ============================================================
# stats/redundancy.py
#
# Redundancy / pair tests (Tests 9, 10):
#   test_spearman          — numeric column-pair redundancy
#   test_categorical_pair  — categorical column-pair redundancy
#
# Every function:
#   - Takes (col1, col2)
#   - Returns a dict
#   - Always includes a "redundant" flag
# ============================================================

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import chi2_contingency


# ── TEST 9: Spearman Correlation (pair redundancy) ────────────
def test_spearman(col1, col2):
    """
    Question: Are these two numeric columns measuring almost
    the same thing?

    If |correlation| > 0.9 → they are essentially duplicates.
    Keep the one that passed more individual tests.
    Operates on rows where BOTH columns have values.
    """
    combined = pd.DataFrame({"a": col1, "b": col2}).dropna()

    if len(combined) < 30:
        return {"test": "Spearman Correlation", "error": "Not enough overlapping rows (<30)"}

    corr, p_value = spearmanr(combined["a"], combined["b"])

    return {
        "test"       : "Spearman Correlation",
        "correlation": round(float(corr), 4),
        "p_value"    : round(float(p_value), 6),
        "n_overlap"  : len(combined),
        "redundant"  : bool(abs(corr) > 0.9),
        "threshold"  : "|correlation| > 0.9 = redundant pair"
    }


# ── TEST 10: Categorical Pair ─────────────────────────────────
def test_categorical_pair(col1, col2):
    """
    Question: Are these two categorical columns so strongly
    associated with each other that one is redundant?

    Uses Cramer's V between the two columns (not vs churn).
    V > 0.8 → one is essentially a copy of the other.
    Keep whichever passed more tests in the individual scan.
    """
    combined = pd.DataFrame({"a": col1, "b": col2}).dropna()

    if len(combined) < 30:
        return {"test": "Categorical Pair", "error": "Not enough overlapping rows (<30)"}
    if combined["a"].nunique() < 2 or combined["b"].nunique() < 2:
        return {"test": "Categorical Pair", "error": "Need at least 2 categories in each column"}

    contingency   = pd.crosstab(combined["a"], combined["b"])
    chi2, _, _, _ = chi2_contingency(contingency)
    n             = int(contingency.sum().sum())
    r, k          = contingency.shape
    v             = float(np.sqrt(chi2 / (n * (min(r, k) - 1))))

    return {
        "test"     : "Categorical Pair (Cramer's V)",
        "cramers_v": round(v, 4),
        "n"        : n,
        "redundant": bool(v > 0.8),
        "threshold": "Cramer's V > 0.8 = redundant pair"
    }
