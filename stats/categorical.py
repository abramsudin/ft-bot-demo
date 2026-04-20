# ============================================================
# stats/categorical.py
#
# Categorical statistical tests (Tests 6, 7):
#   test_chi_square  — does category spread differ by churn?
#   test_cramers_v   — how strong is that association?
#
# Every function:
#   - Takes (column, target)
#   - Returns a dict
#   - Always includes a "meaningful" flag
# ============================================================

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


# ── TEST 6: Chi-Square ────────────────────────────────────────
def test_chi_square(column, target):
    """
    Question: Does the spread of categories differ between
    churners and non-churners?

    How it works: builds a table of category counts × churn,
    then checks if the pattern is random or systematic.
    p < 0.05 = systematic (not random).

    WARNING: with 50,000 rows this test is very sensitive.
    ALWAYS pair with Cramer's V to measure practical size.
    """
    col_clean = column.dropna()
    tgt_clean = target[col_clean.index]

    if col_clean.nunique() < 2:
        return {"test": "Chi-Square", "error": "Need at least 2 categories"}

    contingency           = pd.crosstab(col_clean, tgt_clean)
    chi2, p_value, dof, _ = chi2_contingency(contingency)

    return {
        "test"      : "Chi-Square",
        "chi2"      : round(float(chi2), 4),
        "p_value"   : round(float(p_value), 6),
        "dof"       : int(dof),
        "meaningful": bool(p_value < 0.05),
        "threshold" : "p_value < 0.05 (always check Cramer's V too)"
    }


# ── TEST 7: Cramer's V ────────────────────────────────────────
def test_cramers_v(column, target):
    """
    Question: How STRONG is the association between this
    categorical column and churn?

    Chi-Square says IF an association exists.
    Cramer's V says HOW BIG it is (0 = none, 1 = perfect).
    V > 0.1 = practically meaningful.

    Use both: chi-square filters, cramer's V measures.
    """
    col_clean = column.dropna()
    tgt_clean = target[col_clean.index]

    if col_clean.nunique() < 2:
        return {"test": "Cramer's V", "error": "Need at least 2 categories"}

    contingency   = pd.crosstab(col_clean, tgt_clean)
    chi2, _, _, _ = chi2_contingency(contingency)
    n             = int(contingency.sum().sum())
    r, k          = contingency.shape
    v             = float(np.sqrt(chi2 / (n * (min(r, k) - 1))))

    return {
        "test"      : "Cramer's V",
        "cramers_v" : round(v, 4),
        "n"         : n,
        "meaningful": bool(v > 0.1),
        "threshold" : "Cramer's V > 0.1"
    }
