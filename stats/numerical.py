# ============================================================
# stats/numerical.py
#
# Numeric statistical tests (Tests 1, 2, 3, 5):
#   test_mann_whitney        — rank-based group difference
#   test_mutual_information  — information-theoretic signal
#   test_point_biserial      — linear correlation with churn
#   test_negative_signal     — churn gap for negative values
#
# Every function:
#   - Takes (column, target)
#   - Returns a dict
#   - Always includes a "meaningful" flag
# ============================================================

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pointbiserialr
from sklearn.metrics import mutual_info_score


# ── TEST 1: Mann-Whitney U ────────────────────────────────────
def test_mann_whitney(column, target):
    """
    Question: Do churners and non-churners have different
    value distributions in this numeric column?

    How it works: ranks all values together, then checks if
    churner ranks are systematically higher or lower.

    Pass condition: effect size > 0.1 AND p-value < 0.05
    Effect size tells us HOW different, not just IF different.
    """
    col_clean = column.dropna()
    tgt_clean = target[col_clean.index]
    g1 = col_clean[tgt_clean == 1]   # churners
    g0 = col_clean[tgt_clean == 0]   # not churned

    if len(g1) < 10 or len(g0) < 10:
        return {"test": "Mann-Whitney U", "error": "Not enough data (<10 per group)"}

    stat, p_value = stats.mannwhitneyu(g1, g0, alternative="two-sided")
    n1, n2        = len(g1), len(g0)
    effect_size   = abs(1 - (2 * stat) / (n1 * n2))

    return {
        "test"         : "Mann-Whitney U",
        "statistic"    : round(float(stat), 4),
        "p_value"      : round(float(p_value), 6),
        "effect_size"  : round(float(effect_size), 4),
        "n_churned"    : int(n1),
        "n_not_churned": int(n2),
        "meaningful"   : bool(effect_size > 0.1 and p_value < 0.05),
        "threshold"    : "effect_size > 0.1 AND p_value < 0.05"
    }


# ── TEST 2: Mutual Information ────────────────────────────────
def test_mutual_information(column, target):
    """
    Question: How much does knowing this column's value reduce
    our uncertainty about whether a customer churns?

    Score of 0 = tells us nothing.
    Score > 0.01 = meaningfully informative.
    Works by binning the column into 10 buckets first.
    """
    col_clean = column.dropna()
    tgt_clean = target[col_clean.index]

    if len(col_clean) < 50:
        return {"test": "Mutual Information", "error": "Not enough data (<50 rows)"}

    col_binned = pd.cut(col_clean, bins=10, labels=False, duplicates="drop")
    score      = mutual_info_score(tgt_clean, col_binned)

    return {
        "test"      : "Mutual Information",
        "score"     : round(float(score), 6),
        "meaningful": bool(score > 0.01),
        "threshold" : "score > 0.01"
    }


# ── TEST 3: Point-Biserial Correlation ───────────────────────
def test_point_biserial(column, target):
    """
    Question: Is there a linear relationship between this
    column's values and churn?

    Correlation ranges from -1 to +1.
    |corr| > 0.05 AND p < 0.05 = meaningful.
    Note: linear only. Use Mann-Whitney for non-linear patterns.
    """
    col_clean = column.dropna()
    tgt_clean = target[col_clean.index]

    if len(col_clean) < 30:
        return {"test": "Point-Biserial", "error": "Not enough data (<30 rows)"}
    if col_clean.nunique() < 2:
        return {"test": "Point-Biserial", "error": "Constant column — skip"}

    corr, p_value = pointbiserialr(tgt_clean, col_clean)

    return {
        "test"       : "Point-Biserial Correlation",
        "correlation": round(float(corr), 4),
        "p_value"    : round(float(p_value), 6),
        "meaningful" : bool(abs(corr) > 0.05 and p_value < 0.05),
        "threshold"  : "|correlation| > 0.05 AND p_value < 0.05"
    }


# ── TEST 5: Negative Value Signal ────────────────────────────
def test_negative_signal(column, target):
    """
    Question: Do customers with NEGATIVE values in this column
    churn differently than customers with positive values?

    Why: some columns use negative numbers as error codes or
    reversal markers, not real values. If negatives cluster
    with churners, that pattern is worth capturing.
    Gap > 3 percentage points = meaningful.
    """
    col_clean = column.dropna()
    tgt_clean = target[col_clean.index]
    has_neg   = col_clean < 0
    n_neg     = int(has_neg.sum())
    n_pos     = int((~has_neg).sum())

    if n_neg < 100 or n_pos < 100:
        return {"test": "Negative Signal",
                "error": f"Not enough negatives (n_neg={n_neg}, need 100)"}

    churn_neg = float(tgt_clean[has_neg].mean() * 100)
    churn_pos = float(tgt_clean[~has_neg].mean() * 100)
    gap       = churn_neg - churn_pos

    return {
        "test"              : "Negative Signal",
        "n_negative"        : n_neg,
        "n_non_negative"    : n_pos,
        "churn_negative_pct": round(churn_neg, 2),
        "churn_positive_pct": round(churn_pos, 2),
        "gap_pp"            : round(gap, 2),
        "meaningful"        : bool(abs(gap) > 3),
        "threshold"         : "|gap| > 3 percentage points"
    }
