# ============================================================
# pipeline/m3_scanner.py
#
# Wrapped from milestone3/m3_02_scan.py
#
# Public API:
#   run_m3_scan(df, target, null_group_map, output_path) -> dict
#
# What was changed vs the original:
#   - Import paths updated: stats.numerical / categorical /
#     null_signal / redundancy  (was m3_01_stat_tests)
#   - Entire script body wrapped inside run_m3_scan()
#   - Hardcoded TRAIN_PATH, LABELS_PATH, pd.read_csv removed
#   - Top-level print statements converted to progress markers
#     (kept inside the function so callers can see progress)
#   - OUTPUT_DIR / SCRIPT_DIR setup removed; caller passes output_path
#   - pd.to_pickle path uses output_path arg
#
# What was NOT changed:
#   - compute_confidence()   — logic identical
#   - compute_risk_tag()     — logic identical
#   - compute_profile()      — logic identical
#   - The scan loop          — logic identical
#   - The redundancy check   — logic identical
#   - Verdict rules          — logic identical
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd

from stats.numerical   import (test_mann_whitney, test_mutual_information,
                                test_point_biserial, test_negative_signal)
from stats.categorical import test_chi_square, test_cramers_v
from stats.null_signal import test_null_signal
from stats.redundancy  import test_spearman, test_categorical_pair

warnings.filterwarnings("ignore")


# ── CONFIDENCE SCORE ──────────────────────────────────────────
def compute_confidence(mw_result, mi_result, pb_result,
                       null_gap, neg_gap,
                       chi2_result, cramv_result,
                       col_type):
    """
    Compute a 0-100 confidence score based on HOW STRONGLY
    each test passed, not just whether it passed.

    Numeric columns: scored on MW effect size, MI score, PB correlation,
                     null gap magnitude, negative gap magnitude.
    Categorical columns: scored on Cramer's V magnitude, chi2 p-value,
                         null gap magnitude.

    Each component contributes a partial score, then all are averaged
    and scaled to 0-100.

    A column that barely scraped the threshold scores ~30-40.
    A column with strong effect sizes scores ~70-90.
    """
    scores = []

    if col_type == "numeric":
        if mw_result and "effect_size" in mw_result and "error" not in mw_result:
            es = mw_result["effect_size"]
            s  = min(100, max(0, (es - 0.1) / 0.4 * 80 + 20)) if es >= 0.1 else 0
            scores.append(s)

        if mi_result and "score" in mi_result and "error" not in mi_result:
            ms = mi_result["score"]
            s  = min(100, max(0, (ms - 0.01) / 0.09 * 80 + 20)) if ms >= 0.01 else 0
            scores.append(s)

        if pb_result and "correlation" in pb_result and "error" not in pb_result:
            corr = abs(pb_result["correlation"])
            s    = min(100, max(0, (corr - 0.05) / 0.25 * 80 + 20)) if corr >= 0.05 else 0
            scores.append(s)

    else:  # categorical
        if cramv_result and "cramers_v" in cramv_result and "error" not in cramv_result:
            v = cramv_result["cramers_v"]
            s = min(100, max(0, (v - 0.1) / 0.3 * 80 + 20)) if v >= 0.1 else 0
            scores.append(s)

        if chi2_result and "p_value" in chi2_result and "error" not in chi2_result:
            p = chi2_result["p_value"]
            s = min(100, max(0, (1 - p / 0.05) * 70 + 20)) if p < 0.05 else 0
            scores.append(s)

    if null_gap is not None and abs(null_gap) > 3:
        s = min(100, max(0, (abs(null_gap) - 3) / 12 * 80 + 20))
        scores.append(s)

    if neg_gap is not None and abs(neg_gap) > 3:
        s = min(100, max(0, (abs(neg_gap) - 3) / 12 * 80 + 20))
        scores.append(s)

    if not scores:
        return 0
    return int(round(sum(scores) / len(scores)))


# ── RISK TAG ──────────────────────────────────────────────────
def compute_risk_tag(verdict, null_rate, null_gap, passing_tests,
                     col_type, cramv):
    """
    Assign one of four risk tags:

    Null-Driven   : signal comes entirely from being null, not from values.
    High-Null Risk: real signal exists but null rate > 60%.
    Clean         : passes tests, null rate < 20%, no special concerns.
    Moderate      : passes tests but null rate 20-60%.

    Note: Redundancy Risk is added AFTER the redundancy check in Step 2.
    """
    if verdict in ("DROP", "DROP-NULL"):
        return "No Signal"

    value_tests  = {"MW", "MI", "PB", "CHI2", "CRAMV", "NEG"}
    passing_set  = (set(t.strip() for t in passing_tests.split(","))
                    if passing_tests != "—" else set())

    has_value_signal = bool(passing_set & value_tests)
    has_null_signal  = "NULL" in passing_set
    only_null        = has_null_signal and not has_value_signal

    if only_null:
        return "Null-Driven"
    if null_rate > 60:
        return "High-Null Risk"
    if null_rate <= 20:
        return "Clean"
    return "Moderate"


# ── PROFILE ───────────────────────────────────────────────────
def compute_profile(verdict, col_type, null_rate, null_gap,
                    passing_tests, confidence, risk_tag, n_signals):
    """
    Generate a short plain-English profile string that captures
    the column's signal character in one sentence.
    """
    if verdict == "DROP-NULL":
        return "100% null — no data, drop immediately"
    if verdict == "DROP":
        return "No signal detected across any test — safe to drop"

    passing_set = (set(t.strip() for t in passing_tests.split(","))
                   if passing_tests != "—" else set())

    # Null-driven case
    if risk_tag == "Null-Driven":
        gap_dir = "higher" if (null_gap or 0) > 0 else "lower"
        return (f"Signal comes entirely from being null "
                f"(missing customers churn {gap_dir}) — "
                f"create is_null indicator, raw values carry no signal")

    # Signal description
    if col_type == "numeric":
        if {"MW", "MI", "PB"}.issubset(passing_set):
            sig_desc = "strong signal across distribution, information, and linear tests"
        elif "MW" in passing_set and "MI" in passing_set:
            sig_desc = "non-linear distribution signal confirmed by mutual information"
        elif "MW" in passing_set:
            sig_desc = "distribution differs between churners and non-churners"
        elif "MI" in passing_set:
            sig_desc = "reduces uncertainty about churn (mutual information)"
        elif "PB" in passing_set:
            sig_desc = "linear relationship with churn target"
        elif "NEG" in passing_set:
            sig_desc = "negative values cluster with churn — possible error-code pattern"
        else:
            sig_desc = "marginal signal"
    else:
        if "CHI2" in passing_set and "CRAMV" in passing_set:
            sig_desc = "category distribution significantly and practically associated with churn"
        elif "CRAMV" in passing_set:
            sig_desc = "category distribution strongly associated with churn"
        elif "CHI2" in passing_set:
            sig_desc = "category distribution statistically differs between churn groups"
        else:
            sig_desc = "categorical association with churn"

    # Null context
    if null_rate == 0:
        null_desc = "fully complete"
    elif null_rate < 10:
        null_desc = f"nearly complete ({null_rate}% null)"
    elif null_rate < 30:
        null_desc = f"mostly present ({null_rate}% null)"
    elif null_rate < 60:
        null_desc = f"partially sparse ({null_rate}% null)"
    else:
        null_desc = f"very sparse ({null_rate}% null) — high missing-data risk"

    # Null gap addendum
    null_addendum = ""
    if null_gap is not None and abs(null_gap) > 3:
        null_addendum = f", plus null pattern predicts churn ({null_gap:+.1f}pp gap)"

    # Confidence label
    if confidence >= 70:
        conf_label = "strong"
    elif confidence >= 45:
        conf_label = "moderate"
    else:
        conf_label = "borderline"

    verdict_prefix = "KEEP" if verdict == "KEEP" else "FLAG (borderline)"

    return (f"{verdict_prefix} — {conf_label} {sig_desc}, {null_desc}"
            f"{null_addendum}")


# ── run_m3_scan ───────────────────────────────────────────────
def run_m3_scan(df: pd.DataFrame,
                target: pd.Series,
                null_group_map: dict,
                output_path: str) -> dict:
    """
    Run the full M3 statistical scan over every feature column.

    Steps:
      1. Run all statistical tests across every column.
      2. Assign each column a verdict: KEEP / FLAG / DROP / DROP-NULL.
      3. Check KEEP columns for redundant pairs (Spearman / Cramer's V).
      4. Apply confidence score, null group, risk tag, profile to each row.
      5. Save scan_results.pkl to output_path.

    Parameters
    ----------
    df            : DataFrame from loader.load_data() — must include "churn".
    target        : Binary churn Series (0/1), same index as df.
    null_group_map: dict[col -> group_label] from m2_eda_scanner.find_null_groups().
    output_path   : Directory where scan_results.pkl will be saved.

    Returns
    -------
    dict with keys:
        verdict_df       — full per-column results DataFrame
        pairs_df         — redundant pairs DataFrame
        num_cols         — list of numeric feature column names
        cat_cols         — list of categorical feature column names
        redundancy_drop  — list of columns recommended to drop
    """
    os.makedirs(output_path, exist_ok=True)

    feature_cols = [c for c in df.columns if c != "churn"]
    num_cols     = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols     = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
    null_rates   = df[feature_cols].isnull().mean() * 100

    print(f"Loaded: {df.shape[0]:,} rows | {len(feature_cols)} columns "
          f"({len(num_cols)} numeric, {len(cat_cols)} categorical)")

    # ── STEP 1: FULL SCAN ─────────────────────────────────────
    print("Step 1/3: Running statistical tests... (~1-2 mins)")

    all_results = []

    for col in feature_cols:
        nr       = null_rates[col]
        col_type = "numeric" if col in num_cols else "categorical"

        # Null signal applies to all column types
        is_null   = df[col].isnull()
        n_null    = is_null.sum()
        n_present = (~is_null).sum()
        null_gap  = None
        if n_null >= 100 and n_present >= 100:
            cn       = target[is_null].mean() * 100
            cp       = target[~is_null].mean() * 100
            null_gap = round(cn - cp, 2)

        mw_pass = mi_score = pb_pass = neg_gap = chi2_pass = cramv = None
        mw_result = mi_result = pb_result = chi2_result = cramv_result = None

        if col_type == "numeric":
            col_clean = df[col].dropna()
            if len(col_clean) > 100 and col_clean.nunique() > 1:
                mw_result = test_mann_whitney(df[col], target)
                if "error" not in mw_result:
                    mw_pass = mw_result["meaningful"]

                mi_result = test_mutual_information(df[col], target)
                if "error" not in mi_result:
                    mi_score = mi_result["score"]

                pb_result = test_point_biserial(df[col], target)
                if "error" not in pb_result:
                    pb_pass = pb_result["meaningful"]

                neg_result = test_negative_signal(df[col], target)
                if "error" not in neg_result:
                    neg_gap = neg_result["gap_pp"]

        else:  # categorical
            col_clean = df[col].dropna()
            if len(col_clean) > 100 and col_clean.nunique() >= 2:
                chi2_result = test_chi_square(df[col], target)
                if "error" not in chi2_result:
                    chi2_pass = chi2_result["meaningful"]

                cramv_result = test_cramers_v(df[col], target)
                if "error" not in cramv_result:
                    cramv = cramv_result["cramers_v"]

        # Count passing signals
        passing = []
        if mw_pass  == True:               passing.append("MW")
        if mi_score and mi_score > 0.01:   passing.append("MI")
        if pb_pass  == True:               passing.append("PB")
        if null_gap and abs(null_gap) > 3: passing.append("NULL")
        if neg_gap  and abs(neg_gap)  > 3: passing.append("NEG")
        if chi2_pass == True:              passing.append("CHI2")
        if cramv    and cramv > 0.1:       passing.append("CRAMV")

        n_sig       = len(passing)
        signals_str = ", ".join(passing) if passing else "—"

        # Assign verdict
        if nr == 100:                         verdict = "DROP-NULL"
        elif n_sig >= 2:                      verdict = "KEEP"
        elif n_sig == 1:                      verdict = "FLAG"
        elif null_gap and abs(null_gap) > 3:  verdict = "FLAG"
        else:                                 verdict = "DROP"

        confidence = compute_confidence(
            mw_result, mi_result, pb_result,
            null_gap, neg_gap,
            chi2_result, cramv_result,
            col_type
        )

        null_group = null_group_map.get(col, "—")

        risk_tag = compute_risk_tag(
            verdict, nr, null_gap, signals_str, col_type, cramv
        )

        profile = compute_profile(
            verdict, col_type, nr, null_gap,
            signals_str, confidence, risk_tag, n_sig
        )

        all_results.append({
            "column"    : col,
            "type"      : col_type,
            "null_rate" : round(nr, 1),
            "null_gap"  : null_gap,
            "mw"        : mw_pass,
            "mi_score"  : mi_score,
            "pb"        : pb_pass,
            "neg_gap"   : neg_gap,
            "chi2"      : chi2_pass,
            "cramers_v" : cramv,
            "signals"   : signals_str,
            "n_signals" : n_sig,
            "verdict"   : verdict,
            "confidence": confidence,
            "null_group": null_group,
            "risk_tag"  : risk_tag,
            "profile"   : profile,
        })

    verdict_df = pd.DataFrame(all_results)

    keep_df     = verdict_df[verdict_df.verdict == "KEEP"]
    flag_df     = verdict_df[verdict_df.verdict == "FLAG"]
    drop_df     = verdict_df[verdict_df.verdict == "DROP"]
    dropn_df    = verdict_df[verdict_df.verdict == "DROP-NULL"]
    null_ind    = verdict_df[verdict_df.null_gap.notna() & (verdict_df.null_gap.abs() > 3)]
    keep_num_df = keep_df[keep_df.type == "numeric"]
    keep_cat_df = keep_df[keep_df.type == "categorical"]

    print("\nVERDICT SUMMARY:")
    print(f"  KEEP  (2+ tests)  : {len(keep_df)}")
    print(f"  FLAG  (1 test)    : {len(flag_df)}")
    print(f"  DROP  (no signal) : {len(drop_df)}")
    print(f"  DROP  (100% null) : {len(dropn_df)}")
    print(f"  NULL INDICATORS   : {len(null_ind)} columns need is_null flag")

    if len(keep_df) > 0:
        strong   = (keep_df.confidence >= 70).sum()
        moderate = ((keep_df.confidence >= 45) & (keep_df.confidence < 70)).sum()
        border   = (keep_df.confidence < 45).sum()
        print(f"\n  KEEP confidence breakdown:")
        print(f"    Strong   (70-100) : {strong}")
        print(f"    Moderate (45-69)  : {moderate}")
        print(f"    Borderline (0-44) : {border}")

    print(f"\n  RISK TAG BREAKDOWN (KEEP + FLAG columns):")
    keep_flag_df = verdict_df[verdict_df.verdict.isin(["KEEP", "FLAG"])]
    for tag, count in keep_flag_df["risk_tag"].value_counts().items():
        print(f"    {tag:<18}: {count}")

    # ── STEP 2: REDUNDANCY CHECK ──────────────────────────────
    print("\nStep 2/3: Checking redundant pairs among KEEP columns...")

    keep_num_cols = keep_num_df["column"].tolist()
    keep_cat_cols = keep_cat_df["column"].tolist()
    pair_rows     = []

    for i, c1 in enumerate(keep_num_cols):
        for c2 in keep_num_cols[i + 1:]:
            r = test_spearman(df[c1], df[c2])
            if "error" not in r and r["redundant"]:
                pair_rows.append({
                    "col_1"    : c1,
                    "col_2"    : c2,
                    "type"     : "numeric",
                    "metric"   : "Spearman r",
                    "value"    : r["correlation"],
                    "threshold": "|r| > 0.9",
                })

    for i, c1 in enumerate(keep_cat_cols):
        for c2 in keep_cat_cols[i + 1:]:
            r = test_categorical_pair(df[c1], df[c2])
            if "error" not in r and r["redundant"]:
                pair_rows.append({
                    "col_1"    : c1,
                    "col_2"    : c2,
                    "type"     : "categorical",
                    "metric"   : "Cramer's V",
                    "value"    : r["cramers_v"],
                    "threshold": "V > 0.8",
                })

    pairs_df = (pd.DataFrame(pair_rows) if pair_rows
                else pd.DataFrame(
                    columns=["col_1", "col_2", "type", "metric", "value", "threshold"]))

    print(f"  Redundant pairs found: {len(pairs_df)}")

    # ── STEP 3: REDUNDANCY RISK TAG + TIE-BREAK ───────────────
    print("\nStep 3/3: Applying redundancy risk tags and tie-breaking...")

    redundancy_drop = set()

    if len(pairs_df) > 0:
        for _, pair in pairs_df.iterrows():
            c1, c2 = pair["col_1"], pair["col_2"]

            verdict_df.loc[verdict_df.column == c1, "risk_tag"] = "Redundancy Risk"
            verdict_df.loc[verdict_df.column == c2, "risk_tag"] = "Redundancy Risk"

            row_c1 = verdict_df[verdict_df.column == c1].iloc[0]
            row_c2 = verdict_df[verdict_df.column == c2].iloc[0]

            if row_c1["n_signals"] < row_c2["n_signals"]:
                drop_col = c1
            elif row_c2["n_signals"] < row_c1["n_signals"]:
                drop_col = c2
            elif row_c1["null_rate"] > row_c2["null_rate"]:
                drop_col = c1
            else:
                drop_col = c2

            redundancy_drop.add(drop_col)

        pairs_df["recommended_drop"] = pairs_df.apply(
            lambda r: r["col_1"] if r["col_1"] in redundancy_drop else r["col_2"],
            axis=1
        )

        print(f"  Columns recommended to drop: {sorted(redundancy_drop)}")

        for col in redundancy_drop:
            old_profile = verdict_df.loc[verdict_df.column == col, "profile"].values[0]
            verdict_df.loc[verdict_df.column == col, "profile"] = (
                old_profile + " — REDUNDANT with another KEEP column, recommend drop"
            )

    # ── SAVE ──────────────────────────────────────────────────
    pkl_path = os.path.join(output_path, "scan_results.pkl")
    payload  = {
        "verdict_df"     : verdict_df,
        "pairs_df"       : pairs_df,
        "num_cols"       : num_cols,
        "cat_cols"       : cat_cols,
        "redundancy_drop": list(redundancy_drop),
    }
    pd.to_pickle(payload, pkl_path)
    print(f"\nSaved: {pkl_path}")
    print("  Fields in verdict_df: confidence, null_group, risk_tag, profile")
    print("  Fields in pairs_df  : recommended_drop")

    return payload
