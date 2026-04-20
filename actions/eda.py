# ============================================================
# actions/eda.py
#
# Action: EDA
#
# Deep-dive distribution and missingness analysis on one or
# more columns. Returns structured data for the formatter to
# narrate as plain-English text.
#
# Visualisation removed (v8): chart generation, PNG saving,
# chart_b64, chart_filename, and eda_shown_visuals tracking
# have all been removed. The formatter narrates the numbers
# directly. visual_type is retained as a narrative focus hint
# (e.g. "concentrate on null gap") but no chart is produced.
#
# Public API:
#   run(state: dict) -> dict
#
# No LLM calls. Pure Python.
#
# C-3 FIX (column resolution):
#   - Column resolution now checks ALL three sources in order:
#       1. intent_params["columns"]  (plural list — preferred)
#       2. intent_params["column"]   (singular legacy key)
#       3. state["active_focus"]     (follow-up context)
#     This prevents silent fall-through when the classifier emits
#     the singular key only, which caused eda.run() to report
#     "No column specified" even though the column was present.
#   - All three resolution paths are now logged when EDA_DEBUG=1.
# ============================================================

import os


def run(state: dict) -> dict:
    """
    EDA on one or more columns — returns structured stats for
    the formatter to narrate.
    """
    intent_params = state.get("intent_params", {})
    session       = state["session"]

    # ── C-3 FIX: Robust column resolution — three-stage fallback ─
    #
    # Stage 1: intent_params["columns"] (list, the canonical key)
    columns = list(intent_params.get("columns") or [])

    # Stage 2: intent_params["column"] (singular, legacy/classifier fallback)
    if not columns:
        legacy = intent_params.get("column")
        if legacy:
            columns = [legacy]

    # Stage 3: active_focus (follow-up visual requests with no explicit column)
    if not columns:
        af = state.get("active_focus")
        if isinstance(af, list):
            columns = list(af)
        elif af:
            columns = [af]

    if os.environ.get("EDA_DEBUG", "0") == "1":
        print(
            f"[eda] resolved columns={columns} | "
            f"intent_params keys={list(intent_params.keys())} | "
            f"active_focus={state.get('active_focus')} | "
            f"visual_type={intent_params.get('visual_type', 'default')}"
        )

    if not columns:
        return {"action_result": {
            "error": "No column specified. Say something like: 'deep dive Var22' or 'eda on Var22, Var34'"
        }}

    visual_type  = intent_params.get("visual_type", "default")
    feature_cols = session.get("feature_cols", [])
    df           = session.get("df")
    target       = session.get("target")
    verdict_df   = session.get("verdict_df")
    null_scan_df = session.get("null_scan_df")
    null_group_map = session.get("null_group_map", {})
    num_cols     = session.get("num_cols", [])
    cat_cols     = session.get("cat_cols", [])

    # ── Validate columns ──────────────────────────────────────
    valid_cols   = []
    invalid_msgs = []
    for col in columns:
        if col not in feature_cols:
            close = _find_close(col, feature_cols)
            msg   = f"'{col}' not found." + (f" Did you mean '{close}'?" if close else "")
            invalid_msgs.append(msg)
        else:
            valid_cols.append(col)

    if not valid_cols:
        return {"action_result": {
            "error": " ".join(invalid_msgs) or "None of the specified columns were found."
        }}

    if len(valid_cols) == 1:
        return _run_single(
            valid_cols[0], visual_type, df, target, verdict_df,
            null_scan_df, null_group_map, num_cols, cat_cols
        )

    return _run_multi(
        valid_cols, visual_type, df, target, verdict_df,
        null_scan_df, null_group_map, num_cols, cat_cols
    )


# ── Single column EDA ─────────────────────────────────────────

def _run_single(col, visual_type, df, target, verdict_df,
                null_scan_df, null_group_map, num_cols, cat_cols):

    col_type    = "numeric" if col in num_cols else "categorical"
    series      = df[col]
    verdict_row = _get_verdict_row(col, verdict_df)
    null_rate   = round(series.isnull().mean() * 100, 2)
    null_group  = null_group_map.get(col)
    null_info   = _lookup_null_scan(col, null_scan_df)

    return {
        "action_result": {
            "column"            : col,
            "col_type"          : col_type,
            "null_rate"         : null_rate,
            "null_group"        : null_group,
            "null_gap_pp"       : null_info.get("gap_pp"),
            "null_direction"    : null_info.get("direction"),
            "churn_when_null"   : null_info.get("churn_null"),
            "churn_when_present": null_info.get("churn_present"),
            "verdict"           : verdict_row.get("verdict", "UNKNOWN"),
            "confidence"        : verdict_row.get("confidence", 0),
            "risk_tag"          : verdict_row.get("risk_tag", ""),
            "profile"           : verdict_row.get("profile", ""),
            "signals"           : verdict_row.get("signals", "—"),
            "distribution"      : _compute_distribution(series, col_type),
            "churn_split"       : _compute_churn_split(series, target, col_type),
            "visual_focus"      : visual_type,
            "multi_column"      : False,
        },
        "active_focus": col,
    }


# ── Multi-column EDA ──────────────────────────────────────────

def _run_multi(valid_cols, visual_type, df, target, verdict_df,
               null_scan_df, null_group_map, num_cols, cat_cols):

    per_col_results = []
    for col in valid_cols:
        col_type    = "numeric" if col in num_cols else "categorical"
        series      = df[col]
        verdict_row = _get_verdict_row(col, verdict_df)
        null_rate   = round(series.isnull().mean() * 100, 2)
        null_info   = _lookup_null_scan(col, null_scan_df)

        per_col_results.append({
            "column"            : col,
            "col_type"          : col_type,
            "null_rate"         : null_rate,
            "null_group"        : null_group_map.get(col),
            "null_gap_pp"       : null_info.get("gap_pp"),
            "null_direction"    : null_info.get("direction"),
            "churn_when_null"   : null_info.get("churn_null"),
            "churn_when_present": null_info.get("churn_present"),
            "verdict"           : verdict_row.get("verdict", "UNKNOWN"),
            "confidence"        : verdict_row.get("confidence", 0),
            "risk_tag"          : verdict_row.get("risk_tag", ""),
            "signals"           : verdict_row.get("signals", "—"),
            "distribution"      : _compute_distribution(series, col_type),
            "churn_split"       : _compute_churn_split(series, target, col_type),
        })

    return {
        "action_result": {
            "multi_column" : True,
            "columns"      : valid_cols,
            "results"      : per_col_results,
            "visual_focus" : visual_type,
        },
        "active_focus": valid_cols,
    }


# ── Data computation helpers ──────────────────────────────────

def _compute_distribution(series, col_type: str) -> dict:
    clean = series.dropna()
    if col_type == "numeric":
        if clean.empty:
            return {"error": "No non-null values"}
        try:
            return {
                "count"       : int(len(clean)),
                "mean"        : round(float(clean.mean()), 4),
                "std"         : round(float(clean.std()), 4),
                "min"         : round(float(clean.min()), 4),
                "p25"         : round(float(clean.quantile(0.25)), 4),
                "median"      : round(float(clean.median()), 4),
                "p75"         : round(float(clean.quantile(0.75)), 4),
                "max"         : round(float(clean.max()), 4),
                "skew"        : round(float(clean.skew()), 4),
                "n_unique"    : int(clean.nunique()),
                "pct_negative": round(float((clean < 0).mean() * 100), 2),
            }
        except Exception as e:
            return {"error": str(e)}
    else:
        if clean.empty:
            return {"error": "No non-null values"}
        try:
            vc = clean.value_counts(normalize=True).head(10)
            return {
                "n_categories"  : int(clean.nunique()),
                "top_categories": {str(k): round(float(v * 100), 2) for k, v in vc.items()},
            }
        except Exception as e:
            return {"error": str(e)}


def _compute_churn_split(series, target, col_type: str) -> dict:
    try:
        import pandas as pd
        combined = pd.DataFrame({"val": series, "churn": target}).dropna(subset=["val"])
        if combined.empty or combined["churn"].nunique() < 2:
            return {"error": "Insufficient data for churn split"}
        overall_churn = round(float(combined["churn"].mean() * 100), 2)
        if col_type == "numeric":
            combined["decile"] = pd.qcut(combined["val"], q=10,
                                          labels=False, duplicates="drop")
            by_bucket = (combined.groupby("decile")["churn"]
                         .agg(["mean", "count"]).reset_index())
            buckets = [{"bucket": int(r["decile"]),
                        "churn_rate": round(float(r["mean"] * 100), 2),
                        "count": int(r["count"])}
                       for _, r in by_bucket.iterrows()]
            return {"overall_churn_rate": overall_churn, "by_decile": buckets}
        else:
            by_cat = (combined.groupby("val")["churn"]
                      .agg(["mean", "count"])
                      .sort_values("count", ascending=False).head(15))
            cats = {str(cat): {"churn_rate": round(float(r["mean"] * 100), 2),
                                "count": int(r["count"])}
                    for cat, r in by_cat.iterrows()}
            return {"overall_churn_rate": overall_churn, "by_category": cats}
    except Exception as e:
        return {"error": str(e)}


def _get_verdict_row(col: str, verdict_df) -> dict:
    if verdict_df is None:
        return {}
    try:
        rows = verdict_df[verdict_df["column"] == col]
        if rows.empty:
            return {}
        r = rows.iloc[0]
        return {
            "verdict"   : str(r.get("verdict", "")),
            "confidence": int(r.get("confidence", 0)),
            "risk_tag"  : str(r.get("risk_tag", "")),
            "profile"   : str(r.get("profile", "")),
            "signals"   : str(r.get("signals", "—")),
        }
    except Exception:
        return {}


def _lookup_null_scan(col: str, null_scan_df) -> dict:
    if null_scan_df is None or null_scan_df.empty:
        return {}
    try:
        col_field = "col" if "col" in null_scan_df.columns else "column"
        rows      = null_scan_df[null_scan_df[col_field] == col]
        if rows.empty:
            return {}
        r = rows.iloc[0]
        return {
            "gap_pp"       : _sf(r, ["gap_pp", "gap"]),
            "direction"    : _sv(r, ["direction"]),
            "null_rate"    : _sf(r, ["null_rate"]),
            "churn_null"   : _sf(r, ["churn_null", "churn_when_null"]),
            "churn_present": _sf(r, ["churn_present", "churn_when_present"]),
        }
    except Exception:
        return {}


def _find_close(col: str, feature_cols: list):
    col_lower = col.lower()
    for c in feature_cols:
        if c.lower().startswith(col_lower[:4]):
            return c
    return None


def _sf(row, keys):
    for k in keys:
        try:
            v = row[k]
            if v is not None and str(v) not in ("", "nan", "None"):
                return round(float(v), 2)
        except Exception:
            pass
    return None


def _sv(row, keys):
    for k in keys:
        try:
            v = row[k]
            if v is not None and str(v) not in ("", "nan", "None"):
                return str(v)
        except Exception:
            pass
    return None