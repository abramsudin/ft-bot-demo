# ============================================================
# actions/compare.py
#
# Action: COMPARE
#
# Cross-column comparison. Triggered when the user uses
# comparison language: "compare", "vs", "versus", "side by side",
# "which is better", "contrast", "difference between".
#
# Unlike EDA (per-column deep dive), COMPARE builds a structured
# side-by-side data object so the formatter can narrate direct
# differences in verdicts, null rates, confidence, and churn gap.
#
# Visualisation removed (v8): chart generation, PNG saving,
# chart_b64, chart_filename all removed. visual_type is retained
# as a narrative focus hint for the formatter.
#
# Public API:
#   run(state: dict) -> dict
#
# No LLM calls. Pure Python.
# ============================================================


def run(state: dict) -> dict:
    """
    Compare multiple columns — returns structured data for
    the formatter to narrate as a side-by-side comparison.
    """
    intent_params = state.get("intent_params", {})
    session       = state["session"]

    columns     = intent_params.get("columns", [])
    visual_type = intent_params.get("visual_type", "default")
    zone_param  = intent_params.get("zone")          # M-3: zone-scoped COMPARE

    # M-3: if classifier emitted a zone instead of an explicit column list,
    # resolve it here before the 2-column guard — otherwise the guard fires
    # on a 0- or 1-element list and rejects a perfectly valid zone query.
    if (not columns or len(columns) < 2) and zone_param:
        verdict_df_z = session.get("verdict_df")
        feature_cols_z = session.get("feature_cols", [])
        if verdict_df_z is not None and "verdict" in verdict_df_z.columns:
            zone_upper = str(zone_param).upper()
            mask = verdict_df_z["verdict"].astype(str).str.upper() == zone_upper
            if "column" in verdict_df_z.columns:
                matched = verdict_df_z.loc[mask, "column"].tolist()
            else:
                matched = verdict_df_z.loc[mask].index.tolist()
            columns = [c for c in matched if c in feature_cols_z]

    # Fallback to active_focus
    if not columns:
        af = state.get("active_focus")
        if isinstance(af, list):
            columns = af
        elif af:
            columns = [af]

    if len(columns) < 2:
        return {"action_result": {
            "error": (
                "COMPARE needs at least 2 columns. "
                "Try: 'compare Var22 and Var34' or 'Var7 vs Var83'."
            )
        }}

    feature_cols = session.get("feature_cols", [])
    valid_cols   = [c for c in columns if c in feature_cols]
    bad_cols     = [c for c in columns if c not in feature_cols]

    if len(valid_cols) < 2:
        return {"action_result": {
            "error": (
                f"Could not find enough valid columns to compare. "
                f"Unknown: {bad_cols}. Please check column names."
            )
        }}

    df           = session.get("df")
    target       = session.get("target")
    num_cols     = session.get("num_cols", [])
    null_scan_df = session.get("null_scan_df")
    verdict_df   = session.get("verdict_df")

    # ── Per-column metadata ───────────────────────────────────
    col_meta = {}
    for col in valid_cols:
        series    = df[col]
        col_type  = "numeric" if col in num_cols else "categorical"
        null_rate = round(series.isnull().mean() * 100, 2)
        null_info = _lookup_null_scan(col, null_scan_df)
        vrow      = _get_verdict_row(col, verdict_df)
        col_meta[col] = {
            "col_type"  : col_type,
            "null_rate" : null_rate,
            "null_gap"  : null_info.get("gap_pp"),
            "churn_when_null"   : null_info.get("churn_null"),
            "churn_when_present": null_info.get("churn_present"),
            "verdict"   : vrow.get("verdict", "UNKNOWN"),
            "confidence": vrow.get("confidence", 0),
            "signals"   : vrow.get("signals", "—"),
            "risk_tag"  : vrow.get("risk_tag", ""),
        }

    comparison_notes = _build_comparison_notes(valid_cols, col_meta)

    return {
        "action_result": {
            "multi_column"    : True,
            "compare"         : True,
            "columns"         : valid_cols,
            "col_meta"        : col_meta,
            "visual_focus"    : visual_type,
            "comparison_notes": comparison_notes,
            "bad_cols"        : bad_cols,
        },
        "active_focus": valid_cols,
    }


# ── Comparison narrative helper ───────────────────────────────

def _build_comparison_notes(valid_cols: list, col_meta: dict) -> dict:
    notes = {}

    with_conf = [(c, col_meta[c].get("confidence") or 0) for c in valid_cols]
    best_col, best_conf = max(with_conf, key=lambda x: x[1])
    notes["strongest_signal"] = {"column": best_col, "confidence": best_conf}

    null_rates   = {c: col_meta[c]["null_rate"] for c in valid_cols}
    highest_null = max(null_rates, key=lambda c: null_rates[c])
    notes["highest_null_rate"] = {"column": highest_null, "rate": null_rates[highest_null]}

    verdicts = {c: col_meta[c]["verdict"] for c in valid_cols}
    notes["verdicts"] = verdicts
    notes["keeps"]    = [c for c, v in verdicts.items() if v == "KEEP"]
    notes["drops"]    = [c for c, v in verdicts.items() if v in ("DROP", "DROP-NULL")]
    notes["flags"]    = [c for c, v in verdicts.items() if v == "FLAG"]

    return notes


# ── Shared helpers ────────────────────────────────────────────

def _lookup_null_scan(col: str, null_scan_df) -> dict:
    if null_scan_df is None or null_scan_df.empty:
        return {}
    try:
        col_field = "col" if "col" in null_scan_df.columns else "column"
        rows      = null_scan_df[null_scan_df[col_field] == col]
        if rows.empty:
            return {}
        r = rows.iloc[0]
        gap = None
        for k in ["gap_pp", "gap"]:
            try:
                v = r[k]
                if v is not None and str(v) not in ("", "nan", "None"):
                    gap = round(float(v), 2)
                    break
            except Exception:
                pass
        churn_null    = None
        churn_present = None
        for k in ["churn_null", "churn_when_null"]:
            try:
                v = r[k]
                if v is not None and str(v) not in ("", "nan", "None"):
                    churn_null = round(float(v), 2)
                    break
            except Exception:
                pass
        for k in ["churn_present", "churn_when_present"]:
            try:
                v = r[k]
                if v is not None and str(v) not in ("", "nan", "None"):
                    churn_present = round(float(v), 2)
                    break
            except Exception:
                pass
        return {"gap_pp": gap, "churn_null": churn_null, "churn_present": churn_present}
    except Exception:
        return {}


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
            "signals"   : str(r.get("signals", "—")),
        }
    except Exception:
        return {}