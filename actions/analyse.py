# ============================================================
# actions/analyse.py
#
# Action: ANALYSE
#
# v8: EDA intent merged into ANALYSE.
#   - intent_params["deep_dive"] = True → delegates to eda.run()
#     for full distribution/missingness breakdown
#   - intent_params["deep_dive"] = False (default) → stat profile
#     lookup from verdict_df (original behaviour)
#   - intent_params["visual_focus"] passed through to eda for
#     narrative focus (no chart is generated)
#
# Returns partial state with:
#   - action_result : structured analysis dict / multi-column results
#   - active_focus  : column name (str) or list of names
#
# No LLM calls. No global side effects. Pure Python.
#
# C-3 FIX:
#   - deep_dive is now read exclusively from intent_params (never from
#     state directly), using intent_params.get("deep_dive", False).
#   - Before delegating to eda.run(), the state dict is explicitly
#     patched so intent_params["columns"] is always a list (handles
#     the legacy singular "column" key that the classifier may emit).
#   - A debug log (ANALYSE_DEBUG=1) prints the intent_params on entry
#     so it is easy to verify that deep_dive and columns arrived correctly.
# ============================================================

import os


def run(state: dict) -> dict:
    """
    Route to deep-dive EDA or stat-profile lookup based on deep_dive flag.

    deep_dive=True  → delegates to eda.run() for full distribution/null breakdown.
    deep_dive=False → stat profile from verdict_df (original ANALYSE behaviour).

    Single column → action_result is a flat dict.
    Multiple columns → action_result["results"] as list of per-column dicts.
    Zone path → if intent_params contains "zone", returns zone-level aggregate.
    """
    session       = state["session"]
    intent_params = state["intent_params"]

    # ── C-3 FIX: debug logging ────────────────────────────────
    if os.environ.get("ANALYSE_DEBUG", "0") == "1":
        print(
            f"[analyse] intent_params on entry: {intent_params} | "
            f"active_focus={state.get('active_focus')}"
        )

    # ── C-3 FIX: deep_dive routing ────────────────────────────
    # Always read from intent_params — never from state directly.
    # Coerce to bool to guard against the LLM emitting the string "true".
    raw_deep_dive = intent_params.get("deep_dive", False)
    deep_dive = raw_deep_dive is True or str(raw_deep_dive).lower() == "true"

    if deep_dive:
        import actions.eda as eda_module

        # ── C-3 FIX: normalise columns before handing off to eda ──
        # The classifier may emit "column" (singular, legacy) instead of
        # "columns" (list). If eda.py only checks the plural key it silently
        # falls through to active_focus — potentially None or stale.
        # We patch a shallow copy of intent_params so eda.run() always sees
        # the plural "columns" key as a proper list.
        fixed_params = dict(intent_params)  # shallow copy — do not mutate state

        if not fixed_params.get("columns"):
            # Try to promote singular "column" key → ["column"]
            singular = fixed_params.get("column")
            if singular:
                fixed_params["columns"] = [singular]
            else:
                # Fall back to active_focus so follow-up visual requests work
                af = state.get("active_focus")
                if isinstance(af, list):
                    fixed_params["columns"] = af
                elif af:
                    fixed_params["columns"] = [af]
                # else: leave as [] and let eda.py report the error cleanly

        # Build a patched state copy so eda.run() sees the normalised params
        patched_state = {**state, "intent_params": fixed_params}

        if os.environ.get("ANALYSE_DEBUG", "0") == "1":
            print(
                f"[analyse] delegating to eda.run() | "
                f"fixed columns={fixed_params.get('columns')} | "
                f"visual_type={fixed_params.get('visual_type', 'default')}"
            )

        return eda_module.run(patched_state)

    # ── Issue #6: Check for zone-level analyse request ────────
    zone_param = (intent_params.get("zone") or "").strip().upper()
    if zone_param:
        return _analyse_zone(zone_param, state)

    # ── Resolve column list ───────────────────────────────────
    # Support both new "columns" list and legacy "column" singular
    columns = intent_params.get("columns") or []
    if not columns:
        legacy = intent_params.get("column")
        columns = [legacy] if legacy else []

    # Fall back to active_focus if nothing extracted
    if not columns:
        af = state.get("active_focus")
        if isinstance(af, list):
            columns = af
        elif af:
            columns = [af]

    if not columns:
        return {
            "action_result": {
                "error": "No column name was provided. Please specify which column(s) you'd like to analyse."
            },
            "active_focus": state.get("active_focus"),
        }

    # ── Single-column path (original behaviour) ───────────────
    if len(columns) == 1:
        result = _analyse_single(columns[0], state)
        new_focus = columns[0] if "error" not in result else state.get("active_focus")
        return {
            "action_result": result,
            "active_focus" : new_focus,
        }

    # ── Multi-column path ─────────────────────────────────────
    per_col_results = []
    valid_cols      = []
    for col in columns:
        r = _analyse_single(col, state)
        per_col_results.append(r)
        if "error" not in r:
            valid_cols.append(col)

    # Build a brief comparison summary for the formatter
    summary = _build_comparison_summary(per_col_results)

    action_result = {
        "multi_column": True,
        "columns"     : columns,
        "results"     : per_col_results,
        "summary"     : summary,
    }

    new_focus = valid_cols if len(valid_cols) > 1 else (valid_cols[0] if valid_cols else state.get("active_focus"))

    return {
        "action_result": action_result,
        "active_focus" : new_focus,
    }


# ── Zone-level analysis (Issue #6) ───────────────────────────

def _analyse_zone(zone: str, state: dict) -> dict:
    """
    Aggregate analysis for all columns in a given verdict zone
    (e.g. "FLAG", "KEEP", "DROP").

    Returns a zone-level summary with average confidence, count,
    top risk tags, and a plain-text recommendation so the formatter
    can give a meaningful answer to questions like "Is it a good idea
    to remove all flagged columns?"
    """
    session      = state["session"]
    verdict_df   = session.get("verdict_df")
    feature_cols = session.get("feature_cols", [])

    if verdict_df is None:
        return {
            "action_result": {"error": "Statistical scan results are not available."},
            "active_focus" : state.get("active_focus"),
        }

    # Resolve columns in this zone
    try:
        if "column" in verdict_df.columns:
            mask = verdict_df["verdict"].astype(str).str.upper() == zone
            zone_cols = verdict_df.loc[mask, "column"].tolist()
        else:
            mask = verdict_df["verdict"].astype(str).str.upper() == zone
            zone_cols = verdict_df.loc[mask].index.tolist()
        zone_cols = [c for c in zone_cols if c in feature_cols]
    except Exception:
        zone_cols = []

    if not zone_cols:
        return {
            "action_result": {
                "error": f"No columns found with verdict '{zone}'."
            },
            "active_focus": state.get("active_focus"),
        }

    # Aggregate stats
    confidences = []
    risk_tag_counts: dict = {}
    for col in zone_cols:
        try:
            rows = (
                verdict_df[verdict_df["column"] == col]
                if "column" in verdict_df.columns
                else verdict_df[verdict_df.index == col]
            )
            if rows.empty:
                continue
            r = rows.iloc[0]
            conf = _safe_get(r.to_dict(), ["confidence", "confidence_score", "score"])
            if conf is not None:
                confidences.append(float(conf))
            tag = _safe_get(r.to_dict(), ["risk_tag", "tag", "risk"])
            if tag:
                risk_tag_counts[tag] = risk_tag_counts.get(tag, 0) + 1
        except Exception:
            pass

    avg_confidence = round(sum(confidences) / len(confidences), 3) if confidences else None
    top_tags = sorted(risk_tag_counts.items(), key=lambda x: -x[1])[:5]

# REPLACEMENT — enriched return with per_col_ranked, user_question, and active_focus fix
    # Build per-column ranked list for recommendation questions
    per_col_ranked = []
    for col in zone_cols:
        try:
            rows = (
                verdict_df[verdict_df["column"] == col]
                if "column" in verdict_df.columns
                else verdict_df[verdict_df.index == col]
            )
            if rows.empty:
                continue
            r = rows.iloc[0].to_dict()
            per_col_ranked.append({
                "column"    : col,
                "confidence": _safe_get(r, ["confidence", "confidence_score", "score"]),
                "risk_tag"  : _safe_get(r, ["risk_tag", "tag", "risk"]),
                "null_rate" : _safe_get(r, ["null_rate", "missing_rate", "null_pct"]),
                "verdict"   : str(_safe_get(r, ["verdict", "decision", "recommendation"], "UNKNOWN")).upper(),
            })
        except Exception:
            pass

    # Sort by confidence descending so formatter can slice top/bottom
    per_col_ranked.sort(key=lambda x: x.get("confidence") or 0, reverse=True)

    # Extract user's latest message so formatter can branch on intent
    user_question = _get_latest_user_message(state)

    return {
        "action_result": {
            "zone_analysis"  : True,
            "zone"           : zone,
            "column_count"   : len(zone_cols),
            "columns"        : zone_cols,
            "avg_confidence" : avg_confidence,
            "top_risk_tags"  : [{"tag": t, "count": c} for t, c in top_tags],
            "per_col_ranked" : per_col_ranked,      # ← NEW: ranked per-column data
            "user_question"  : user_question,        # ← NEW: what the user actually asked
        },
        "active_focus": zone_cols,  # ← FIXED: was state.get("active_focus") — always None for zones
    }

# ── Single column analysis ────────────────────────────────────

def _analyse_single(column: str, state: dict) -> dict:
    """
    Look up one column and return a structured result dict.
    Returns a dict with "error" key on failure.
    """
    session        = state["session"]
    feature_cols   = session.get("feature_cols", [])
    verdict_df     = session.get("verdict_df")
    null_scan_df   = session.get("null_scan_df")
    null_group_map = session.get("null_group_map", {})

    if column not in feature_cols:
        close = _fuzzy_match(column, feature_cols)
        suggestion = f" Did you mean '{close}'?" if close else ""
        return {
            "error" : f"'{column}' was not found in the dataset.{suggestion}",
            "column": column,
        }

    if verdict_df is None:
        return {
            "error" : "Statistical scan results are not available.",
            "column": column,
        }

    col_rows = (
        verdict_df[verdict_df["column"] == column]
        if "column" in verdict_df.columns
        else verdict_df[verdict_df.index == column]
    )

    if col_rows.empty:
        return {
            "error" : f"No scan results found for '{column}'.",
            "column": column,
        }

    row      = col_rows.iloc[0]
    row_dict = row.to_dict()

    num_cols = session.get("num_cols", [])
    cat_cols = session.get("cat_cols", [])
    profile  = "numeric" if column in num_cols else "categorical" if column in cat_cols else "unknown"

    null_rate  = _safe_get(row_dict, ["null_rate", "missing_rate", "null_pct"])
    null_group = null_group_map.get(column)
    signals    = _extract_signals(row_dict, profile)
    risk_tag   = _safe_get(row_dict, ["risk_tag", "tag", "risk"])
    verdict    = str(_safe_get(row_dict, ["verdict", "decision", "recommendation"], "UNKNOWN")).upper()
    confidence = _safe_get(row_dict, ["confidence", "confidence_score", "score"])

    null_signal_info = {}
    if null_scan_df is not None:
        try:
            col_field = "col" if "col" in null_scan_df.columns else "column"
            ns_rows   = null_scan_df[null_scan_df[col_field] == column]
            if not ns_rows.empty:
                ns_row = ns_rows.iloc[0].to_dict()
                null_signal_info = {
                    "churn_when_null"   : _safe_get(ns_row, ["churn_when_null", "churn_null"]),
                    "churn_when_present": _safe_get(ns_row, ["churn_when_present", "churn_present"]),
                    "gap_pp"            : _safe_get(ns_row, ["gap_pp", "gap", "churn_gap"]),
                }
        except Exception:
            pass

    return {
        "column"      : column,
        "verdict"     : verdict,
        "confidence"  : confidence,
        "profile"     : profile,
        "risk_tag"    : risk_tag,
        "null_rate"   : null_rate,
        "null_group"  : null_group,
        "signals"     : signals,
        "test_details": row_dict,
        **null_signal_info,
    }


def _build_comparison_summary(results: list[dict]) -> str:
    """
    Build a brief plain-text comparison summary for the formatter
    to use when narrating multi-column ANALYSE results.
    """
    valid = [r for r in results if "error" not in r]
    if not valid:
        return "No valid results to compare."

    keeps  = [r["column"] for r in valid if r.get("verdict") == "KEEP"]
    flags  = [r["column"] for r in valid if r.get("verdict") == "FLAG"]
    drops  = [r["column"] for r in valid if r.get("verdict") in ("DROP", "DROP-NULL")]

    parts = []
    if keeps:
        parts.append(f"{', '.join(keeps)} recommended to KEEP")
    if flags:
        parts.append(f"{', '.join(flags)} flagged for review")
    if drops:
        parts.append(f"{', '.join(drops)} recommended to DROP")

    # Highest confidence column
    with_conf = [(r["column"], r.get("confidence") or 0) for r in valid]
    if with_conf:
        best_col, best_conf = max(with_conf, key=lambda x: x[1])
        parts.append(f"strongest signal: {best_col} (confidence {best_conf})")

    return "; ".join(parts) if parts else "Results retrieved for all columns."


# ── Helpers ───────────────────────────────────────────────────

def _safe_get(d: dict, keys: list[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _extract_signals(row_dict: dict, profile: str) -> list[dict]:
    signals = []

    if profile == "numeric":
        candidates = [
            ("mann_whitney_p",   "Mann-Whitney p-value",    "lower is stronger signal"),
            ("mutual_info",      "Mutual Information",      "higher means more predictive"),
            ("point_biserial_r", "Point-Biserial r",        "magnitude indicates correlation"),
            ("negative_signal",  "Negative signal flag",    "True means inverse relationship"),
            ("spearman_r",       "Spearman redundancy",     "high value means redundant with another kept column"),
        ]
    else:
        candidates = [
            ("chi2_p",      "Chi-Square p-value", "lower is stronger signal"),
            ("cramers_v",   "Cramer's V",         "higher means stronger association"),
            ("null_signal", "Null signal",        "True means missingness is informative"),
        ]

    for key, label, interpretation in candidates:
        val = _safe_get(row_dict, [key, key.replace("_", "-")])
        if val is not None:
            signals.append({
                "test"          : label,
                "value"         : val,
                "interpretation": interpretation,
            })

    return signals


def _fuzzy_match(target: str, candidates: list[str]) -> str | None:
    target_lower = target.lower()
    for c in candidates:
        if target_lower in c.lower() or c.lower() in target_lower:
            return c
    return None

def _get_latest_user_message(state: dict) -> str:
    """Extract the most recent user message content, lowercased."""
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "user":
            return msg.get("content", "").lower()
    return ""