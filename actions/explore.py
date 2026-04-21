# ============================================================
# actions/explore.py
#
# Action: EXPLORE
#
# Filters verdict_df by one or more criteria and returns a
# ranked summary of matching columns. The formatter turns this
# into a browsable list for the user.
#
# Public API:
#   run(state: dict) -> dict
#
# Supported filters (from intent_params["filter"]):
#   verdict        — "KEEP" | "FLAG" | "DROP"
#   risk_tag       — any risk_tag value in verdict_df
#   null_group     — any null_group value (G1, G2, …)
#   type           — "numeric" | "categorical"
#   confidence     — "high" | "medium" | "low"   (band)
#
# Returns partial state with:
#   - action_result : filtered column list + summary stats
#
# No LLM calls. No state mutations. Pure Python read-only.
# ============================================================

import re


# Confidence band thresholds — pipeline uses 0–100 scale
# high ≥ 70, medium 45–69, low < 45  (mirrors KT section 6.10 thresholds)
CONFIDENCE_BANDS = {
    "high"  : 70.0,
    "medium": 45.0,
    "low"   : 0.0,
}
CONFIDENCE_BAND_UPPER = {
    "high"  : 101.0,
    "medium": 70.0,
    "low"   : 45.0,
}

# ── Keyword normalisation maps ────────────────────────────────
# Maps any natural-language variant the LLM might produce → canonical value

_VERDICT_KEYWORDS = {
    # KEEP variants
    "keep": "KEEP", "kept": "KEEP", "keeping": "KEEP",
    "recommended keep": "KEEP", "recommended to keep": "KEEP",
    "good": "KEEP", "useful": "KEEP", "worth keeping": "KEEP",
    "strong signal": "KEEP", "strong": "KEEP",

    # FLAG variants
    "flag": "FLAG", "flagged": "FLAG", "flagging": "FLAG",
    "borderline": "FLAG", "uncertain": "FLAG", "ambiguous": "FLAG",
    "mixed signal": "FLAG", "mixed": "FLAG", "unclear": "FLAG",
    # Issue #9: natural synonym expansion for FLAG zone
    "under review": "FLAG", "needs review": "FLAG", "review": "FLAG",
    "maybe": "FLAG", "possibly": "FLAG", "not sure": "FLAG",
    "could go either way": "FLAG", "on the fence": "FLAG",
    "marginal": "FLAG", "moderate": "FLAG", "questionable": "FLAG",

    # DROP variants
    "drop": "DROP", "dropped": "DROP", "dropping": "DROP",
    "no signal": "DROP", "no-signal": "DROP", "nosignal": "DROP",
    "weak": "DROP", "useless": "DROP", "discard": "DROP",
    "remove": "DROP", "eliminate": "DROP", "exclude": "DROP",
    "recommended drop": "DROP", "recommended to drop": "DROP",
}

_CONFIDENCE_KEYWORDS = {
    "high confidence": "high", "high": "high",
    "medium confidence": "medium", "medium": "medium", "med": "medium", "moderate": "medium",
    "low confidence": "low", "low": "low",
}

_TYPE_KEYWORDS = {
    "numeric": "numeric", "numerical": "numeric", "number": "numeric", "numbers": "numeric",
    "categorical": "categorical", "cat": "categorical", "category": "categorical",
}

_SPECIAL_KEYWORDS = {
    # null-related
    "null": "null_driven", "null driven": "null_driven", "null-driven": "null_driven",
    "null heavy": "null_driven", "high null": "null_driven", "missing": "null_driven",
    "null signal": "null_driven", "null indicator": "null_driven",
    "drop null": "drop_null", "drop-null": "drop_null", "drop_null": "drop_null",

    # redundancy
    "redundant": "redundant", "redundancy": "redundant", "duplicate signal": "redundant",
    "correlated": "redundant", "collinear": "redundant",

    # Bug 5: pending / undecided
    "pending": "pending", "undecided": "pending", "left to decide": "pending",
    "not decided": "pending", "no decision": "pending",
}


def run(state: dict) -> dict:
    """
    Filter verdict_df and return a ranked summary of matching columns.

    intent_params schema:
        filter : str | None  — the filter expression from the user

    Returns
    -------
    dict — partial state update with action_result
    """
    session       = state["session"]
    intent_params = state.get("intent_params", {})

    verdict_df   = session.get("verdict_df")
    feature_cols = session.get("feature_cols", [])
    num_cols     = session.get("num_cols", [])
    cat_cols     = session.get("cat_cols", [])
    decisions    = state.get("decisions", {})

    if verdict_df is None:
        return {
            "action_result": {
                "error": (
                    "Statistical scan results are not available. "
                    "Please ensure the pipeline ran successfully."
                )
            }
        }

    filter_expr = (intent_params.get("filter") or "").strip()

    # ── Resolve filter type and apply ────────────────────────
    filter_type, matched_cols = _apply_filter(
        filter_expr, verdict_df, feature_cols, num_cols, cat_cols,
        session=session, decisions=decisions        # Bug 5
    )

    if not matched_cols and filter_expr:
        # Bug 2: for null_group misses, list valid group labels
        if filter_type == "null_group":
            valid_groups: list[str] = []
            if verdict_df is not None and "null_group" in verdict_df.columns:
                valid_groups = sorted(
                    set(
                        str(v).upper()
                        for v in verdict_df["null_group"].dropna()
                        if str(v) not in ("", "nan", "None")
                    )
                )
            note = (
                f"No columns found in null group '{filter_expr.upper()}'. "
                + (f"Valid groups: {', '.join(valid_groups)}."
                   if valid_groups else "No null groups are defined in this dataset.")
            )
        else:
            note = (
                f"No columns matched '{filter_expr}'. "
                "Try: KEEP, FLAG, DROP, a risk tag, null_group, "
                "'numeric', 'categorical', or 'high/medium/low confidence'."
            )
        return {
            "action_result": {
                "filter_used"  : filter_expr,
                "filter_type"  : filter_type,
                "matches"      : [],
                "total_matches": 0,
                "note"         : note,
            }
        }

    # ── Build result rows ────────────────────────────────────
    matches = [
        _build_row(col, verdict_df, num_cols, cat_cols, decisions)
        for col in matched_cols
    ]

    # Sort: verdict order (KEEP first for KEEP filter, else DROP first), then confidence desc
    verdict_order = {"DROP": 0, "FLAG": 1, "KEEP": 2, "UNKNOWN": 3}
    if filter_type == "verdict" and matched_cols:
        # When filtering by a specific verdict, sort by confidence desc only
        matches.sort(key=lambda r: -(r.get("confidence") or 0.0))
    else:
        matches.sort(
            key=lambda r: (
                verdict_order.get(r.get("verdict", "UNKNOWN"), 3),
                -(r.get("confidence") or 0.0),
            )
        )

    return {
        "action_result": {
            "filter_used"  : filter_expr or "all",
            "filter_type"  : filter_type,
            "matches"      : matches,
            "total_matches": len(matches),
        }
    }


# ── Filter dispatch ───────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return " ".join(text.lower().split())


def _apply_filter(
    filter_expr: str,
    verdict_df,
    feature_cols: list,
    num_cols: list,
    cat_cols: list,
    session: dict = None,
    decisions: dict = None, 
) -> tuple[str, list[str]]:
    """
    Resolve filter_expr to (filter_type, list_of_column_names).
    Handles natural language variants robustly.
    """
    if not filter_expr:
        return "all", list(feature_cols)

    norm = _normalise(filter_expr)

    # ── 0. Compound filters: type + confidence (e.g. "numeric high confidence") ──
    # Detect if BOTH a type keyword AND a confidence keyword are present
    detected_type = None
    detected_conf = None
    for keyword, col_type in _TYPE_KEYWORDS.items():
        if keyword in norm:
            detected_type = col_type
            break
    for keyword, band in _CONFIDENCE_KEYWORDS.items():
        if keyword in norm:
            detected_conf = band
            break

    if detected_type and detected_conf:
        type_cols = (
            [c for c in feature_cols if c in num_cols]
            if detected_type == "numeric"
            else [c for c in feature_cols if c in cat_cols]
        )
        conf_cols = set(_filter_by_confidence(verdict_df, detected_conf, feature_cols))
        combined  = [c for c in type_cols if c in conf_cols]
        if combined:
            return "compound", combined

    # ── 1. Verdict match (with natural language variants) ─────
    for keyword, verdict in _VERDICT_KEYWORDS.items():
        if keyword in norm:
            cols = _filter_by_column_value(verdict_df, "verdict", verdict, feature_cols, upper=True)
            if cols:
                # Cross-reference live decisions — exclude columns the user has
                # overridden since the scan (e.g. user dropped a bot-KEEP column).
                if verdict == "KEEP":
                    cols = [c for c in cols if decisions.get(c) != "drop"]
                elif verdict == "DROP":
                    cols = [c for c in cols if decisions.get(c) != "keep"]
                if cols:
                    return "verdict", cols

    # ── 2. Confidence band ────────────────────────────────────
    for keyword, band in _CONFIDENCE_KEYWORDS.items():
        if keyword in norm:
            cols = _filter_by_confidence(verdict_df, band, feature_cols)
            if cols:
                return "confidence", cols

    # ── 3. Type match ─────────────────────────────────────────
    for keyword, col_type in _TYPE_KEYWORDS.items():
        if keyword in norm:
            if col_type == "numeric":
                return "type", [c for c in feature_cols if c in num_cols]
            else:
                return "type", [c for c in feature_cols if c in cat_cols]

    # ── 4. Special semantic filters ───────────────────────────
    for keyword, special in _SPECIAL_KEYWORDS.items():
        if keyword in norm:
            if special == "null_driven":
                # Columns that have a null_group assigned (null signal matters)
                cols = _filter_by_null_driven(verdict_df, feature_cols)
                return "null_driven", cols
            if special == "drop_null":
                # Columns with DROP-NULL verdict — use contains to handle DROP-NULL / DROP_NULL variants
                cols = []
                if "verdict" in verdict_df.columns:
                    mask = (
                        verdict_df["verdict"]
                        .astype(str)
                        .str.upper()
                        .str.contains("DROP.NULL", regex=True, na=False)
                    )
                    if "column" in verdict_df.columns:
                        matched = verdict_df.loc[mask, "column"].tolist()
                    else:
                        matched = verdict_df.loc[mask].index.tolist()
                    cols = [c for c in matched if c in feature_cols]
                return "drop_null", cols
            if special == "redundant":
                # Pull from session redundancy_drop list
                if session:
                    redundant = session.get("redundancy_drop", [])
                    cols = [c for c in redundant if c in feature_cols]
                    if not cols:
                        # Fall back: filter by risk_tag containing 'redundant'
                        cols = _filter_risk_tag_contains(verdict_df, "redundant", feature_cols)
                return "redundant", cols if session else []

            if special == "pending":
                # Bug 5: columns in feature_cols not yet in decisions
                decided = set(decisions.keys()) if decisions else set()
                cols = [c for c in feature_cols if c not in decided]
                return "pending", cols

    # ── 5. Null group match (e.g. "G1", "G2", "null_group G1", "in G2", "columns in G1") ──
    # Strip known prefixes before the G-label: "null_group", "null group", and leading "in "
    clean = norm.replace("null_group", "").replace("null group", "").strip()
    # Strip a leading "in " so "which columns are in G2" → "G2" (not "IN G2" → miss)
    clean = re.sub(r"^in\s+", "", clean, flags=re.IGNORECASE).strip().upper()
    if clean.startswith("G") and len(clean) <= 3:
        # Normalise both sides to uppercase before comparison (guards against "g2" vs "G2" mismatch)
        cols = _filter_by_null_group_upper(verdict_df, clean, feature_cols)
        if not cols:
            # Collect valid group labels for the fallback message
            valid_groups: list[str] = []
            if "null_group" in verdict_df.columns:
                valid_groups = sorted(
                    set(
                        str(v).upper()
                        for v in verdict_df["null_group"].dropna()
                        if str(v) not in ("", "nan", "None")
                    )
                )
            note = (
                f"No columns found in null group '{clean}'. "
                + (f"Valid groups are: {', '.join(valid_groups)}." if valid_groups
                   else "No null groups are defined in this dataset.")
            )
            return "null_group", []  # caller handles empty + note via existing no-match path
        return "null_group", cols

    # ── 6. Risk tag match ─────────────────────────────────────
    for tag_col in ("risk_tag", "tag", "risk"):
        try:
            if tag_col in verdict_df.columns:
                cols = _filter_by_column_value(verdict_df, tag_col, filter_expr.upper(), feature_cols, upper=True)
                if cols:
                    return "risk_tag", cols
                # Also try contains match
                cols = _filter_risk_tag_contains(verdict_df, norm, feature_cols)
                if cols:
                    return "risk_tag", cols
        except Exception:
            pass

    # ── 7. Partial / fuzzy fallback — substring in col name ──
    substring_matches = [c for c in feature_cols if norm in c.lower()]
    if substring_matches:
        return "name_search", substring_matches

    return "unknown", []


def _filter_by_column_value(
    verdict_df, col: str, value: str, feature_cols: list, upper: bool = False
) -> list[str]:
    """Return feature_cols whose verdict_df[col] == value."""
    try:
        if col not in verdict_df.columns:
            return []
        series = verdict_df[col].astype(str)
        if upper:
            series = series.str.upper()
        mask = series == value
        if "column" in verdict_df.columns:
            matched = verdict_df.loc[mask, "column"].tolist()
        else:
            matched = verdict_df.loc[mask].index.tolist()
        return [c for c in matched if c in feature_cols]
    except Exception:
        return []


def _filter_by_null_group_upper(verdict_df, group_label: str, feature_cols: list) -> list[str]:
    """
    Return feature_cols in the given null_group, normalising BOTH
    the lookup value AND the stored map keys to uppercase before comparison.

    This guards against mismatches like stored "g2" vs requested "G2",
    or stored "group_2" vs "G2" (which would still miss — but at least
    eliminates the pure case-sensitivity failure that was the root cause).
    """
    try:
        if "null_group" not in verdict_df.columns:
            return []
        series = verdict_df["null_group"].astype(str).str.strip().str.upper()
        mask   = series == group_label.strip().upper()
        if "column" in verdict_df.columns:
            matched = verdict_df.loc[mask, "column"].tolist()
        else:
            matched = verdict_df.loc[mask].index.tolist()
        return [c for c in matched if c in feature_cols]
    except Exception:
        return []



def _filter_by_confidence(verdict_df, band: str, feature_cols: list) -> list[str]:
    """Return feature_cols in the given confidence band."""
    try:
        if "confidence" not in verdict_df.columns:
            return []
        lo = CONFIDENCE_BANDS[band]
        hi = CONFIDENCE_BAND_UPPER[band]
        mask = (verdict_df["confidence"] >= lo) & (verdict_df["confidence"] < hi)
        if "column" in verdict_df.columns:
            matched = verdict_df.loc[mask, "column"].tolist()
        else:
            matched = verdict_df.loc[mask].index.tolist()
        return [c for c in matched if c in feature_cols]
    except Exception:
        return []


# Issue #7: threshold for "null-driven" filter
NULL_DRIVEN_RATE_THRESHOLD = 0.50   # column must have > 50% nulls OR explicit risk_tag


def _filter_by_null_driven(verdict_df, feature_cols: list) -> list[str]:
    """
    Return columns that are genuinely null-heavy.

    Issue #7 fix: previously returned ANY column with a null_group assigned,
    which could flood results with hundreds of columns that have just one null.
    Now a column qualifies only if it meets AT LEAST ONE of:
      1. null_rate > NULL_DRIVEN_RATE_THRESHOLD (> 50% missing)
      2. risk_tag explicitly contains "null" or "high_null"
      3. verdict is DROP-NULL
    """
    try:
        results = set()

        # Path 1: null_rate threshold
        if "null_rate" in verdict_df.columns or "missing_rate" in verdict_df.columns:
            rate_col = "null_rate" if "null_rate" in verdict_df.columns else "missing_rate"
            mask = verdict_df[rate_col].astype(float) > NULL_DRIVEN_RATE_THRESHOLD
            if "column" in verdict_df.columns:
                matched = verdict_df.loc[mask, "column"].tolist()
            else:
                matched = verdict_df.loc[mask].index.tolist()
            results.update(c for c in matched if c in feature_cols)

        # Path 2: risk_tag contains "null" or "high_null"
        for tag_col in ("risk_tag", "tag", "risk"):
            if tag_col in verdict_df.columns:
                mask = verdict_df[tag_col].astype(str).str.lower().str.contains("null", na=False)
                if "column" in verdict_df.columns:
                    matched = verdict_df.loc[mask, "column"].tolist()
                else:
                    matched = verdict_df.loc[mask].index.tolist()
                results.update(c for c in matched if c in feature_cols)
                break

        # Path 3: DROP-NULL verdict — use contains so "DROP-NULL" / "DROP_NULL" / "drop-null" all match
        for verdict_col in ("verdict",):
            if verdict_col in verdict_df.columns:
                mask = (
                    verdict_df[verdict_col]
                    .astype(str)
                    .str.upper()
                    .str.contains("DROP.NULL", regex=True, na=False)
                )
                if "column" in verdict_df.columns:
                    matched = verdict_df.loc[mask, "column"].tolist()
                else:
                    matched = verdict_df.loc[mask].index.tolist()
                results.update(c for c in matched if c in feature_cols)
                break

        return [c for c in feature_cols if c in results]   # preserve original order
    except Exception:
        return []


def _filter_risk_tag_contains(verdict_df, substring: str, feature_cols: list) -> list[str]:
    """Return columns whose risk_tag contains the given substring."""
    try:
        for tag_col in ("risk_tag", "tag", "risk"):
            if tag_col not in verdict_df.columns:
                continue
            mask = verdict_df[tag_col].astype(str).str.lower().str.contains(substring, na=False)
            if "column" in verdict_df.columns:
                matched = verdict_df.loc[mask, "column"].tolist()
            else:
                matched = verdict_df.loc[mask].index.tolist()
            result = [c for c in matched if c in feature_cols]
            if result:
                return result
    except Exception:
        pass
    return []


# ── Row builder ───────────────────────────────────────────────

def _build_row(
    col: str,
    verdict_df,
    num_cols: list,
    cat_cols: list,
    decisions: dict,
) -> dict:
    """Build a compact summary dict for one matched column."""
    row = {}
    try:
        if "column" in verdict_df.columns:
            rows = verdict_df[verdict_df["column"] == col]
        else:
            rows = verdict_df[verdict_df.index == col]

        if not rows.empty:
            r = rows.iloc[0]
            row = {
                "column"    : col,
                "verdict"   : str(r.get("verdict", "UNKNOWN")).upper() if hasattr(r, "get") else str(r["verdict"]).upper(),
                "confidence": _safe_float(r, ["confidence", "confidence_score"]),
                "risk_tag"  : _safe_str(r, ["risk_tag", "tag", "risk"]),
                "null_rate" : _safe_float(r, ["null_rate", "missing_rate", "null_pct"]),
                "null_group": _safe_str(r, ["null_group"]),
                "profile"   : "numeric" if col in num_cols else "categorical" if col in cat_cols else "unknown",
                "decision"  : decisions.get(col),
            }
        else:
            row = {
                "column"  : col,
                "verdict" : "UNKNOWN",
                "profile" : "numeric" if col in num_cols else "categorical" if col in cat_cols else "unknown",
                "decision": decisions.get(col),
            }
    except Exception:
        row = {"column": col, "verdict": "UNKNOWN", "decision": decisions.get(col)}

    return row


def _safe_float(row, keys: list):
    for k in keys:
        try:
            v = row[k]
            if v is not None and str(v) not in ("", "nan", "None"):
                return float(v)
        except Exception:
            pass
    return None


def _safe_str(row, keys: list):
    for k in keys:
        try:
            v = row[k]
            if v is not None and str(v) not in ("", "nan", "None", "NaN"):
                return str(v)
        except Exception:
            pass
    return None
