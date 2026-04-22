# ============================================================
# actions/conditional_decide.py
#
# Action: CONDITIONAL_DECIDE
#
# Applies keep/drop decisions based on user-specified conditions
# evaluated against the pre-computed verdict_df. Supports:
#   - Single conditions ("drop if null rate > 60%")
#   - Compound AND/OR conditions ("drop if null rate high AND signal low")
#   - Zone scoping ("drop FLAGGED columns if null rate high")
#   - Dry run / preview mode ("what would happen if I dropped...")
#   - All condition types: null_rate, confidence, mi_score, p_value,
#     churn_gap, risk_tag, verdict
#
# Public API:
#   run(state: dict) -> dict
#
# I-4 fix: Added debug logging to _evaluate_condition so field-alias
# resolution failures are immediately visible in the terminal rather
# than silently returning passed=False with matched_cols=[].
# Enable with:
#   CONDITIONAL_DECIDE_DEBUG=1 python main.py          (environment)
#   intent_params["debug"] = True                      (from tests)
#
# No LLM calls. Pure Python.
# ============================================================

import os
from datetime import datetime, timezone
import copy

# I-4: debug flag — set CONDITIONAL_DECIDE_DEBUG=1 in environment,
# or pass intent_params["debug"] = True for per-call tracing.
_ENV_DEBUG = os.environ.get("CONDITIONAL_DECIDE_DEBUG", "0") == "1"

# Guardrail — same threshold as decide.py
_DROP_GUARDRAIL = 0.75

# Default thresholds for vague conditions
# (fallback when classifier can't determine a precise value)
_DEFAULTS = {
    "null_rate_high"      : 0.60,
    "confidence_good"     : 60,
    "confidence_low"      : 45,
    "confidence_strong"   : 70,
    "p_value_significant" : 0.05,
    "churn_gap_strong"    : 3.0,
    "churn_gap_moderate"  : 1.0,
}

# Map field names the classifier might send → actual verdict_df column names
_FIELD_ALIASES = {
    "null_rate"  : ["null_rate", "missing_rate", "null_pct"],
    "confidence" : ["confidence", "confidence_score", "score"],
    "mi_score"   : ["mutual_info", "mi_score", "mutual_information"],
    "p_value"    : ["mw", "mann_whitney_p", "chi2_p", "p_value", "p_val"],
    "churn_gap"  : ["null_gap", "gap_pp", "churn_gap", "gap", "neg_gap"],
    "risk_tag"   : ["risk_tag", "tag", "risk"],
    "verdict"    : ["verdict", "decision", "recommendation"],
}


def run(state: dict) -> dict:
    """
    Evaluate conditions against verdict_df and apply (or preview) decisions.
    """
    session       = state["session"]
    intent_params = state["intent_params"]
    verdict_df    = session.get("verdict_df")
    feature_cols  = session.get("feature_cols", [])
    decisions     = dict(state.get("decisions", {}))
    undo_stack    = list(state.get("undo_stack", []))
    decision_log  = list(state.get("decision_log", []))

    # I-4: honour per-call debug flag (useful from pytest without env var)
    debug = _ENV_DEBUG or bool(intent_params.get("debug", False))

    if verdict_df is None:
        return {"action_result": {
            "error": "Scan results not available — cannot evaluate conditions."
        }}

    conditions      = intent_params.get("conditions", [])
    condition_logic = (intent_params.get("condition_logic") or "AND").upper()
    decision        = (intent_params.get("decision") or "drop").lower()
    scope           = (intent_params.get("scope") or "").strip().upper() or None
    dry_run         = bool(intent_params.get("dry_run", False))

    if decision not in ("keep", "drop"):
        return {"action_result": {
            "error": f"Unknown decision '{decision}'. Please say 'keep' or 'drop'."
        }}

    if not conditions:
        return {"action_result": {
            "error": "No conditions were provided. Please specify what to check (e.g. 'if null rate is high')."
        }}

    # ── Build working set: all feature cols, optionally scoped ──
    if scope:
        working_set = _columns_for_zone(scope, verdict_df, feature_cols)
        if not working_set:
            return {"action_result": {
                "error": f"No columns found in zone '{scope}'."
            }}
    else:
        working_set = feature_cols

    if debug:
        print(f"[CONDITIONAL_DECIDE] decision={decision!r} logic={condition_logic} "
              f"scope={scope!r} dry_run={dry_run} "
              f"working_set={len(working_set)} cols  conditions={conditions}")

    # ── Evaluate conditions against each column ───────────────
    matched_cols   = []
    unmatched_cols = []
    condition_details = []

    for col in working_set:
        row_dict = _get_row(col, verdict_df)
        if row_dict is None:
            if debug:
                print(f"[CONDITIONAL_DECIDE]   {col}: not found in verdict_df — skipped")
            continue

        results = [_evaluate_condition(cond, row_dict, col, debug=debug)
                   for cond in conditions]

        if condition_logic == "OR":
            passed = any(r["passed"] for r in results)
        else:  # AND
            passed = all(r["passed"] for r in results)

        if passed:
            matched_cols.append(col)
        else:
            unmatched_cols.append(col)

        condition_details.append({
            "column"    : col,
            "passed"    : passed,
            "conditions": results,
        })

    if debug:
        print(f"[CONDITIONAL_DECIDE] matched={len(matched_cols)} "
              f"unmatched={len(unmatched_cols)}")

    if not matched_cols:
        return {
            "action_result": {
                "dry_run"         : dry_run,
                "decision"        : decision,
                "scope"           : scope,
                "conditions"      : conditions,
                "condition_logic" : condition_logic,
                "matched_count"   : 0,
                "matched_columns" : [],
                "message"         : (
                    f"No columns matched your condition(s). "
                    f"{'Nothing to preview.' if dry_run else 'No changes made.'}"
                ),
            }
        }

    # ── Dry run: return preview without writing ───────────────
    if dry_run:
        already_decided_same = [c for c in matched_cols if decisions.get(c) == decision]
        would_change         = [c for c in matched_cols if decisions.get(c) != decision]

        # Show sample of matched (max 10 for readability)
        sample = matched_cols[:10]
        sample_details = []
        for col in sample:
            row_dict = _get_row(col, verdict_df)
            if row_dict:
                sample_details.append({
                    "column"    : col,
                    "verdict"   : _safe_get(row_dict, _FIELD_ALIASES["verdict"]),
                    "null_rate" : _safe_get(row_dict, _FIELD_ALIASES["null_rate"]),
                    "confidence": _safe_get(row_dict, _FIELD_ALIASES["confidence"]),
                    "risk_tag"  : _safe_get(row_dict, _FIELD_ALIASES["risk_tag"]),
                })

        return {
            "action_result": {
                "dry_run"              : True,
                "decision"             : decision,
                "scope"                : scope,
                "conditions"           : conditions,
                "condition_logic"      : condition_logic,
                "matched_count"        : len(matched_cols),
                "would_change_count"   : len(would_change),
                "already_same_count"   : len(already_decided_same),
                "matched_columns"      : matched_cols,
                "would_change_columns" : would_change,
                "sample_details"       : sample_details,
                "total_features"       : len(feature_cols),
                "pct_of_total"         : round(len(matched_cols) / len(feature_cols) * 100, 1) if feature_cols else 0,
                "message": (
                    f"Preview: {len(matched_cols)} columns match your condition(s). "
                    f"{len(would_change)} would change to '{decision}' "
                    f"({len(already_decided_same)} already marked '{decision}'). "
                    f"Say 'confirm' or repeat without 'what would happen' to apply."
                ),
            }
        }

    # ── Live run: guardrail then apply ────────────────────────
    if decision == "drop":
        current_drops   = sum(1 for v in decisions.values() if v == "drop")
        new_drops       = [c for c in matched_cols if decisions.get(c) != "drop"]
        projected_total = current_drops + len(new_drops)
        total_cols      = len(feature_cols)
        projected_pct   = projected_total / total_cols if total_cols else 0.0

        force_confirm = intent_params.get("force_confirm", False)
        if projected_pct > _DROP_GUARDRAIL and not force_confirm:
            return {
                "action_result": {
                    "guardrail_triggered": True,
                    "dry_run"            : False,
                    "decision"           : decision,
                    "scope"              : scope,
                    "conditions"         : conditions,
                    "matched_count"      : len(matched_cols),
                    "projected_drops"    : projected_total,
                    "total_cols"         : total_cols,
                    "drop_pct_after"     : round(projected_pct * 100, 1),
                    "columns_affected"   : matched_cols,
                    "message": (
                        f"This would drop {projected_total} of {total_cols} columns "
                        f"({projected_pct*100:.0f}%) — above the 75% guardrail. "
                        f"Type 'confirm' to proceed anyway."
                    ),
                }
            }

    # ── 5. Push to Undo Stack & Return ───────────────────────────
    force_confirm = intent_params.get("force_confirm", False)
    snapshot_label = "guardrail_force_confirm" if force_confirm else f"conditional_{decision}"
    undo_stack.append({
        "decisions"   : copy.deepcopy(decisions),
        "active_focus": state.get("active_focus"),
        "label"       : snapshot_label,
    })

    # Apply decisions
    applied, skipped = [], []
    ts = datetime.now(timezone.utc).isoformat()

    # Build human-readable condition description for the log
    cond_desc = _describe_conditions(conditions, condition_logic)

    for col in matched_cols:
        if decisions.get(col) == decision:
            skipped.append(col)
            continue
        bot_verdict = _safe_get(
            _get_row(col, verdict_df) or {},
            _FIELD_ALIASES["verdict"]
        )
        override = False
        if bot_verdict:
            override = (decision == "keep") != (str(bot_verdict).upper() == "KEEP")

        decisions[col] = decision
        decision_log.append({
            "col"      : col,
            "decision" : decision,
            "reason"   : f"conditional rule: {cond_desc}",
            "source"   : "user",
            "timestamp": ts,
            "override" : override,
        })
        applied.append(col)

    current_drops = sum(1 for v in decisions.values() if v == "drop")

    return {
        "action_result": {
            "dry_run"         : False,
            "decision"        : decision,
            "scope"           : scope,
            "conditions"      : conditions,
            "condition_logic" : condition_logic,
            "matched_count"   : len(matched_cols),
            "applied_count"   : len(applied),
            "skipped_count"   : len(skipped),
            "applied_columns" : applied,
            "skipped_columns" : skipped,
            "total_features"  : len(feature_cols),
            "new_drop_total"  : current_drops,
            "condition_desc"  : cond_desc,
            "message": (
                f"Applied: {len(applied)} columns marked '{decision}' "
                f"based on rule — {cond_desc}. "
                f"({len(skipped)} already had that decision.)"
            ),
        },
        "decisions"   : decisions,
        "undo_stack"  : undo_stack,
        "decision_log": decision_log,
        "active_focus": None,
    }


# ── Condition evaluation ──────────────────────────────────────

def _evaluate_condition(cond: dict, row_dict: dict, col: str, *, debug: bool = False) -> dict:
    """
    Evaluate one condition against a column's row_dict.

    Returns:
        {
            "passed"         : bool,
            "field"          : str,
            "actual_value"   : any,
            "threshold"      : any,
            "resolved_alias" : str | None,  # I-4: which alias key actually resolved
            "note"           : str | None,  # I-4: populated on miss or error
        }

    I-4 debug logging:
        When debug=True (or CONDITIONAL_DECIDE_DEBUG=1 in environment), prints
        a trace line per condition showing: column, field requested, aliases tried,
        which alias resolved (or NONE), actual value, operator, threshold, and
        pass/fail result.  This makes silent field-mapping failures visible
        immediately instead of producing a mysterious "no columns matched" result.
    """
    field     = cond.get("field", "")
    operator  = cond.get("operator", ">")
    threshold = cond.get("threshold")

    aliases = _FIELD_ALIASES.get(field, [field])

    # I-4: walk aliases explicitly so we know which one resolved
    resolved_alias = None
    actual = None
    for alias in aliases:
        if alias in row_dict and row_dict[alias] is not None:
            resolved_alias = alias
            actual = row_dict[alias]
            break

    if debug:
        print(
            f"[CONDITIONAL_DECIDE]     col={col!r}  field={field!r}  "
            f"aliases_tried={aliases}  resolved={resolved_alias!r}  "
            f"actual={actual!r}  op={operator!r}  threshold={threshold!r}",
            end="",
        )

    if actual is None:
        # I-4: log clearly — this is a field-mapping problem, not a data problem
        note = (
            f"field '{field}' not found in verdict_df row "
            f"(tried aliases: {aliases})"
        )
        if debug:
            print(f"  → FAIL ({note})")
        return {
            "passed"         : False,
            "field"          : field,
            "actual_value"   : None,
            "threshold"      : threshold,
            "resolved_alias" : None,
            "note"           : note,
        }

    try:
        passed = _compare(actual, operator, threshold)
    except Exception as exc:
        note = f"compare error: {exc}"
        if debug:
            print(f"  → FAIL ({note})")
        return {
            "passed"         : False,
            "field"          : field,
            "actual_value"   : actual,
            "threshold"      : threshold,
            "resolved_alias" : resolved_alias,
            "note"           : note,
        }

    if debug:
        print(f"  → {'PASS' if passed else 'FAIL'}")

    return {
        "passed"         : passed,
        "field"          : field,
        "actual_value"   : actual,
        "threshold"      : threshold,
        "resolved_alias" : resolved_alias,
        "note"           : None,
    }


def _compare(actual, operator: str, threshold) -> bool:
    """Evaluate actual <operator> threshold. Handles string contains."""
    if operator == "contains":
        return str(threshold).lower() in str(actual).lower()
    if operator == "==":
        return str(actual).upper() == str(threshold).upper()

    # Numeric comparisons
    actual_f    = float(actual)
    threshold_f = float(threshold)
    if operator == ">":
        return actual_f > threshold_f
    if operator == "<":
        return actual_f < threshold_f
    if operator == ">=":
        return actual_f >= threshold_f
    if operator == "<=":
        return actual_f <= threshold_f
    return False


# ── Zone resolution ───────────────────────────────────────────

def _columns_for_zone(zone: str, verdict_df, feature_cols: list) -> list[str]:
    zone_upper = zone.upper()
    try:
        col_field = "column" if "column" in verdict_df.columns else None
        if "verdict" in verdict_df.columns:
            matched = verdict_df[verdict_df["verdict"].str.upper() == zone_upper]
            if not matched.empty:
                cols = matched[col_field].tolist() if col_field else matched.index.tolist()
                return [c for c in cols if c in feature_cols]
        for tag_col in ("risk_tag", "tag", "risk"):
            if tag_col in verdict_df.columns:
                matched = verdict_df[verdict_df[tag_col].str.upper() == zone_upper]
                if not matched.empty:
                    cols = matched[col_field].tolist() if col_field else matched.index.tolist()
                    return [c for c in cols if c in feature_cols]
    except Exception:
        pass
    return []


# ── Helpers ───────────────────────────────────────────────────

def _get_row(col: str, verdict_df) -> dict | None:
    """Return the verdict_df row for a column as a dict, or None."""
    try:
        if "column" in verdict_df.columns:
            rows = verdict_df[verdict_df["column"] == col]
        else:
            rows = verdict_df[verdict_df.index == col]
        if rows.empty:
            return None
        return rows.iloc[0].to_dict()
    except Exception:
        return None


def _safe_get(d: dict, keys: list, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _describe_conditions(conditions: list, logic: str) -> str:
    """Build a readable string describing the conditions."""
    parts = []
    for c in conditions:
        field = c.get("field", "?")
        op    = c.get("operator", "?")
        thr   = c.get("threshold", "?")
        parts.append(f"{field} {op} {thr}")
    joiner = f" {logic} "
    return joiner.join(parts)
