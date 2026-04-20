# ============================================================
# actions/decide.py
#
# Action: DECIDE
#
# v7 changes:
#   - C2: Guardrail now correctly fires on zone-based bulk drops.
#         Previously the guardrail check in the multi-zone path
#         existed but the projected drop count didn't account for
#         columns already decided as drop. Fixed to use the same
#         pattern as _decide_bulk().
#   - P1: undo_stack snapshots now include label field so time-travel
#         undo can find named snapshots.
# ============================================================

from datetime import datetime, timezone
import copy


# ── Constants ─────────────────────────────────────────────────
DROP_GUARDRAIL_PCT = 0.75


def run(state: dict) -> dict:
    session       = state["session"]
    intent_params = state["intent_params"]
    decisions     = dict(state.get("decisions", {}))
    undo_stack    = list(state.get("undo_stack", []))
    decision_log  = list(state.get("decision_log", []))

    feature_cols = session.get("feature_cols", [])
    verdict_df   = session.get("verdict_df")

    decision = (intent_params.get("decision") or "").lower()
    if decision not in ("keep", "drop"):
        return {
            "action_result": {
                "error": (
                    f"Unknown decision '{decision}'. "
                    "Please say 'keep' or 'drop'."
                )
            }
        }

    confirmed = bool(intent_params.get("confirmed", False))

    # ── Initialise resolution lists ───────────────────────────
    resolved_columns: list = []
    resolved_zones:   list = []

    # ── Resolve column(s) ─────────────────────────────────────
    columns_param = intent_params.get("columns")
    column_param  = intent_params.get("column")

    if columns_param and isinstance(columns_param, list):
        resolved_columns = [c for c in columns_param if c]
    elif column_param:
        resolved_columns = [column_param]

    # ── Resolve zone(s) ───────────────────────────────────────
    zones_param = intent_params.get("zones")
    zone_param  = intent_params.get("zone")
    # Normalise: filter out None/empty values in either param
    if zones_param and isinstance(zones_param, list):
        resolved_zones = [z for z in zones_param if z]
    elif zone_param:
        resolved_zones = [zone_param]

    # Fall back to active_focus ONLY for pronoun-style requests.
    # SAFETY: do NOT fall back when a zone is present.
    if not resolved_columns and not resolved_zones:
        af = state.get("active_focus")
        if isinstance(af, list):
            resolved_columns = af
        elif af:
            resolved_columns = [af]

    # ── Route ─────────────────────────────────────────────────
    if resolved_columns:
        if len(resolved_columns) == 1:
            return _decide_single(
                resolved_columns[0], decision, decisions, undo_stack,
                decision_log, feature_cols, verdict_df, state,
            )

        # Multiple columns — push one shared undo snapshot
        undo_stack.append({
            "decisions"   : dict(decisions),
            "active_focus": state.get("active_focus"),
            "label"       : "multi_decide",
        })
        applied, skipped_errors = [], []
        for col in resolved_columns:
            if col not in feature_cols:
                close = _fuzzy_match(col, feature_cols)
                msg   = f"'{col}' not found." + (f" Did you mean '{close}'?" if close else "")
                skipped_errors.append(msg)
                continue
            bot_verdict = _lookup_verdict(col, verdict_df)
            override    = False
            if bot_verdict:
                override = (decision == "keep") != (bot_verdict.upper() == "KEEP")
            decisions[col] = decision
            decision_log.append({
                "col"      : col,
                "decision" : decision,
                "reason"   : "user instruction",
                "source"   : "user",
                "timestamp": _now(),
                "override" : override,
            })
            applied.append(col)

        last_focus = applied[-1] if applied else state.get("active_focus")
        return {
            "action_result": {
                "mode"          : "multi",
                "decision"      : decision,
                "applied_to"    : applied,
                "skipped_errors": skipped_errors,
                "total_applied" : len(applied),
            },
            "decisions"   : decisions,
            "undo_stack"  : undo_stack,
            "decision_log": decision_log,
            "active_focus": last_focus,
        }

    elif resolved_zones:
        # ── Multi-zone bulk ────────────────────────────────────
        # Collect all target columns across all zones first
        all_target_cols = []
        for z in resolved_zones:
            zone_cols = _columns_for_zone(z, verdict_df, feature_cols)
            for c in zone_cols:
                if c not in all_target_cols:
                    all_target_cols.append(c)

        if not all_target_cols:
            return {
                "action_result": {
                    "error": (
                        f"No columns matched the filter(s) {resolved_zones}. "
                        "Try verdict labels KEEP, FLAG, or DROP."
                    )
                }
            }

        # ── C2 FIX: Guardrail now correctly counts existing drops ──
        # Previously used len(new_drops) alone, ignoring existing drop decisions.
        # Now matches the same pattern as _decide_bulk() below.
        if decision == "drop":
            current_drops   = sum(1 for v in decisions.values() if v == "drop")
            new_drops       = [c for c in all_target_cols if decisions.get(c) != "drop"]
            projected_total = current_drops + len(new_drops)
            total_cols      = len(feature_cols)
            projected_pct   = projected_total / total_cols if total_cols else 0.0

            if projected_pct > DROP_GUARDRAIL_PCT and not confirmed:
                return {
                    "action_result": {
                        "mode"               : "bulk",
                        "guardrail_triggered": True,
                        "decision"           : decision,
                        "filter_used"        : " + ".join(resolved_zones),
                        "projected_drops"    : projected_total,
                        "total_cols"         : total_cols,
                        "drop_pct_after"     : round(projected_pct * 100, 1),
                        "columns_affected"   : all_target_cols,
                        "message": (
                            f"This would drop {projected_total} of {total_cols} columns "
                            f"({projected_pct*100:.0f}%) — above the 75% guardrail. "
                            f"Type 'confirm' to proceed anyway."
                        ),
                    }
                }

        # Apply
        applied_to, skipped = [], []
        undo_stack.append({
            "decisions"   : dict(decisions),
            "active_focus": state.get("active_focus"),
            "label"       : f"bulk_{decision}_{'_'.join(resolved_zones).lower()}",
        })
        now = _now()
        for col in all_target_cols:
            if decisions.get(col) == decision:
                skipped.append(col)
            else:
                decisions[col] = decision
                decision_log.append({
                    "col"      : col,
                    "decision" : decision,
                    "reason"   : f"bulk {decision} — filter: {' + '.join(resolved_zones)}",
                    "source"   : "user",
                    "timestamp": now,
                    "override" : False,
                })
                applied_to.append(col)

        current_drops = sum(1 for v in decisions.values() if v == "drop")
        return {
            "action_result": {
                "mode"          : "bulk",
                "decision"      : decision,
                "filter_used"   : " + ".join(resolved_zones),
                "applied_to"    : applied_to,
                "skipped"       : skipped,
                "total_cols"    : len(feature_cols),
                "new_drop_total": current_drops,
            },
            "decisions"   : decisions,
            "undo_stack"  : undo_stack,
            "decision_log": decision_log,
        }

    else:
        return {
            "action_result": {
                "error": (
                    "No column or filter was specified. "
                    "Tell me which column to decide on, or use a filter "
                    "like 'drop all FLAG columns'."
                )
            }
        }


# ── Single-column path ────────────────────────────────────────

def _decide_single(
    column: str,
    decision: str,
    decisions: dict,
    undo_stack: list,
    decision_log: list,
    feature_cols: list,
    verdict_df,
    state: dict,
) -> dict:
    if column not in feature_cols:
        close = _fuzzy_match(column, feature_cols)
        suggestion = f" Did you mean '{close}'?" if close else ""
        return {
            "action_result": {
                "error": f"'{column}' was not found in the dataset.{suggestion}"
            },
            "active_focus": state.get("active_focus"),
        }

    bot_verdict = _lookup_verdict(column, verdict_df)
    override = False
    if bot_verdict:
        user_keep = decision == "keep"
        bot_keep  = bot_verdict.upper() == "KEEP"
        override  = user_keep != bot_keep

    # Push undo snapshot with label
    undo_stack.append({
        "decisions"   : dict(decisions),
        "active_focus": state.get("active_focus"),
        "label"       : f"decide_{column}_{decision}",
    })

    decisions[column] = decision
    decision_log.append({
        "col"      : column,
        "decision" : decision,
        "reason"   : "user instruction",
        "source"   : "user",
        "timestamp": _now(),
        "override" : override,
    })

    return {
        "action_result": {
            "mode"       : "single",
            "column"     : column,
            "decision"   : decision,
            "override"   : override,
            "bot_verdict": bot_verdict,
        },
        "decisions"   : decisions,
        "undo_stack"  : undo_stack,
        "decision_log": decision_log,
        "active_focus": column,
    }


# ── Helpers ───────────────────────────────────────────────────

def _decide_bulk(
    zone: str,
    decision: str,
    confirmed: bool,
    decisions: dict,
    undo_stack: list,
    decision_log: list,
    feature_cols: list,
    verdict_df,
    state: dict,
) -> dict:
    """Apply a decision to all columns matching a single zone filter."""
    if verdict_df is None:
        return {
            "action_result": {
                "error": (
                    "Statistical scan results are not available. "
                    "Bulk decisions require verdict_df from the pipeline."
                )
            }
        }

    target_cols = _columns_for_zone(zone, verdict_df, feature_cols)

    if not target_cols:
        return {
            "action_result": {
                "error": (
                    f"No columns matched the filter '{zone}'. "
                    "Try verdict labels KEEP, FLAG, or DROP, "
                    "or a specific risk tag."
                )
            }
        }

    if decision == "drop":
        current_drops   = sum(1 for v in decisions.values() if v == "drop")
        new_drops       = [c for c in target_cols if decisions.get(c) != "drop"]
        projected_total = current_drops + len(new_drops)
        total_cols      = len(feature_cols)
        projected_pct   = projected_total / total_cols if total_cols else 0.0

        if projected_pct > DROP_GUARDRAIL_PCT and not confirmed:
            return {
                "action_result": {
                    "mode"               : "bulk",
                    "guardrail_triggered": True,
                    "decision"           : decision,
                    "filter_used"        : zone,
                    "projected_drops"    : projected_total,
                    "total_cols"         : total_cols,
                    "drop_pct_after"     : round(projected_pct * 100, 1),
                    "columns_affected"   : target_cols,
                }
            }

    applied_to = []
    skipped    = []
    for col in target_cols:
        if decisions.get(col) == decision:
            skipped.append(col)
        else:
            applied_to.append(col)

    if not applied_to:
        return {
            "action_result": {
                "mode"       : "bulk",
                "decision"   : decision,
                "filter_used": zone,
                "applied_to" : [],
                "skipped"    : skipped,
                "note"       : "All matched columns already have that decision.",
            }
        }

    undo_stack.append({
        "decisions"   : dict(decisions),
        "active_focus": state.get("active_focus"),
        "label"       : f"bulk_{decision}_{zone.lower()}",
    })

    now = _now()
    for col in applied_to:
        decisions[col] = decision
        decision_log.append({
            "col"      : col,
            "decision" : decision,
            "reason"   : f"bulk {decision} — filter: {zone}",
            "source"   : "user",
            "timestamp": now,
            "override" : False,
        })

    current_drops = sum(1 for v in decisions.values() if v == "drop")

    return {
        "action_result": {
            "mode"          : "bulk",
            "decision"      : decision,
            "filter_used"   : zone,
            "applied_to"    : applied_to,
            "skipped"       : skipped,
            "total_cols"    : len(feature_cols),
            "new_drop_total": current_drops,
        },
        "decisions"   : decisions,
        "undo_stack"  : undo_stack,
        "decision_log": decision_log,
    }


def _lookup_verdict(column: str, verdict_df) -> str | None:
    if verdict_df is None:
        return None
    try:
        if "column" in verdict_df.columns:
            rows = verdict_df[verdict_df["column"] == column]
        else:
            rows = verdict_df[verdict_df.index == column]
        if rows.empty:
            return None
        return str(rows.iloc[0]["verdict"]).upper()
    except Exception:
        return None


def _columns_for_zone(zone: str, verdict_df, feature_cols: list) -> list[str]:
    zone_upper = zone.upper()
    try:
        if "column" in verdict_df.columns:
            col_field = "column"
        else:
            col_field = None

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

        if "null_group" in verdict_df.columns:
            matched = verdict_df[verdict_df["null_group"].str.upper() == zone_upper]
            if not matched.empty:
                cols = matched[col_field].tolist() if col_field else matched.index.tolist()
                return [c for c in cols if c in feature_cols]

    except Exception:
        pass

    return []


def _fuzzy_match(target: str, candidates: list[str]) -> str | None:
    target_lower = target.lower()
    for c in candidates:
        if target_lower in c.lower() or c.lower() in target_lower:
            return c
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()