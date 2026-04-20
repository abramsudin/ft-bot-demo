# ============================================================
# actions/auto_decide.py
#
# Action: AUTO_DECIDE
#
# v7 changes:
#   - D1: Draft mode — bot verdicts are loaded as a starting draft
#         that the user can edit before finalising via REPORT.
#         Behaviour is identical to before (writes to decisions ledger)
#         but the action_result now clearly signals "draft" mode so
#         the formatter tells the user these are editable, not final.
#   - P1: undo snapshot now includes label="auto_decide" so time-travel
#         undo ("undo back to auto-decide") can find it. (KT section 7)
# ============================================================

from datetime import datetime, timezone
import copy


_DROP_GUARDRAIL = 0.75


def run(state: dict) -> dict:
    """
    Load bot verdicts as a draft decisions state.
    KEEP → keep, DROP/DROP-NULL → drop, FLAG stays pending.
    User can freely override any of these before generating report.
    """
    session      = state["session"]
    verdict_df   = session.get("verdict_df")
    decisions    = state.get("decisions", {})
    undo_stack   = state.get("undo_stack", [])
    decision_log = state.get("decision_log", [])

    if verdict_df is None:
        return {"action_result": {
            "error": "Scan results not available — cannot apply auto-decisions."
        }}

    feature_cols = session.get("feature_cols", [])
    total = len(feature_cols)

    keep_cols  = verdict_df[verdict_df["verdict"] == "KEEP"]["column"].tolist()
    flag_cols  = verdict_df[verdict_df["verdict"] == "FLAG"]["column"].tolist()
    drop_cols  = verdict_df[verdict_df["verdict"] == "DROP"]["column"].tolist()
    dropn_cols = verdict_df[verdict_df["verdict"] == "DROP-NULL"]["column"].tolist()
    all_drop   = drop_cols + dropn_cols

    already_decided = set(decisions.keys())
    to_keep       = [c for c in keep_cols  if c not in already_decided]
    to_drop       = [c for c in all_drop   if c not in already_decided]
    pending_flags = [c for c in flag_cols  if c not in already_decided]

    # ── Idempotency guard ─────────────────────────────────────
    # If there is nothing new to apply (all bot verdicts already loaded),
    # return a clear informative message rather than writing 0 decisions
    # and misleadingly showing 0 kept / 0 dropped.
    if not to_keep and not to_drop:
        kept_count    = sum(1 for v in decisions.values() if v == "keep")
        dropped_count = sum(1 for v in decisions.values() if v == "drop")
        has_overrides = any(e.get("override") for e in decision_log)
        return {
            "action_result": {
                "already_applied"  : True,
                "draft_mode"       : True,
                "has_user_overrides": has_overrides,
                "kept"             : kept_count,
                "dropped"          : dropped_count,
                "pending_flags"    : len(pending_flags),
                "flag_columns"     : pending_flags,
                "total_features"   : total,
                "message": (
                    f"Bot recommendations are already loaded — "
                    f"{kept_count} columns marked keep and {dropped_count} marked drop. "
                    f"{len(pending_flags)} flagged columns still need your review. "
                    f"You can override any decision before generating the report."
                ),
            }
        }

    # ── Guardrail check ───────────────────────────────────────
    existing_drops        = sum(1 for v in decisions.values() if v == "drop")
    projected_total_drops = existing_drops + len(to_drop)

    if total > 0 and (projected_total_drops / total) > _DROP_GUARDRAIL:
        return {
            "action_result": {
                "guardrail_triggered" : True,
                "projected_drops"     : projected_total_drops,
                "total_features"      : total,
                "drop_pct"            : round(projected_total_drops / total * 100, 1),
                "to_keep"             : len(to_keep),
                "to_drop"             : len(to_drop),
                "pending_flags"       : len(pending_flags),
                "message": (
                    f"Loading bot recommendations as a draft would mark "
                    f"{projected_total_drops} of {total} columns as drop "
                    f"({projected_total_drops/total*100:.0f}%) — above the 75% guardrail. "
                    f"Type 'confirm auto-decide' to proceed anyway."
                ),
            }
        }

    # ── Push pre-write snapshot (for plain step-back undo) ────
    # label="pre_auto_decide" lets a plain 1-step undo fully reverse
    # the auto-decide run without needing time-travel.
    undo_stack = undo_stack + [{
        "decisions"   : copy.deepcopy(decisions),
        "active_focus": state.get("active_focus"),
        "label"       : "pre_auto_decide",
    }]

    # ── Apply draft decisions ─────────────────────────────────
    new_decisions = dict(decisions)
    ts    = datetime.now(timezone.utc).isoformat()
    new_log = list(decision_log)

    for col in to_keep:
        override = decisions.get(col) == "drop"
        new_decisions[col] = "keep"
        new_log.append({
            "col"      : col,
            "decision" : "keep",
            "reason"   : "Bot recommendation: 2+ statistical tests passed",
            "source"   : "bot",
            "timestamp": ts,
            "override" : override,
        })

    for col in to_drop:
        override = decisions.get(col) == "keep"
        reason = (
            "Bot recommendation: 100% null — no data"
            if col in dropn_cols
            else "Bot recommendation: no statistical signal detected"
        )
        new_decisions[col] = "drop"
        new_log.append({
            "col"      : col,
            "decision" : "drop",
            "reason"   : reason,
            "source"   : "bot",
            "timestamp": ts,
            "override" : override,
        })
        
    # ── Push post-write snapshot WITH label "auto_decide" ─────
    # Time-travel undo ("undo back to auto-decide") must restore the
    # state *after* decisions were written, not the empty pre-write
    # state. Pushing here, after new_decisions is fully populated,
    # means _undo_to_target restores a non-empty decisions dict.
    undo_stack = undo_stack + [{
        "decisions"   : copy.deepcopy(new_decisions),
        "active_focus": None,
        "label"       : "auto_decide",
    }]

    total_decided = len(new_decisions)
    coverage_pct  = round(total_decided / total * 100, 1) if total > 0 else 0

    return {
        "action_result": {
            "auto_decided"         : True,
            "draft_mode"           : True,       # D1: signals formatter to present as draft
            "kept"                 : len(to_keep),
            "dropped"              : len(to_drop),
            "pending_flags"        : len(pending_flags),
            "flag_columns"         : pending_flags,
            "total_decided"        : total_decided,
            "total_features"       : total,
            "coverage_pct"         : coverage_pct,
            "already_had_decisions": len(already_decided),
            "message": (
                f"Loaded bot recommendations as a draft: {len(to_keep)} marked keep, "
                f"{len(to_drop)} marked drop. "
                f"{len(pending_flags)} flagged columns still need your review. "
                f"You can override any of these before generating the report."
            ),
        },
        "decisions"   : new_decisions,
        "undo_stack"  : undo_stack,
        "decision_log": new_log,
        "active_focus": None,
    }