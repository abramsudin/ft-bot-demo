# ============================================================
# actions/undo.py
#
# Action: UNDO
#
# Restores the decisions dict to the state it was in before the
# most recent change. Supports two modes:
#
#   Full undo  — pop the top undo_stack snapshot (no column named)
#   Column undo — reverse only the named column to its prior value
#
# Appends undo events to decision_log (append-only — never deletes).
#
# Public API:
#   run(state: dict) -> dict
#
# Returns partial state with:
#   - action_result  : outcome summary for the formatter
#   - decisions      : restored decisions dict
#   - undo_stack     : updated (top snapshot consumed)
#   - decision_log   : undo event(s) appended
#
# No LLM calls. No global side effects. Pure Python.
# ============================================================

from datetime import datetime, timezone


def run(state: dict) -> dict:
    """
    Restore the decisions dict to the previous snapshot.

    intent_params schema:
        column : str | None — if provided, only undo that column

    Parameters
    ----------
    state : dict — full GraphState

    Returns
    -------
    dict — partial state update:

        Full undo success:
            action_result : {
                mode         : 'full',
                reverted     : list[str],   # columns whose value changed
                snapshot_age : int,         # how many ops ago (stack depth)
            }
            decisions    : restored dict
            undo_stack   : top snapshot removed
            decision_log : undo entries appended

        Column undo success:
            action_result : {
                mode         : 'column',
                column       : str,
                prior_value  : 'keep' | 'drop' | None,
                removed      : bool,   # True if decision was fully removed
            }
            decisions    : updated dict
            undo_stack   : top snapshot removed
            decision_log : undo entry appended

        Nothing to undo:
            action_result : {
                error : str
            }
    """
    decisions    = dict(state.get("decisions", {}))
    undo_stack   = list(state.get("undo_stack", []))
    decision_log = list(state.get("decision_log", []))

    intent_params = state.get("intent_params", {})
    column        = intent_params.get("column")
    # Issue #10: multi-step undo — extract steps count and target_action
    steps         = intent_params.get("steps")
    target_action = (intent_params.get("target_action") or "").strip().lower()

    # Normalise steps: default plain undo to 1 step
    if steps is None and not column and not target_action:
        steps = 1
    try:
        steps = int(steps) if steps is not None else None
    except (TypeError, ValueError):
        steps = 1

    # ── Guard: nothing to undo ────────────────────────────────
    if not undo_stack:
        return {
            "action_result": {
                "error": (
                    "Nothing to undo — no decisions have been recorded yet."
                )
            }
        }

    # ── Route ────────────────────────────────────────────────
    if column:
        # Column-level undo (single step, specific column)
        return _undo_column(column, decisions, undo_stack, decision_log)
    elif target_action:
        # Time-travel: rewind until we find a snapshot labelled with target_action
        return _undo_to_target(target_action, decisions, undo_stack, decision_log)
    else:
        # Multi-step full undo: pop N snapshots
        steps = min(steps or 1, len(undo_stack))   # clamp to stack depth
        return _undo_multi(steps, decisions, undo_stack, decision_log)


# ── Multi-step full undo (Issue #10) ─────────────────────────

def _undo_multi(
    steps: int,
    decisions: dict,
    undo_stack: list,
    decision_log: list,
) -> dict:
    """
    Pop N snapshots from the undo stack in sequence, restoring the
    decisions dict to where it was N operations ago.

    Collects all diffs across the N pops so the formatter can report
    every column that changed during the time-travel.
    """
    original_decisions = dict(decisions)
    total_reverted: set = set()
    now = _now()
    prior_active_focus = None   # initialise before loop in case stack is empty

    for step_i in range(steps):
        if not undo_stack:
            break
        raw_snapshot = undo_stack.pop()
        if isinstance(raw_snapshot, dict) and "decisions" in raw_snapshot:
            prior_snapshot     = raw_snapshot["decisions"]
            prior_active_focus = raw_snapshot.get("active_focus")
        else:
            prior_snapshot     = raw_snapshot
            prior_active_focus = None

        decisions = dict(prior_snapshot)

    # Compute the true diff: final restored state vs original state.
    # Doing this AFTER the loop (rather than accumulating per-step diffs) ensures
    # reverted_count is accurate — intermediate oscillations (drop→keep→drop across
    # multiple steps) don't inflate the count if the column ends up at the same value.
    all_keys = set(original_decisions.keys()) | set(decisions.keys())
    for col in all_keys:
        if original_decisions.get(col) != decisions.get(col):
            total_reverted.add(col)

    # Append one audit entry per reverted column
    for col in total_reverted:
        prior_val   = decisions.get(col)
        current_val = original_decisions.get(col)
        decision_log.append({
            "col"      : col,
            "decision" : "undo",
            "reason"   : (
                f"multi-step undo ({steps} step(s)) — "
                f"reverted from '{current_val}' to '{prior_val or 'undecided'}'"
            ),
            "source"   : "undo",
            "timestamp": now,
            "override" : False,
        })

    result = {
        "action_result": {
            "mode"        : "multi_step",
            "steps"       : steps,
            "reverted"    : sorted(total_reverted),
            # total_reverted is columns whose value actually changed;
            # the snapshot may have covered more columns that were already
            # at the same value (e.g. pre-existing drops in a bulk-drop undo).
            "reverted_count"         : len(total_reverted),
        },
        "decisions"   : decisions,
        "undo_stack"  : undo_stack,
        "decision_log": decision_log,
    }
    if prior_active_focus is not None:
        result["active_focus"] = prior_active_focus
    return result


def _undo_to_target(
    target_action: str,
    decisions: dict,
    undo_stack: list,
    decision_log: list,
) -> dict:
    """
    Rewind the stack until a snapshot whose label matches target_action
    is found (e.g. "auto_decide"). Each snapshot may carry an optional
    "label" key set by the action that pushed it.

    Falls back to a full 1-step undo if no matching label is found.
    """
    original_decisions = dict(decisions)
    total_reverted: set = set()
    steps_taken = 0
    now = _now()
    prior_active_focus = None

    while undo_stack:
        raw_snapshot = undo_stack[-1]   # peek first
        label = ""
        if isinstance(raw_snapshot, dict):
            label = raw_snapshot.get("label", "").lower()

        raw_snapshot = undo_stack.pop()
        steps_taken += 1

        if isinstance(raw_snapshot, dict) and "decisions" in raw_snapshot:
            prior_snapshot     = raw_snapshot["decisions"]
            prior_active_focus = raw_snapshot.get("active_focus")
        else:
            prior_snapshot     = raw_snapshot
            prior_active_focus = None

        decisions = dict(prior_snapshot)

        if _normalize(target_action) in _normalize(label):
            break   # found the target snapshot — stop here

    # True diff: final restored state vs original (same rationale as _undo_multi)
    all_keys = set(original_decisions.keys()) | set(decisions.keys())
    for col in all_keys:
        if original_decisions.get(col) != decisions.get(col):
            total_reverted.add(col)

    for col in total_reverted:
        prior_val   = decisions.get(col)
        current_val = original_decisions.get(col)
        decision_log.append({
            "col"      : col,
            "decision" : "undo",
            "reason"   : (
                f"time-travel undo to '{target_action}' "
                f"({steps_taken} step(s)) — "
                f"reverted from '{current_val}' to '{prior_val or 'undecided'}'"
            ),
            "source"   : "undo",
            "timestamp": now,
            "override" : False,
        })

    result = {
        "action_result": {
            "mode"         : "time_travel",
            "target_action": target_action,
            "steps_taken"  : steps_taken,
            "reverted"     : sorted(total_reverted),
        },
        "decisions"   : decisions,
        "undo_stack"  : undo_stack,
        "decision_log": decision_log,
    }
    if prior_active_focus is not None:
        result["active_focus"] = prior_active_focus
    return result


# ── Full undo (1-step alias kept for backward compatibility) ──

def _undo_full(
    decisions: dict,
    undo_stack: list,
    decision_log: list,
) -> dict:
    """
    Pop the top snapshot and restore the entire decisions dict.

    Computes the diff so the formatter can tell the user exactly
    which columns were reverted.
    """
    raw_snapshot   = undo_stack.pop()        # consume top snapshot
    snapshot_age   = len(undo_stack) + 1     # 1-indexed depth before pop

    # Support both old format (bare dict) and new format ({decisions, active_focus})
    if isinstance(raw_snapshot, dict) and "decisions" in raw_snapshot:
        prior_snapshot    = raw_snapshot["decisions"]
        prior_active_focus = raw_snapshot.get("active_focus")
    else:
        prior_snapshot     = raw_snapshot   # legacy bare dict
        prior_active_focus = None

    # Diff: which columns actually changed?
    reverted = []
    all_keys = set(decisions.keys()) | set(prior_snapshot.keys())
    for col in all_keys:
        if decisions.get(col) != prior_snapshot.get(col):
            reverted.append(col)

    # Append one undo log entry per reverted column
    now = _now()
    for col in reverted:
        prior_val   = prior_snapshot.get(col)
        current_val = decisions.get(col)
        decision_log.append({
            "col"       : col,
            "decision"  : "undo",
            "reason"    : (
                f"full undo — reverted from '{current_val}' "
                f"to '{prior_val or 'undecided'}'"
            ),
            "source"    : "undo",
            "timestamp" : now,
            "override"  : False,
        })

    result = {
        "action_result": {
            "mode"        : "full",
            "reverted"    : sorted(reverted),
            "reverted_count": len(reverted),
            "snapshot_age": snapshot_age,
            # If the undo snapshot came from a bulk-drop action, the zone may have
            # contained more columns than were actually changed (some were already
            # dropped before the bulk ran). reverted_count is the true diff;
            # the zone size reported in the original DECIDE result may be larger.
            "note": (
                "Only columns whose value changed are counted in reverted_count. "
                "Columns already at the same value before the operation are excluded."
            ) if reverted else None,
        },
        "decisions"   : prior_snapshot,   # fully restored
        "undo_stack"  : undo_stack,
        "decision_log": decision_log,
    }
    # Restore active_focus from snapshot if it was recorded there.
    # This fixes the undo bug where active_focus stayed as the last single
    # column instead of restoring the previous list (e.g. ["Var10","Var12","Var15"]).
    if prior_active_focus is not None:
        result["active_focus"] = prior_active_focus
    return result


# ── Column-level undo ─────────────────────────────────────────

def _undo_column(
    column: str,
    decisions: dict,
    undo_stack: list,
    decision_log: list,
) -> dict:
    """
    Reverse the last recorded decision for a specific column.

    Looks up the column's value in the most recent undo snapshot
    (which captured state just before the last write) and restores
    only that column. The rest of the decisions dict is untouched.

    If the column did not exist in the prior snapshot, the decision
    is removed entirely (equivalent to "forget this decision").
    """
    raw_snapshot = undo_stack[-1]     # peek — pop only after validation
    # Support both old format (bare dict) and new format ({decisions, active_focus})
    if isinstance(raw_snapshot, dict) and "decisions" in raw_snapshot:
        prior_snapshot     = raw_snapshot["decisions"]
        prior_active_focus = raw_snapshot.get("active_focus")
    else:
        prior_snapshot     = raw_snapshot
        prior_active_focus = None
    current_val    = decisions.get(column)

    # If the column has no current decision, nothing to undo for it
    if current_val is None and column not in prior_snapshot:
        return {
            "action_result": {
                "error": (
                    f"'{column}' has no recorded decision to undo."
                )
            }
        }

    prior_val = prior_snapshot.get(column)   # None means "not yet decided"

    # Apply the column-level change
    if prior_val is None:
        # Column wasn't in scope before — remove it
        decisions.pop(column, None)
        removed = True
    else:
        decisions[column] = prior_val
        removed = False

    # Consume the snapshot (the write already happened — snapshot is stale
    # now; further undos should go to the next entry on the stack)
    undo_stack.pop()

    # Append undo log entry
    decision_log.append({
        "col"      : column,
        "decision" : "undo",
        "reason"   : (
            f"column undo — reverted from '{current_val}' "
            f"to '{prior_val or 'undecided'}'"
        ),
        "source"   : "undo",
        "timestamp": _now(),
        "override" : False,
    })

    result = {
        "action_result": {
            "mode"       : "column",
            "column"     : column,
            "prior_value": prior_val,
            "removed"    : removed,
        },
        "decisions"   : decisions,
        "undo_stack"  : undo_stack,
        "decision_log": decision_log,
    }
    # Restore active_focus from snapshot so multi-column context
    # is preserved correctly after a column-level undo.
    if prior_active_focus is not None:
        result["active_focus"] = prior_active_focus
    return result


# ── Helpers ───────────────────────────────────────────────────

def _now() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()

def _normalize(s: str) -> str:
    """Unify spaces, hyphens, and underscores for robust string matching."""
    return str(s or "").replace("_", " ").replace("-", " ").strip().lower()