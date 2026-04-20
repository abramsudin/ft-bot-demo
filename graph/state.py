# ============================================================
# graph/state.py
#
# Defines the single shared state TypedDict that flows through
# every node in the LangGraph agent.
#
# Fields are grouped by concern:
#   - Pipeline data      : session
#   - Conversation       : messages, last_response
#   - Decision ledger    : decisions, undo_stack, decision_log
#   - Focus tracking     : active_focus
#   - Intent routing     : intent, intent_params, action_result
#   - Overview mode      : overview_mode
#
# RULES:
#   - session is loaded once at startup and NEVER mutated
#   - decision_log is append-only (undo adds entries, never deletes)
#   - active_focus can now be a single column name, a list of column
#     names, or None. A list is set when the user references multiple
#     columns in one turn (e.g. "analyse Var22, Var34, Var216").
#     Pronoun resolution ("drop them all") resolves against the list.
#   - chart_b64 / chart_filename are cleared by UNDERSTAND at the
#     start of each new turn, not by RESPOND (so main.py can read
#     them after the full turn completes). Both can now be a single
#     string OR a list of strings when multi-column EDA generates
#     more than one chart.
#   - overview_mode tracks whether the last OVERVIEW was "fresh"
#     (no user decisions yet) or "post_tweak" (user has made manual
#     changes on top of bot recommendations). Set by overview.run().
#   - eda_shown_visuals maps column_name -> list of visual_type
#     strings already shown, so progressive disclosure knows what
#     panels remain to offer the user.
# ============================================================

from __future__ import annotations
from typing import Any, Optional, Union
from typing_extensions import TypedDict


class GraphState(TypedDict):

    # ── 1. Pipeline data ──────────────────────────────────────
    # The full session dict produced by pipeline/session.py.
    # Contains: df, target, feature_cols, num_cols, cat_cols,
    #           null_group_map, null_scan_df, verdict_df,
    #           pairs_df, redundancy_drop, output_dir.
    # Treat as READ-ONLY — no node should mutate this dict.
    session: dict[str, Any]

    # ── 2. Conversation history ───────────────────────────────
    # Full message log in OpenAI-style format.
    # Each entry: {"role": "user" | "assistant", "content": str}
    # RESPOND node appends the assistant reply after every turn.
    messages: list[dict[str, str]]

    # ── 3. Decision ledger ────────────────────────────────────
    # Live map of column_name -> 'keep' | 'drop'.
    # Written by decide.run() and auto_decide.run().
    # Read by status.run() and report.run().
    decisions: dict[str, str]

    # ── 4. Undo stack ─────────────────────────────────────────
    # Stack of snapshots pushed BEFORE every write operation.
    # Each snapshot is a dict with keys:
    #   "decisions"    : dict[str, str]   — full decisions copy
    #   "active_focus" : str | list | None
    #   "label"        : str              — optional tag for time-travel undo
    #                                       e.g. "auto_decide", "bulk_drop"
    # Issue #10: undo.run() now supports popping N snapshots (multi-step)
    # and rewinding to a named snapshot via target_action / label matching.
    undo_stack: list[dict[str, Any]]

    # ── 5. Decision log ───────────────────────────────────────
    # Append-only audit trail of every decision event.
    # Each entry: {
    #   col        : str,
    #   decision   : 'keep' | 'drop' | 'undo',
    #   reason     : str,
    #   source     : 'user' | 'bot' | 'undo',
    #   timestamp  : str  (ISO format),
    #   override   : bool  (True when user overrides bot verdict)
    # }
    # NOTHING is ever deleted from this list — undo adds a new entry.
    decision_log: list[dict[str, Any]]

    # ── 6. Active focus ───────────────────────────────────────
    # The column(s) currently in scope, or None.
    # - Single column str  : set when one column is resolved
    # - List of str        : set when multiple columns are resolved
    #                        in one turn (e.g. "analyse Var22, Var34")
    # - None               : cleared for overview / ambiguous ops
    #
    # Powers pronoun resolution:
    #   "drop it"       resolves to active_focus (single str)
    #   "drop them all" resolves to active_focus (list)
    active_focus: Optional[Union[str, list[str]]]

    # ── 7. Intent ─────────────────────────────────────────────
    # One of 13 valid intent strings:
    #   ANALYSE | EXPLORE | EDA | DECIDE | UNDO | STATUS |
    #   EXPLAIN | REPORT | OVERVIEW | AUTO_DECIDE | COMPARE |
    #   CONDITIONAL_DECIDE | AMBIGUOUS
    # Set by UNDERSTAND; consumed by ACT.
    intent: Optional[str]

    # ── 8. Intent params ──────────────────────────────────────
    # Structured parameters extracted alongside the intent.
    # Schema varies by intent (see llm/classifier.py):
    #   ANALYSE     -> {columns: list[str], zone: str | None}
    #                  zone set for zone-level queries (Issue #6 / v7 C1)
    #   EXPLORE     -> {filter: str}
    #   EDA         -> {columns: list[str], visual_type: str}
    #                  visual_type: "default" | "null_distribution" |
    #                  "churn_rate" | "distribution" | "null_gap"
    #   COMPARE     -> {columns: list[str], visual_type: str}
    #   DECIDE      -> {column: str | None,
    #                   columns: list[str] | None,  # v7 C3: multi-col from focus
    #                   zone: str | None,
    #                   zones: list[str] | None,    # multi-zone (Issue #1)
    #                   decision: 'keep' | 'drop'}
    #   UNDO        -> {column: str | None,
    #                   steps: int | None,           # multi-step (Issue #10)
    #                   target_action: str | None}   # time-travel (Issue #10)
    #   EXPLAIN     -> {concepts: list[str]}          # multi-concept (Issue #5)
    #   AMBIGUOUS   -> {reason: str | None}           # conditional (Issue #4)
    #   OVERVIEW    -> {}
    #   AUTO_DECIDE -> {}
    #   STATUS / REPORT -> {}
    #   CONDITIONAL_DECIDE -> {                       # v7 N1
    #     conditions: [{field, operator, threshold}],
    #     condition_logic: "AND" | "OR",
    #     decision: "keep" | "drop",
    #     scope: str | None,   # zone to restrict to, or null = all columns
    #     dry_run: bool        # True = preview only, False = apply
    #   }
    intent_params: dict[str, Any]

    # ── 9. Action result ──────────────────────────────────────
    # Raw result dict returned by the action function in ACT node.
    # Passed directly to formatter.format_response() in RESPOND node.
    # Reset to None after RESPOND consumes it.
    action_result: Optional[dict[str, Any]]

    # ── 10. Last response ─────────────────────────────────────
    # Plain-English string produced by formatter.format_response().
    # Stored so main.py / app.py can read it without parsing messages.
    last_response: Optional[str]

    # ── 11. Overview mode ─────────────────────────────────────
    # Tracks whether the most recent OVERVIEW was:
    #   "fresh"      — no user decisions exist yet (pure bot scan view)
    #   "post_tweak" — user has made manual keep/drop decisions that
    #                  differ from the bot's original verdicts
    # Set by overview.run() by comparing decisions ledger vs verdict_df.
    # Read by formatter.py to adjust narrative tone.
    overview_mode: Optional[str]
    
    # ── 12. Focus Age (I-3 Fix) ───────────────────────────────
    # Tracks how many turns the current active_focus has been active
    # without changing. Resets to 0 when focus changes.
    # Used by classifier to detect stale pronouns and trigger AMBIGUOUS.
    focus_age: int

    # ── 13. Draft Mode (I-2 Fix) ──────────────────────────────
    # True if AUTO_DECIDE has loaded a batch of recommendations
    # that the user is currently reviewing/tweaking.
    # Tells the formatter to remind the user they are in a draft state.
    draft_mode: bool