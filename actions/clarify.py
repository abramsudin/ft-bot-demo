# ============================================================
# actions/clarify.py
#
# Action: AMBIGUOUS
#
# Called when the classifier cannot resolve intent with sufficient
# confidence. Stores the ambiguous user message and produces a
# structured result the formatter uses to ask exactly ONE
# targeted clarifying question.
#
# Rules (from blueprint §5.9):
#   - Ask exactly one question. Never two.
#   - The question must be targeted — specific to what is actually
#     ambiguous, not a generic "what did you mean?" fallback.
#   - The formatter crafts the natural-language question from the
#     structured result this action returns.
#
# Public API:
#   run(state: dict) -> dict
#
# Returns partial state with:
#   - action_result : structured disambiguation context
#
# No LLM calls. No state mutations. Pure Python.
# ============================================================


def run(state: dict) -> dict:
    """
    Produce a structured disambiguation payload for the formatter.

    The formatter receives this and crafts exactly one targeted
    question based on the ambiguity_type field.

    Parameters
    ----------
    state : dict — full GraphState

    Returns
    -------
    dict — partial state update:
        action_result : {
            ambiguity_type   : str,       # see AMBIGUITY_TYPES below
            original_message : str,       # the user's raw message
            active_focus     : str | None,# column currently in scope
            candidates       : list[str], # options to present (if any)
            hint             : str,       # one-line prompt for formatter
        }
    """
    messages     = state.get("messages", [])
    active_focus = state.get("active_focus")
    intent_params = state.get("intent_params", {})
    session      = state.get("session", {})
    feature_cols = session.get("feature_cols", [])

    # Extract the last user message
    original_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            original_message = msg.get("content", "")
            break

    # ── Determine ambiguity type ──────────────────────────────
    ambiguity_type, candidates, hint = _classify_ambiguity(
        original_message  = original_message,
        intent_params     = intent_params,
        active_focus      = active_focus,
        feature_cols      = feature_cols,
    )

    return {
        "action_result": {
            "ambiguity_type"  : ambiguity_type,
            "original_message": original_message,
            "active_focus"    : active_focus,
            "candidates"      : candidates,
            "hint"            : hint,
        }
    }


# ── Ambiguity classification ──────────────────────────────────

# Ambiguity type strings (formatter uses these to shape its question)
AT_NO_COLUMN       = "no_column"         # action needs a column, none resolved
AT_AMBIGUOUS_COLUMN= "ambiguous_column"  # partial name matches multiple columns
AT_NO_DECISION     = "no_decision"       # keep or drop not clear
AT_PRONOUN_UNCLEAR = "pronoun_unclear"   # "it"/"this" with no active_focus
AT_INTENT_UNCLEAR  = "intent_unclear"    # could be ANALYSE, EXPLORE, or DECIDE
AT_FILTER_UNCLEAR  = "filter_unclear"    # EXPLORE filter not understood
AT_GENERIC         = "generic"           # fallback — formatter asks open question


def _classify_ambiguity(
    original_message: str,
    intent_params: dict,
    active_focus: str | None,
    feature_cols: list,
) -> tuple[str, list[str], str]:
    """
    Return (ambiguity_type, candidates, hint).

    Tries to give the formatter the most specific disambiguation
    type possible so it can ask a single, pinpointed question.
    """
    msg_lower = original_message.lower()

    # ── Pronoun with no focus ─────────────────────────────────
    PRONOUNS = ("it", "this", "that", "the column", "this column", "that column")
    if any(p in msg_lower for p in PRONOUNS) and not active_focus:
        return (
            AT_PRONOUN_UNCLEAR,
            [],
            "User used a pronoun but no column is in scope — ask which column.",
        )

    # ── Ambiguous column name (partial match → multiple hits) ─
    partial = intent_params.get("column") or ""
    if partial:
        matches = _partial_matches(partial, feature_cols)
        if len(matches) > 1:
            return (
                AT_AMBIGUOUS_COLUMN,
                matches[:5],  # cap at 5 options
                f"'{partial}' matches multiple columns — ask which one.",
            )

    # ── Decision missing (DECIDE intent without keep/drop) ────
    if _mentions_decision_verb(msg_lower) and not intent_params.get("decision"):
        return (
            AT_NO_DECISION,
            ["keep", "drop"],
            "Decision verb detected but keep/drop not clear — ask which.",
        )

    # ── Analyse/Decide/Explore without a column ───────────────
    if _mentions_column_action(msg_lower) and not partial and not active_focus:
        return (
            AT_NO_COLUMN,
            [],
            "Action requires a column but none was named or in scope — ask which column.",
        )

    # ── Explore filter not understood ────────────────────────
    if _mentions_explore(msg_lower) and not intent_params.get("filter"):
        return (
            AT_FILTER_UNCLEAR,
            ["KEEP", "FLAG", "DROP", "numeric", "categorical", "high confidence"],
            "EXPLORE action but filter is unclear — ask what to filter by.",
        )

    # ── Could be ANALYSE or DECIDE (ambiguous action) ─────────
    if _mentions_both_actions(msg_lower):
        return (
            AT_INTENT_UNCLEAR,
            ["Analyse (deep-dive stats)", "Decide (keep or drop)"],
            "Message could mean ANALYSE or DECIDE — ask what the user wants to do.",
        )

    # ── Generic fallback ──────────────────────────────────────
    return (
        AT_GENERIC,
        [],
        "Intent unclear — ask the user to rephrase or be more specific.",
    )


# ── Keyword helpers ───────────────────────────────────────────

def _partial_matches(partial: str, feature_cols: list) -> list[str]:
    p = partial.lower()
    return [c for c in feature_cols if p in c.lower()]


def _mentions_decision_verb(msg: str) -> bool:
    return any(v in msg for v in ("keep", "drop", "remove", "retain", "include", "exclude"))


def _mentions_column_action(msg: str) -> bool:
    return any(v in msg for v in (
        "analyse", "analyze", "look at", "tell me about",
        "what about", "show me", "deep dive", "eda",
        "decide", "keep", "drop",
    ))


def _mentions_explore(msg: str) -> bool:
    return any(v in msg for v in (
        "show all", "list", "explore", "which columns", "what columns",
        "show me all", "filter",
    ))


def _mentions_both_actions(msg: str) -> bool:
    has_analyse = any(v in msg for v in ("analyse", "analyze", "what does", "tell me about", "stats"))
    has_decide  = any(v in msg for v in ("keep", "drop", "decide", "remove"))
    return has_analyse and has_decide
