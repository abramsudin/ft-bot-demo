# ============================================================
# graph/nodes.py
#
# The three LangGraph nodes every message passes through:
#   UNDERSTAND → ACT → RESPOND
#
# Changes vs previous version:
#   - UNDERSTAND clears eda_shown_visuals is NOT reset each turn
#     (it persists across the session by design)
#   - overview_mode is preserved across turns (set by overview.run)
#   - active_focus can now be str | list[str] | None
#   - chart_b64 / chart_filename cleared at start of each new turn
#   - active_focus is only updated when resolved_focus is not None,
#     so AMBIGUOUS clarification turns no longer wipe the in-scope
#     column(s) from the previous turn
#
# Fixes applied:
#   - I-1: Consume focus_clear from classifier to explicitly clear
#     active_focus on clean topic pivots (EXPLAIN, STATUS, OVERVIEW, etc.)
#   - I-2: Preserve active_focus as None when draft_mode is active
#     and AUTO_DECIDE just ran; set draft_mode flag from action_result
#   - I-3: Track focus_age counter; reset to 0 when focus changes,
#     increment otherwise. Used by classifier for staleness detection.
# ============================================================

from llm   import classifier, formatter
from graph import edges


def understand_node(state: dict) -> dict:
    """
    LLM Call #1.

    Classifies intent, resolves pronouns, updates active_focus.
    active_focus may now be a single str, a list of strs, or None.

    Focus update priority (I-1 + I-3 fix):
      1. focus_clear=True  → clear active_focus to None (clean topic pivot)
      2. resolved_focus is not None → update to resolved value, reset focus_age
      3. otherwise → keep existing active_focus, increment focus_age
    """
    result = classifier.classify(
        messages     = state["messages"],
        active_focus = state["active_focus"],
        session      = state["session"],
        focus_age    = state.get("focus_age", 0),   # I-3: pass staleness counter
    )

    resolved   = result.get("resolved_focus")
    focus_clear = result.get("focus_clear", False)  # I-1: clean pivot signal

    # ── I-1 + I-3: Determine new active_focus and focus_age ──────────────
    current_age = state.get("focus_age", 0)

    # BUG 4 FIX: Don't clear focus if this is an ANALYSE deep-dive follow-up.
    # focus_clear fires on ANALYSE turns too, but deep_dive follow-ups need
    # active_focus intact so eda.py can resolve the column.
    intent_from_classifier = result.get("intent")
    is_deep_dive = result.get("params", {}).get("deep_dive", False)

    if focus_clear and not (intent_from_classifier == "ANALYSE" and is_deep_dive):
        if resolved is not None:
            print(
                f"[understand_node] WARNING: focus_clear=True but resolved_focus={resolved!r} "
                f"was also set (classifier contradiction). Honouring focus_clear."
            )
        new_active_focus = None
        new_focus_age    = 0
    elif resolved is not None:
        # A real column was resolved this turn — update and reset age.
        new_active_focus = resolved
        new_focus_age    = 0
        
        # P0-2 FIX: If active_focus was None coming in (just cleared),
        # and the resolved column doesn't literally appear in the current
        # user message, the classifier pulled it from history — force AMBIGUOUS.
        if state["active_focus"] is None:
            user_msg = ""
            for msg in reversed(state.get("messages", [])):
                if msg.get("role") == "user":
                    user_msg = msg.get("content", "")
                    break
            if isinstance(new_active_focus, str) and new_active_focus not in user_msg:
                result["intent"] = "AMBIGUOUS"
                result["params"] = {
                    "ambiguity_type"       : "no_column",
                    "stale_focus_candidate": new_active_focus,
                }
                new_active_focus = None
                new_focus_age    = 0
    
    else:
        # No new column info — keep existing focus, age it by one turn.
        new_active_focus = state["active_focus"]
        new_focus_age    = current_age + 1

    # ── Debug logging (set UNDERSTAND_DEBUG=1 to enable) ─────────────────
    import os
    if os.environ.get("UNDERSTAND_DEBUG", "0") == "1":
        print(
            f"[understand_node] intent={result['intent']} | "
            f"focus_clear={focus_clear} | resolved={resolved!r} | "
            f"active_focus: {state['active_focus']!r} → {new_active_focus!r} | "
            f"focus_age: {current_age} → {new_focus_age}"
        )

    return {
        "intent"        : result["intent"],
        "intent_params" : result["params"],
        "active_focus"  : new_active_focus,
        "focus_age"     : new_focus_age,
        # Clear chart fields from previous turn
        "chart_b64"     : None,
        "chart_filename": None,
        # BUG-1 FIX: explicitly carry forward decision state
        # so LangGraph doesn't reset them between nodes
        "decisions"     : state.get("decisions", {}),
        "undo_stack"    : state.get("undo_stack", []),
        "decision_log"  : state.get("decision_log", []),
    }


def act_node(state: dict) -> dict:
    """
    Pure Python routing — no LLM call.
    Routes to the correct action via edges.route() and returns
    the partial state update from that action.

    Issue #8: Wrapped in try/except so raw Python tracebacks never
    reach the chat UI. Errors are logged to console for debugging
    and returned as a clean {"error": True} dict for the formatter
    to turn into a polite apology message.
    """
    try:
        action_fn = edges.route(state["intent"])
        return action_fn(state)
    except Exception as exc:
        import traceback
        # Log full traceback to console for developer debugging
        print(f"[act_node ERROR] intent={state.get('intent')} | {exc}")
        traceback.print_exc()
        # Return a clean error payload — formatter will narrate politely
        return {
            "action_result": {
                "error"      : True,
                "error_type" : type(exc).__name__,
                "error_msg"  : str(exc),
                "intent"     : state.get("intent"),
            }
        }


def respond_node(state: dict) -> dict:
    """
    LLM Call #2.

    Formats action_result into a plain-English reply.
    Appends it to messages. Clears action_result.
    Does NOT clear chart_b64, overview_mode, or eda_shown_visuals.

    I-2 fix: Reads draft_mode from action_result if the key is present
    so AUTO_DECIDE can set it and REPORT can clear it without nodes.py
    needing to know about every action's internals.
    """
    # BUG 5 FIX: Pass the user's latest message so formatter can answer stat questions directly
    latest_user_msg = ""
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "user":
            latest_user_msg = msg.get("content", "")
            break

    response = formatter.format_response(
        intent            = state["intent"],
        action_result     = state["action_result"],
        draft_mode        = state.get("draft_mode", False),
        user_message      = latest_user_msg,
        guardrail_pending = state.get("guardrail_pending", False),
    )

    updated_messages = state["messages"] + [
        {"role": "assistant", "content": response}
    ]

    # I-2: propagate draft_mode changes written by actions
    action_result = state["action_result"] or {}
    new_draft_mode = state.get("draft_mode", False)
    if "draft_mode" in action_result:
        new_draft_mode = action_result["draft_mode"]

    # Build base return
    respond_result = {
        "last_response": response,
        "messages"     : updated_messages,
        "action_result": None,
        "draft_mode"   : new_draft_mode,
        # BUG 1 FIX: Explicitly forward decision state so LangGraph doesn't wipe it
        "decisions"    : state.get("decisions", {}),
        "undo_stack"   : state.get("undo_stack", []),
        "decision_log" : state.get("decision_log", []),
    }

    # Forward active_focus if an action explicitly set it (e.g. undo.py restoring
    # from snapshot, or overview.py clearing it). Without this, LangGraph keeps
    # whatever active_focus was set by understand_node, ignoring the action's update.
    FOCUS_UPDATING_INTENTS = {"UNDO", "OVERVIEW", "AUTO_DECIDE"}
    if action_result and "active_focus" in action_result:
        if state.get("intent") in FOCUS_UPDATING_INTENTS:
            respond_result["active_focus"] = action_result["active_focus"]
    # For all other intents (DECIDE, ANALYSE, etc.), trust understand_node's value

    # Forward overview_mode if an action set it (overview.py sets this explicitly)
    if action_result and "overview_mode" in action_result:
        respond_result["overview_mode"] = action_result["overview_mode"]

    # ── Issue 5: Track guardrail_pending across turns ─────────
    # guardrail_pending is True for exactly the turn a guardrail fires and
    # no further. It clears unconditionally on the very next turn regardless
    # of what the user says — STATUS, EXPLORE, UNDO, ACKNOWLEDGE, anything.
    # If the user wants to confirm, they must do so immediately on the next
    # turn (classifier detects "confirm" via dry-run context scan).
    action_result_raw = action_result or {}
    if action_result_raw.get("guardrail_triggered"):
        respond_result["guardrail_pending"] = True
    else:
        respond_result["guardrail_pending"] = False

    return respond_result
