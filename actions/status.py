# ============================================================
# actions/status.py
#
# Action: STATUS
#
# Reads the current decisions dict and verdict_df to produce
# a complete snapshot of where the conversation stands.
#
# Public API:
#   run(state: dict) -> dict
#
# Returns partial state with:
#   - action_result : full status snapshot for the formatter
#
# No LLM calls. No state mutations. Pure Python read-only.
# ============================================================


def run(state: dict) -> dict:
    """
    Summarise the current decision state across all columns.

    Parameters
    ----------
    state : dict — full GraphState

    Returns
    -------
    dict — partial state update:
        action_result : {
            total_cols    : int,
            kept          : int,
            dropped       : int,
            pending       : int,
            coverage_pct  : float,    # % of feature_cols with a decision
            kept_list     : list[str],
            dropped_list  : list[str],
            pending_list  : list[str],
            override_count: int,      # decisions that overrode bot verdict
            error         : str       # only present on failure
        }
    """
    session      = state["session"]
    decisions    = state.get("decisions", {})
    decision_log = state.get("decision_log", [])

    feature_cols = session.get("feature_cols", [])
    verdict_df   = session.get("verdict_df")

    if not feature_cols:
        return {
            "action_result": {
                "error": "No feature columns found in the session. Please check that the dataset loaded correctly."
            }
        }

    # ── Partition columns into kept / dropped / pending ───────
    feature_set  = set(feature_cols)
    kept_list    = sorted([col for col, dec in decisions.items() if dec == "keep" and col in feature_set])
    dropped_list = sorted([col for col, dec in decisions.items() if dec == "drop" and col in feature_set])

    # Pending = analysed columns (in verdict_df) not yet decided
    # + any feature column with no decision at all
    decided_set  = set(decisions.keys())
    pending_list = sorted([col for col in feature_cols if col not in decided_set])

    total_cols   = len(feature_cols)
    kept         = len(kept_list)
    dropped      = len(dropped_list)
    pending      = len(pending_list)
    decided      = kept + dropped
    coverage_pct = round((decided / total_cols) * 100, 1) if total_cols > 0 else 0.0

    # ── Count human overrides from the decision log ────────────
    override_count = sum(
        1 for entry in decision_log
        if entry.get("override") is True
    )

    # ── Bot verdict alignment stats ───────────────────────────
    agreement_stats = _compute_agreement(decisions, verdict_df)

    # ── Pending bot recommendations ───────────────────────────
    # Columns the bot has a clear verdict for (KEEP or DROP/DROP-NULL)
    # but the user hasn't confirmed yet. These are NOT the same as
    # "pending" (which just means no decision at all).
    # Surfacing these prevents the "starting over" illusion when the
    # user checks status before running AUTO_DECIDE.
    bot_recs = _compute_bot_recommendations(pending_list, verdict_df)

    action_result = {
        "total_cols"              : total_cols,
        "kept"                    : kept,
        "dropped"                 : dropped,
        "decided"                 : decided,
        "pending"                 : pending,
        "coverage_pct"            : coverage_pct,
        "kept_list"               : kept_list,
        "dropped_list"            : dropped_list,
        "pending_list"            : pending_list,
        "override_count"          : override_count,
        # Bot recommendation breakdown for pending columns
        "bot_pending_keep"        : bot_recs["pending_keep"],
        "bot_pending_drop"        : bot_recs["pending_drop"],
        "bot_pending_flag"        : bot_recs["pending_flag"],
        "bot_pending_keep_list"   : bot_recs["pending_keep_list"],
        "bot_pending_drop_list"   : bot_recs["pending_drop_drop_null_list"],
        # Suggest AUTO_DECIDE when bot has pending KEEP/DROP recs
        # and the user hasn't run it yet (i.e. no bot-sourced entries
        # in decision_log). Coverage threshold removed — the illusion
        # can happen at any coverage level before AUTO_DECIDE is run.
        "suggest_auto_decide"     : (
            (bot_recs["pending_keep"] + bot_recs["pending_drop"]) > 0 and
            not any(e.get("source") == "bot" for e in decision_log)
        ),
        **agreement_stats,
    }

    return {"action_result": action_result}


# ── Helpers ───────────────────────────────────────────────────

def _compute_bot_recommendations(pending_list: list, verdict_df) -> dict:
    """
    For columns the user hasn't decided yet, bucket them by what the
    bot's own verdict_df says about them.

    Returns:
        pending_keep               : int   — bot says KEEP, user hasn't confirmed
        pending_drop               : int   — bot says DROP/DROP-NULL, user hasn't confirmed
        pending_flag               : int   — bot says FLAG (needs human review)
        pending_keep_list          : list[str]
        pending_drop_drop_null_list: list[str]
    """
    empty = {
        "pending_keep"               : 0,
        "pending_drop"               : 0,
        "pending_flag"               : 0,
        "pending_keep_list"          : [],
        "pending_drop_drop_null_list": [],
    }

    if verdict_df is None or not pending_list:
        return empty

    try:
        if "column" in verdict_df.columns:
            verdict_map = dict(
                zip(verdict_df["column"], verdict_df["verdict"].str.upper())
            )
        else:
            verdict_map = verdict_df["verdict"].str.upper().to_dict()

        keep_list      = []
        drop_null_list = []
        flag_count     = 0

        for col in pending_list:
            v = verdict_map.get(col, "")
            if v == "KEEP":
                keep_list.append(col)
            elif v in ("DROP", "DROP-NULL"):
                drop_null_list.append(col)
            elif v == "FLAG":
                flag_count += 1

        return {
            "pending_keep"               : len(keep_list),
            "pending_drop"               : len(drop_null_list),
            "pending_flag"               : flag_count,
            "pending_keep_list"          : keep_list,
            "pending_drop_drop_null_list": drop_null_list,
        }

    except Exception:
        return empty


def _compute_agreement(decisions: dict, verdict_df) -> dict:
    """
    Compare user decisions against the bot's own verdict_df recommendations.

    Returns a dict with:
        agreed   : int  — decisions that match bot verdict
        disagreed: int  — decisions that differ (human overrides)

    Returns empty dict if verdict_df is unavailable.
    """
    if verdict_df is None or decisions is None:
        return {}

    try:
        # Normalise verdict_df: try "column" as a column or as the index
        if "column" in verdict_df.columns:
            verdict_map = dict(
                zip(verdict_df["column"], verdict_df["verdict"].str.upper())
            )
        else:
            verdict_map = verdict_df["verdict"].str.upper().to_dict()

        agreed    = 0
        disagreed = 0

        for col, user_decision in decisions.items():
            bot_verdict = verdict_map.get(col, "").upper()
            if not bot_verdict:
                continue

            # Bot says KEEP → user says keep  ✓
            # Bot says DROP/FLAG → user says drop  ✓
            user_keep = user_decision == "keep"
            bot_keep  = bot_verdict == "KEEP"

            if user_keep == bot_keep:
                agreed += 1
            else:
                disagreed += 1

        return {"agreed": agreed, "disagreed": disagreed}

    except Exception:
        return {}   # best-effort — never crash status on a stats failure