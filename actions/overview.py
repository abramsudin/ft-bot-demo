# ============================================================
# actions/overview.py
#
# Action: OVERVIEW
#
# Full dataset-level summary. Returns structured stats and
# recommendations for the formatter to narrate.
#
# v7 changes (preserved):
#   - D2: Fresh overview offers two explicit paths:
#         (1) bot-led draft mode via AUTO_DECIDE
#         (2) user-led independent work
#
# Visualisation removed (v8): chart generation, PNG saving,
# chart_b64, chart_filename all removed. The formatter narrates
# the numbers directly.
#
# No LLM calls. Pure Python.
# ============================================================

from datetime import datetime, timezone


def run(state: dict) -> dict:
    session          = state["session"]
    verdict_df       = session.get("verdict_df")
    pairs_df         = session.get("pairs_df")
    null_group_map   = session.get("null_group_map", {})
    null_scan_df     = session.get("null_scan_df")
    redundancy_drop  = session.get("redundancy_drop", [])
    feature_cols     = session.get("feature_cols", [])
    num_cols         = session.get("num_cols", [])
    cat_cols         = session.get("cat_cols", [])
    decisions        = state.get("decisions", {})

    if verdict_df is None:
        return {"action_result": {
            "error": "Scan results not available. The pipeline may not have completed."
        }}

    # ── Detect post-tweak mode ────────────────────────────────
    tweaked_decisions = []
    decision_log      = state.get("decision_log", [])

    bot_verdict_map = {}
    if "column" in verdict_df.columns:
        for _, row in verdict_df.iterrows():
            v = str(row["verdict"]).upper()
            bot_verdict_map[row["column"]] = "keep" if v == "KEEP" else "drop"

    for entry in decision_log:
        if entry.get("source") == "user":
            col      = entry.get("col", "")
            user_dec = entry.get("decision", "")
            bot_rec  = bot_verdict_map.get(col, "")
            if bot_rec and user_dec != bot_rec:
                tweaked_decisions.append({
                    "column"       : col,
                    "user_decision": user_dec,
                    "bot_verdict"  : bot_rec,
                })

    overview_mode = "post_tweak" if tweaked_decisions or decisions else "fresh"

    # ── Verdict counts ────────────────────────────────────────
    keep_df  = verdict_df[verdict_df["verdict"] == "KEEP"]
    flag_df  = verdict_df[verdict_df["verdict"] == "FLAG"]
    drop_df  = verdict_df[verdict_df["verdict"] == "DROP"]
    dropn_df = verdict_df[verdict_df["verdict"] == "DROP-NULL"]

    n_keep       = len(keep_df)
    n_flag       = len(flag_df)
    n_drop       = len(drop_df)
    n_dropn      = len(dropn_df)
    n_drop_total = n_drop + n_dropn
    total        = len(verdict_df)

    decided_keep = sum(1 for v in decisions.values() if v == "keep")
    decided_drop = sum(1 for v in decisions.values() if v == "drop")
    pending      = total - len(decisions)

    null_ind_cols = []
    if "null_gap" in verdict_df.columns:
        null_ind_cols = verdict_df[
            verdict_df["null_gap"].notna() & (verdict_df["null_gap"].abs() > 3)
        ]["column"].tolist()

    conf_strong   = int((keep_df["confidence"] >= 70).sum())  if "confidence" in keep_df.columns else 0
    conf_moderate = int(((keep_df["confidence"] >= 45) & (keep_df["confidence"] < 70)).sum()) if "confidence" in keep_df.columns else 0
    conf_border   = int((keep_df["confidence"] < 45).sum())   if "confidence" in keep_df.columns else 0

    risk_counts = {}
    if "risk_tag" in verdict_df.columns:
        kf = verdict_df[verdict_df["verdict"].isin(["KEEP", "FLAG"])]
        risk_counts = kf["risk_tag"].value_counts().to_dict()

    top_keeps = []
    if "confidence" in keep_df.columns:
        top_k = keep_df.sort_values("confidence", ascending=False).head(10)
        for _, r in top_k.iterrows():
            col     = r["column"]
            top_keeps.append({
                "column"          : col,
                "confidence"      : int(r.get("confidence", 0)),
                "signals"         : str(r.get("signals", "—")),
                "null_rate"       : round(float(r.get("null_rate", 0)), 1),
                "risk_tag"        : str(r.get("risk_tag", "")),
                "profile"         : str(r.get("profile", "")),
                "current_decision": decisions.get(col, "pending"),
            })

    flag_summary = []
    for _, r in flag_df.iterrows():
        col = r["column"]
        flag_summary.append({
            "column"          : col,
            "confidence"      : int(r.get("confidence", 0)) if "confidence" in flag_df.columns else None,
            "signals"         : str(r.get("signals", "—")),
            "null_rate"       : round(float(r.get("null_rate", 0)), 1),
            "risk_tag"        : str(r.get("risk_tag", "")) if "risk_tag" in flag_df.columns else "",
            "profile"         : str(r.get("profile", "")) if "profile" in flag_df.columns else "",
            "current_decision": decisions.get(col, "pending"),
        })

    redundancy_pairs = []
    if pairs_df is not None and not pairs_df.empty:
        for _, r in pairs_df.iterrows():
            redundancy_pairs.append({
                "col_1"           : r.get("col_1", ""),
                "col_2"           : r.get("col_2", ""),
                "metric"          : str(r.get("metric", "")),
                "value"           : round(float(r.get("value", 0)), 4),
                "recommended_drop": r.get("recommended_drop", ""),
            })

    n_null_groups  = len(set(null_group_map.values())) if null_group_map else 0
    cols_in_groups = len(null_group_map)

    target_series = session.get("target")
    overall_churn_rate = None
    if target_series is not None:
        try:
            overall_churn_rate = round(float(target_series.mean() * 100), 2)
        except Exception:
            pass

    # ── D2: Two-path next step ────────────────────────────────
    if overview_mode == "fresh":
        next_step = (
            "How would you like to proceed? You can: "
            "(1) say 'load your recommendations' to use my verdicts as a starting draft "
            "that you can then edit freely before exporting, or "
            "(2) work independently from scratch — analyse columns, set your own rules, "
            "and build your own keep/drop list without following my recommendations."
        )
        flow_options = {
            "bot_led" : "load your recommendations as a draft (say 'load your recommendations' or 'accept your recommendations')",
            "user_led": "work independently from scratch (just start analysing or deciding columns directly)",
        }
    else:
        n_pending_flags = sum(1 for r in flag_summary if r["current_decision"] == "pending")
        if n_pending_flags > 0:
            next_step = (
                f"There are still {n_pending_flags} flagged columns awaiting your review. "
                f"Want to go through them, make more tweaks, run conditional rules, or export the report now?"
            )
        else:
            next_step = (
                "All columns have been reviewed. "
                "Want to export the report, or would you like to make any more changes?"
            )
        flow_options = {}

    return {
        "action_result": {
            "overview_mode"     : overview_mode,
            "total_columns"     : total,
            "n_numeric"         : len(num_cols),
            "n_categorical"     : len(cat_cols),
            "overall_churn_rate": overall_churn_rate,

            "verdict_summary": {
                "KEEP"      : n_keep,
                "FLAG"      : n_flag,
                "DROP"      : n_drop,
                "DROP-NULL" : n_dropn,
                "drop_total": n_drop_total,
            },

            "live_decisions": {
                "decided_keep": decided_keep,
                "decided_drop": decided_drop,
                "pending"     : pending,
            },

            "tweaked_decisions"  : tweaked_decisions,
            "n_tweaks"           : len(tweaked_decisions),

            "confidence_breakdown": {
                "strong"    : conf_strong,
                "moderate"  : conf_moderate,
                "borderline": conf_border,
            },

            "risk_tag_breakdown"   : risk_counts,
            "null_indicator_cols"  : null_ind_cols,
            "n_null_indicator_cols": len(null_ind_cols),
            "n_null_groups"        : n_null_groups,
            "cols_in_groups"       : cols_in_groups,
            "redundancy_pairs"     : redundancy_pairs,
            "redundancy_drop"      : redundancy_drop,
            "top_keep_columns"     : top_keeps,
            "flag_columns"         : flag_summary,

            "bot_recommendation": (
                f"Keep {n_keep} columns, drop {n_drop_total} "
                f"({n_drop} no-signal + {n_dropn} fully null), "
                f"and review {n_flag} borderline columns with you. "
                f"{next_step}"
            ),
            "next_step"   : next_step,
            "flow_options": flow_options,
        },
        "active_focus" : None,
        "overview_mode": overview_mode,
    }