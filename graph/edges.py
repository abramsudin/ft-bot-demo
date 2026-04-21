# ============================================================
# graph/edges.py
#
# Intent → action function routing table.
#
# v7 changes:
#   - N4: Added CONDITIONAL_DECIDE intent → conditional_decide.run
#   - EDA merged into ANALYSE (deep_dive=True param)
#   - Total intents: 12
#
# Current intents (12 total):
#   ANALYSE            — statistical deep-dive on one or more columns / zone
#   EXPLORE            — filter/browse column list
#   DECIDE             — keep/drop a column or bulk zone
#   UNDO               — reverse last decision
#   STATUS             — how many kept/dropped/pending so far
#   EXPLAIN            — explain a stat concept
#   REPORT             — export Excel report
#   OVERVIEW           — full dataset summary
#   AUTO_DECIDE        — load bot verdicts as draft, leave FLAGs pending
#   COMPARE            — explicit cross-column comparison
#   CONDITIONAL_DECIDE — rule-based bulk decisions with optional dry_run preview
#   AMBIGUOUS          — clarify unclear message
# ============================================================

from actions import (
    analyse,
    explore,
    decide,
    undo,
    status,
    explain,
    report,
    overview,
    auto_decide,
    compare,
    conditional_decide,
    clarify,
    acknowledge,
)

INTENT_MAP: dict = {
    "ANALYSE"           : analyse.run,
    "EXPLORE"           : explore.run,
    "DECIDE"            : decide.run,
    "UNDO"              : undo.run,
    "STATUS"            : status.run,
    "EXPLAIN"           : explain.run,
    "REPORT"            : report.run,
    "OVERVIEW"          : overview.run,
    "AUTO_DECIDE"       : auto_decide.run,
    "COMPARE"           : compare.run,
    "CONDITIONAL_DECIDE": conditional_decide.run,
    "AMBIGUOUS"         : clarify.run,
    "ACKNOWLEDGE"       : acknowledge.run,
}


def route(intent: str):
    """
    Return the action function for the given intent string.
    Falls back to clarify.run for unknown intents.
    """
    return INTENT_MAP.get(intent, clarify.run)
