# ============================================================
# actions/acknowledge.py
#
# Intent: ACKNOWLEDGE
#
# Handles conversational dismissals and stand-by phrases:
#   "no need", "never mind", "nvm", "skip it", "forget it",
#   "don't do it", "actually no", "ok thanks", "got it",
#   "alright", "noted", "cool", "stop"
#
# This is a read-only, no-op action. It performs no writes,
# pushes no undo snapshots, and appends nothing to decision_log.
#
# The formatter receives {"acknowledged": True} and outputs
# a single short acknowledgement sentence with no follow-up
# question. active_focus is preserved (focus_clear=False in
# the classifier) so the next substantive turn can still
# resolve pronouns against the same column context.
# ============================================================


def run(state: dict) -> dict:
    """
    No-op action for conversational dismissals.

    Returns a minimal action_result so the formatter knows
    this was an acknowledgement turn and not an error.
    Does NOT return decisions, undo_stack, or decision_log —
    this is a read-only action and those fields are passed
    through by respond_node automatically (BUG-1 pattern).
    """
    return {
        "acknowledged": True,
    }
