# ============================================================
# llm/classifier.py
#
# LLM Call #1 — Intent Detection + Pronoun Resolution
#
# Public API:
#   classify(messages, active_focus, session, focus_age) -> dict
#
# Returns:
#   {
#     "intent"        : str,                    # one of 13 valid intent strings
#     "params"        : dict,                   # structured params for that intent
#     "resolved_focus": str | list[str] | None  # column(s) in scope after this turn
#     "focus_clear"   : bool                    # True when classifier signals a clean pivot
#   }
#
# Changes vs previous version:
#   - Added COMPARE to VALID_INTENTS (was missing — caused COMPARE to silently
#     downgrade to AMBIGUOUS)
#   - Updated ANALYSE, EDA, COMPARE params schema to use "columns" (list)
#     instead of "column" (single str) to support multi-column requests
#   - Added multi-column and COMPARE examples to the prompt
#   - Fixed resolved_focus backfill to handle list[str] from "columns" param
#   - AMBIGUOUS turns no longer wipe active_focus (partial context preserved)
#
# Fixes applied (Issues I-1, I-3, I-4):
#   - I-1: Classifier JSON output now includes focus_clear boolean.
#     Prompt instructs model to emit focus_clear=true for clean topic pivots
#     (EXPLAIN, STATUS, OVERVIEW, REPORT, bare UNDO, AUTO_DECIDE).
#   - I-3: focus_age is passed in and injected into the system prompt.
#     When focus_age > 2 and a pronoun is detected, the classifier is
#     instructed to emit AMBIGUOUS with a staleness note rather than
#     silently resolving to a column discussed several turns ago.
#   - I-4: Added "confirm conditional" / "yes, apply it" few-shot examples
#     so the confirmation flow reliably re-routes to CONDITIONAL_DECIDE
#     with dry_run=False rather than going AMBIGUOUS.
# ============================================================

import json
import re
import os
import requests
from dotenv import load_dotenv
load_dotenv()

# ── OpenRouter config ─────────────────────────────────────────────────────────
OPENROUTER_URL    = "https://openrouter.ai/api/v1/chat/completions"
CLASSIFIER_MODEL = "openai/gpt-4o-mini"
CLASSIFIER_WINDOW = 8 # Only last N messages sent to classifier (Issue #11)

# ── Fix #1: COMPARE added to VALID_INTENTS ────────────────────────────────────
VALID_INTENTS = {
    "ANALYSE",
    "EXPLORE",
    "DECIDE",
    "CONDITIONAL_DECIDE",  # ← rule-based bulk decisions (e.g. "drop if null rate > 60%")
    "UNDO",
    "STATUS",
    "EXPLAIN",
    "REPORT",
    "OVERVIEW",
    "AUTO_DECIDE",
    "COMPARE",
    "AMBIGUOUS",
    "EDA",                 # ← kept for regex fallback only; routes via ANALYSE deep_dive
    "ACKNOWLEDGE",
}

_INTENT_PATTERNS = [
    (r"\bANALYSE\b",           "ANALYSE"),
    (r"\bEXPLORE\b",           "EXPLORE"),
    (r"\bCONDITIONAL_DECIDE\b","CONDITIONAL_DECIDE"),
    (r"\bDECIDE\b",            "DECIDE"),
    (r"\bUNDO\b",              "UNDO"),
    (r"\bSTATUS\b",            "STATUS"),
    (r"\bEXPLAIN\b",           "EXPLAIN"),
    (r"\bREPORT\b",            "REPORT"),
    (r"\bOVERVIEW\b",          "OVERVIEW"),
    (r"\bAUTO_DECIDE\b",       "AUTO_DECIDE"),
    (r"\bCOMPARE\b",           "COMPARE"),
    (r"\bAMBIGUOUS\b",         "AMBIGUOUS"),
    (r"\bACKNOWLEDGE\b",       "ACKNOWLEDGE"),
    (r"\bEDA\b",               "EDA"),  # fallback only
]


def _build_system_prompt(
    active_focus         : "str | list[str] | None",
    column_names         : list[str],
    last_3_turns         : list[dict],
    focus_age            : int = 0,
    prior_dry_run_context: str = "",           # BUG 2 FIX: injected guardrail context
    prior_guardrail_context : str = "",           # ADD THIS
) -> str:

    column_list_str = ", ".join(column_names) if column_names else "(none yet)"
    history_lines = []
    for msg in last_3_turns:
        role    = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        history_lines.append(f"  {role}: {content}")
    history_str = "\n".join(history_lines) if history_lines else "  (no history yet)"

    if isinstance(active_focus, list):
        focus_str = str(active_focus)
    elif active_focus:
        focus_str = f'"{active_focus}"'
    else:
        focus_str = "null (no column in scope)"

    # I-3: Build staleness warning block for the prompt
    if active_focus and focus_age > 2:
        staleness_warning = f"""
⚠️ STALE FOCUS WARNING: active_focus has not changed for {focus_age} turns.
If the current message contains a pronoun ("it", "this", "that", "them") and
does NOT contain an explicit column name, you MUST return AMBIGUOUS instead of
silently resolving to the stale focus. Include the stale column as a candidate:
  {{"intent": "AMBIGUOUS", "params": {{"stale_focus_candidate": {focus_str}}}, "resolved_focus": null, "focus_clear": false}}
This prevents the user from accidentally deciding the wrong column.
"""
    else:
        staleness_warning = ""

    # BUG 2 FIX: build dry-run context block for the prompt
    if prior_dry_run_context:
        dry_run_block = (
            f"\n⚠️ PRIOR DRY-RUN / GUARDRAIL CONTEXT FOUND:\n"
            f"{prior_dry_run_context}\n"
            f"If the user says 'confirm', 'yes', 'go ahead', or 'do it', you MUST route to "
            f"CONDITIONAL_DECIDE with dry_run=false and reconstruct the exact same conditions. "
            f"DO NOT route to AUTO_DECIDE.\n"
        )
    else:
        dry_run_block = ""

    return f"""You are the intent classifier for a feature selection assistant.
Your job: read the user's latest message and return exactly one JSON object.
{dry_run_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALID INTENTS (pick exactly one)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OVERVIEW    — User wants the full dataset-level summary: all verdicts, top keeps,
              flagged columns, confidence breakdown, risk tags, visual chart.
              ALSO covers dataset-level stat questions with no specific column named.
              This is bot-led — the bot presents its findings without the user
              going column by column.
              Examples: "do the feature selection", "what do you recommend?",
                        "show me the full picture", "give me the overview",
                        "what's your verdict on everything?",
                        "show me what you found", "summarise the scan results",
                        "what did the scan find?", "show me the dataset analysis",
                        "show me your results", "what should I keep?",
                        "give me a summary of all columns",
                        "how many numeric vs categorical?",
                        "what is the null rate distribution?",
                        "what is the class distribution?",
                        "how many columns have high null rates?",
                        "how many features are in the dataset?",
                        "what is the churn rate overall?",
                        "what is the target distribution?",
                        "how many columns are there?",
                        "how many features do we have?",
                        "what percentage of columns are flagged?"
              Params: {{}}

              ⚠️ This is the DEFAULT when the user asks for a big-picture summary
              without naming a specific column. If no column is mentioned and the
              user wants to see the bot's analysis of all columns → OVERVIEW.
              ⚠️ Dataset-level stat questions (null rate distribution, class distribution,
              numeric vs categorical count, overall churn rate) with NO column name
              → ALWAYS OVERVIEW, NEVER AMBIGUOUS. The overview result contains all
              of this information already.

AUTO_DECIDE — User wants to accept the bot's bulk recommendations and apply them
              all at once. KEEP columns get marked keep, DROP/DROP-NULL get marked
              drop, FLAG columns stay pending for review.
              Examples: "accept your recommendations", "apply your decisions",
                        "auto-decide everything", "accept all bot verdicts",
                        "keep your KEEPs and drop your DROPs",
                        "just apply what you recommend",
                        "confirm auto-decide",
                        "yes go ahead and apply them",
                        "yes, apply all your recommendations"
              Params: {{}}

EXPLORE     — User wants to filter or browse the column list by a property
              (verdict zone, risk tag, null group, confidence band, type,
               pending status). Returns a ranked list of matching columns.
              Examples: "show me flagged columns", "list null-driven columns",
                        "which columns are borderline", "show borderline columns",
                        "null-driven columns", "show me drop-null columns",
                        "list redundant columns", "show numeric columns",
                        "high confidence columns", "low confidence columns",
                        "pending columns", "what's left to decide",
                        "undecided columns", "what still needs a decision",
                        "columns not yet decided", "what's left",
                        "show me the flagged ones", "list everything flagged as null-driven",
                        "borderline columns", "which columns are on the fence"
              Params: {{ "filter": "<natural language filter expression>" }}
              filter_type guidance (for the action, not the JSON):
                "pending"    — undecided / pending / what's left
                "null_driven"— null-driven / null heavy / high null
                "FLAG"       — borderline / flagged / uncertain / on the fence
                "DROP"       — drop / no signal / weak
                "KEEP"       — keep / strong signal
              ⚠️ "show flagged columns" → EXPLORE (browsing list).
                 "analyse flagged columns" → ANALYSE (stats on that zone).
              ⚠️ "pending columns" / "what's left" / "undecided" → ALWAYS EXPLORE
                 with filter="pending". NEVER AMBIGUOUS.
              ⚠️ "null-driven columns" / "borderline columns" → ALWAYS EXPLORE.
              ⚠️ focus_clear=false for EXPLORE (user stays in browse context).

ANALYSE     — User wants to SEE statistical results or get a recommendation for
              ONE OR MORE specific columns, OR for an entire verdict zone,
              OR wants a deep-dive distribution/missingness analysis (deep_dive=True).
              Examples (column): "check Var83", "what does Var22 look like",
                        "is Var22 worth keeping?", "tell me about Var7",
                        "analyse Var22, Var34", "check Var1, Var2, Var3",
                        "analysis on Var2 and Var3", "analyse Var22 and Var34",
                        "tell me about Var1 and Var2"
              Examples (zone): "analyse flagged columns", "analyse keep columns",
                        "analyse drop columns", "what do the flagged columns look like",
                        "tell me about the KEEP zone", "analyse the FLAG group",
                        "give me a summary of flagged ones", "check the drop columns"
              Examples (deep_dive): "deep dive Var22", "eda on Var10",
                        "explore missingness for Var83", "show null distribution for Var1",
                        "show distribution of Var5", "show class imbalance",
                        "full distribution of Var22", "churn split for Var10",
                        "show churn split", "full breakdown of Var5",
                        "what's the distribution of Var22"
              Params (column): {{ "columns": ["<column_name>", ...], "zone": null, "deep_dive": false }}
              Params (zone):   {{ "columns": [], "zone": "FLAG" | "KEEP" | "DROP", "deep_dive": false }}
              Params (deep_dive): {{ "columns": ["<column_name>", ...], "zone": null, "deep_dive": true,
                                    "visual_type": "default" | "null_distribution" | "churn_rate" |
                                                   "distribution" | "null_gap" | "class_imbalance" }}
              ⚠️ Always use "columns" as a list, even for a single column.
              ⚠️ "and" between column names means MULTIPLE columns — extract ALL of them.
              ⚠️ When the user says "flagged/flag columns", "keep columns", "drop columns"
                 WITHOUT naming a specific column → set zone to "FLAG"/"KEEP"/"DROP"
                 and columns to []. Do NOT route to EXPLORE for these — ANALYSE is correct.
              ⚠️ CRITICAL: Questions ABOUT a column/zone = ANALYSE. Commands = DECIDE.
              ⚠️ deep_dive=True for: "deep dive", "deep-dive", "eda on", "eda",
                 "explore missingness", "show distribution", "show null distribution",
                 "show class imbalance", "full distribution", "full breakdown",
                 "churn split", "churn rate breakdown", "distribution of",
                 "distribution breakdown", "show me the distribution",
                 "null breakdown", "missingness breakdown", "show nulls",
                 "full analysis", "detailed breakdown".
              ⚠️ Any phrase containing "distribution", "churn split", or "full breakdown"
                 → deep_dive=true, even without "deep dive" verbatim.
              ⚠️ ALWAYS emit deep_dive as a JSON boolean true — NOT the string "true".
              ⚠️ FOLLOW-UP VISUAL REQUESTS (deep_dive=True): If user names a visual type
                 without a column, resolve columns from active_focus. Map:
                 "churn rate" → "churn_rate", "distribution" → "distribution",
                 "null gap" → "null_gap". NEVER use visual_type="default" for these.
              ⚠️ ALWAYS emit "columns" as a JSON array even for a single column:
                 correct: "columns": ["Var10"]   wrong: "column": "Var10"

CONDITIONAL_DECIDE — User wants to apply keep/drop decisions based on a condition
              evaluated against scan data (null rate, confidence, etc.).
              Supports single, compound AND/OR conditions, zone scoping, and dry run.
              Examples: "drop columns where null rate is above 60%",
                        "drop flagged columns if null rate > 60%",
                        "keep columns with confidence above 70",
                        "drop if null rate high and signal low",
                        "what would happen if I dropped columns with null rate > 50%",
                        "drop columns with confidence below 45",
                        "keep columns where p-value is significant",
                        "drop all flagged columns where confidence is low",
                        "drop if churn gap is below 1"
              Params: {{
                "decision"       : "keep" | "drop",
                "conditions"     : [
                  {{"field": "null_rate"|"confidence"|"mi_score"|"p_value"|"churn_gap"|"risk_tag"|"verdict",
                    "operator": ">"|"<"|">="|"<="|"=="|"contains",
                    "threshold": <value>}}
                ],
                "condition_logic": "AND" | "OR",
                "scope"          : "FLAG"|"KEEP"|"DROP"|null,
                "dry_run"        : true | false
              }}
              ⚠️ "null rate is high" → field="null_rate", operator=">", threshold=0.60
              ⚠️ "confidence is low" → field="confidence", operator="<", threshold=45
              ⚠️ "signal is low" → field="confidence", operator="<", threshold=45
              ⚠️ "confidence is good/strong" → field="confidence", operator=">=", threshold=70
              ⚠️ "p-value is significant" → field="p_value", operator="<", threshold=0.05
              ⚠️ "churn gap is strong" → field="churn_gap", operator=">", threshold=3.0
              ⚠️ "and" between conditions → condition_logic="AND"
              ⚠️ "or" between conditions → condition_logic="OR"
              ⚠️ "flagged columns if ..." → scope="FLAG" plus the condition(s)
              ⚠️ "what would happen if I dropped/kept ..." → dry_run=true
              ⚠️ CRITICAL: "drop all FLAG columns" (no condition) → DECIDE.
                 "drop columns where null rate > 60%" (has condition) → CONDITIONAL_DECIDE.

COMPARE     — User explicitly wants to compare the SAME metric across two or more
              columns side-by-side. Separate from ANALYSE — comparison implies
              shared axis, same scale, cross-column insight.
              Examples: "compare null distribution of Var22, Var34",
                        "compare churn rate of Var1, Var2",
                        "compare Var5 and Var9",
                        "compare Var22 and Var34",
                        "which of Var22, Var34 has more nulls?"
              Params: {{
                "columns"     : ["<column_name>", ...],
                "visual_type" : "default" | "null_distribution" | "churn_rate" |
                                "distribution" | "null_gap"
              }}
              ⚠️ Requires at least 2 columns. If only 1 column mentioned → ANALYSE.
              ⚠️ "and" between column names means multiple columns — extract ALL of them.
              ⚠️ If no specific visual_type is mentioned, use "default".

DECIDE      — User is EXPLICITLY making a keep/drop decision (imperative statement).
              Examples: "keep Var83", "drop it", "drop all FLAG columns"
              Zone examples: "drop all FLAG columns", "drop all KEEP and FLAG columns",
                        "keep all DROP columns", "keep all flagged columns",
                        "drop everything flagged", "keep the drop columns"
              Params: {{ "column": "<column_name or null>", "zone": "<zone or null>",
                         "zones": ["<zone1>", "<zone2>"] or null,
                         "decision": "keep" | "drop" }}

              ⚠️ CRITICAL: "is X worth keeping?" = ANALYSE. "keep X" = DECIDE.
              ⚠️ When user says "keep all DROP columns" → decision="keep", zone="DROP".
              ⚠️ When user says "drop all FLAG and KEEP columns" → decision="drop",
                 zones=["FLAG","KEEP"] (use zones list for multiple zones).
              ⚠️ Zone name is the verdict label the columns currently have, NOT the action.
                 "keep all DROP columns" means: take columns in the DROP zone → mark keep.
              ⚠️ CRITICAL: Any phrase mentioning a column PROPERTY as the reason to drop
                 ("with null values", "with no signal", "with low confidence", "with high nulls")
                 → CONDITIONAL_DECIDE, NOT DECIDE. Only pure zone/column commands → DECIDE.

UNDO        — User wants to reverse a previous decision.
              Examples: "undo that", "revert Var83", "go back",
                        "undo the last 3 decisions", "revert the last 2 changes",
                        "undo 5 steps", "go back 3 steps",
                        "undo back to auto-decide", "revert to before auto-decide"
              Params: {{
                "column"       : "<column_name or null>",
                "steps"        : <integer N for multi-step undo, or null for single step>,
                "target_action": "<snapshot label for time-travel e.g. 'auto_decide', or null>"
              }}
              ⚠️ "undo the last 3 decisions" → steps=3, column=null, target_action=null
              ⚠️ "revert the last N changes" → steps=N
              ⚠️ "undo back to auto-decide" → target_action="auto_decide", steps=null
              ⚠️ "undo that" / "go back" / bare undo → steps=null (single step, default)

STATUS      — User wants a summary of decisions made so far.
              Examples: "what have we decided", "show summary", "how many kept"
              Params: {{}}

EXPLAIN     — User wants a concept explained. Supports single or multiple concepts.
              Examples: "what is mutual information", "explain Cramer's V",
                        "what's the difference between MI and Cramer's V",
                        "explain mutual information and chi-square",
                        "compare MI vs Cramer's V"
              Single concept: {{ "concept": "<concept_name>" }}
              Multi concept:  {{ "concepts": ["<concept_1>", "<concept_2>", ...] }}
              ⚠️ If the user names TWO OR MORE concepts in one message, always emit
                 "concepts" as a list — never just the last one in "concept".
              ⚠️ "What risk tag types are there?" / "list all risk tags" / "what are the
                 possible risk tags?" → EXPLAIN with concept="risk_tag". These are
                 metadata questions about the taxonomy, not dataset-level stats.
                 Do NOT route to OVERVIEW or STATUS for these.

REPORT      — User wants to export/generate the final Excel report.
              Examples: "export report", "I'm done", "generate the output"
              Params: {{}}

ACKNOWLEDGE — User is dismissing, declining, or standing by. No action needed.
              Examples: "no need", "never mind", "nvm", "skip it", "forget it",
                        "don't do it", "actually no", "ok thanks", "got it",
                        "alright", "noted", "cool", "stop", "no don't"
              ⚠️ CRITICAL: If the message contains "undo", "revert", or "go back" —
                 even prefixed with "actually no" or "no wait" — route to UNDO, NOT
                 ACKNOWLEDGE. Example: "actually no, undo that" → UNDO.
              Params: {{}}
              ⚠️ This is NOT the same as AMBIGUOUS. The user's meaning is clear —
                 they are closing the topic or declining a suggestion. Do NOT ask
                 a clarifying question. Do NOT route to AMBIGUOUS.
              ⚠️ focus_clear=False — the topic has not changed, just the action
                 was declined. active_focus should be preserved.

AMBIGUOUS   — Message is unclear, multi-intent, or has an unresolvable pronoun,
              OR contains gibberish/unrecognisable tokens with no clear intent.
              Params: {{ "ambiguity_type": "generic" | "ambiguous_column" | "pronoun_unclear" | "intent_unclear" | "no_column" }}
              ⚠️ ALWAYS use AMBIGUOUS (never OVERVIEW) when the message contains
                 gibberish tokens or is a vague continuation like "what are my options?",
                 "what else?", "what now?" with no active_focus.
              ⚠️ Partial column names (e.g. "keep Var", "drop Var1") that match
                 multiple columns → AMBIGUOUS with ambiguity_type="ambiguous_column".
              ⚠️ Message mentions BOTH "drop" and "keep" as possible actions →
                 AMBIGUOUS with ambiguity_type="intent_unclear".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY ROUTING DECISIONS — READ CAREFULLY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OVERVIEW vs ANALYSE vs EXPLORE:
  "what do you recommend?"         → OVERVIEW  (no specific column, wants full picture)
  "show me the results"            → OVERVIEW  (full dataset view)
  "what should I keep?"            → OVERVIEW  (asking for bot's full recommendation)
  "how many numeric vs categorical?"→ OVERVIEW  (dataset-level stat, no column named)
  "what is the null rate distribution?" → OVERVIEW (dataset-level stat)
  "what is the class distribution?" → OVERVIEW (dataset-level stat)
  "how many columns have high nulls?" → OVERVIEW (dataset-level stat)
  "how many features do we have?"   → OVERVIEW  (dataset-level stat)
  "show me flagged columns"        → EXPLORE   (filtering column list by property)
  "is Var22 worth keeping?"        → ANALYSE   (specific column)
  "what does Var83 look like?"     → ANALYSE   (specific column)

  ⚠️ DATASET-LEVEL STAT RULE: If the user asks a statistical question about the
  WHOLE DATASET (distributions, counts, breakdowns) WITHOUT naming a specific column
  → OVERVIEW. NEVER return AMBIGUOUS for these. Examples:
    "how many numeric columns are there?" → OVERVIEW
    "what is the overall churn rate?"     → OVERVIEW
    "how is the null rate distributed?"   → OVERVIEW
    "what percentage are flagged?"        → OVERVIEW

ANALYSE vs COMPARE:
  "analyse Var22, Var34"           → ANALYSE   (inspect both, separate outputs)
  "compare Var22, Var34"           → COMPARE   (side-by-side on same scale)
  "which of Var22, Var34 is better?" → COMPARE (cross-column judgment)

ANALYSE vs CONDITIONAL_DECIDE vs DECIDE:
  "is Var22 worth keeping?"        → ANALYSE          (asking for opinion)
  "drop all FLAG columns"          → DECIDE            (zone-based, no condition)
  "drop flagged columns if null rate > 60%"
                                   → CONDITIONAL_DECIDE (zone + condition)
  "drop columns where null rate > 60%"
                                   → CONDITIONAL_DECIDE (condition, no zone)
CONFIRM / YES / GO AHEAD (CRITICAL — check conversation history FIRST):
  Bare confirmation phrases: "confirm", "yes", "go ahead", "do it", "apply it", "yes apply it"
    STEP 1 — Scan the RECENT CONVERSATION HISTORY above (last 3 turns).
    STEP 2 — Look for a PREVIOUS ASSISTANT message that describes a dry-run preview
             OR a guardrail block (which functions as an implicit dry-run).
             Dry-run markers: "would affect", "would change", "matched", "dry run",
             "preview", "columns would be dropped", "columns would be kept",
             "guardrail", "above the", "% guardrail", "would mark", "would drop",
             "would flag", "blocked", "exceeds the".
    STEP 3 — Branch:
      • Dry-run preview IS found in history:
          → CONDITIONAL_DECIDE with dry_run=False.
             Reconstruct the SAME conditions, decision, scope, condition_logic
             from that prior dry-run. Do NOT invent new conditions.
      • NO dry-run preview found in history:
          → AUTO_DECIDE.
  ⚠️ "confirm auto-decide" ALWAYS → AUTO_DECIDE (explicit label overrides).
  ⚠️ Bare "confirm"/"yes"/"go ahead" AFTER a dry-run preview → CONDITIONAL_DECIDE.
  ⚠️ Bare "confirm"/"yes" with NO prior dry-run → AUTO_DECIDE.

AUTO_DECIDE vs DECIDE:
  "accept your recommendations"    → AUTO_DECIDE  (bulk, bot-led)
  "apply everything you suggest"   → AUTO_DECIDE  (bulk)
  "keep Var22"                     → DECIDE       (single column, explicit)
  "drop all flagged columns"       → DECIDE       (zone-level, explicit)

ANALYSE vs DECIDE:
  "is X worth keeping?"            → ANALYSE  (asking for opinion)
  "should I keep X?"               → ANALYSE  (asking for opinion)
  "keep X"                         → DECIDE   (commanding)
  "drop it"                        → DECIDE   (commanding)
  When in doubt + question mark    → ANALYSE

ANALYSE deep_dive follow-up resolution:
  Prior turn showed deep dive for Var10; user says "churn rate breakdown"
  → ANALYSE, columns=["Var10"] (from active_focus), deep_dive=true, visual_type="churn_rate"
  Prior turn showed deep dive for Var10; user says "show the distribution"
  → ANALYSE, columns=["Var10"] (from active_focus), deep_dive=true, visual_type="distribution"
  Prior turn showed deep dive for Var10; user says "null gap"
  → ANALYSE, columns=["Var10"] (from active_focus), deep_dive=true, visual_type="null_gap"
  ⚠️ NEVER return visual_type="default" for a follow-up that names a visual type.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRONOUN RESOLUTION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Current active_focus: {focus_str}
Focus age (turns since focus last changed): {focus_age}

- "it", "this", "that", "that one", "this column" → resolve to active_focus (single)
- "them", "they", "those", "them all"             → resolve to active_focus (list)
- If active_focus is null and pronoun is unresolvable → AMBIGUOUS
- Explicit column name(s) stated → use those, set resolved_focus to them
- Zone/tag-level DECIDE or OVERVIEW/AUTO_DECIDE → set resolved_focus to null

{staleness_warning}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KNOWN COLUMN NAMES (validate column references against this list)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{column_list_str}

If the user mentions a column name not in this list, return AMBIGUOUS.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECENT CONVERSATION HISTORY (last 3 turns)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{history_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — STRICT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return ONLY a JSON object. No preamble. No explanation. No markdown.

Schema:
{{
  "intent"        : "<one of the 13 valid intent strings>",
  "params"        : {{ ... }},
  "resolved_focus": "<column_name>" | ["<col1>", "<col2>", ...] | null,
  "focus_clear"   : true | false
}}

focus_clear rules (I-1 fix):
  Set focus_clear=true when the message is a clean topic pivot away from column context.
  ALWAYS true for: EXPLAIN, STATUS, OVERVIEW, REPORT, AUTO_DECIDE, and bare UNDO.
  ALWAYS false for: ANALYSE, DECIDE, CONDITIONAL_DECIDE, COMPARE, EDA, AMBIGUOUS, EXPLORE.

Examples:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL OVERRIDE FOR VAGUE FOLLOW-UPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When the user sends a vague continuation like:
  "what else can you show me?", "show me more", "what's next?",
  "what else is there?", "anything else?", "keep going"

YOU MUST follow this two-step rule — DO NOT default to OVERVIEW:

  STEP 1 — Check active_focus (shown at the top of this prompt).
  STEP 2 — Branch:
    • active_focus IS NOT EMPTY (user is mid-column review)
      → Return EDA with visual_type="default" and columns from active_focus.
         NEVER return OVERVIEW in this case, even though the phrase
         "what else can you show me?" sounds like a broad summary request.
    • active_focus IS EMPTY / null
      → Return OVERVIEW (user has no column context, broad summary is correct).

User: "what else can you show me?"  (active_focus = "Var10")
→ {{"intent": "ANALYSE", "params": {{"columns": ["Var10"], "deep_dive": true, "visual_type": "default"}}, "resolved_focus": "Var10", "focus_clear": false}}

User: "what else can you show me?"  (active_focus = null)
→ {{"intent": "OVERVIEW", "params": {{}}, "resolved_focus": null, "focus_clear": true}}

User: "show me more"  (active_focus = ["Var22", "Var34"])
→ {{"intent": "ANALYSE", "params": {{"columns": ["Var22", "Var34"], "deep_dive": true, "visual_type": "default"}}, "resolved_focus": ["Var22", "Var34"], "focus_clear": false}}

User: "do the feature selection"
→ {{"intent": "OVERVIEW", "params": {{}}, "resolved_focus": null, "focus_clear": true}}

User: "what do you recommend?"
→ {{"intent": "OVERVIEW", "params": {{}}, "resolved_focus": null, "focus_clear": true}}

User: "accept your recommendations"
→ {{"intent": "AUTO_DECIDE", "params": {{}}, "resolved_focus": null, "focus_clear": true}}

User: "analyse Var83"
→ {{"intent": "ANALYSE", "params": {{"columns": ["Var83"]}}, "resolved_focus": "Var83", "focus_clear": false}}

User: "full distribution of Var22"
→ {{"intent": "ANALYSE", "params": {{"columns": ["Var22"], "deep_dive": true, "visual_type": "distribution"}}, "resolved_focus": "Var22", "focus_clear": false}}

User: "churn split for Var10"
→ {{"intent": "ANALYSE", "params": {{"columns": ["Var10"], "deep_dive": true, "visual_type": "churn_rate"}}, "resolved_focus": "Var10", "focus_clear": false}}

User: "show distribution Var5"
→ {{"intent": "ANALYSE", "params": {{"columns": ["Var5"], "deep_dive": true, "visual_type": "distribution"}}, "resolved_focus": "Var5", "focus_clear": false}}

User: "churn split"  (active_focus = "Var10")
→ {{"intent": "ANALYSE", "params": {{"columns": ["Var10"], "deep_dive": true, "visual_type": "churn_rate"}}, "resolved_focus": "Var10", "focus_clear": false}}

User: "distribution breakdown"  (active_focus = "Var22")
→ {{"intent": "ANALYSE", "params": {{"columns": ["Var22"], "deep_dive": true, "visual_type": "distribution"}}, "resolved_focus": "Var22", "focus_clear": false}}

User: "show me the distribution of Var5"
→ {{"intent": "ANALYSE", "params": {{"columns": ["Var5"], "deep_dive": true, "visual_type": "distribution"}}, "resolved_focus": "Var5", "focus_clear": false}}

User: "full distribution"  (active_focus = "Var10")
→ {{"intent": "ANALYSE", "params": {{"columns": ["Var10"], "deep_dive": true, "visual_type": "distribution"}}, "resolved_focus": "Var10", "focus_clear": false}}

User: "show churn split"  (active_focus = "Var83")
→ {{"intent": "ANALYSE", "params": {{"columns": ["Var83"], "deep_dive": true, "visual_type": "churn_rate"}}, "resolved_focus": "Var83", "focus_clear": false}}

User: "analyse Var22, Var34"
→ {{"intent": "ANALYSE", "params": {{"columns": ["Var22", "Var34"]}}, "resolved_focus": ["Var22", "Var34"], "focus_clear": false}}

User: "analysis on Var2 and Var3"
→ {{"intent": "ANALYSE", "params": {{"columns": ["Var2", "Var3"]}}, "resolved_focus": ["Var2", "Var3"], "focus_clear": false}}

User: "analyse Var22 and Var34"
→ {{"intent": "ANALYSE", "params": {{"columns": ["Var22", "Var34"]}}, "resolved_focus": ["Var22", "Var34"], "focus_clear": false}}

User: "check nulls in Var1, Var2, Var3"
→ {{"intent": "ANALYSE", "params": {{"columns": ["Var1", "Var2", "Var3"], "deep_dive": true, "visual_type": "null_distribution"}}, "resolved_focus": ["Var1", "Var2", "Var3"], "focus_clear": false}}

User: "eda on Var10"
→ {{"intent": "ANALYSE", "params": {{"columns": ["Var10"], "deep_dive": true, "visual_type": "default"}}, "resolved_focus": "Var10", "focus_clear": false}}

User: "deep dive Var22"
→ {{"intent": "ANALYSE", "params": {{"columns": ["Var22"], "deep_dive": true, "visual_type": "default"}}, "resolved_focus": "Var22", "focus_clear": false}}

User: "deep dive on Var10"
→ {{"intent": "ANALYSE", "params": {{"columns": ["Var10"], "deep_dive": true, "visual_type": "default"}}, "resolved_focus": "Var10", "focus_clear": false}}
 
User: "explore missingness for Var83"
→ {{"intent": "ANALYSE", "params": {{"columns": ["Var83"], "deep_dive": true, "visual_type": "null_distribution"}}, "resolved_focus": "Var83", "focus_clear": false}}

User: "show class imbalance"
→ {{"intent": "ANALYSE", "params": {{"columns": [], "deep_dive": true, "visual_type": "class_imbalance"}}, "resolved_focus": null, "focus_clear": true}}

User: "churn rate breakdown"  (active_focus = "Var10")
→ {{"intent": "ANALYSE", "params": {{"columns": [], "deep_dive": true, "visual_type": "churn_rate"}}, "resolved_focus": "Var10", "focus_clear": false}}

User: "show the distribution"  (active_focus = "Var10")
→ {{"intent": "ANALYSE", "params": {{"columns": [], "deep_dive": true, "visual_type": "distribution"}}, "resolved_focus": "Var10", "focus_clear": false}}

User: "null gap"  (active_focus = "Var10")
→ {{"intent": "ANALYSE", "params": {{"columns": [], "deep_dive": true, "visual_type": "null_gap"}}, "resolved_focus": "Var10", "focus_clear": false}}

User: "now show churn rate"  (active_focus = "Var10")
→ {{"intent": "ANALYSE", "params": {{"columns": [], "deep_dive": true, "visual_type": "churn_rate"}}, "resolved_focus": "Var10", "focus_clear": false}}

User: "drop columns where null rate is above 60%"
→ {{"intent": "CONDITIONAL_DECIDE", "params": {{"decision": "drop", "conditions": [{{"field": "null_rate", "operator": ">", "threshold": 0.60}}], "condition_logic": "AND", "scope": null, "dry_run": false}}, "resolved_focus": null, "focus_clear": false}}

User: "drop flagged columns if null rate > 60%"
→ {{"intent": "CONDITIONAL_DECIDE", "params": {{"decision": "drop", "conditions": [{{"field": "null_rate", "operator": ">", "threshold": 0.60}}], "condition_logic": "AND", "scope": "FLAG", "dry_run": false}}, "resolved_focus": null, "focus_clear": false}}

User: "keep columns with confidence above 70"
→ {{"intent": "CONDITIONAL_DECIDE", "params": {{"decision": "keep", "conditions": [{{"field": "confidence", "operator": ">=", "threshold": 70}}], "condition_logic": "AND", "scope": null, "dry_run": false}}, "resolved_focus": null, "focus_clear": false}}

User: "what would happen if I dropped columns with null rate above 50%"
→ {{"intent": "CONDITIONAL_DECIDE", "params": {{"decision": "drop", "conditions": [{{"field": "null_rate", "operator": ">", "threshold": 0.50}}], "condition_logic": "AND", "scope": null, "dry_run": true}}, "resolved_focus": null, "focus_clear": false}}

User: "drop if null rate high and signal low"
→ {{"intent": "CONDITIONAL_DECIDE", "params": {{"decision": "drop", "conditions": [{{"field": "null_rate", "operator": ">", "threshold": 0.60}}, {{"field": "confidence", "operator": "<", "threshold": 45}}], "condition_logic": "AND", "scope": null, "dry_run": false}}, "resolved_focus": null, "focus_clear": false}}

User: "no need"
→ {{"intent": "ACKNOWLEDGE", "params": {{}}, "resolved_focus": null, "focus_clear": false}}

User: "cancel"
→ {{"intent": "ACKNOWLEDGE", "params": {{}}, "resolved_focus": null, "focus_clear": false}}

User: "never mind"
→ {{"intent": "ACKNOWLEDGE", "params": {{}}, "resolved_focus": null, "focus_clear": false}}

User: "ok thanks"
→ {{"intent": "ACKNOWLEDGE", "params": {{}}, "resolved_focus": null, "focus_clear": false}}

User: "forget it"
→ {{"intent": "ACKNOWLEDGE", "params": {{}}, "resolved_focus": null, "focus_clear": false}}

User: "confirm"  (history shows prior dry-run: "drop if null rate > 0.60, 14 columns would change")
→ {{"intent": "CONDITIONAL_DECIDE", "params": {{"decision": "drop", "conditions": [{{"field": "null_rate", "operator": ">", "threshold": 0.60}}], "condition_logic": "AND", "scope": null, "dry_run": false}}, "resolved_focus": null, "focus_clear": false}}

User: "yes apply it"  (history shows prior dry-run: "drop flagged columns if confidence < 45, 8 would change")
→ {{"intent": "CONDITIONAL_DECIDE", "params": {{"decision": "drop", "conditions": [{{"field": "confidence", "operator": "<", "threshold": 45}}], "condition_logic": "AND", "scope": "FLAG", "dry_run": false}}, "resolved_focus": null, "focus_clear": false}}

User: "go ahead"  (history shows prior dry-run: "drop if null rate > 0.50 AND confidence < 45, 6 would change")
→ {{"intent": "CONDITIONAL_DECIDE", "params": {{"decision": "drop", "conditions": [{{"field": "null_rate", "operator": ">", "threshold": 0.50}}, {{"field": "confidence", "operator": "<", "threshold": 45}}], "condition_logic": "AND", "scope": null, "dry_run": false}}, "resolved_focus": null, "focus_clear": false}}

User: "do it"  (history shows prior dry-run: "keep if confidence >= 70, 22 would change")
→ {{"intent": "CONDITIONAL_DECIDE", "params": {{"decision": "keep", "conditions": [{{"field": "confidence", "operator": ">=", "threshold": 70}}], "condition_logic": "AND", "scope": null, "dry_run": false}}, "resolved_focus": null, "focus_clear": false}}

User: "confirm"  (history shows guardrail block: "This would drop 205 of 230 columns — above the 75% guardrail. Type 'confirm' to proceed anyway.")
→ {{"intent": "CONDITIONAL_DECIDE", "params": {{"decision": "drop", "conditions": [{{"field": "null_rate", "operator": ">", "threshold": 0.70}}], "condition_logic": "AND", "scope": null, "dry_run": false, "force_confirm": true}}, "resolved_focus": null, "focus_clear": false}}
User: "yes go ahead"  (history shows guardrail block mentioning "guardrail" or "blocked" or "exceeds" with a condition)
→ reconstruct the SAME conditions and emit CONDITIONAL_DECIDE with dry_run=false AND force_confirm=true. Do NOT route to AUTO_DECIDE.

User: "confirm"  (NO prior dry-run in history AND no guardrail message)
→ {{"intent": "AUTO_DECIDE", "params": {{}}, "resolved_focus": null, "focus_clear": true}}

User: "compare null distribution of Var22, Var34"
→ {{"intent": "COMPARE", "params": {{"columns": ["Var22", "Var34"], "visual_type": "null_distribution"}}, "resolved_focus": ["Var22", "Var34"], "focus_clear": false}}

User: "compare churn rate of Var1, Var2"
→ {{"intent": "COMPARE", "params": {{"columns": ["Var1", "Var2"], "visual_type": "churn_rate"}}, "resolved_focus": ["Var1", "Var2"], "focus_clear": false}}

User: "side by side Var22 and Var34"
→ {{"intent": "COMPARE", "params": {{"columns": ["Var22", "Var34"], "visual_type": "default"}}, "resolved_focus": ["Var22", "Var34"], "focus_clear": false}}

User: "which one is better, Var22 or Var34?"
→ {{"intent": "COMPARE", "params": {{"columns": ["Var22", "Var34"], "visual_type": "default"}}, "resolved_focus": ["Var22", "Var34"], "focus_clear": false}}

User: "contrast Var1 and Var5"
→ {{"intent": "COMPARE", "params": {{"columns": ["Var1", "Var5"], "visual_type": "default"}}, "resolved_focus": ["Var1", "Var5"], "focus_clear": false}}

User: "compare the FLAG columns"
→ {{"intent": "COMPARE", "params": {{"columns": [], "zone": "FLAG", "visual_type": "default"}}, "resolved_focus": null, "focus_clear": false}}

User: "compare all DROP columns side by side"
→ {{"intent": "COMPARE", "params": {{"columns": [], "zone": "DROP", "visual_type": "default"}}, "resolved_focus": null, "focus_clear": false}}

User: "is Var22 worth keeping?"
→ {{"intent": "ANALYSE", "params": {{"columns": ["Var22"], "zone": null}}, "resolved_focus": "Var22", "focus_clear": false}}

User: "analyse flagged columns"
→ {{"intent": "ANALYSE", "params": {{"columns": [], "zone": "FLAG"}}, "resolved_focus": null, "focus_clear": false}}

User: "analyse keep columns"
→ {{"intent": "ANALYSE", "params": {{"columns": [], "zone": "KEEP"}}, "resolved_focus": null, "focus_clear": false}}

User: "tell me about the drop columns"
→ {{"intent": "ANALYSE", "params": {{"columns": [], "zone": "DROP"}}, "resolved_focus": null, "focus_clear": false}}

User: "what do the flagged columns look like"
→ {{"intent": "ANALYSE", "params": {{"columns": [], "zone": "FLAG"}}, "resolved_focus": null, "focus_clear": false}}

User: "keep Var22"
→ {{"intent": "DECIDE", "params": {{"column": "Var22", "zone": null, "zones": null, "decision": "keep"}}, "resolved_focus": "Var22", "focus_clear": false}}

User: "drop it"  (active_focus = "Var83")
→ {{"intent": "DECIDE", "params": {{"column": "Var83", "zone": null, "zones": null, "decision": "drop"}}, "resolved_focus": "Var83", "focus_clear": false}}

User: "drop it"  (active_focus = null)
→ {{"intent": "AMBIGUOUS", "params": {{}}, "resolved_focus": null, "focus_clear": false}}

User: "drop them all"  (active_focus = ["Var22", "Var34"])
→ {{"intent": "DECIDE", "params": {{"column": null, "zone": null, "zones": null, "decision": "drop"}}, "resolved_focus": ["Var22", "Var34"], "focus_clear": false}}

User: "drop all FLAG columns"
→ {{"intent": "DECIDE", "params": {{"column": null, "zone": "FLAG", "zones": null, "decision": "drop"}}, "resolved_focus": null, "focus_clear": false}}

User: "drop all FLAG and KEEP columns"
→ {{"intent": "DECIDE", "params": {{"column": null, "zone": null, "zones": ["FLAG", "KEEP"], "decision": "drop"}}, "resolved_focus": null, "focus_clear": false}}

User: "keep all DROP columns"
→ {{"intent": "DECIDE", "params": {{"column": null, "zone": "DROP", "zones": null, "decision": "keep"}}, "resolved_focus": null, "focus_clear": false}}

User: "keep all flagged columns"
→ {{"intent": "DECIDE", "params": {{"column": null, "zone": "FLAG", "zones": null, "decision": "keep"}}, "resolved_focus": null, "focus_clear": false}}

User: "drop all columns with the null value"
→ {{"intent": "CONDITIONAL_DECIDE", "params": {{"decision": "drop", "conditions": [{{"field": "null_rate", "operator": ">", "threshold": 0.0}}], "condition_logic": "AND", "scope": null, "dry_run": false}}, "resolved_focus": null, "focus_clear": false}}

User: "drop columns with no signal"
→ {{"intent": "CONDITIONAL_DECIDE", "params": {{"decision": "drop", "conditions": [{{"field": "confidence", "operator": "<", "threshold": 45}}], "condition_logic": "AND", "scope": null, "dry_run": false}}, "resolved_focus": null, "focus_clear": false}}

User: "drop all columns with high null rates"
→ {{"intent": "CONDITIONAL_DECIDE", "params": {{"decision": "drop", "conditions": [{{"field": "null_rate", "operator": ">", "threshold": 0.60}}], "condition_logic": "AND", "scope": null, "dry_run": false}}, "resolved_focus": null, "focus_clear": false}}

User: "list all the flagged columns"
→ {{"intent": "EXPLORE", "params": {{"filter": "flagged columns"}}, "resolved_focus": null, "focus_clear": false}}

User: "what have we decided so far?"
→ {{"intent": "STATUS", "params": {{}}, "resolved_focus": null, "focus_clear": true}}

User: "what is mutual information?"
→ {{"intent": "EXPLAIN", "params": {{"concept": "mutual_information"}}, "resolved_focus": null, "focus_clear": true}}

User: "what's the difference between Mutual Information and Cramer's V?"
→ {{"intent": "EXPLAIN", "params": {{"concepts": ["mutual_information", "cramers_v"]}}, "resolved_focus": null, "focus_clear": true}}

User: "explain mutual information and chi-square"
→ {{"intent": "EXPLAIN", "params": {{"concepts": ["mutual_information", "chi_square"]}}, "resolved_focus": null, "focus_clear": true}}

User: "MI vs Cramer's V — what's the difference?"
→ {{"intent": "EXPLAIN", "params": {{"concepts": ["mutual_information", "cramers_v"]}}, "resolved_focus": null, "focus_clear": true}}

User: "what risk tag types are there?"
→ {{"intent": "EXPLAIN", "params": {{"concept": "risk_tag"}}, "resolved_focus": null, "focus_clear": true}}

User: "list all the risk tags"
→ {{"intent": "EXPLAIN", "params": {{"concept": "risk_tag"}}, "resolved_focus": null, "focus_clear": true}}

User: "what are the possible risk tags?"
→ {{"intent": "EXPLAIN", "params": {{"concept": "risk_tag"}}, "resolved_focus": null, "focus_clear": true}}

User: "export the report"
→ {{"intent": "REPORT", "params": {{}}, "resolved_focus": null, "focus_clear": true}}

User: "what is the overall churn rate?"  (active_focus = "Var22", focus_age = 4)
→ {{"intent": "STATUS", "params": {{}}, "resolved_focus": null, "focus_clear": true}}

User: "keep it"  (active_focus = "Var22", focus_age = 4 — STALE, pronoun with no recent column context)
→ {{"intent": "AMBIGUOUS", "params": {{"stale_focus_candidate": "Var22"}}, "resolved_focus": null, "focus_clear": false}}

User: "keep it"  (active_focus = "Var83", focus_age = 0 — FRESH, just analysed Var83)
→ {{"intent": "DECIDE", "params": {{"column": "Var83", "zone": null, "zones": null, "decision": "keep"}}, "resolved_focus": "Var83", "focus_clear": false}}

User: "drop it"  (active_focus = "Var83", focus_age = 1)
→ {{"intent": "DECIDE", "params": {{"column": "Var83", "zone": null, "zones": null, "decision": "drop"}}, "resolved_focus": "Var83", "focus_clear": false}}

User: "yeah drop it"  (active_focus = "Var22", focus_age = 0)
→ {{"intent": "DECIDE", "params": {{"column": "Var22", "zone": null, "zones": null, "decision": "drop"}}, "resolved_focus": "Var22", "focus_clear": false}}

User: "ok keep it"  (active_focus = "Var10", focus_age = 0)
→ {{"intent": "DECIDE", "params": {{"column": "Var10", "zone": null, "zones": null, "decision": "keep"}}, "resolved_focus": "Var10", "focus_clear": false}}

User: "drop them all"  (active_focus = ["Var22", "Var34"])
→ {{"intent": "DECIDE", "params": {{"column": null, "zone": null, "zones": null, "decision": "drop"}}, "resolved_focus": ["Var22", "Var34"], "focus_clear": false}}

User: "drop them all"  (active_focus = null — no context)
→ {{"intent": "AMBIGUOUS", "params": {{}}, "resolved_focus": null, "focus_clear": false}}

User: "undo the last 3 decisions"
→ {{"intent": "UNDO", "params": {{"column": null, "steps": 3, "target_action": null}}, "resolved_focus": null, "focus_clear": true}}

User: "revert the last 2 changes"
→ {{"intent": "UNDO", "params": {{"column": null, "steps": 2, "target_action": null}}, "resolved_focus": null, "focus_clear": true}}

User: "undo 5 steps"
→ {{"intent": "UNDO", "params": {{"column": null, "steps": 5, "target_action": null}}, "resolved_focus": null, "focus_clear": true}}

User: "undo back to auto-decide"
→ {{"intent": "UNDO", "params": {{"column": null, "steps": null, "target_action": "auto_decide"}}, "resolved_focus": null, "focus_clear": true}}

User: "undo that"
→ {{"intent": "UNDO", "params": {{"column": null, "steps": null, "target_action": null}}, "resolved_focus": null, "focus_clear": true}}

User: "which numeric columns have high confidence?"
→ {{"intent": "EXPLORE", "params": {{"filter": "numeric high confidence"}}, "resolved_focus": null, "focus_clear": false}}

User: "list everything flagged as null-driven"
→ {{"intent": "EXPLORE", "params": {{"filter": "null driven"}}, "resolved_focus": null, "focus_clear": false}}

User: "show me numeric columns with high confidence scores"
→ {{"intent": "EXPLORE", "params": {{"filter": "numeric high confidence"}}, "resolved_focus": null, "focus_clear": false}}

User: "pending columns"
→ {{"intent": "EXPLORE", "params": {{"filter": "pending"}}, "resolved_focus": null, "focus_clear": false}}

User: "what's left to decide"
→ {{"intent": "EXPLORE", "params": {{"filter": "pending"}}, "resolved_focus": null, "focus_clear": false}}

User: "undecided columns"
→ {{"intent": "EXPLORE", "params": {{"filter": "pending"}}, "resolved_focus": null, "focus_clear": false}}

User: "what's left"
→ {{"intent": "EXPLORE", "params": {{"filter": "pending"}}, "resolved_focus": null, "focus_clear": false}}

User: "columns not yet decided"
→ {{"intent": "EXPLORE", "params": {{"filter": "pending"}}, "resolved_focus": null, "focus_clear": false}}

User: "null-driven columns"
→ {{"intent": "EXPLORE", "params": {{"filter": "null driven"}}, "resolved_focus": null, "focus_clear": false}}

User: "borderline columns"
→ {{"intent": "EXPLORE", "params": {{"filter": "borderline"}}, "resolved_focus": null, "focus_clear": false}}

User: "show me borderline columns"
→ {{"intent": "EXPLORE", "params": {{"filter": "borderline"}}, "resolved_focus": null, "focus_clear": false}}

User: "which columns are in G2?"
→ {{"intent": "EXPLORE", "params": {{"filter": "G2"}}, "resolved_focus": null, "focus_clear": false}}

User: "show me G1 columns"
→ {{"intent": "EXPLORE", "params": {{"filter": "G1"}}, "resolved_focus": null, "focus_clear": false}}

User: "how many numeric vs categorical?"
→ {{"intent": "OVERVIEW", "params": {{}}, "resolved_focus": null, "focus_clear": true}}

User: "what is the null rate distribution?"
→ {{"intent": "OVERVIEW", "params": {{}}, "resolved_focus": null, "focus_clear": true}}

User: "what is the class distribution?"
→ {{"intent": "OVERVIEW", "params": {{}}, "resolved_focus": null, "focus_clear": true}}

User: "how many columns have high null rates?"
→ {{"intent": "OVERVIEW", "params": {{}}, "resolved_focus": null, "focus_clear": true}}

User: "how many features are in the dataset?"
→ {{"intent": "OVERVIEW", "params": {{}}, "resolved_focus": null, "focus_clear": true}}

User: "what percentage of columns are flagged?"
→ {{"intent": "OVERVIEW", "params": {{}}, "resolved_focus": null, "focus_clear": true}}

User: "what is the overall churn rate?"
→ {{"intent": "OVERVIEW", "params": {{}}, "resolved_focus": null, "focus_clear": true}}

User: "how is the null rate spread across columns?"
→ {{"intent": "OVERVIEW", "params": {{}}, "resolved_focus": null, "focus_clear": true}}

User: "xklqpzrt maybe do something with the data"
→ {{"intent": "AMBIGUOUS", "params": {{"ambiguity_type": "generic"}}, "resolved_focus": null, "focus_clear": false}}

User: "keep Var"  (partial column name — matches multiple columns, no exact match)
→ {{"intent": "AMBIGUOUS", "params": {{"ambiguity_type": "ambiguous_column"}}, "resolved_focus": null, "focus_clear": false}}

User: "should I drop or keep these?"  (mentions both drop and keep — intent unclear)
→ {{"intent": "AMBIGUOUS", "params": {{"ambiguity_type": "intent_unclear"}}, "resolved_focus": null, "focus_clear": false}}

User: "what are my options from here?"  (vague continuation, active_focus=null)
→ {{"intent": "AMBIGUOUS", "params": {{"ambiguity_type": "generic"}}, "resolved_focus": null, "focus_clear": false}}

User: "actually no, undo that"
→ {{"intent": "UNDO", "params": {{"column": null, "steps": null, "target_action": null}}, "resolved_focus": null, "focus_clear": true}}
"""


def _regex_fallback(raw_text: str) -> str:
    upper = raw_text.upper()
    for pattern, intent in _INTENT_PATTERNS:
        if re.search(pattern, upper):
            return intent
    return "AMBIGUOUS"


def _parse_response(raw_text: str) -> dict:
    cleaned = raw_text.strip().strip("```json").strip("```").strip()
    try:
        parsed = json.loads(cleaned)
        intent = parsed.get("intent", "AMBIGUOUS").upper()
        if intent not in VALID_INTENTS:
            intent = "AMBIGUOUS"
        return {
            "intent"        : intent,
            "params"        : parsed.get("params", {}),
            "resolved_focus": parsed.get("resolved_focus", None),
            "focus_clear"   : bool(parsed.get("focus_clear", False)),  # I-1
        }
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            intent = parsed.get("intent", "AMBIGUOUS").upper()
            if intent not in VALID_INTENTS:
                intent = "AMBIGUOUS"
            return {
                "intent"        : intent,
                "params"        : parsed.get("params", {}),
                "resolved_focus": parsed.get("resolved_focus", None),
                "focus_clear"   : bool(parsed.get("focus_clear", False)),  # I-1
            }
        except json.JSONDecodeError:
            pass

    intent = _regex_fallback(raw_text)
    return {"intent": intent, "params": {}, "resolved_focus": None, "focus_clear": False}


def classify(
    messages    : list[dict],
    active_focus: "str | list[str] | None",
    session     : dict,
    focus_age   : int = 0,                    # I-3: staleness counter from GraphState
) -> dict:
    """
    Classify the latest user message into one of 13 intents.

    Returns a dict with keys: intent, params, resolved_focus, focus_clear.
    focus_clear=True signals understand_node to clear active_focus (I-1).
    focus_age is injected into the prompt to enable staleness-based AMBIGUOUS (I-3).
    """
    user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break

    if not user_message.strip():
        return {"intent": "AMBIGUOUS", "params": {}, "resolved_focus": active_focus, "focus_clear": False}

    column_names    = session.get("feature_cols", [])
    last_n_messages = messages[-CLASSIFIER_WINDOW:]

    prior_dry_run_context = ""
    prior_guardrail_context = ""
    for msg in reversed(last_n_messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "").lower()
            # Check guardrail markers FIRST — separate from dry-run markers
            if any(term in content for term in [
                "above the 75% guardrail", "% guardrail",
                "blocked", "exceeds the", "type 'confirm' to proceed",
            ]):
                prior_guardrail_context = msg.get("content", "")
                break
            if any(term in content for term in [
                "would affect", "would change", "matched", "dry run",
                "preview", "would be dropped", "would be kept",
                "would mark", "say confirm", "say 'confirm'",
                "columns match", "would apply", "repeat the command",
                "columns would", "would drop", "would keep", "to apply",
            ]):
                prior_dry_run_context = msg.get("content", "")
                break

    system_prompt = _build_system_prompt(
        active_focus          = active_focus,
        column_names          = column_names,
        last_3_turns          = last_n_messages,
        focus_age             = focus_age,
        prior_dry_run_context = prior_dry_run_context,   # BUG 2 FIX
        prior_guardrail_context = prior_guardrail_context,   # ADD THIS
    )

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("[classifier] OPENROUTER_API_KEY not set — returning AMBIGUOUS.")
        return {"intent": "AMBIGUOUS", "params": {}, "resolved_focus": active_focus}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type" : "application/json",
        "HTTP-Referer" : "https://github.com/feature-selection-bot",
        "X-Title"      : "Feature Selection Bot",
    }
    
    payload = {
        "model"      : CLASSIFIER_MODEL,
        "temperature": 0.1,
        "messages"   : [
            {
                "role"   : "user",
                "content": system_prompt + "\n\nUser message to classify:\n" + user_message,
            }
        ],
    }

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )
        if response.status_code != 200:
            print(f"[classifier] OpenRouter Error Details: {response.text}")
        response.raise_for_status()
        raw_text = response.json()["choices"][0]["message"]["content"]
        
    except requests.exceptions.Timeout:
        print("[classifier] Request timed out — returning AMBIGUOUS.")
        return {"intent": "AMBIGUOUS", "params": {}, "resolved_focus": active_focus, "focus_clear": False}
    except Exception as e:
        print(f"[classifier] OpenRouter API error: {e}")
        return {"intent": "AMBIGUOUS", "params": {}, "resolved_focus": active_focus, "focus_clear": False}

    result = _parse_response(raw_text)

    # ── ISSUE 1: DEEP_DIVE SAFETY NET ────────────────────────────
    # If LLM classified as ANALYSE but missed deep_dive=True,
    # catch it here using keyword matching on the raw user message.
    _DEEP_DIVE_KEYWORDS = [
        "distribution", "deep dive", "deep-dive", "eda on", "eda ",
        "churn split", "churn rate breakdown", "null gap", "null distribution",
        "class imbalance", "full breakdown", "full distribution",
        "show me the distribution", "missingness", "show nulls",
    ]
    if result["intent"] == "ANALYSE":
        msg_lower = user_message.lower()
        if not result["params"].get("deep_dive"):
            if any(kw in msg_lower for kw in _DEEP_DIVE_KEYWORDS):
                result["params"]["deep_dive"] = True
                if not result["params"].get("visual_type"):
                    if "churn" in msg_lower:
                        result["params"]["visual_type"] = "churn_rate"
                    elif "null gap" in msg_lower:
                        result["params"]["visual_type"] = "null_gap"
                    elif "class imbalance" in msg_lower:
                        result["params"]["visual_type"] = "class_imbalance"
                    elif "null" in msg_lower:
                        result["params"]["visual_type"] = "null_distribution"
                    else:
                        result["params"]["visual_type"] = "distribution"

    # ── ISSUE 9: REPORT SAFETY NET ───────────────────────────────
    # If LLM returned AMBIGUOUS for a clear report/export request,
    # override it here.
    _REPORT_KEYWORDS = [
        "export the report", "generate the report", "export report",
        "generate report", "export the output", "generate the output",
        "export now", "generate now", "export it", "get the report",
    ]
    if result["intent"] == "AMBIGUOUS":
        msg_lower = user_message.lower()
        if any(kw in msg_lower for kw in _REPORT_KEYWORDS):
            result["intent"] = "REPORT"
            result["params"] = {}
            result["resolved_focus"] = None
            result["focus_clear"] = True

    # ── ISSUE 4: UNDO vs ACKNOWLEDGE SAFETY NET ──────────────────
    # "actually no, undo that" was being caught as ACKNOWLEDGE
    # because "actually no" appears in the ACKNOWLEDGE examples.
    # If the message contains an undo verb, always route to UNDO.
    _UNDO_OVERRIDE_KEYWORDS = ["undo", "revert", "go back", "undo that", "undo it"]
    if result["intent"] == "ACKNOWLEDGE":
        msg_lower = user_message.lower()
        if any(kw in msg_lower for kw in _UNDO_OVERRIDE_KEYWORDS):
            result["intent"] = "UNDO"
            result["params"] = {"column": None, "steps": None, "target_action": None}
            result["resolved_focus"] = None
            result["focus_clear"] = True

    # ── EXPLAIN focus_clear SAFETY NET ───────────────────────────
    # EXPLAIN must always set focus_clear=True. If the LLM missed it, enforce it.
    if result["intent"] == "EXPLAIN":
        result["focus_clear"] = True
        result["resolved_focus"] = None
        
    # ── resolved_focus backfill ─────────────────────────────────────────────
    # Handles both "columns" (list) and "column" (single str).
    # Preserves active_focus on AMBIGUOUS turns so context is not lost.
    #
    # C-2 FIX: DECIDE must NOT overwrite active_focus when the user used a
    # pronoun that resolved to the current focus column. The LLM resolves
    # "drop it" → params["column"] = active_focus (e.g. "Var83"), so we
    # cannot tell after-the-fact whether the user named a new column or used
    # a pronoun. Rule: for DECIDE, only advance active_focus when
    # params["column"] is DIFFERENT from the existing active_focus. If it
    # matches (pronoun scenario), leave resolved_focus=None so nodes.py keeps
    # the existing focus without resetting focus_age.
    if result["resolved_focus"] is None:
        if result["intent"] == "AMBIGUOUS":
            # Preserve existing active_focus so clarification round-trips
            # don't lose the columns the user already mentioned.
            result["resolved_focus"] = active_focus

        elif result["intent"] not in (
            "EXPLORE", "STATUS", "REPORT", "OVERVIEW", "AUTO_DECIDE"
        ):
            # Multi-column intents (ANALYSE, EDA, COMPARE)
            cols_from_params = result["params"].get("columns")
            if cols_from_params and isinstance(cols_from_params, list):
                # Issue B extension: for ANALYSE/EDA, if active_focus is None,
                # only accept columns that literally appear in the current message.
                # Prevents silent resolution to a stale column from conversation history.
                if active_focus is None:
                    cols_in_msg = [c for c in cols_from_params if c in user_message]
                    if not cols_in_msg:
                        result["intent"] = "AMBIGUOUS"
                        result["params"] = {
                            "ambiguity_type"       : "no_column",
                            "stale_focus_candidate": cols_from_params[0] if cols_from_params else None,
                        }
                        result["resolved_focus"] = None
                        result["focus_clear"]    = False
                    elif len(cols_in_msg) == 1:
                        result["resolved_focus"] = cols_in_msg[0]
                    else:
                        result["resolved_focus"] = cols_in_msg
                else:
                    if len(cols_from_params) == 1:
                        result["resolved_focus"] = cols_from_params[0]
                    elif len(cols_from_params) > 1:
                        result["resolved_focus"] = cols_from_params
             
            else:
                # Single-column intents (DECIDE, UNDO, EXPLAIN)
                col_from_params = result["params"].get("column")
                if col_from_params:
                    if result["intent"] == "DECIDE":
                        # C-2 FIX: Only advance focus if the user named a NEW column.
                        # If params["column"] == active_focus the user used a pronoun
                        # ("drop it") — leave resolved_focus=None so nodes.py does NOT
                        # reset focus_age or overwrite active_focus.
                        existing = (
                            active_focus
                            if isinstance(active_focus, str)
                            else None  # list focus: pronoun handled via cols_from_params above
                        )
                        if col_from_params != existing:
                            # Issue B FIX: If active_focus was None coming in, the LLM may
                            # have resolved the column from conversation history rather than
                            # the current user message. Check whether the resolved column
                            # name literally appears in the current message. If it does not,
                            # it came from history — force AMBIGUOUS so the user is asked
                            # to confirm which column they mean.
                            if active_focus is None:
                                if col_from_params not in user_message:
                                    result["intent"] = "AMBIGUOUS"
                                    result["params"] = {
                                        "ambiguity_type"       : "ambiguous_column",
                                        "stale_focus_candidate": col_from_params,
                                    }
                                    result["resolved_focus"] = None
                                    result["focus_clear"]    = False
                                else:
                                    # Column was named explicitly in this message — safe to use
                                    result["resolved_focus"] = col_from_params
                            else:
                                # User named a different/new column explicitly
                                result["resolved_focus"] = col_from_params
                        # else: pronoun pointed at current focus — leave resolved_focus=None
                    else:
                        result["resolved_focus"] = col_from_params
                else:
                    result["resolved_focus"] = active_focus

    # ── Debug logging (set CLASSIFIER_DEBUG=1 to enable) ───────────────────
    if os.environ.get("CLASSIFIER_DEBUG", "0") == "1":
        print(
            f"[classifier] intent={result['intent']} | "
            f"focus_clear={result['focus_clear']} | "
            f"resolved_focus={result['resolved_focus']} | "
            f"active_focus_in={active_focus} | "
            f"focus_age={focus_age}"
        )

    return result
