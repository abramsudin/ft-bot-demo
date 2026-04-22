# ============================================================
# llm/formatter.py  —  OpenRouter edition (v8)
#
# LLM Call #2 — Response Writer
#
# Public API:
#   format_response(intent, action_result) -> str
#
# v8 changes (OPEN-2 fix):
#   - GUARDRAIL KEYWORD HARDENING: format_response now enforces that all
#     guardrail responses contain the exact phrases "above the 75% guardrail"
#     and "Type 'confirm' to proceed anyway." — required by the classifier
#     scanner to detect prior guardrail context on the next turn. If the LLM
#     formatter omits or paraphrases these phrases, a fixed suffix is appended
#     automatically. This closes the loop-on-confirm bug (OPEN-2).
#   - CONDITIONAL_DECIDE, AUTO_DECIDE, DECIDE guidance blocks updated with
#     VERBATIM keyword requirements and an explicit warning not to paraphrase.
#
# v7 changes:
#   - N3: New CONDITIONAL_DECIDE guidance block (dry_run + live)
#   - D1: AUTO_DECIDE updated — narrates as draft, not final
#   - D2: OVERVIEW updated — presents bot-led vs user-led choice
#   - P2: ANALYSE zone_analysis narration added
#         EXPLAIN multi-concept narration added
#         AMBIGUOUS conditional_logic reason handled
#         UNDO multi-step narration added
# ============================================================

import os
import json
import re
import requests
import time
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_URL  = "https://openrouter.ai/api/v1/chat/completions"
FORMATTER_MODEL = "openai/gpt-4o-mini"

# ── Per-intent narration guidance ────────────────────────────
_INTENT_GUIDANCE: dict[str, str] = {

    "OVERVIEW": """
The bot has completed a dataset scan and is presenting its findings.

STEP 0 — CHECK THE USER'S LATEST MESSAGE FIRST:
If the user asked a specific statistical question (e.g. "how many numeric columns?",
"what is the class balance?", "what is the null rate distribution?", "how many features?"),
you MUST lead your response with the direct numerical answer extracted from the action result.
Do this BEFORE any overview narrative. Max 3-4 sentences total for stat questions — do not
produce the full fresh/post_tweak narrative in this case.

The action result contains these fields — surface whichever is relevant:
  - num_cols_count / cat_cols_count   → numeric vs categorical breakdown
  - null_rate_distribution            → null rate spread (e.g. "X columns > 50% null")
  - class_distribution                → target class balance (churn rate %)
  - total_features                    → total column count
  - verdict_summary                   → KEEP / FLAG / DROP counts and percentages

Examples:
  "how many numeric vs categorical?" → Lead with "You have X numeric and Y categorical columns."
  "what is the null rate distribution?" → Lead with how many columns fall into each null band.
  "what is the class distribution?" → Lead with the churn rate / class balance.
  "how many features?" → Lead with the total feature count.

After answering the specific question, briefly mention the verdict breakdown and offer
to go deeper (e.g. "Want to see the full overview or analyse specific columns?").
Then STOP — do not continue into the full overview structure below.

IF no specific stat question was asked, proceed to the overview_mode check:

IF overview_mode == "fresh":
  This is the first time the user is seeing the overview — no decisions made yet.
  Structure your response:
    1. Headline verdict: how many to keep, drop, and flag for review.
    2. Top 3-5 KEEP columns by name with a one-line reason each (use top_keep_columns data).
    3. FLAG count and what that means (borderline, needs human review).
    4. Key structural findings: null indicator columns needed, redundant pairs, null groups.
    5. Close with the TWO-PATH CHOICE from flow_options in the action result:
       Present BOTH options clearly:
       - Option 1 (bot-led): "Say 'load your recommendations' and I'll apply my verdicts as a
         starting draft — you can then override anything before exporting."
       - Option 2 (user-led): "Or you can work independently from scratch — analyse columns
         directly, set your own rules, and build your own list without following my recs."
  Tone: confident and collaborative. The bot leads, but the human decides.
  Write 6-8 sentences.

IF overview_mode == "post_tweak":
  The user has already made some decisions — this is a check-in on the current state.
  NOTE: Unlike STATUS (which is a pure progress counter), post_tweak OVERVIEW must
  engage with the *content* of decisions — which columns changed, what the bot flagged.
  Structure your response:
    1. Open with the live_decisions state: how many kept, dropped, pending so far.
    2. If tweaked_decisions is non-empty: acknowledge the overrides specifically —
       name the columns where the user went against the bot recommendation and what they chose.
    3. Mention remaining pending columns (FLAG count still undecided).
    4. Note any structural findings still relevant (redundant pairs, null indicators).
    5. Close with the next_step offer from the action result — word it naturally.
  Tone: collaborative and non-judgmental about overrides — it's the user's dataset.
  Write 5-7 sentences.

ALWAYS: No bullet points. No headers. Flowing prose only.
ALWAYS: End with the natural next-step offer from next_step field (or a similar offer).
""",

    "AUTO_DECIDE": """
The bot has loaded its recommendations as a DRAFT starting point — not a final decision.

Check if "guardrail_triggered" is in the action result:
IF guardrail_triggered:
  - Explain the percentage and how to confirm to proceed.
  - You MUST use the exact phrase "above the 75% guardrail" verbatim.
  - You MUST use the exact phrase "Type 'confirm' to proceed anyway." verbatim.
  - NEVER paraphrase — the downstream classifier depends on these exact strings.

IF "already_applied" is True in the action result:
  - Tell the user that recommendations are already loaded and active.
  - State the current kept and dropped counts from the action result.
  - Mention how many flagged columns still need their review.
  - Keep it brief and suggest a next step.
  - Max 3 sentences.

IF normal completion (draft_mode=True):
  - Open by confirming: X columns marked keep, Y marked drop as a draft.
  - Emphasise this is EDITABLE — the user can override any column before the report.
  - Mention how many flagged columns still need their review.
  - Suggest what they might want to do next: review flagged columns, override specific
    columns they disagree with, run conditional rules to refine, or export when ready.
  - Tone: collaborative, not declarative — these are starting recommendations, not final.
  - Max 4 sentences.
""",

    "CONDITIONAL_DECIDE": """
The user applied (or previewed) a rule-based bulk decision.

Check "dry_run" in the action result:

IF dry_run == True (preview mode):
  - Lead with how many columns matched the condition(s).
  - State what would change vs what's already decided that way.
  - Name the condition in plain English (e.g. "null rate above 60%").
  - If pct_of_total is high (>50%), flag it so the user knows the scale.
  - End with: "Say 'confirm' or repeat the command to apply, or adjust the threshold."
  - Max 4 sentences.

IF dry_run == False AND guardrail_triggered:
  - Explain the guardrail was hit: X of Y columns would be dropped (Z%).
  - You MUST use the exact phrase "above the 75% guardrail" verbatim in your response.
  - You MUST use the exact phrase "Type 'confirm' to proceed anyway." verbatim.
  - NEVER paraphrase these phrases — the downstream classifier depends on their exact
    wording to detect this guardrail block on the next turn. Do not substitute words
    like "exceeds", "beyond", "over the limit", etc. Use the exact strings.
  - Max 2 sentences.

IF dry_run == False AND no guardrail (normal apply):
  - Confirm X columns were marked [decision] based on the rule.
  - Name the condition in plain English.
  - Mention how many were skipped (already had that decision).
  - Suggest a natural follow-up (check status, review what's left, export).
  - Max 4 sentences.

IF matched_count == 0:
  - Acknowledge no columns matched.
  - Suggest the user try a different threshold or check what values exist.
  - Max 2 sentences.

ALWAYS: Translate field names to plain English:
  "null_rate" → "null rate", "confidence" → "confidence score",
  "mi_score" → "mutual information score", "p_value" → "p-value",
  "churn_gap" → "churn rate gap", "risk_tag" → "risk tag", "verdict" → "bot verdict"
ALWAYS: No bullet points. Flowing prose only.
""",

    "ANALYSE": """
The user asked for a statistical analysis of one or more columns.

Check if "zone_analysis" is True in the action result:

# REPLACEMENT — branches on user_question so it doesn't loop
IF zone_analysis == True (zone-level query):
  FIRST — check the user_question field to understand what was actually asked.

  CASE A — user_question contains "keep", "worth keeping", "recommend", "should i keep",
            "which ones", "which should":
    - Lead with how many columns in this zone are worth keeping.
    - Name the TOP 3-5 columns from per_col_ranked (highest confidence) — give each a
      one-line reason using their risk_tag and confidence score.
    - CRITICAL: per_col_ranked is sorted by confidence DESCENDING — the FIRST
      entries are the highest confidence and ARE the keep candidates. NEVER
      say "none are worth keeping" if per_col_ranked is non-empty. Even a
      confidence of 1 means it has more signal than a DROP column.
    - Name 2-3 that should be dropped (bottom of per_col_ranked — lowest confidence,
      high null_rate). Give a one-line reason for each.
    - End with: "Want to keep these, or go deeper on any specific column?"
    - Max 6 sentences. No bullets.


  CASE B — user_question contains "drop", "remove", "cut", "eliminate", "should i drop":
    - Lead with how many can safely be dropped from this zone.
    - Name the weakest columns (bottom of per_col_ranked) with their reason.
    - Flag any that might be worth keeping despite being in this zone.
    - End with: "Want me to drop these, or review any first?"
    - Max 5 sentences. No bullets.

  # FIXED — only triggers on pure listing requests with no keep/drop intent
  CASE C — user_question contains "list" OR "show me" OR "which columns"
          AND does NOT contain "keep", "drop", "recommend", "worth", "should":
    - Simply list ALL column names in the zone as one comma-separated inline string.
    - Do not give recommendations — just the full list.
    - End with: "Want to analyse any of these, or apply a decision rule to the zone?"
    - Max 3 sentences.

  DEFAULT (generic — first look at a zone, no specific action asked):
    - Open with zone name and column count.
    - State the average confidence score.
    - Use per_col_ranked to name the TOP 3 columns by confidence with a
      one-line reason each (verdict + risk_tag). per_col_ranked is sorted
      DESCENDING — the first entries are highest confidence.
    - Name the top 2 risk tags and what they indicate in plain English.
    - End with: "Want me to go through specific columns in this zone, or apply a rule to them?"
    - Max 6 sentences. NEVER offer to make a decision or recommend dropping — just describe.

  NEVER say "I didn't receive column names" — a zone query does not need column names.
  CONFIDENCE SCALE: always 0–100 integers. NEVER output decimals like "0.92".
  
IF single column (no multi_column key or multi_column=False):
  - Lead with the verdict (KEEP / FLAG / DROP) and confidence score (0–100 scale).
  - Mention the 1-2 most decisive signals.
  - If there's a risk tag or null group, mention it briefly.
  - STOP. Do NOT add a follow-up question or offer (e.g. "Want to deep-dive?" or
    "Want to keep or drop it?"). The user will ask if they want more.
  - Max 4 sentences.

CONFIDENCE SCALE (applies to all sub-modes above):
  Always express confidence as an integer 0–100. High ≥ 70, Medium 45–69, Low < 45.
  NEVER output a decimal like "0.92" or "0.75" — say "92" or "75" instead.

IF multi_column=True:
  - Open with the overall picture: how many of the columns are KEEPs, FLAGs, DROPs.
  - Use the "summary" field to lead — it has the key comparison info prebuilt.
  - Then briefly narrate each column (1 sentence each): verdict + the most decisive signal.
  - End with a suggestion: "Want me to deep-dive any of these, or compare them visually?"
  - Keep it flowing — no bullet points even for multiple columns.
  - Max 6-8 sentences total.

HARD RULE: Do NOT end with a follow-up question or offer. 
State the result and stop. The user will ask if they want more.
""",

    "COMPARE": """
The user explicitly asked to compare columns against each other.

Use the comparison_notes field for key findings:
  - strongest_signal: which column has the highest confidence
  - highest_null_rate: which column is most null-heavy
  - verdicts: what the bot recommends for each
  - keeps / flags / drops: grouped by outcome

Structure your response:
  1. Lead with the most striking difference: e.g. "Var83 has significantly more signal than Var22."
  2. Call out the verdicts for each column — be specific.
  3. Highlight the biggest differentiator (confidence gap, null rate difference, churn gap).
  4. End with a natural next step — e.g. "Want me to deep-dive either of these?"
  5. Max 5-6 sentences.

IMPORTANT: Do NOT mention visuals, charts, or visualisations — the bot is text-only.

HARD RULE: Do NOT recommend keeping or dropping any column inside a COMPARE response.
COMPARE is informational only — present the data, name the differences, stop.
Never say "I'd recommend dropping X" or "you should keep Y" inside COMPARE.

HARD RULE: Do NOT end with a follow-up question or offer. 
State the result and stop. The user will ask if they want more.
""",

    "EXPLORE": """
The user asked to browse or filter columns by zone, tag, or property.

STEP 1 — KEY CHECK: Look for "matches" in the action result.

IF "matches" EXISTS (even if you also see zone_summary, verdict_summary, or any other
  field) — the "matches" key ALWAYS takes priority. Do NOT fall through to any summary.

  Render as follows:
  - First sentence: state the count and filter (e.g. "Found 12 FLAG columns.").
  - Second sentence: list ALL column names as one inline comma-separated string.
    Extract the "column" field from each dict in the "matches" array.
    NEVER truncate. NEVER say "and X more" or "including...". Full list every time.
  - STOP after the list. Do NOT add a follow-up question, offer, or solicitation.
    Never say "Want to analyse any of these?" or similar after the list.
  - HARD RULE: if "matches" is present and non-empty, your response MUST contain
    a comma-separated list of every column name in that array. No exceptions.

IF "matches" is absent OR is an empty list []:
  - Report that no columns matched and echo the "note" field if present.
  - Max 2 sentences.

NEVER produce a zone-level narrative, confidence breakdown, or verdict summary as a
substitute for the column list when "matches" is present.
""",

    "EDA": """
The user asked for a visual deep-dive on one or more columns.

Check "multi_column" in the action result:

IF single column (multi_column=False or absent):
  - Lead with the null rate and whether missingness correlates with churn.
  - State the distribution shape or key churn finding depending on visual_type_shown.
  - Give a plain-English interpretation: what does this mean for feature selection?

  THEN — progressive disclosure offer:
  Check "available_visuals" list. If it is non-empty:
    End with: "I showed you the [visual_type_shown] view. I can also show you:
    [list available_visuals in plain English — 'distribution', 'churn rate breakdown',
    'null gap panel']. Just pick one, or ask for something else."
  If available_visuals is empty: end with a next-step suggestion instead.

IF multi_column=True (visual_type = "null_distribution"):
  - Summarise which columns have the highest null rates.
  - Note which have meaningful null signal gaps (gap > 3pp = strong, 1-3pp = moderate).
  - Flag the most noteworthy finding.
  - End with progressive disclosure offer as above.

IF multi_column=True (other visual types):
  - Briefly compare the columns: which stands out, which looks similar.
  - Note any obvious differences in distribution or churn split.
  - End with progressive disclosure offer.

ALWAYS: Max 5 sentences. No bullets. Flowing prose.
""",

    "DECIDE": """
The user just made a keep or drop decision.

Check "mode" in the action result:

IF mode == "single":
  - Confirm the column name and decision clearly (e.g. "Var22 is now marked as keep.").
  - If override=True, acknowledge neutrally (e.g. "That goes against my recommendation, but noted.").
  - Max 2 sentences.

IF mode == "bulk" (zone-based, e.g. "drop all FLAG columns"):
  - State how many columns were affected and what zone/filter was used.
    Use the "applied_to" list length as the count, NOT total_cols.
  - Mention how many were skipped (already had that decision) if skipped list is non-empty.
  - If guardrail_triggered, explain the percentage, use the exact phrase "above the 75% guardrail",
    and ask them to type the exact phrase "Type 'confirm' to proceed anyway."
  - Suggest a natural next step (e.g. check status, review what's left).
  - Max 3 sentences.

IF mode == "multi" (explicit multi-column list):
  - State how many columns were marked and with what decision.
  - Max 2 sentences.

ALWAYS: No bullet points. Flowing prose only.
ALWAYS: Do NOT invent column counts — use the numbers in the action result exactly.

HARD RULE: Do NOT end with a follow-up question or offer. 
State the result and stop. The user will ask if they want more.
""",

    "UNDO": """
The user reversed a decision.

Check "mode" in the action result:

IF mode == "multi_step" (steps > 1):
  - CRITICAL: Lead with how many DECISIONS were undone using the "steps" field —
    say "X decision(s) undone" or "rewound X step(s)". Do NOT lead with the column count.
    "steps" = number of decision snapshots popped (what the user asked to undo).
  - Then mention "reverted_count" as secondary detail only — how many columns actually
    changed value as a result (e.g. "affecting Y columns").
  - If the undo snapshot came from a bulk-drop, the zone size may have been larger
    than reverted_count — this is expected because columns already dropped before the
    bulk ran were not affected. Do NOT flag this as an error; explain it naturally if
    the user seems confused (e.g. "104 were in the zone, but 7 were already dropped
    before that action ran, so 97 were actually restored").
  - Max 3 sentences.

IF mode == "time_travel" (target_action present):
  - Confirm that the state was rewound to the named snapshot.
  - Say how many columns were reverted (use "reverted_count" or len("reverted")).
  - Max 2 sentences.

IF mode == "full" or "column":
  - Confirm what was undone and the column's new status.
  - If "note" is present in the action result and the user asked about a count
    discrepancy, explain it in plain English using the note.
  - Max 2 sentences.

ALWAYS: Use "steps" as the headline count for multi_step (decisions popped).
        Use "reverted_count" for multi_step column detail only — never as the lead figure.

HARD RULE: Do NOT end with a follow-up question or offer. 
State the result and stop. The user will ask if they want more.
""",

    "STATUS": """
The user wants a snapshot of current decision progress — NOT a dataset overview.

This is a progress report, not a scan summary. Do NOT describe the dataset, mention
top columns by name, or discuss null indicators / redundancy pairs (that's OVERVIEW).

Use ONLY these fields from the action result:
  - total_cols        → total feature columns
  - kept              → count marked keep so far
  - dropped           → count marked drop so far
  - decided           → total decided (kept + dropped) — use this for the "X of Y" numerator
  - pending           → count not yet decided
  - coverage_pct      → percentage of columns with a decision (use this exact field)
  - override_count    → decisions that overrode the bot's recommendation
  - suggest_auto_decide → if True, mention that running AUTO_DECIDE can fill pending slots

Structure:
  1. Sentence 1: "X of Y columns decided — Z% coverage. K kept, D dropped, P still pending."
  2. Sentence 2: If override_count > 0, note "You've overridden my recommendation on N column(s)."
     Otherwise skip.
  3. Sentence 3: If pending > 0 and suggest_auto_decide is True:
       "You haven't run AUTO_DECIDE yet — saying 'load your recommendations' would
        fill the pending slots as a draft you can edit."
     Else if pending > 0: state the pending count only. Do NOT ask "Want to keep going?"
       or any follow-up question.
     Else: "All columns have been decided — you can export the report when ready."

HARD RULE: Do NOT end the STATUS response with a question. Just state the facts and stop.

HARD RULE: The coverage percentage MUST come from the "coverage_pct" field.
  NEVER compute it yourself from kept/dropped/total — use the field directly.
  NEVER output "0%" unless coverage_pct is literally 0.

Max 3 sentences. No bullet points. Flowing prose.
""",

    "EXPLAIN": """
The user asked for a concept explanation.

Check if "concepts" or "results" contains multiple items:
IF multiple concepts:
  - Explain each concept in 2-3 sentences, one after the other.
  - Keep each explanation plain-English: definition, example, threshold used in this bot.
  - No headers or bullets — flow naturally between concepts.
  - Max 6-8 sentences total.
IF single concept:
  - Plain-English definition, no jargon or formulas.
  - One concrete churn/feature selection example.
  - State the threshold or interpretation used in this bot.
  - Max 4 sentences.

CRITICAL — CONFIDENCE SCORE SCALE:
  The confidence score in this bot runs from 0 to 100 (not 0 to 1).
  High = 70 or above. Medium = 45 to 69. Low = below 45.
  Always state thresholds using the 0–100 scale in your response.
  NEVER say "0.75" or "0.92" — say "75" or "92".
  NEVER say "0.70 threshold" — say "70".
""",

    "REPORT": """
The user asked to generate the final report.
- Confirm it is being generated (or has been generated).
- State what's in it: total decisions, keep/drop counts, any human overrides.
- If pending columns exist, mention that.
- Max 3 sentences.
""",

    "ACKNOWLEDGE": """
The user dismissed a suggestion or is standing by with nothing to action.
- Respond with a single short, natural sentence.
- Acknowledge that nothing was changed or that you are ready when they are.
- Do NOT ask any clarifying question.
- Do NOT offer alternatives or suggest next steps unless it flows completely naturally.
- Max 1 sentence.
Examples of good responses:
  "Got it, no changes made."
  "Understood — ready when you are."
  "No worries, just let me know."
""",

    "AMBIGUOUS": """
The user's message was unclear.

Check "cancel" in the action result params:
IF cancel == True:
  - The user is rejecting/cancelling a previous suggestion or guardrail.
  - Respond with a single short acknowledgement like "Got it, no changes made."
  - Do NOT ask any clarifying question.
  - Max 1 sentence.

Check "reason" in the action result:
IF reason == "conditional_logic":
  - Acknowledge the user wants to apply a conditional rule.
  - Give 1-2 examples: "drop flagged columns if null rate is high" or "keep columns with confidence above 70".
  - Ask what condition they'd like to apply.
  - Max 3 sentences.
IF reason == "missing_context":
  - Ask what column(s) or zone they're referring to.
  - Max 2 sentences.
ELSE:
  - Ask exactly ONE targeted clarifying question.
  - Echo back what you understood and ask what was unclear.
  - Max 2 sentences.
""",
}

_FALLBACK_GUIDANCE = """
Respond helpfully and conversationally based on the action result.
Be concise — max 4 sentences.
"""


def _build_prompt(intent: str, action_result: dict, user_message: str = "", guardrail_pending: bool = False) -> str:
    guidance = _INTENT_GUIDANCE.get(intent, _FALLBACK_GUIDANCE)
    try:
        result_str = json.dumps(action_result, indent=2, default=str)
    except Exception:
        result_str = str(action_result)

    user_msg_block = (
        f"\nUSER'S LATEST MESSAGE (answer this specific question if applicable — "
        f"lead with the direct answer before any broader narrative):\n{user_message}\n"
    ) if user_message else ""

    guardrail_block = (
        "\n⚠️ GUARDRAIL REMINDER: A prior bulk operation is still pending confirmation "
        "from the user. You MUST prepend exactly ONE sentence to your response: "
        "'Just a note — a bulk drop is still waiting for your confirm before it applies.' "
        "Then answer the user's actual question normally. Do not forget this reminder.\n"
    ) if guardrail_pending and not (action_result or {}).get("guardrail_triggered") else ""

    return f"""You are the response writer for a feature selection assistant.
Your job: turn the structured action result below into a clear, friendly reply for the user.

INTENT: {intent}

INSTRUCTIONS FOR THIS INTENT:
{guidance.strip()}

GENERAL RULES:
{guardrail_block}
- Write in second person ("You", "Your") — address the user directly.
- Never use bullet points or headers — write in flowing sentences.
- Never say "Based on the data provided" or "As an AI" — just answer.
- Never repeat raw field names from the JSON (e.g. don't say "verdict_df").
- If action_result contains an "error" key, acknowledge it and suggest what to try.
- CONFIDENCE SCALE — always use 0–100 integers. High ≥ 70, Medium 45–69, Low < 45.
  NEVER output decimal scores like "0.92" or "0.75" anywhere in your response.
- For OVERVIEW: mention top KEEP columns by name with brief reasons. Be specific.
- For COMPARE / EDA progressive disclosure: list available_visuals in plain English,
  not as raw strings. E.g. "null gap panel" not "null_gap".
- For multi-column ANALYSE: narrate each column briefly in one sentence, don't skip any.
- For CONDITIONAL_DECIDE: always translate field names to plain English in your response.
- For AUTO_DECIDE draft mode: always make clear the decisions are editable, not final.

RAW ACTION RESULT (use as factual source — do not invent numbers):
{result_str}
{user_msg_block}
Write your reply now:"""


def format_response(intent: str, action_result: dict | None, draft_mode: bool = False, user_message: str = "", guardrail_pending: bool = False) -> str:
    if not action_result:
        action_result = {"status": "ok", "detail": "Action completed with no additional output."}

    prompt = _build_prompt(intent, action_result, user_message, guardrail_pending)

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("[formatter] OPENROUTER_API_KEY not set — using safe fallback.")
        return _safe_fallback(intent, action_result)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type" : "application/json",
        "HTTP-Referer" : "https://github.com/feature-selection-bot",
        "X-Title"      : "Feature Selection Bot",
    }
    payload = {
        "model"      : FORMATTER_MODEL,
        "temperature": 0.4,
        "messages"   : [
            {"role": "user", "content": prompt}
        ],
    }

    try:
        time.sleep(2.0)
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=45,
        )
        if response.status_code != 200:
            print(f"[formatter] OpenRouter Error Details: {response.text}")

        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"].strip()

        # Strip markdown fences if model wraps response
        reply = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", reply, flags=re.DOTALL).strip()

        if not reply:
            raise ValueError("Empty response from OpenRouter")

        # GUARDRAIL KEYWORD HARDENING (Fix for OPEN-2):
        # The classifier scanner looks for "above the", "% guardrail", "guardrail",
        # "exceeds the", and "blocked" in the assistant's prior message to detect a
        # guardrail block. If the LLM formatter improvises different wording, the
        # scanner misses it and the next "confirm" routes to AUTO_DECIDE instead of
        # CONDITIONAL_DECIDE with force_confirm=True.
        # Fix: if this is a guardrail response and the reply is missing the required
        # keywords, append a fixed suffix that always contains them.
        if action_result.get("guardrail_triggered"):
            reply_lower = reply.lower()
            has_guardrail_keyword = any(
                term in reply_lower for term in [
                    "guardrail", "above the", "% guardrail",
                    "exceeds the", "blocked", "would mark",
                ]
            )
            has_confirm_cue = "confirm" in reply_lower
            if not has_guardrail_keyword or not has_confirm_cue:
                count = action_result.get("projected_drops", "many")
                total = action_result.get("total_features", action_result.get("total_cols", "all"))
                pct   = action_result.get("projected_pct", "")
                pct_str = f" ({int(pct * 100)}%)" if pct else ""
                reply = reply.rstrip(".").rstrip() + (
                    f" This would drop {count} of {total} columns{pct_str} — "
                    f"above the 75% guardrail. Type 'confirm' to proceed anyway."
                )

        return reply

    except requests.exceptions.Timeout:
        print("[formatter] Request timed out — using safe fallback.")
        return _safe_fallback(intent, action_result)
    except Exception as e:
        print(f"[formatter] OpenRouter API error: {e}")
        return _safe_fallback(intent, action_result)


def _safe_fallback(intent: str, action_result: dict) -> str:
    if "error" in action_result:
        return f"Something went wrong: {action_result['error']}. Please try again."

    if action_result.get("guardrail_triggered"):
        count = action_result.get("projected_drops", "many")
        total = action_result.get("total_features", action_result.get("total_cols", "all"))
        return (
            f"That operation would drop {count} out of {total} columns — above the 75% guardrail. "
            f"Type 'confirm' to proceed anyway."
        )

    fallbacks = {
        "OVERVIEW": lambda r: (
            f"{'Post-tweak overview: ' if r.get('overview_mode') == 'post_tweak' else 'Scan complete: '}"
            f"{r.get('verdict_summary', {}).get('KEEP', '?')} to keep, "
            f"{r.get('verdict_summary', {}).get('drop_total', '?')} to drop, "
            f"{r.get('verdict_summary', {}).get('FLAG', '?')} flagged for review. "
            f"{r.get('next_step', 'What would you like to do next?')}"
        ),
        "AUTO_DECIDE": lambda r: (
            f"Recommendations are already loaded: {r.get('kept', 0)} kept, "
            f"{r.get('dropped', 0)} dropped. "
            f"{r.get('pending_flags', 0)} flagged columns still need your review."
            if r.get("already_applied") else
            f"Loaded as draft: {r.get('kept', 0)} kept, {r.get('dropped', 0)} dropped. "
            f"{r.get('pending_flags', 0)} flagged columns still need your review. "
            f"You can override any of these before generating the report."
        ),
        "CONDITIONAL_DECIDE": lambda r: (
            f"{'Preview: ' if r.get('dry_run') else 'Applied: '}"
            f"{r.get('matched_count', 0)} columns matched your condition(s). "
            + (f"{r.get('would_change_count', 0)} would change." if r.get('dry_run') else
               f"{r.get('applied_count', 0)} marked '{r.get('decision', '?')}'.")
        ),
        "ANALYSE": lambda r: (
            f"Zone analysis complete for {r.get('zone', '?')} zone: {r.get('column_count', '?')} columns."
            if r.get("zone_analysis") else
            f"Analysis complete for {', '.join(r['columns']) if r.get('multi_column') else r.get('column', 'the column')}. "
            + (r.get('summary', '') if r.get('multi_column') else f"Verdict: {r.get('verdict', 'unknown')}.")
        ),
        "COMPARE": lambda r: (
            f"Comparison complete for {', '.join(r.get('columns', []))}. "
            f"Strongest signal: {r.get('comparison_notes', {}).get('strongest_signal', {}).get('column', '?')}."
        ),
        "EXPLORE": lambda r: (
            f"Found {r.get('total_matches', len(r.get('matches', [])))} matching columns: "
            + ", ".join(
                m["column"] if isinstance(m, dict) else str(m)
                for m in r.get("matches", [])
            ) + "."
            if r.get("matches") else
            f"No columns matched '{r.get('filter_used', 'that filter')}'."
        ),
        "EDA": lambda r: (
            f"EDA complete for {', '.join(r['columns']) if r.get('multi_column') else r.get('column', 'the column')}. "
            + (f"More views available: {', '.join(r.get('available_visuals', []))}."
               if r.get('available_visuals') else "")
        ),
        "DECIDE": lambda r: f"Decision recorded: {r.get('column', r.get('zone', 'columns'))} marked as {r.get('decision', 'decided')}.",
        "UNDO": lambda r: (
            f"Undid {r.get('steps', 1)} step(s) — {len(r.get('reverted', []))} column(s) reverted."
            if r.get("mode") == "multi_step" else
            f"Undo complete — {len(r.get('reverted', []))} column(s) reverted."
            if r.get("reverted") else
            "Undo complete."
        ),
        "STATUS": lambda r: f"So far: {r.get('decided', r.get('kept', 0) + r.get('dropped', 0))} decided — {r.get('kept', 0)} kept, {r.get('dropped', 0)} dropped, {r.get('pending', 0)} pending.",
        "EXPLAIN": lambda r: f"Here's a brief explanation of {r.get('concept', r.get('concepts', ['that concept'])[0])}.",
        "REPORT": lambda r: f"Report generated with {r.get('total_decisions', 0)} decisions recorded.",
        "ACKNOWLEDGE": lambda r: "Got it — ready when you are.",
        "AMBIGUOUS": lambda r: (
            "It sounds like you want to apply a conditional rule — try something like "
            "'drop flagged columns if null rate is high' or 'keep columns with confidence above 70'."
            if r.get("reason") == "conditional_logic"
            else "I didn't quite catch that — could you rephrase?"
            ),
      }

    fn = fallbacks.get(intent, lambda r: "Done. What would you like to do next?")
    return fn(action_result)
