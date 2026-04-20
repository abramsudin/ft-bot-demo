# ============================================================
# actions/explain.py
#
# Action: EXPLAIN
#
# Returns a structured explanation for a statistical or domain
# concept used by the bot. The formatter narrates this in plain
# English; the formatter also handles any concept not in this
# lookup (flagged via "unknown_concept": True).
#
# Public API:
#   run(state: dict) -> dict
#
# Concept coverage:
#   mann_whitney, mutual_information, point_biserial,
#   chi_square, cramers_v, spearman, null_signal,
#   confidence_score, risk_tag, null_group,
#   verdict, feature_selection
#
# No LLM calls. No state mutations. Pure Python.
# ============================================================


# ── Concept lookup table ──────────────────────────────────────
# Each entry:
#   name        : display name
#   what        : one-sentence definition
#   how_used    : how this bot uses it
#   threshold   : decision boundary (if applicable)
#   example     : concrete illustration
#   aliases     : list of strings that map to this entry

CONCEPTS: dict[str, dict] = {

    "mann_whitney": {
        "name"     : "Mann-Whitney U Test",
        "what"     : (
            "A non-parametric test that checks whether numeric values "
            "tend to be higher in one group (churned vs retained) than the other, "
            "without assuming a normal distribution."
        ),
        "how_used" : (
            "Used for numeric features. A low p-value means the feature "
            "separates churned and retained customers — evidence it is predictive."
        ),
        "threshold": "p < 0.05 is treated as a significant signal.",
        "example"  : (
            "If 'tenure_months' has p = 0.001, churned customers tend to have "
            "meaningfully different tenure than retained ones."
        ),
        "aliases"  : ["mann whitney", "mannwhitney", "mann-whitney", "mw test", "mw p", "mann_whitney_p"],
    },

    "mutual_information": {
        "name"     : "Mutual Information (MI)",
        "what"     : (
            "Measures how much knowing a feature reduces uncertainty about the target. "
            "Zero means the feature is statistically independent of churn; "
            "higher values mean stronger dependency."
        ),
        "how_used" : (
            "Used for numeric features alongside Mann-Whitney. "
            "A high MI score confirms a feature carries real predictive content, "
            "even if the relationship is non-linear."
        ),
        "threshold": "MI > 0.01 is treated as a non-trivial signal; > 0.05 is strong.",
        "example"  : (
            "A feature like 'days_since_last_login' might have MI = 0.08, "
            "indicating it shares substantial information with churn."
        ),
        "aliases"  : ["mutual info", "mi", "mutual information", "mutual_info"],
    },

    "point_biserial": {
        "name"     : "Point-Biserial Correlation",
        "what"     : (
            "Measures the linear correlation between a continuous numeric feature "
            "and a binary target (churn). Ranges from -1 to +1."
        ),
        "how_used" : (
            "A positive value means higher feature values correlate with churn; "
            "negative means higher values correlate with retention. "
            "Magnitude indicates strength of the linear relationship."
        ),
        "threshold": "|r| > 0.1 is treated as a meaningful linear signal.",
        "example"  : (
            "'num_support_calls' with r = 0.22 means customers who churn tend "
            "to have made more support calls."
        ),
        "aliases"  : ["point biserial", "point_biserial", "point_biserial_r", "pb correlation", "pb corr"],
    },

    "chi_square": {
        "name"     : "Chi-Square Test",
        "what"     : (
            "Tests whether the distribution of a categorical feature's values "
            "differs between churned and retained customers, beyond what random "
            "chance would produce."
        ),
        "how_used" : (
            "Used for categorical features. A low p-value means the category "
            "breakdown is meaningfully different between churned and retained groups."
        ),
        "threshold": "p < 0.05 is treated as a significant association.",
        "example"  : (
            "If 'contract_type' has chi2 p = 0.0003, the mix of Monthly/Annual "
            "contracts differs substantially between churners and non-churners."
        ),
        "aliases"  : ["chi2", "chi square", "chi-square", "chi_square", "chi2_p"],
    },

    "cramers_v": {
        "name"     : "Cramér's V",
        "what"     : (
            "A measure of association strength between two categorical variables, "
            "derived from the Chi-Square statistic. Ranges from 0 (no association) "
            "to 1 (perfect association)."
        ),
        "how_used" : (
            "Used alongside Chi-Square for categorical features. Chi-Square tells us "
            "if an association exists; Cramér's V tells us how strong it is."
        ),
        "threshold": "V > 0.1 is moderate; V > 0.3 is strong.",
        "example"  : (
            "'payment_method' with V = 0.28 has a moderately strong association "
            "with churn — the method customers use matters."
        ),
        "aliases"  : ["cramers v", "cramer v", "cramers_v", "cramers", "cramer's v"],
    },

    "spearman": {
        "name"     : "Spearman Rank Correlation (Redundancy)",
        "what"     : (
            "Measures the monotonic relationship between two numeric features. "
            "Used here to detect redundancy: if two KEEP features are highly "
            "correlated, one adds little extra information."
        ),
        "how_used" : (
            "High Spearman correlation between two features that are both marked "
            "KEEP flags a potential redundancy. One may be dropped without "
            "losing predictive power."
        ),
        "threshold": "|r| > 0.85 between two KEEP columns triggers a redundancy flag.",
        "example"  : (
            "If 'total_charges' and 'monthly_charges × tenure' have r = 0.97, "
            "keeping both adds minimal new information."
        ),
        "aliases"  : ["spearman", "spearman r", "spearman_r", "redundancy", "correlation redundancy"],
    },

    "null_signal": {
        "name"     : "Null Signal",
        "what"     : (
            "Whether the fact that a value is missing is itself predictive of churn, "
            "independent of what the value would have been."
        ),
        "how_used" : (
            "Compares churn rate among rows where the feature is null vs present. "
            "A large gap suggests engineering an is_null binary indicator feature."
        ),
        "threshold": "A gap of > 3 percentage points (pp) is treated as a meaningful null signal.",
        "example"  : (
            "If 'last_upgrade_date' is null for 40% of customers, and churners "
            "are null at 60% vs 25% for retained — a 35pp gap — missingness is informative."
        ),
        "aliases"  : ["null signal", "null_signal", "missing signal", "is null", "nulls predictive"],
    },

    "confidence_score": {
        "name"     : "Confidence Score",
        "what"     : (
            "A 0–100 score summarising how consistent the statistical evidence is "
            "for the bot's verdict. High confidence means multiple tests agree; "
            "low confidence means mixed or weak signals."
        ),
        "how_used" : (
            "Displayed alongside KEEP/FLAG/DROP to help you judge how much to "
            "trust the verdict. A FLAG with confidence 85 warrants more scrutiny "
            "than a FLAG with confidence 40."
        ),
        "threshold": "≥ 70 is high; 45–69 is medium; < 45 is low.",
        "example"  : (
            "A feature with Mann-Whitney p = 0.0001, MI = 0.08, and a strong "
            "point-biserial correlation would score ~92 confidence for KEEP."
        ),
        "aliases"  : ["confidence", "confidence score", "confidence_score", "how confident", "score"],
    },

    "risk_tag": {
        "name"     : "Risk Tag",
        "what"     : (
            "A short label summarising the dominant risk or characteristic "
            "associated with a column's statistical profile."
        ),
        "how_used" : (
            "Helps you quickly understand why a column was flagged or dropped "
            "without reading every test result. Common tags: HIGH_NULL, "
            "LOW_SIGNAL, REDUNDANT, NEAR_ZERO_VARIANCE, INFORMATIVE_NULL."
        ),
        "threshold": None,
        "example"  : (
            "A column with 80% missing values and no predictive signal would "
            "be tagged HIGH_NULL. A column nearly identical to another KEEP "
            "column would be tagged REDUNDANT."
        ),
        "aliases"  : ["risk tag", "risk_tag", "tag", "risk label", "risk"],
    },

    "null_group": {
        "name"     : "Null Group",
        "what"     : (
            "A cluster label (G1, G2, …) assigned to columns that have correlated "
            "missingness patterns — their values tend to be absent at the same time "
            "for the same customers."
        ),
        "how_used" : (
            "Columns in the same null group likely share a root cause for missingness "
            "(e.g., all optional at signup, or all populated by the same data feed). "
            "Useful for understanding data collection issues."
        ),
        "threshold": None,
        "example"  : (
            "If 'last_login', 'session_duration', and 'pages_visited' are all G2, "
            "they're probably missing together for customers who never used the portal."
        ),
        "aliases"  : ["null group", "null_group", "missing group", "g1", "g2", "g3", "null cluster"],
    },

    "verdict": {
        "name"     : "Verdict (KEEP / FLAG / DROP / DROP-NULL)",
        "what"     : (
            "The bot's recommendation for each feature column based on the "
            "combined weight of all statistical tests."
        ),
        "how_used" : (
            "KEEP: evidence suggests this feature is predictive and should be "
            "retained for modelling. "
            "FLAG: mixed or moderate evidence — review before deciding. "
            "DROP: weak or no evidence of predictive value, or redundant. "
            "DROP-NULL: column is entirely or near-entirely null — no usable data. "
            "DROP-NULL columns are handled separately because the reason for "
            "dropping is data absence, not low signal."
        ),
        "threshold": None,
        "example"  : (
            "A column with significant Mann-Whitney, high MI, and no redundancy "
            "gets KEEP. A column with high null rate and no signal gets DROP. "
            "A column that is 100% null gets DROP-NULL."
        ),
        "aliases"  : [
            "verdict", "keep flag drop", "recommendation", "bot recommendation",
            "drop null", "drop-null", "drop_null", "dropnull",
            "what is drop null", "difference between drop and drop null",
            "drop vs drop null", "drop-null verdict",
        ],
    },

    "feature_selection": {
        "name"     : "Feature Selection",
        "what"     : (
            "The process of choosing which input variables to include in a "
            "predictive model — keeping informative features and removing noise, "
            "redundancy, or data-quality problems."
        ),
        "how_used" : (
            "This entire bot is a feature selection assistant. The goal is to "
            "produce a final KEEP list that maximises signal quality for a "
            "churn prediction model."
        ),
        "threshold": None,
        "example"  : (
            "Starting from 80 raw features, good feature selection might yield "
            "30–40 high-quality inputs that outperform using all 80."
        ),
        "aliases"  : ["feature selection", "feature engineering", "variable selection", "what is this bot"],
    },
}

# ── Alias index ───────────────────────────────────────────────
# Built once at import time for fast O(1) lookups.
_ALIAS_INDEX: dict[str, str] = {}
for _key, _entry in CONCEPTS.items():
    _ALIAS_INDEX[_key] = _key
    for _alias in _entry.get("aliases", []):
        _ALIAS_INDEX[_alias.lower()] = _key


def run(state: dict) -> dict:
    """
    Return structured explanations for one or more statistical concepts.

    intent_params schema (Issue #5 — multi-concept support):
        concepts : list[str] | None — concepts the user asked about
        concept  : str | None       — legacy single-concept fallback

    Returns
    -------
    dict — partial state update:
        action_result : {
            multi_concept : bool,
            results       : list of per-concept dicts, each containing:
                concept         : str,
                resolved_key    : str | None,
                name            : str | None,
                what            : str | None,
                how_used        : str | None,
                threshold       : str | None,
                example         : str | None,
                unknown_concept : bool
        }
    """
    intent_params = state.get("intent_params", {})

    # Support new "concepts" list AND legacy "concept" singular
    concepts_raw = intent_params.get("concepts") or []
    if not concepts_raw:
        legacy = (intent_params.get("concept") or "").strip()
        concepts_raw = [legacy] if legacy else []

    if not concepts_raw:
        return {
            "action_result": {
                "error": (
                    "No concept specified. Try asking about Mann-Whitney, "
                    "Mutual Information, Cramér's V, confidence score, "
                    "null signal, risk tag, or null group."
                )
            }
        }

    results = [_explain_one(c) for c in concepts_raw]

    return {
        "action_result": {
            "multi_concept": len(results) > 1,
            "results"      : results,
        }
    }


def _explain_one(concept_raw: str) -> dict:
    """Resolve and return explanation dict for a single concept string."""
    concept_raw = (concept_raw or "").strip()
    resolved_key = _resolve(concept_raw)

    if resolved_key is None:
        return {
            "concept"        : concept_raw,
            "resolved_key"   : None,
            "unknown_concept": True,
        }

    entry = CONCEPTS[resolved_key]
    return {
        "concept"        : concept_raw,
        "resolved_key"   : resolved_key,
        "name"           : entry["name"],
        "what"           : entry["what"],
        "how_used"       : entry["how_used"],
        "threshold"      : entry.get("threshold"),
        "example"        : entry["example"],
        "unknown_concept": False,
    }


# ── Helpers ───────────────────────────────────────────────────

def _resolve(concept: str) -> str | None:
    """
    Map a user-supplied concept string to a key in CONCEPTS.

    Tries, in order:
      1. Exact alias match (case-insensitive)
      2. Substring match against alias index
    Returns None if nothing found.
    """
    lower = concept.lower().strip()

    # Exact match
    if lower in _ALIAS_INDEX:
        return _ALIAS_INDEX[lower]

    # Substring: user said "explain the mann whitney test"
    for alias, key in _ALIAS_INDEX.items():
        if alias in lower or lower in alias:
            return key

    return None