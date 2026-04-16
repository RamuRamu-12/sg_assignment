"""
The four node functions that make up the LangGraph Email Generation Agent.

Each function follows the LangGraph node contract:
  - Accepts the current EmailState dict as its only argument
  - Returns a dict containing ONLY the state keys it updates
    (LangGraph merges this partial update into the full state automatically)

Node responsibilities:
  1. input_validator   — Validates user inputs before any LLM call is made
  2. email_drafter     — Calls the LLM to produce (or refine) an email draft
  3. quality_checker   — Runs deterministic quality checks on the draft (no LLM)
  4. refiner           — Builds a targeted feedback prompt for the next draft attempt

Routing decisions are defined as separate functions at the bottom so they
can be imported cleanly into graph.py.
"""

import re
from typing import List

from src.state import EmailState
from src.models import call_llm, MODEL_GEMINI, MODEL_GROQ
from src.prompt_templates import (
    SYSTEM_PROMPT,
    build_draft_prompt,
    build_refinement_prompt,
)

# Maximum number of refinement attempts before accepting the best available draft.
# Set to 2 to allow one retry while preventing infinite loops.
MAX_ATTEMPTS = 2

# Minimum average semantic similarity score required for a draft to pass the
# fact-presence check inside QualityChecker. Set deliberately lower than the
# full FRS metric threshold so the agent only retries on clear fact omissions.
QUALITY_FACT_THRESHOLD = 0.60


# ── Node 1: InputValidator ───────────────────────────────────────────────────

def input_validator(state: EmailState) -> dict:
    """
    Gate-keep bad inputs before making any LLM API call.

    Checks performed:
      - intent is a non-empty, non-whitespace string
      - facts is a non-empty list with at least one non-empty string item
      - tone is a non-empty, non-whitespace string
      - No single fact exceeds 500 characters (catches accidental essay pastes)
      - model_name is one of the two supported values

    Returns
    -------
    dict with key 'validation_error':
      - Empty string  → all checks passed, continue to EmailDrafter
      - Error message → something is wrong, graph will terminate at END
    """
    intent     = (state.get("intent") or "").strip()
    facts      = state.get("facts") or []
    tone       = (state.get("tone") or "").strip()
    model_name = (state.get("model_name") or "").strip()

    # Check intent
    if not intent:
        return {"validation_error": "Intent cannot be empty. Please describe the purpose of the email."}

    # Check facts
    if not facts or not isinstance(facts, list):
        return {"validation_error": "Facts must be a non-empty list of strings."}

    non_empty_facts = [f for f in facts if isinstance(f, str) and f.strip()]
    if not non_empty_facts:
        return {"validation_error": "Facts list contains no valid (non-empty) items."}

    oversized = [f for f in non_empty_facts if len(f.strip()) > 500]
    if oversized:
        return {
            "validation_error": (
                f"One or more facts exceed 500 characters. "
                f"Please break them into shorter, specific bullet points."
            )
        }

    # Check tone
    if not tone:
        return {"validation_error": "Tone cannot be empty. Examples: formal, casual, persuasive, empathetic."}

    # Check model_name
    supported = {MODEL_GEMINI, MODEL_GROQ}
    if model_name not in supported:
        return {
            "validation_error": (
                f"Unsupported model_name: '{model_name}'. "
                f"Use one of: {sorted(supported)}"
            )
        }

    return {"validation_error": ""}


# ── Node 2: EmailDrafter ─────────────────────────────────────────────────────

def email_drafter(state: EmailState) -> dict:
    """
    Call the LLM to produce or refine an email draft.

    Behaviour:
      - On the FIRST attempt (attempts == 0 or refinement_prompt is empty):
        Uses the standard triple-layer draft prompt.
      - On SUBSEQUENT attempts (refinement_prompt is non-empty):
        Uses the targeted refinement prompt that includes the previous draft
        and a list of specific issues to fix.

    The node increments `attempts` and writes `draft_email`.

    Returns
    -------
    dict with keys: 'draft_email', 'attempts'
    """
    intent            = state["intent"]
    facts             = state["facts"]
    tone              = state["tone"]
    model_name        = state["model_name"]
    refinement_prompt = (state.get("refinement_prompt") or "").strip()
    attempts          = state.get("attempts", 0)

    # Choose prompt based on whether this is a retry
    if refinement_prompt:
        prompt = refinement_prompt
    else:
        prompt = build_draft_prompt(intent, facts, tone)

    # Call the LLM — temperature 0.7 for creativity, deterministic enough for quality
    draft = call_llm(
        prompt=prompt,
        model_name=model_name,
        temperature=0.7,
        system_prompt=SYSTEM_PROMPT,
    )

    return {
        "draft_email": draft,
        "attempts": attempts + 1,
    }


# ── Node 3: QualityChecker ───────────────────────────────────────────────────

def quality_checker(state: EmailState) -> dict:
    """
    Evaluate the current draft against four deterministic quality checks.

    No LLM call is made here. All checks are fast and local.

    Checks:
      1. Fact presence   — Each key fact must be semantically present in the draft.
                           Uses cosine similarity with sentence-transformers.
                           Threshold: QUALITY_FACT_THRESHOLD (0.60).
      2. Subject line    — Draft must contain a line starting with "Subject:".
      3. Greeting        — Draft must contain a greeting in the opening lines.
      4. Closing         — Draft must contain a professional closing near the end.

    Returns
    -------
    dict with keys: 'quality_issues' (list of str), 'quality_passed' (bool)
    """
    draft  = (state.get("draft_email") or "").strip()
    facts  = state.get("facts") or []
    issues = []

    if not draft:
        return {
            "quality_issues": ["Draft email is empty — no content was generated."],
            "quality_passed": False,
        }

    # ── Check 1: Semantic fact presence ─────────────────────────────────────
    missing_facts = _check_fact_presence(draft, facts)
    issues.extend(missing_facts)

    # ── Check 2: Subject line ────────────────────────────────────────────────
    if not _has_subject_line(draft):
        issues.append("No subject line found. Email must start with 'Subject: ...'")

    # ── Check 3: Greeting ────────────────────────────────────────────────────
    if not _has_greeting(draft):
        issues.append(
            "No greeting found. Email must include 'Dear', 'Hello', or 'Hi' near the top."
        )

    # ── Check 4: Professional closing ────────────────────────────────────────
    if not _has_closing(draft):
        issues.append(
            "No professional closing found. "
            "Email must end with 'Regards', 'Sincerely', 'Best', 'Warm regards', "
            "'Thank you', or similar."
        )

    quality_passed = len(issues) == 0
    return {
        "quality_issues": issues,
        "quality_passed": quality_passed,
    }


# ── Node 4: Refiner ──────────────────────────────────────────────────────────

def refiner(state: EmailState) -> dict:
    """
    Build a targeted feedback prompt that the EmailDrafter will use on its
    next attempt.

    The Refiner does NOT call the LLM — it only prepares the refinement prompt.
    This keeps the LLM-calling logic in one place (email_drafter) and makes
    each node single-responsibility and easy to unit-test.

    Returns
    -------
    dict with key 'refinement_prompt' (str)
    """
    draft_email    = state.get("draft_email", "")
    quality_issues = state.get("quality_issues", [])
    intent         = state["intent"]
    facts          = state["facts"]
    tone           = state["tone"]

    feedback_prompt = build_refinement_prompt(
        draft_email=draft_email,
        quality_issues=quality_issues,
        intent=intent,
        facts=facts,
        tone=tone,
    )

    return {"refinement_prompt": feedback_prompt}


# ── Helper functions for QualityChecker ─────────────────────────────────────

def _check_fact_presence(draft: str, facts: List[str]) -> List[str]:
    """
    Return a list of facts that are NOT semantically present in the draft.

    Uses sentence-transformers (all-MiniLM-L6-v2) to compute cosine similarity
    between each fact and the draft sentences. A fact is considered present if
    its max similarity to any sentence exceeds QUALITY_FACT_THRESHOLD.

    This is a lighter version of the full FRS metric used for fast in-graph
    quality gating (lower threshold, same technique).
    """
    try:
        from sentence_transformers import SentenceTransformer, util

        # Load model once per process; sentence-transformers caches it internally
        _model = SentenceTransformer("all-MiniLM-L6-v2")

        # Split draft into sentences for granular comparison
        draft_sentences = [s.strip() for s in re.split(r"[.!?\n]", draft) if s.strip()]
        if not draft_sentences:
            return [f"Fact missing (draft has no valid sentences): '{f}'" for f in facts]

        missing = []
        for fact in facts:
            fact_stripped = fact.strip()
            if not fact_stripped:
                continue

            fact_embedding     = _model.encode(fact_stripped, convert_to_tensor=True)
            sentence_embeddings = _model.encode(draft_sentences, convert_to_tensor=True)
            similarities       = util.cos_sim(fact_embedding, sentence_embeddings)[0]
            max_sim            = float(similarities.max())

            if max_sim < QUALITY_FACT_THRESHOLD:
                missing.append(
                    f"Fact not clearly present (similarity={max_sim:.2f}): '{fact_stripped}'"
                )

        return missing

    except ImportError:
        # If sentence-transformers is not installed, skip this check gracefully
        # rather than crashing the entire agent.
        return []
    except Exception:
        # Any other error (e.g. model download failure on first run) — skip gracefully
        return []


def _has_subject_line(draft: str) -> bool:
    """
    Check whether the draft contains a Subject line.
    Accepts 'Subject:', 'SUBJECT:', 'subject:' (case-insensitive).
    """
    return bool(re.search(r"(?i)^\s*subject\s*:", draft, re.MULTILINE))


def _has_greeting(draft: str) -> bool:
    """
    Check whether the draft contains a greeting in the first 10 lines.
    Accepts: Dear, Hello, Hi, Good morning/afternoon/evening.
    """
    first_lines = "\n".join(draft.splitlines()[:10])
    return bool(re.search(
        r"(?i)\b(dear|hello|hi\b|good\s+(morning|afternoon|evening))\b",
        first_lines
    ))


def _has_closing(draft: str) -> bool:
    """
    Check whether the draft contains a professional closing in the last 8 lines.
    Accepts: Regards, Sincerely, Best, Warm regards, Thank you, Yours, Cheers.
    """
    last_lines = "\n".join(draft.splitlines()[-8:])
    return bool(re.search(
        r"(?i)\b(regards|sincerely|best|warm regards|thank you|yours|cheers|faithfully)\b",
        last_lines
    ))


# ── Routing functions for conditional edges ──────────────────────────────────

def route_after_validation(state: EmailState) -> str:
    """
    Conditional edge after InputValidator.
    Returns 'valid' if inputs passed, 'invalid' if they did not.
    """
    if state.get("validation_error", ""):
        return "invalid"
    return "valid"


def route_after_quality(state: EmailState) -> str:
    """
    Conditional edge after QualityChecker.

    Returns:
      'pass'     → quality passed; set final_email and terminate
      'refine'   → quality failed but we still have attempts left; go to Refiner
      'give_up'  → quality failed and max attempts reached; accept best draft
    """
    quality_passed = state.get("quality_passed", False)
    attempts       = state.get("attempts", 0)

    if quality_passed:
        return "pass"
    elif attempts < MAX_ATTEMPTS:
        return "refine"
    else:
        return "give_up"
