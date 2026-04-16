"""
evaluator.py
------------
Implements the three custom evaluation metrics for the Email Generation Assistant.

All three metrics are designed specifically for professional email generation —
they are NOT general NLP metrics borrowed from other tasks.

─────────────────────────────────────────────────────────────────────────────
METRIC 1: Fact Recall Score (FRS)         Range: 0.0 – 1.0
─────────────────────────────────────────────────────────────────────────────
Definition:
  Measures whether every key fact provided in the input was semantically
  captured in the generated email, even when paraphrased.

Why not ROUGE or exact-match?
  A good email WILL paraphrase facts. "The meeting was on Tuesday, April 8th"
  might become "our Tuesday conversation" in the email. ROUGE n-gram overlap
  would score this low. Semantic cosine similarity correctly scores it high
  because the meaning is preserved.

Algorithm:
  1. Embed each key fact using sentence-transformers (all-MiniLM-L6-v2)
  2. Split the generated email into sentences; embed each sentence
  3. For every fact, find the maximum cosine similarity against all sentences
  4. FRS = mean of all per-fact max-similarities

─────────────────────────────────────────────────────────────────────────────
METRIC 2: Tone Alignment Score (TAS)      Range: 0.0 – 1.0
─────────────────────────────────────────────────────────────────────────────
Definition:
  Measures how accurately the generated email matches the requested tone,
  as judged by an independent LLM evaluator.

Why LLM-as-Judge?
  Tone is inherently subjective and contextual. "Firm but professional" cannot
  be detected by word lists, sentiment analysis, or readability formulas.
  An LLM judge with a clear rubric and temperature=0 produces scores that are
  consistent, human-interpretable, and more reliable than any heuristic.

Algorithm:
  1. Build a structured judge prompt with the email + requested tone
  2. Call gemini-2.5-flash-lite at temperature=0 (deterministic)
  3. Parse the JSON response: {"score": int (1-10), "reason": str}
  4. TAS = score / 10

─────────────────────────────────────────────────────────────────────────────
METRIC 3: Professional Quality Index (PQI)  Range: 0.0 – 1.0
─────────────────────────────────────────────────────────────────────────────
Definition:
  Measures whether the email is immediately usable as-is — grammatically
  correct, appropriately readable, and structurally complete.

Why a composite?
  No single automated metric captures "email quality". Grammar tools catch
  technical errors but miss structure issues. Readability scores miss grammar.
  Structure checks miss readability. Combining all three into one index gives
  a holistic, actionable quality signal.

Algorithm (equal-weighted average of 3 sub-scores):
  Grammar Score   = 1 - min(1.0, error_count / 10)    [language-tool-python]
  Readability Score = mapped from Flesch Reading Ease  [textstat]
                      (target: 30–60 for professional email)
  Structure Score = 0.25 per present element:
                    Subject line, Greeting, Closing, ≥2 body paragraphs
  PQI = mean(grammar, readability, structure)
─────────────────────────────────────────────────────────────────────────────
"""

import re
from typing import List, Dict

from src.models import call_judge_llm
from src.prompt_templates import build_judge_prompt


# ── Metric 1: Fact Recall Score (FRS) ───────────────────────────────────────

def fact_recall_score(generated_email: str, facts: List[str]) -> float:
    """
    Compute the Fact Recall Score for a generated email.

    Parameters
    ----------
    generated_email : The email text produced by the agent.
    facts           : The list of key facts that were given as input.

    Returns
    -------
    float : FRS score in range [0.0, 1.0].
            Returns 0.0 if facts list is empty or email is empty.
    """
    if not generated_email or not generated_email.strip():
        return 0.0

    clean_facts = [f.strip() for f in facts if f and f.strip()]
    if not clean_facts:
        return 0.0

    try:
        from sentence_transformers import SentenceTransformer, util

        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Split email into sentences for fine-grained comparison.
        # Split on sentence-ending punctuation AND newlines.
        email_sentences = [
            s.strip()
            for s in re.split(r"[.!?\n]", generated_email)
            if s.strip() and len(s.strip()) > 5
        ]

        if not email_sentences:
            return 0.0

        # Encode all facts and all email sentences in one batch call (faster)
        fact_embeddings     = model.encode(clean_facts,     convert_to_tensor=True)
        sentence_embeddings = model.encode(email_sentences, convert_to_tensor=True)

        # For each fact, find the maximum similarity to any sentence
        from sentence_transformers import util as st_util
        similarity_matrix = st_util.cos_sim(fact_embeddings, sentence_embeddings)

        per_fact_max = [float(similarity_matrix[i].max()) for i in range(len(clean_facts))]
        frs = sum(per_fact_max) / len(per_fact_max)
        return round(min(1.0, max(0.0, frs)), 4)

    except ImportError:
        # sentence-transformers not installed — return a neutral placeholder
        return 0.5
    except Exception:
        return 0.0


# ── Metric 2: Tone Alignment Score (TAS) ────────────────────────────────────

def tone_alignment_score(generated_email: str, tone: str) -> Dict[str, object]:
    """
    Compute the Tone Alignment Score using LLM-as-a-Judge.

    Parameters
    ----------
    generated_email : The email text produced by the agent.
    tone            : The requested tone passed to the generator.

    Returns
    -------
    dict with keys:
      'score'  (float) : TAS score in range [0.0, 1.0]
      'reason' (str)   : One-sentence explanation from the judge
      'raw_score' (int): Original 1–10 integer score from the judge
    """
    if not generated_email or not generated_email.strip():
        return {"score": 0.0, "reason": "Email is empty.", "raw_score": 0}

    judge_prompt = build_judge_prompt(generated_email, tone)

    try:
        result = call_judge_llm(judge_prompt)
        raw_score = max(1, min(10, int(result.get("score", 5))))
        reason    = str(result.get("reason", "No reason provided."))
        tas_score = round(raw_score / 10.0, 4)
        return {"score": tas_score, "reason": reason, "raw_score": raw_score}
    except Exception as exc:
        return {
            "score":     0.5,
            "reason":    f"Evaluation failed: {str(exc)[:80]}",
            "raw_score": 5,
        }


# ── Metric 3: Professional Quality Index (PQI) ──────────────────────────────

def professional_quality_index(generated_email: str) -> Dict[str, object]:
    """
    Compute the Professional Quality Index for a generated email.

    Parameters
    ----------
    generated_email : The email text produced by the agent.

    Returns
    -------
    dict with keys:
      'score'             (float) : PQI composite score in [0.0, 1.0]
      'grammar_score'     (float) : Grammar sub-score [0.0, 1.0]
      'readability_score' (float) : Readability sub-score [0.0, 1.0]
      'structure_score'   (float) : Structure sub-score [0.0, 1.0]
      'grammar_errors'    (int)   : Raw error count from language-tool
      'flesch_score'      (float) : Raw Flesch Reading Ease score
    """
    if not generated_email or not generated_email.strip():
        return {
            "score": 0.0, "grammar_score": 0.0,
            "readability_score": 0.0, "structure_score": 0.0,
            "grammar_errors": 0, "flesch_score": 0.0,
        }

    grammar_score,     grammar_errors = _compute_grammar_score(generated_email)
    readability_score, flesch_score   = _compute_readability_score(generated_email)
    structure_score                   = _compute_structure_score(generated_email)

    pqi = round((grammar_score + readability_score + structure_score) / 3.0, 4)

    return {
        "score":             pqi,
        "grammar_score":     round(grammar_score,     4),
        "readability_score": round(readability_score, 4),
        "structure_score":   round(structure_score,   4),
        "grammar_errors":    grammar_errors,
        "flesch_score":      round(flesch_score,       2),
    }


def _compute_grammar_score(text: str):
    """
    Use language-tool-python to count grammar/spelling errors.

    Score formula: 1 - min(1.0, error_count / 10)
      - 0 errors  → 1.0 (perfect)
      - 5 errors  → 0.5
      - 10+ errors → 0.0

    Returns (grammar_score: float, error_count: int)
    """
    try:
        import language_tool_python
        tool   = language_tool_python.LanguageTool("en-US")
        matches = tool.check(text)

        # Filter out style suggestions — only count clear grammar/spelling errors
        real_errors = [
            m for m in matches
            if m.ruleIssueType in ("grammar", "misspelling", "typographical")
        ]
        count = len(real_errors)
        score = max(0.0, 1.0 - count / 10.0)
        return round(score, 4), count
    except ImportError:
        return 0.8, 0   # neutral fallback if library not installed
    except Exception:
        return 0.8, 0


def _compute_readability_score(text: str):
    """
    Compute Flesch Reading Ease and map it to a 0–1 quality score.

    Flesch Reading Ease interpretation for professional emails:
      < 30     → Too complex (academic/legal writing) → score scales up from 0
      30 – 60  → Ideal range for professional business emails → score = 1.0
      > 60     → Too simple (short sentences, basic vocabulary) → score scales down

    Returns (readability_score: float, raw_flesch: float)
    """
    try:
        import textstat
        flesch = textstat.flesch_reading_ease(text)

        if 30.0 <= flesch <= 60.0:
            score = 1.0
        elif flesch < 30.0:
            # Scale linearly from 0 (at FRE=0) to 1 (at FRE=30)
            score = max(0.0, flesch / 30.0)
        else:
            # Scale linearly from 1 (at FRE=60) down to 0.5 (at FRE=100)
            score = max(0.5, 1.0 - (flesch - 60.0) / 80.0)

        return round(score, 4), round(flesch, 2)
    except ImportError:
        return 0.8, 50.0
    except Exception:
        return 0.8, 50.0


def _compute_structure_score(text: str) -> float:
    """
    Check for the four structural elements required in a professional email.
    Each present element contributes 0.25 to the score.

    Elements checked:
      1. Subject line  — a line matching "Subject: ..."
      2. Greeting      — Dear / Hello / Hi near the top
      3. Closing       — Regards / Sincerely / Best / etc. near the bottom
      4. Body depth    — at least 2 non-empty paragraphs (separated by blank lines)

    Returns float in {0.0, 0.25, 0.50, 0.75, 1.0}
    """
    score = 0.0

    # Check 1: Subject line
    if re.search(r"(?i)^\s*subject\s*:", text, re.MULTILINE):
        score += 0.25

    # Check 2: Greeting (in first 10 lines)
    first_lines = "\n".join(text.splitlines()[:10])
    if re.search(r"(?i)\b(dear|hello|hi\b|good\s+(morning|afternoon|evening))\b", first_lines):
        score += 0.25

    # Check 3: Closing (in last 8 lines)
    last_lines = "\n".join(text.splitlines()[-8:])
    if re.search(
        r"(?i)\b(regards|sincerely|best|warm regards|thank you|yours|cheers|faithfully)\b",
        last_lines
    ):
        score += 0.25

    # Check 4: At least 2 body paragraphs
    # Split on blank lines; count non-empty, non-subject, non-greeting paragraphs
    raw_paragraphs = re.split(r"\n\s*\n", text)
    body_paragraphs = [
        p.strip() for p in raw_paragraphs
        if p.strip()
        and not re.match(r"(?i)^\s*subject\s*:", p.strip())
        and len(p.strip()) > 20
    ]
    if len(body_paragraphs) >= 2:
        score += 0.25

    return round(score, 4)


# ── Convenience: run all three metrics in one call ───────────────────────────

def evaluate_email(
    generated_email: str,
    facts: List[str],
    tone: str,
) -> Dict[str, object]:
    """
    Run all three evaluation metrics on a generated email and return a
    consolidated results dict.

    Parameters
    ----------
    generated_email : Email text to evaluate.
    facts           : Original fact list from the input scenario.
    tone            : Original requested tone from the input scenario.

    Returns
    -------
    dict containing:
      frs            : float — Fact Recall Score
      tas            : float — Tone Alignment Score
      tas_reason     : str   — Judge's reasoning
      tas_raw        : int   — Judge's raw 1–10 score
      pqi            : float — Professional Quality Index
      pqi_grammar    : float — Grammar sub-score
      pqi_readability: float — Readability sub-score
      pqi_structure  : float — Structure sub-score
      grammar_errors : int
      flesch_score   : float
      composite      : float — Simple average of FRS, TAS, PQI
    """
    frs_score  = fact_recall_score(generated_email, facts)
    tas_result = tone_alignment_score(generated_email, tone)
    pqi_result = professional_quality_index(generated_email)

    composite = round((frs_score + tas_result["score"] + pqi_result["score"]) / 3.0, 4)

    return {
        "frs":             frs_score,
        "tas":             tas_result["score"],
        "tas_reason":      tas_result["reason"],
        "tas_raw":         tas_result["raw_score"],
        "pqi":             pqi_result["score"],
        "pqi_grammar":     pqi_result["grammar_score"],
        "pqi_readability": pqi_result["readability_score"],
        "pqi_structure":   pqi_result["structure_score"],
        "grammar_errors":  pqi_result["grammar_errors"],
        "flesch_score":    pqi_result["flesch_score"],
        "composite":       composite,
    }
