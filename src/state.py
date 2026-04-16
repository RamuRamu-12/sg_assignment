"""
Defines the central shared state (EmailState) that flows through every node
in the LangGraph agent.

Every field has a clear owner:
  - Inputs      : set once at graph entry, never mutated by nodes
  - Working     : updated progressively as nodes execute
  - Output      : written only when the graph is ready to terminate
"""

from typing import List
from typing_extensions import TypedDict


class EmailState(TypedDict):
    # ── Inputs (set once at graph.invoke(), never changed by nodes) ──────────
    intent: str
    """The core purpose of the email. E.g. 'Follow up after job interview'."""

    facts: List[str]
    """Bullet-point facts that MUST appear in the final email.
    E.g. ['Interview was on Monday', 'Role is Senior ML Engineer', ...]"""

    tone: str
    """Desired communication style. E.g. 'formal', 'persuasive', 'empathetic'."""

    model_name: str
    """Which LLM backend to use.
    Supported values:
      'gemini-2.5-flash-lite'      → Google Gemini API (free tier)
      'llama-3.1-70b-versatile'    → Groq API (free tier)
    """

    # ── Working fields (mutated by nodes as the graph executes) ─────────────
    draft_email: str
    """The most recent email draft produced by the EmailDrafter node.
    Empty string before the first drafting attempt."""

    attempts: int
    """Number of drafting attempts made so far.
    The graph allows a maximum of 2 attempts to prevent infinite refinement."""

    quality_issues: List[str]
    """Human-readable list of issues found by the QualityChecker node.
    E.g. ['Missing Fact 2: meeting date', 'No subject line found', 'No closing']
    Empty list means the draft passed all quality checks."""

    quality_passed: bool
    """True if the QualityChecker found zero issues. False otherwise."""

    refinement_prompt: str
    """The structured feedback prompt built by the Refiner node.
    Passed back to the EmailDrafter on the next iteration so the LLM knows
    exactly what was wrong and what to fix. Empty string on first attempt."""

    # ── Output (written when the graph is done) ──────────────────────────────
    final_email: str
    """The approved, ready-to-send email.
    Set when quality_passed=True OR when max attempts are exhausted."""

    validation_error: str
    """Set by InputValidator if the inputs are invalid.
    Non-empty string causes the graph to terminate early without any LLM call.
    Empty string means inputs are valid."""
