"""
Assembles the LangGraph StateGraph for the Email Generation Agent.

Graph topology:
                      ┌──────────────────┐
              START → │  InputValidator  │
                      └────────┬─────────┘
                               │
              ┌────────────────┴────────────────┐
        VALID │                                 │ INVALID
              ▼                                 ▼
      ┌──────────────┐                        END
      │ EmailDrafter │◄────────────────────┐
      └──────┬───────┘                     │
             │ (always)                    │
             ▼                             │
     ┌───────────────┐                     │
     │ QualityChecker│                     │
     └───────┬───────┘                     │
             │                             │
    ┌────────┴─────────────┐               │
    │ PASS                 │ FAIL          │
    ▼                      ▼  (attempts    │
   END           ┌──────────────┐ < MAX)   │
  (final_email   │   Refiner    │──────────┘
   = draft)      └──────────────┘
                      │ GIVE_UP (attempts >= MAX)
                      ▼
                     END
                  (final_email = best draft)

The graph is compiled once and reused across all invocations.
Thread-safe: each invoke() call gets an isolated state copy.

"""

from langgraph.graph import StateGraph, END

from src.state import EmailState
from src.nodes import (
    input_validator,
    email_drafter,
    quality_checker,
    refiner,
    route_after_validation,
    route_after_quality,
)

# ── Terminal node wrappers ───────────────────────────────────────────────────
# These small wrapper nodes handle the final state write before the graph ends.
# Separating terminal logic into dedicated nodes keeps the routing functions
# pure (they only decide direction, never mutate state).

def _finalize_success(state: EmailState) -> dict:
    """Called when QualityChecker passes. Sets final_email from the approved draft."""
    return {"final_email": state.get("draft_email", "")}


def _finalize_give_up(state: EmailState) -> dict:
    """
    Called when max attempts are exhausted without passing quality.
    Returns the best available draft rather than an empty result.
    """
    return {"final_email": state.get("draft_email", "")}


def _finalize_invalid(state: EmailState) -> dict:
    """Called when InputValidator rejects the inputs. final_email stays empty."""
    return {"final_email": ""}


# ── Graph factory ────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    """
    Build and compile the LangGraph StateGraph.

    Called once at module import time. The compiled graph object is cached
    in the module-level variable `_compiled_graph`.
    """
    graph = StateGraph(EmailState)

    # ── Register nodes ───────────────────────────────────────────────────────
    graph.add_node("InputValidator",   input_validator)
    graph.add_node("EmailDrafter",     email_drafter)
    graph.add_node("QualityChecker",   quality_checker)
    graph.add_node("Refiner",          refiner)
    graph.add_node("FinalizeSuccess",  _finalize_success)
    graph.add_node("FinalizeGiveUp",   _finalize_give_up)
    graph.add_node("FinalizeInvalid",  _finalize_invalid)

    # ── Entry point ──────────────────────────────────────────────────────────
    graph.set_entry_point("InputValidator")

    # ── Edges ────────────────────────────────────────────────────────────────

    # After validation: branch on valid/invalid
    graph.add_conditional_edges(
        "InputValidator",
        route_after_validation,
        {
            "valid":   "EmailDrafter",
            "invalid": "FinalizeInvalid",
        },
    )

    # After drafting: always move to quality check
    graph.add_edge("EmailDrafter", "QualityChecker")

    # After quality check: branch on pass / refine / give_up
    graph.add_conditional_edges(
        "QualityChecker",
        route_after_quality,
        {
            "pass":     "FinalizeSuccess",
            "refine":   "Refiner",
            "give_up":  "FinalizeGiveUp",
        },
    )

    # Refiner always feeds back into EmailDrafter
    graph.add_edge("Refiner", "EmailDrafter")

    # Terminal nodes all go to END
    graph.add_edge("FinalizeSuccess", END)
    graph.add_edge("FinalizeGiveUp",  END)
    graph.add_edge("FinalizeInvalid", END)

    return graph.compile()


# Compile once, reuse everywhere
_compiled_graph = _build_graph()


# ── Public interface ─────────────────────────────────────────────────────────

def run_agent(
    intent: str,
    facts: list,
    tone: str,
    model_name: str,
) -> dict:
    """
    Run the Email Generation Agent for a single request.

    Parameters
    ----------
    intent     : Core purpose of the email.
    facts      : List of key facts that must appear in the final email.
    tone       : Desired communication tone.
    model_name : 'gemini-2.5-flash-lite' or 'llama-3.3-70b-versatile'.

    Returns
    -------
    dict — the final EmailState after the graph completes, containing:
      final_email      : The generated email (empty if validation failed)
      attempts         : Number of LLM calls made
      quality_passed   : Whether quality checks were satisfied
      quality_issues   : List of issues found (empty if quality passed)
      validation_error : Non-empty string if inputs were invalid
    """
    initial_state: EmailState = {
        "intent":            intent,
        "facts":             facts,
        "tone":              tone,
        "model_name":        model_name,
        "draft_email":       "",
        "attempts":          0,
        "quality_issues":    [],
        "quality_passed":    False,
        "refinement_prompt": "",
        "final_email":       "",
        "validation_error":  "",
    }

    result = _compiled_graph.invoke(initial_state)
    return result
