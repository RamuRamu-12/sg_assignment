"""
models.py
---------
Centralises all LLM client initialisation and model-name constants.

Design decisions:
  - Clients are created lazily (on first call) to avoid import-time failures
    when API keys are not yet set.
  - All model names are defined as constants so changing a model only
    requires editing one line in this file.
  - Both clients expose a single unified interface: call_llm(prompt, model_name)
    that abstracts away SDK differences between Gemini and Groq.
"""

import os
import time
import json
from typing import Optional

from dotenv import load_dotenv

load_dotenv()  # loads GEMINI_API_KEY and GROQ_API_KEY from .env

# ── Model name constants ─────────────────────────────────────────────────────
MODEL_GEMINI = "gemini-2.5-flash-lite"
MODEL_GROQ   = "llama-3.3-70b-versatile"

# The model used as the LLM-judge inside the TAS evaluator metric.
# Using Gemini Flash-Lite here because it is free, fast, and deterministic
# at temperature=0, which is required for reproducible evaluation.
MODEL_JUDGE  = "gemini-2.5-flash-lite"

# Maximum tokens for generated emails — keeps outputs focused and concise.
MAX_TOKENS_EMAIL = 800
MAX_TOKENS_JUDGE = 200

# Retry settings for transient API errors (rate limits, 5xx responses).
MAX_RETRIES   = 3
RETRY_DELAY_S = 5   # seconds between retries (increases linearly)


# ── Lazy-loaded clients ──────────────────────────────────────────────────────
_gemini_client = None
_groq_client   = None


def _get_gemini_client():
    """Initialise and cache the Google Generative AI client."""
    global _gemini_client
    if _gemini_client is None:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. "
                "Get a free key at https://aistudio.google.com and add it to .env"
            )
        genai.configure(api_key=api_key)
        _gemini_client = genai
    return _gemini_client


def _get_groq_client():
    """Initialise and cache the Groq client."""
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Get a free key at https://console.groq.com and add it to .env"
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ── Unified LLM caller ───────────────────────────────────────────────────────

def call_llm(
    prompt: str,
    model_name: str,
    temperature: float = 0.7,
    max_tokens: int = MAX_TOKENS_EMAIL,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Route a prompt to the correct LLM backend and return the response text.

    Parameters
    ----------
    prompt       : The user-facing prompt content.
    model_name   : One of MODEL_GEMINI or MODEL_GROQ.
    temperature  : Sampling temperature. Use 0.0 for deterministic judge calls.
    max_tokens   : Maximum output tokens.
    system_prompt: Optional system-level instruction (role-playing persona).
                   If None, the role is already embedded in the prompt itself.

    Returns
    -------
    str : The raw text response from the LLM, stripped of leading/trailing whitespace.

    Raises
    ------
    ValueError        : If model_name is not recognised.
    RuntimeError      : If all retries are exhausted due to API errors.
    EnvironmentError  : If the required API key is missing.
    """
    if model_name == MODEL_GEMINI or model_name == MODEL_JUDGE:
        return _call_gemini(prompt, model_name, temperature, max_tokens, system_prompt)
    elif model_name == MODEL_GROQ:
        return _call_groq(prompt, model_name, temperature, max_tokens, system_prompt)
    else:
        raise ValueError(
            f"Unrecognised model_name: '{model_name}'. "
            f"Use '{MODEL_GEMINI}' or '{MODEL_GROQ}'."
        )


def _call_gemini(
    prompt: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    system_prompt: Optional[str],
) -> str:
    """Call the Google Gemini API with retry logic."""
    genai = _get_gemini_client()

    generation_config = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "candidate_count": 1,
    }

    # Gemini handles system instructions as a separate parameter
    model_kwargs = {"generation_config": generation_config}
    if system_prompt:
        model_kwargs["system_instruction"] = system_prompt

    model = genai.GenerativeModel(model_name=model_name, **model_kwargs)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Gemini API failed after {MAX_RETRIES} attempts: {exc}"
                ) from exc
            wait = RETRY_DELAY_S * attempt
            print(f"  [Gemini] Attempt {attempt} failed ({exc}). Retrying in {wait}s…")
            time.sleep(wait)


def _call_groq(
    prompt: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    system_prompt: Optional[str],
) -> str:
    """Call the Groq API with retry logic."""
    client = _get_groq_client()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Groq API failed after {MAX_RETRIES} attempts: {exc}"
                ) from exc
            wait = RETRY_DELAY_S * attempt
            print(f"  [Groq] Attempt {attempt} failed ({exc}). Retrying in {wait}s…")
            time.sleep(wait)


def call_judge_llm(prompt: str) -> dict:
    """
    Call the judge LLM for evaluation purposes (TAS metric).

    Always uses temperature=0 to ensure deterministic, reproducible scores.
    Parses and returns the JSON response as a Python dict.

    Returns
    -------
    dict with keys: 'score' (int 1-10) and 'reason' (str).
    Falls back to {'score': 5, 'reason': 'parse error'} on failure.
    """
    raw = call_llm(
        prompt=prompt,
        model_name=MODEL_JUDGE,
        temperature=0.0,
        max_tokens=MAX_TOKENS_JUDGE,
    )

    # Strip markdown code fences if the model wraps the JSON in ```json ... ```
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        result = json.loads(cleaned)
        # Normalise: score must be int in range 1–10
        score = max(1, min(10, int(result.get("score", 5))))
        reason = str(result.get("reason", "no reason provided"))
        return {"score": score, "reason": reason}
    except (json.JSONDecodeError, ValueError, TypeError):
        # If JSON parsing fails entirely, return a neutral fallback score
        return {"score": 5, "reason": f"JSON parse error — raw response: {raw[:100]}"}
