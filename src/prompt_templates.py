"""
All prompt templates used by the Email Generation Agent.

Three prompting techniques are combined (Triple-Layer Prompting):

  1. ROLE-PLAYING
     The LLM is assigned a specific expert persona before any task is given.
     This anchors the model's output style and prevents tone drift toward
     generic or casual language.

  2. FEW-SHOT EXAMPLES
     Two fully-worked input→output examples are embedded in the prompt.
     This shows the model the exact format expected (Subject line, greeting,
     body paragraphs, professional closing) without having to describe it
     in abstract rules. The model learns format by demonstration.

  3. CHAIN-OF-THOUGHT (CoT)
     The user section instructs the model to reason step-by-step BEFORE
     writing the email. This explicit reasoning phase reduces fact omission
     (the model consciously maps each fact to a sentence before drafting)
     and reduces hallucination (it commits facts before writing).

Why this combination specifically?
  - Role-Playing alone → good tone, but unreliable fact inclusion
  - Few-Shot alone → good format, but tone can drift across scenarios
  - CoT alone → careful but verbose; may over-explain
  - All three together → each layer compensates for the other's weakness

Functions exported:
  build_draft_prompt(intent, facts, tone)     → str
  build_refinement_prompt(draft, issues, intent, facts, tone) → str
  build_judge_prompt(email, tone)             → str
  SYSTEM_PROMPT                               → str (shared role text)
"""

from typing import List

# ── Shared system prompt (Role-Playing layer) ────────────────────────────────
# This is passed as the system instruction to both Gemini and Groq.
# It defines the persona, the non-negotiable rules, and the output contract.
SYSTEM_PROMPT = """You are a world-class business communication specialist with 20 years of \
experience writing professional emails for Fortune 500 companies, government agencies, and \
high-growth startups. You have written thousands of emails spanning sales outreach, executive \
communication, client relationship management, and internal leadership.

Your writing is:
  - Precise: every sentence has a purpose; no filler words
  - Tone-perfect: you calibrate language exactly to the requested tone
  - Fact-faithful: you include every provided fact and NEVER invent new ones
  - Structurally complete: every email has a Subject line, an appropriate greeting,
    well-formed body paragraphs, and a professional closing with a placeholder signature

Your absolute rules:
  1. ALWAYS start with a Subject line in the format:  Subject: <subject text>
  2. ALWAYS include a greeting (Dear / Hello / Hi [Name or Title])
  3. ALWAYS end with a closing (Regards / Sincerely / Best / Warm regards / Thank you)
     followed by a signature placeholder: [Your Name] / [Your Title] / [Your Company]
  4. NEVER add facts, names, numbers, or details that were not explicitly provided
  5. NEVER use hollow filler phrases like "I hope this email finds you well" unless
     the tone specifically calls for warmth
  6. Keep emails between 120 and 300 words unless the content genuinely requires more"""


# ── Few-shot examples (embedded inside the draft prompt) ────────────────────
# Two carefully chosen examples that demonstrate:
#   Example 1: Formal tone — structured, restrained, clear call-to-action
#   Example 2: Casual/warm tone — friendly, personal, conversational
# Both show the complete CoT reasoning trace so the model sees the pattern.

_FEW_SHOT_EXAMPLES = """
════════════════════════════════════════════════════════════
EXAMPLE 1  (Formal — Post-Interview Follow-Up)
════════════════════════════════════════════════════════════
INPUT
  Intent : Follow up after a job interview
  Facts  :
    - The interview took place on Tuesday, April 8th
    - The interviewer's name is Mr. James Parker
    - The role is Senior Data Scientist
    - The conversation covered a machine-learning pipeline project
    - The applicant is very interested in the position
  Tone   : Formal

THINKING
  Step 1 — Primary goal: Express sincere gratitude, reference the specific conversation,
           and reaffirm the applicant's interest without sounding desperate.
  Step 2 — Structure: Subject line → formal salutation → short thank-you paragraph
           (facts: interview date, interviewer name, role) → second paragraph (ML pipeline
           discussion, continued strong interest) → professional closing with CTA.
  Step 3 — Fact mapping:
    • "Tuesday, April 8th" + "James Parker" + "Senior Data Scientist" → Para 1
    • "ML pipeline project" + "very interested" → Para 2
  Step 4 — Write the email.

OUTPUT
Subject: Follow-Up – Senior Data Scientist Interview – April 8th

Dear Mr. Parker,

Thank you for taking the time to meet with me on Tuesday, April 8th, to discuss the Senior \
Data Scientist position. I found our conversation both insightful and energising, and I am \
grateful for the opportunity to learn more about your team's work.

In particular, our discussion around the machine-learning pipeline project reinforced my \
enthusiasm for this role. I believe my background aligns closely with the challenges you \
described, and I remain very interested in contributing to your organisation.

Please do not hesitate to reach out should you require any further information. I look \
forward to hearing from you.

Yours sincerely,
[Your Name]
[Your Title]
[Your Contact Information]

════════════════════════════════════════════════════════════
EXAMPLE 2  (Warm/Casual — Thank You After Networking Event)
════════════════════════════════════════════════════════════
INPUT
  Intent : Thank someone after meeting at a networking event
  Facts  :
    - Met at the TechConnect Summit in Austin on Friday
    - The person's name is Priya Sharma
    - Priya works as a Product Manager at Notion
    - They discussed AI-powered productivity tools
    - The applicant wants to stay in touch and possibly collaborate
  Tone   : Warm and casual

THINKING
  Step 1 — Primary goal: Reinforce the personal connection, reference the specific event
           and conversation, and leave a door open for future collaboration.
  Step 2 — Structure: Casual subject line → friendly first-name greeting → warm opening
           referencing the event (not "I hope this email finds you well") → mention
           the discussion topic → express interest in staying connected → light closing.
  Step 3 — Fact mapping:
    • "TechConnect Summit", "Austin", "Friday" → opening line
    • "Priya Sharma", "Product Manager at Notion" → greeting + para 1
    • "AI-powered productivity tools" → para 1 connection point
    • "stay in touch, possibly collaborate" → closing thought
  Step 4 — Write the email.

OUTPUT
Subject: Great Connecting at TechConnect Austin!

Hi Priya,

It was such a pleasure meeting you at the TechConnect Summit in Austin on Friday! \
Our conversation about AI-powered productivity tools was one of the highlights of \
the event for me — you shared some genuinely fascinating perspectives from your work \
at Notion.

I would love to stay in touch and explore whether there are ways we might collaborate \
down the line. I have a few ideas I think could be interesting given what you described \
— happy to share them over a coffee call whenever works for you.

Thanks again for the great chat, and hope the rest of the summit went well!

Warm regards,
[Your Name]
[Your LinkedIn / Contact]
════════════════════════════════════════════════════════════
"""


# ── Draft prompt builder ─────────────────────────────────────────────────────

def build_draft_prompt(intent: str, facts: List[str], tone: str) -> str:
    """
    Build the full Triple-Layer prompt for first-draft email generation.

    The prompt contains all three technique layers:
      - Role-Playing : embedded as a preamble reminding the model of its persona
      - Few-Shot     : two worked examples showing format + CoT reasoning
      - Chain-of-Thought : explicit Step 1–4 reasoning instructions

    Parameters
    ----------
    intent  : Core purpose of the email.
    facts   : List of specific facts that must appear in the email.
    tone    : Desired communication tone.

    Returns
    -------
    str : The complete user-facing prompt (system prompt is passed separately).
    """
    facts_formatted = "\n".join(f"  - {f.strip()}" for f in facts)

    prompt = f"""You are an expert business communication specialist. \
Review the two examples below carefully — they show you the exact thinking process \
and output format required.

{_FEW_SHOT_EXAMPLES}

════════════════════════════════════════════════════════════
YOUR TASK  (follow the same thinking + output format as the examples above)
════════════════════════════════════════════════════════════
INPUT
  Intent : {intent}
  Facts  :
{facts_formatted}
  Tone   : {tone}

THINKING
  Work through Steps 1–4 explicitly before writing the email.
  Step 1 — What is the single primary goal of this email?
  Step 2 — What structure best fits this tone and intent?
           (decide: subject line style, number of paragraphs, need for CTA)
  Step 3 — Map EACH fact above to a specific sentence or paragraph.
           Every fact must be accounted for. Do not omit any.
  Step 4 — Write the complete, ready-to-send email.

OUTPUT
[Write the complete email here, starting with "Subject:"]"""

    return prompt


# ── Refinement prompt builder ────────────────────────────────────────────────

def build_refinement_prompt(
    draft_email: str,
    quality_issues: List[str],
    intent: str,
    facts: List[str],
    tone: str,
) -> str:
    """
    Build the refinement prompt used when QualityChecker finds issues.

    The refinement prompt includes:
      - The original draft (so the model does not rewrite from scratch)
      - A numbered list of specific issues to fix
      - Clear constraints: fix ONLY the listed issues, keep everything else

    Parameters
    ----------
    draft_email    : The previous draft that failed quality checks.
    quality_issues : List of specific issues identified by QualityChecker.
    intent         : Original email intent (context for the LLM).
    facts          : Original fact list (context for the LLM).
    tone           : Original tone (context for the LLM).

    Returns
    -------
    str : The refinement prompt to be passed to EmailDrafter on retry.
    """
    issues_formatted = "\n".join(f"  {i+1}. {issue}" for i, issue in enumerate(quality_issues))
    facts_formatted  = "\n".join(f"  - {f.strip()}" for f in facts)

    prompt = f"""You previously wrote the following email draft for this task:

  Intent : {intent}
  Facts  :
{facts_formatted}
  Tone   : {tone}

══════════════════════
PREVIOUS DRAFT
══════════════════════
{draft_email}
══════════════════════

The draft has the following quality issues that MUST be fixed in your rewrite:

{issues_formatted}

INSTRUCTIONS FOR YOUR REWRITE:
  1. Fix every issue listed above — do not skip any.
  2. Keep all correctly included facts exactly as they are.
  3. Keep the same tone: {tone}.
  4. Do NOT add any new facts that were not in the original fact list.
  5. Ensure the final email has:
       - A Subject line (format: "Subject: ...")
       - A proper greeting
       - Well-formed body paragraphs
       - A professional closing and signature placeholder

Write only the corrected email — no preamble, no explanation, just the email starting \
with "Subject:"."""

    return prompt


# ── Judge prompt builder (for TAS metric) ───────────────────────────────────

def build_judge_prompt(email: str, tone: str) -> str:
    """
    Build the evaluation prompt for the LLM-as-a-Judge (Tone Alignment Score).

    The judge is instructed to:
      - Focus specifically on tone match, not grammar or content
      - Respond in strict JSON format (no prose, no markdown fences)
      - Use temperature=0 (set in models.py) for reproducible scoring

    Parameters
    ----------
    email : The generated email to evaluate.
    tone  : The requested tone that was passed to the generator.

    Returns
    -------
    str : The judge prompt string.
    """
    prompt = f"""You are an expert evaluator of professional business writing.

Your task: Rate how well the email below matches the requested tone.

Requested tone: "{tone}"

Email to evaluate:
══════════════════════════════════════
{email}
══════════════════════════════════════

Evaluation criteria for tone alignment:
  10 — The tone is a perfect, unmistakable match. Every sentence, word choice,
       and level of formality is exactly right for "{tone}".
  7-9 — Strong match. The overall tone is correct with only minor deviations.
  4-6 — Partial match. The tone is approximately right but inconsistent in places.
  1-3 — Poor match. The email's tone is noticeably different from "{tone}".

Respond ONLY as valid JSON with exactly two keys — no markdown, no extra text:
{{"score": <integer 1 to 10>, "reason": "<one concise sentence explaining the score>"}}"""

    return prompt
