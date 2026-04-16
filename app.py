"""
Streamlit web application for the Email Generation Agent.

Features:
  - Input panel: Intent, Key Facts, Tone (8 presets + custom)
  - Model selector: Model A, Model B, or Both side by side
  - Output panel: Generated email(s) with visual metric scores
  - Agent transparency: Shows number of drafting attempts
  - Quality issues display: Explains what the agent had to fix
  - Full evaluation runner: Runs all 10 scenarios and provides CSV/JSON download

Run with:
  streamlit run app.py
"""

import html
import json
import os
import time
from pathlib import Path

import pandas as pd

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Email Generation Agent",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { color: #e0e0e0; margin: 0; font-size: 2rem; }
    .main-header p  { color: #a0aec0; margin: 0.5rem 0 0 0; font-size: 0.95rem; }

    /* Email display box */
    .email-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.5rem;
        font-family: 'Georgia', serif;
        font-size: 0.92rem;
        line-height: 1.7;
        white-space: pre-wrap;
        color: #2d3748;
        min-height: 300px;
    }

    /* Metric card */
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
    }
    .metric-label { font-weight: 600; font-size: 0.85rem; color: #4a5568; }
    .metric-score { font-size: 1.1rem; font-weight: 700; }

    /* Badge styles */
    .badge-green  { background:#c6f6d5; color:#276749; padding:3px 10px; border-radius:999px; font-size:0.8rem; font-weight:600; }
    .badge-yellow { background:#fefcbf; color:#744210; padding:3px 10px; border-radius:999px; font-size:0.8rem; font-weight:600; }
    .badge-red    { background:#fed7d7; color:#742a2a; padding:3px 10px; border-radius:999px; font-size:0.8rem; font-weight:600; }
    .badge-blue   { background:#bee3f8; color:#2a4365; padding:3px 10px; border-radius:999px; font-size:0.8rem; font-weight:600; }

    /* Section divider */
    .section-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #718096;
        margin: 1rem 0 0.4rem 0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_A_NAME = "gemini-2.5-flash-lite"
MODEL_B_NAME = "llama-3.3-70b-versatile"

TONE_PRESETS = [
    "Formal",
    "Persuasive and Professional",
    "Apologetic and Professional",
    "Warm and Casual",
    "Firm but Professional",
    "Empathetic",
    "Energetic and Persuasive",
    "Neutral and Informative",
    "Respectful and Concise",
    "Custom (type below)",
]

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>✉️ Email Generation Agent</h1>
  <p>
    Powered by <strong>LangGraph</strong> · Triple-Layer Prompting (Role + Few-Shot + Chain-of-Thought)
    · Self-Refinement Loop · Custom Evaluation Metrics
  </p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar: Configuration ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    st.markdown("**Model Selection**")
    model_choice = st.radio(
        label="Select model(s) to use:",
        options=["Model A only", "Model B only", "Both (side by side)"],
        index=2,
        help=(
            f"Model A: {MODEL_A_NAME} (Google Gemini, free)\n"
            f"Model B: {MODEL_B_NAME} (Groq, free)"
        ),
    )

    st.markdown(f"""
    <div style="font-size:0.82rem; color:#718096; margin-top:0.5rem;">
        <b>Model A:</b> {MODEL_A_NAME}<br>
        <b>Model B:</b> {MODEL_B_NAME}
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("**Agent Settings**")
    show_agent_trace = st.checkbox(
        "Show agent trace (attempts + issues)",
        value=True,
        help="Display how many drafting attempts the agent made and any issues found.",
    )

    st.divider()
    st.markdown("**About**")
    st.markdown("""
    <div style="font-size:0.82rem; color:#718096;">
    This assistant uses a <b>4-node LangGraph state machine</b>:
    <ol style="margin:0.5rem 0; padding-left:1.2rem;">
      <li>InputValidator</li>
      <li>EmailDrafter (LLM call)</li>
      <li>QualityChecker</li>
      <li>Refiner (if needed)</li>
    </ol>
    The agent self-corrects up to <b>2 times</b> before returning the best available draft.
    </div>
    """, unsafe_allow_html=True)


# ── Helper functions ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_graph():
    """Load and cache the LangGraph compiled graph — done once per session."""
    from src.graph import run_agent as _run_agent
    return _run_agent


def _score_color(score: float) -> str:
    """Return a hex colour based on score value for visual feedback."""
    if score >= 0.80:
        return "#38a169"   # green
    elif score >= 0.60:
        return "#d69e2e"   # amber
    else:
        return "#e53e3e"   # red


def _attempts_badge(n: int) -> str:
    if n <= 1:
        return f'<span class="badge-green">✓ 1st draft passed</span>'
    elif n == 2:
        return f'<span class="badge-yellow">⟳ Refined once ({n} attempts)</span>'
    else:
        return f'<span class="badge-red">⟳ Refined {n-1}× ({n} attempts)</span>'


def _generate_and_display(
    intent: str,
    facts_list: list,
    tone: str,
    model_name: str,
    container,
    show_trace: bool = True,
):
    """
    Run the agent for one model and render results in the given container.
    """
    run_agent = _load_graph()

    model_label = "Model A (Gemini 2.5 Flash-Lite)" if model_name == MODEL_A_NAME else "Model B (Llama 3.3 70B)"

    with container:
        st.markdown(f"#### {model_label}")

        with st.spinner(f"Running agent for {model_name}…"):
            try:
                result = run_agent(
                    intent=intent,
                    facts=facts_list,
                    tone=tone,
                    model_name=model_name,
                )
            except Exception as exc:
                st.error(f"Agent error: {exc}")
                return

        # ── Validation error ─────────────────────────────────────────────────
        if result.get("validation_error"):
            st.error(f"Input validation failed: {result['validation_error']}")
            return

        final_email = result.get("final_email", "")
        if not final_email.strip():
            st.warning("The agent did not produce an email. Check your API keys.")
            return

        # ── Display the email ────────────────────────────────────────────────
        # HTML-escape the email content so characters like <, >, & don't
        # break the surrounding HTML div or cause unexpected rendering.
        safe_email = html.escape(final_email)
        st.markdown(
            f'<div class="email-box">{safe_email}</div>',
            unsafe_allow_html=True,
        )

        col_copy, col_attempts = st.columns([3, 2])
        with col_copy:
            st.download_button(
                label="⬇ Download email",
                data=final_email,
                file_name=f"email_{model_name[:6]}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with col_attempts:
            if show_trace:
                attempts = result.get("attempts", 1)
                st.markdown(
                    _attempts_badge(attempts),
                    unsafe_allow_html=True,
                )

        # ── Agent trace ──────────────────────────────────────────────────────
        if show_trace:
            quality_issues = result.get("quality_issues", [])
            if quality_issues:
                with st.expander("🔧 Issues found by QualityChecker (before refinement)", expanded=False):
                    for issue in quality_issues:
                        st.markdown(f"- {issue}")
            quality_passed = result.get("quality_passed", False)
            if quality_passed:
                st.success("Quality check: passed on final draft", icon="✅")

        # ── Compute and display metrics ──────────────────────────────────────
        st.markdown('<div class="section-title">Evaluation Metrics</div>', unsafe_allow_html=True)

        with st.spinner("Computing evaluation scores…"):
            try:
                from src.evaluator import evaluate_email
                metrics = evaluate_email(
                    generated_email=final_email,
                    facts=facts_list,
                    tone=tone,
                )
            except Exception as exc:
                st.warning(f"Metric computation failed: {exc}")
                return

        # FRS
        frs = metrics["frs"]
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">📌 Fact Recall Score (FRS)</div>
          <div style="color:{_score_color(frs)}; font-size:1.4rem; font-weight:700;">{frs:.3f}</div>
          <div style="font-size:0.78rem; color:#718096;">
            Measures whether all provided facts appear in the email (semantic similarity).
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(frs)

        # TAS
        tas = metrics["tas"]
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">🎭 Tone Alignment Score (TAS)</div>
          <div style="color:{_score_color(tas)}; font-size:1.4rem; font-weight:700;">{tas:.3f}</div>
          <div style="font-size:0.78rem; color:#718096;">
            LLM-as-Judge verdict: <em>"{metrics['tas_reason']}"</em>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(tas)

        # PQI
        pqi = metrics["pqi"]
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">📋 Professional Quality Index (PQI)</div>
          <div style="color:{_score_color(pqi)}; font-size:1.4rem; font-weight:700;">{pqi:.3f}</div>
          <div style="font-size:0.78rem; color:#718096;">
            Grammar: {metrics['pqi_grammar']:.2f} &nbsp;|&nbsp;
            Readability: {metrics['pqi_readability']:.2f} &nbsp;|&nbsp;
            Structure: {metrics['pqi_structure']:.2f}
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(pqi)

        # Composite
        composite = metrics["composite"]
        st.markdown(f"""
        <div style="background:linear-gradient(90deg,#ebf8ff,#e9d8fd); border-radius:8px;
             padding:0.8rem 1rem; margin-top:0.6rem;">
          <div class="metric-label">⭐ Composite Score</div>
          <div style="color:{_score_color(composite)}; font-size:1.6rem; font-weight:800;">
            {composite:.3f} / 1.000
          </div>
          <div style="font-size:0.78rem; color:#718096;">Mean of FRS + TAS + PQI</div>
        </div>
        """, unsafe_allow_html=True)


# ── Main UI: Input Panel ─────────────────────────────────────────────────────
st.markdown("### 📝 Email Request")

col_input_left, col_input_right = st.columns([3, 2])

with col_input_left:
    intent = st.text_input(
        label="Intent *",
        placeholder="e.g. Follow up after a job interview",
        help="Describe the core purpose of the email in one sentence.",
    )

    facts_raw = st.text_area(
        label="Key Facts *",
        placeholder=(
            "Enter one fact per line. Example:\n"
            "Interview was on Monday, April 14th\n"
            "Interviewer's name is Ms. Rachel Chen\n"
            "Role applied for is Senior AI Engineer\n"
            "Discussed a real-time anomaly detection project"
        ),
        height=160,
        help="List every specific fact that MUST appear in the email. One per line.",
    )

with col_input_right:
    tone_preset = st.selectbox(
        label="Tone *",
        options=TONE_PRESETS,
        index=0,
        help="Select a preset tone or choose 'Custom' to type your own.",
    )

    if tone_preset == "Custom (type below)":
        tone = st.text_input(
            label="Custom tone",
            placeholder="e.g. Diplomatic but firm",
        )
    else:
        tone = tone_preset
        st.markdown(f"""
        <div style="background:#ebf8ff; border-radius:8px; padding:0.6rem 0.8rem; margin-top:0.5rem;
             font-size:0.82rem; color:#2c5282;">
            Selected tone: <b>{tone}</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.button(
        label="✨ Generate Email",
        type="primary",
        use_container_width=True,
    )

# ── Generate on button click ─────────────────────────────────────────────────
if generate_btn:
    # Validate inputs
    if not intent.strip():
        st.error("Please enter the email intent.")
        st.stop()

    facts_list = [f.strip() for f in facts_raw.strip().splitlines() if f.strip()]
    if not facts_list:
        st.error("Please enter at least one key fact.")
        st.stop()

    if not tone.strip():
        st.error("Please select or enter a tone.")
        st.stop()

    st.divider()
    st.markdown("### 📧 Generated Email(s)")

    # Determine which models to run
    if model_choice == "Model A only":
        models_to_run = [MODEL_A_NAME]
    elif model_choice == "Model B only":
        models_to_run = [MODEL_B_NAME]
    else:
        models_to_run = [MODEL_A_NAME, MODEL_B_NAME]

    if len(models_to_run) == 1:
        container = st.container()
        _generate_and_display(intent, facts_list, tone, models_to_run[0], container, show_trace=show_agent_trace)
    else:
        col_a, col_b = st.columns(2)
        _generate_and_display(intent, facts_list, tone, MODEL_A_NAME, col_a, show_trace=show_agent_trace)
        _generate_and_display(intent, facts_list, tone, MODEL_B_NAME, col_b, show_trace=show_agent_trace)


# ── Full evaluation section ──────────────────────────────────────────────────
st.divider()
st.markdown("### 📊 Run Full Evaluation (All 10 Scenarios)")

st.markdown("""
<div style="font-size:0.88rem; color:#4a5568; margin-bottom:1rem;">
Click the button below to run both models through all 10 benchmark scenarios,
compute all 3 metrics for every result, and generate downloadable CSV + JSON reports.
<br><br>
⚠️ This makes approximately <b>20 LLM calls + 20 judge calls</b>. Estimated time: <b>3–6 minutes</b>
depending on API response times.
</div>
""", unsafe_allow_html=True)

run_eval_btn = st.button(
    label="▶ Run Full Evaluation",
    type="secondary",
    use_container_width=False,
)

if run_eval_btn:
    scenarios_path = Path("data/test_scenarios.json")
    if not scenarios_path.exists():
        st.error("Scenarios file not found at data/test_scenarios.json")
        st.stop()

    with open(scenarios_path, encoding="utf-8") as f:
        scenarios = json.load(f)

    from src.graph import run_agent as _run_agent
    from src.evaluator import evaluate_email

    progress_text = st.empty()
    progress_bar  = st.progress(0)
    results_rows  = []
    total         = len(scenarios) * 2

    for i, scenario in enumerate(scenarios):
        for j, model_name in enumerate([MODEL_A_NAME, MODEL_B_NAME]):
            step = i * 2 + j + 1
            progress_text.markdown(
                f"Running scenario **{scenario['scenario_id']}/10** | "
                f"Model: `{model_name}` | Step {step}/{total}"
            )
            progress_bar.progress(step / total)

            try:
                result = _run_agent(
                    intent=scenario["intent"],
                    facts=scenario["facts"],
                    tone=scenario["tone"],
                    model_name=model_name,
                )
                final_email = result.get("final_email", "")
                if final_email.strip():
                    metrics = evaluate_email(final_email, scenario["facts"], scenario["tone"])
                else:
                    metrics = {
                        "frs": 0.0, "tas": 0.0, "tas_reason": "No email generated.",
                        "tas_raw": 0, "pqi": 0.0, "pqi_grammar": 0.0,
                        "pqi_readability": 0.0, "pqi_structure": 0.0,
                        "grammar_errors": 0, "flesch_score": 0.0, "composite": 0.0,
                    }
                results_rows.append({
                    "scenario_id":     scenario["scenario_id"],
                    "intent":          scenario["intent"],
                    "tone":            scenario["tone"],
                    "model":           model_name,
                    "attempts_needed": result.get("attempts", 0),
                    "quality_passed":  result.get("quality_passed", False),
                    "final_email":     final_email,
                    "FRS":             metrics["frs"],
                    "TAS":             metrics["tas"],
                    "TAS_reason":      metrics["tas_reason"],
                    "PQI":             metrics["pqi"],
                    "composite_score": metrics["composite"],
                })
            except Exception as exc:
                st.warning(f"Error on scenario {scenario['scenario_id']} / {model_name}: {exc}")

            time.sleep(2)   # Rate limit buffer

    progress_text.markdown("✅ Evaluation complete!")
    progress_bar.progress(1.0)

    if results_rows:
        df = pd.DataFrame(results_rows)

        # Save to results/
        Path("results").mkdir(exist_ok=True)
        csv_path  = "results/evaluation_results.csv"
        json_path = "results/comparison_report.json"
        df.to_csv(csv_path, index=False)

        # Build comparison summary
        summary = {}
        for model_name in [MODEL_A_NAME, MODEL_B_NAME]:
            mdf = df[df["model"] == model_name]
            summary[model_name] = {
                "avg_FRS":       round(float(mdf["FRS"].mean()), 4),
                "avg_TAS":       round(float(mdf["TAS"].mean()), 4),
                "avg_PQI":       round(float(mdf["PQI"].mean()), 4),
                "avg_composite": round(float(mdf["composite_score"].mean()), 4),
                "avg_attempts":  round(float(mdf["attempts_needed"].mean()), 2),
            }
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Download buttons
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="⬇ Download evaluation_results.csv",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="evaluation_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_dl2:
            st.download_button(
                label="⬇ Download comparison_report.json",
                data=json.dumps(summary, indent=2).encode("utf-8"),
                file_name="comparison_report.json",
                mime="application/json",
                use_container_width=True,
            )

        # Quick summary table
        st.markdown("#### Results Summary")
        summary_display = []
        for model_name, vals in summary.items():
            summary_display.append({
                "Model":        model_name,
                "Avg FRS":      f"{vals['avg_FRS']:.3f}",
                "Avg TAS":      f"{vals['avg_TAS']:.3f}",
                "Avg PQI":      f"{vals['avg_PQI']:.3f}",
                "Composite":    f"{vals['avg_composite']:.3f}",
                "Avg Attempts": f"{vals['avg_attempts']:.1f}",
            })
        st.dataframe(
            pd.DataFrame(summary_display).set_index("Model"),
            use_container_width=True,
        )
