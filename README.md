# Email Generation Agent

A production-quality AI email generation assistant built with **LangGraph**, featuring a self-refinement loop, triple-layer prompting, and a custom 3-metric evaluation system.

---

## What This Project Does

Given three inputs — **Intent**, **Key Facts**, and **Tone** — the agent generates a professional, ready-to-send email. It uses:

- A **4-node LangGraph state machine** with self-correction
- **Triple-Layer Prompting**: Role-Playing + Few-Shot Examples + Chain-of-Thought
- Two free LLMs compared head-to-head:
  - **Model A**: `gemini-2.5-flash-lite` (Google AI Studio, free tier)
  - **Model B**: `llama-3.3-70b-versatile` (Groq, free tier)
- **3 custom evaluation metrics** measuring fact recall, tone accuracy, and professional quality

---

## Agent Architecture

```
START → InputValidator → EmailDrafter → QualityChecker → END (pass)
                              ↑                ↓ (fail, <2 attempts)
                           Refiner ←──────────┘
```

The agent can self-correct up to **2 times** before returning the best available draft.

---

## Setup (5 minutes)

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd email-gen-assistant
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> First run downloads the `all-MiniLM-L6-v2` sentence-transformer model (~80 MB). This is automatic.

### 4. Set up API keys (both are FREE, no credit card required)

Copy the example file:
```bash
cp .env.example .env
```

Then edit `.env` and fill in your keys:

```
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

**Get keys:**
- **Gemini**: [aistudio.google.com](https://aistudio.google.com) → click "Get API Key" → no billing required
- **Groq**: [console.groq.com](https://console.groq.com) → sign up → API Keys → Create

---

## Running the Application

### Option A: Streamlit Web UI (recommended)
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`. Enter Intent, Facts, Tone → click Generate.

### Option B: Batch Evaluation Script
```bash
python run_evaluation.py
```
Runs all 10 benchmark scenarios through both models and saves:
- `results/evaluation_results.csv` — full row-per-scenario results
- `results/comparison_report.json` — aggregated model comparison

Optional flags:
```bash
python run_evaluation.py --delay 3       # increase if hitting rate limits
python run_evaluation.py --output-dir my_results
```

---

## Project Structure

```
email-gen-assistant/
├── app.py                        # Streamlit UI
├── run_evaluation.py             # Batch evaluation runner
├── src/
│   ├── state.py                  # EmailState TypedDict (LangGraph state)
│   ├── models.py                 # Gemini + Groq client setup
│   ├── prompt_templates.py       # Triple-layer prompt + refiner + judge prompts
│   ├── nodes.py                  # 4 LangGraph node functions
│   ├── graph.py                  # StateGraph wiring + run_agent() entry point
│   └── evaluator.py              # FRS + TAS + PQI metric implementations
├── data/
│   └── test_scenarios.json       # 10 benchmark scenarios + human reference emails
├── results/                      # Auto-created by run_evaluation.py
│   ├── evaluation_results.csv
│   └── comparison_report.json
├── .env.example
├── requirements.txt
└── README.md
```

---

## Prompting Strategy: Triple-Layer Prompting

Three techniques are combined, each solving a different failure mode:

| Layer | Technique | Problem It Solves |
|---|---|---|
| 1 | **Role-Playing** | Prevents casual/generic tone drift |
| 2 | **Few-Shot Examples** | Shows exact format expected (subject, greeting, body, closing) |
| 3 | **Chain-of-Thought** | Forces fact mapping before writing; reduces omission and hallucination |

The **Refiner** node builds a targeted feedback prompt that lists specific issues (e.g., "Missing Fact 3", "No closing found") for the LLM to fix on retry.

---

## Evaluation Metrics

### Metric 1: Fact Recall Score (FRS)
- **What**: Did the email include all the provided facts, even if paraphrased?
- **How**: Cosine similarity between sentence-transformer embeddings of each fact vs. email sentences
- **Why not ROUGE**: A good email paraphrases facts. Semantic similarity handles this; n-gram overlap does not.

### Metric 2: Tone Alignment Score (TAS)
- **What**: Does the email's tone match what was requested?
- **How**: `gemini-2.5-flash-lite` judges the email at `temperature=0`, returns `{"score": 1-10, "reason": "..."}`
- **Why LLM-judge**: Tone is subjective — no heuristic or word list can reliably detect "firm but empathetic"

### Metric 3: Professional Quality Index (PQI)
- **What**: Is the email send-ready without editing?
- **How**: Composite of three equal sub-scores:
  - Grammar: `language-tool-python` error count → `1 - min(1, errors/10)`
  - Readability: Flesch Reading Ease mapped to 0–1 (target: 30–60 for professional email)
  - Structure: 0.25 each for subject line, greeting, closing, ≥2 body paragraphs

### Output CSV Columns
```
scenario_id, intent, tone, model, attempts_needed, quality_passed,
final_email, FRS, TAS, TAS_reason, TAS_raw, PQI, PQI_grammar,
PQI_readability, PQI_structure, grammar_errors, flesch_score, composite_score
```

The `attempts_needed` column is unique to the LangGraph approach — it reveals first-draft reliability, which simple LLM calling cannot measure.

---

## Models Compared

| | Model A | Model B |
|---|---|---|
| Name | `gemini-2.5-flash-lite` | `llama-3.3-70b-versatile` |
| Provider | Google AI Studio | Groq |
| Type | Proprietary | Open-Source (Meta) |
| Free tier | 15 RPM, 1,000 RPD | Generous free tier |
| SDK | `google-generativeai` | `groq` |

Both models run on the **exact same prompt template and 10 scenarios**, isolating model capability as the only variable.

---

## Requirements

- Python 3.11+
- Internet access (for API calls and first-run model download)
- ~200 MB disk space (sentence-transformer model cache)
