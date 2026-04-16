"""
run_evaluation.py
-----------------
Batch evaluation script for the Email Generation Agent.

What this script does:
  1. Loads all 10 test scenarios from data/test_scenarios.json
  2. Runs each scenario through the LangGraph agent for BOTH models:
       - Model A: gemini-2.5-flash-lite  (Google Gemini, free tier)
       - Model B: llama-3.3-70b-versatile (Groq, free tier)
  3. Computes all 3 custom metrics for every generated email:
       - FRS  (Fact Recall Score)        — semantic similarity
       - TAS  (Tone Alignment Score)     — LLM-as-a-Judge
       - PQI  (Professional Quality Index) — hybrid composite
  4. Writes results to:
       - results/evaluation_results.csv   (row per scenario per model)
       - results/comparison_report.json   (aggregated comparison summary)

Run with:
  python run_evaluation.py

Optional flags:
  --scenarios-file PATH   Path to scenarios JSON (default: data/test_scenarios.json)
  --output-dir PATH       Output directory (default: results/)
  --model-a NAME          Model A name (default: gemini-2.5-flash-lite)
  --model-b NAME          Model B name (default: llama-3.3-70b-versatile)
  --delay SECONDS         Seconds to wait between API calls (default: 2)
                          Increase this if you hit rate limits.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from colorama import Fore, Style, init as colorama_init

from src.graph import run_agent
from src.evaluator import evaluate_email
from src.models import MODEL_GEMINI, MODEL_GROQ

colorama_init(autoreset=True)

# ── Constants ────────────────────────────────────────────────────────────────

METRIC_DEFINITIONS = {
    "FRS": {
        "name":        "Fact Recall Score",
        "description": (
            "Measures whether every key fact provided in the input was semantically "
            "captured in the generated email, even when paraphrased. Uses cosine "
            "similarity between sentence-transformer embeddings of each fact and the "
            "email sentences. Score = mean of per-fact maximum similarities. "
            "Range: 0.0 (no facts present) to 1.0 (all facts fully present)."
        ),
        "technique":   "Automated — sentence-transformers (all-MiniLM-L6-v2)",
        "threshold":   "≥ 0.75 considered good; < 0.60 triggers in-agent refinement",
    },
    "TAS": {
        "name":        "Tone Alignment Score",
        "description": (
            "Measures how accurately the generated email matches the requested tone, "
            "judged by an independent LLM (gemini-2.5-flash-lite at temperature=0). "
            "The judge rates tone match 1–10 using a structured rubric and returns "
            "JSON with score + one-sentence justification. TAS = score / 10. "
            "Range: 0.0 (completely wrong tone) to 1.0 (perfect tone match)."
        ),
        "technique":   "LLM-as-a-Judge — gemini-2.5-flash-lite, temperature=0",
        "threshold":   "≥ 0.80 considered good; < 0.60 indicates significant tone mismatch",
    },
    "PQI": {
        "name":        "Professional Quality Index",
        "description": (
            "Measures whether the email is immediately usable as-is. Composite of "
            "three equally weighted sub-scores: "
            "(1) Grammar Score = 1 - min(1, error_count/10) via language-tool-python; "
            "(2) Readability Score mapped from Flesch Reading Ease (target: 30–60); "
            "(3) Structure Score = 0.25 per present element: subject line, greeting, "
            "closing, ≥2 body paragraphs. PQI = mean(grammar, readability, structure). "
            "Range: 0.0 (unusable) to 1.0 (send-ready without any editing)."
        ),
        "technique":   "Hybrid — language-tool-python + textstat + regex rules",
        "threshold":   "≥ 0.80 considered send-ready; < 0.65 needs significant editing",
    },
}

# ── CLI argument parsing ─────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Email Generation Agent evaluation across 10 scenarios × 2 models."
    )
    parser.add_argument(
        "--scenarios-file", default="data/test_scenarios.json",
        help="Path to the test scenarios JSON file."
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Directory where evaluation_results.csv and comparison_report.json will be saved."
    )
    parser.add_argument(
        "--model-a", default=MODEL_GEMINI,
        help=f"Model A identifier (default: {MODEL_GEMINI})"
    )
    parser.add_argument(
        "--model-b", default=MODEL_GROQ,
        help=f"Model B identifier (default: {MODEL_GROQ})"
    )
    parser.add_argument(
        "--delay", type=float, default=2.0,
        help="Seconds to pause between API calls to respect rate limits (default: 2)."
    )
    return parser.parse_args()


# ── Main evaluation loop ─────────────────────────────────────────────────────

def run_evaluation(
    scenarios_file: str,
    output_dir:     str,
    model_a:        str,
    model_b:        str,
    delay_seconds:  float,
) -> None:
    """
    Execute the full evaluation pipeline.

    For each of the 10 scenarios, runs both models through the LangGraph agent,
    evaluates with all 3 metrics, and writes structured output files.
    """
    # ── Load scenarios ───────────────────────────────────────────────────────
    scenarios_path = Path(scenarios_file)
    if not scenarios_path.exists():
        print(f"{Fore.RED}Error: Scenarios file not found at {scenarios_path}")
        sys.exit(1)

    with open(scenarios_path, encoding="utf-8") as f:
        scenarios = json.load(f)

    print(f"\n{Fore.CYAN}{'='*65}")
    print(f"  EMAIL GENERATION AGENT — BATCH EVALUATION")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}{Style.RESET_ALL}")
    print(f"  Scenarios : {len(scenarios)}")
    print(f"  Model A   : {model_a}")
    print(f"  Model B   : {model_b}")
    print(f"  API delay : {delay_seconds}s between calls")
    print(f"  Output    : {output_dir}/\n")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    models = [model_a, model_b]

    total_calls = len(scenarios) * len(models)
    progress_bar = tqdm(
        total=total_calls,
        desc="Evaluating",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

    # ── Iterate scenarios × models ───────────────────────────────────────────
    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        intent      = scenario["intent"]
        facts       = scenario["facts"]
        tone        = scenario["tone"]

        for model_name in models:
            progress_bar.set_description(
                f"Scenario {scenario_id:02d} | {model_name[:25]}"
            )

            # ── Run the LangGraph agent ──────────────────────────────────────
            try:
                agent_result = run_agent(
                    intent=intent,
                    facts=facts,
                    tone=tone,
                    model_name=model_name,
                )
            except Exception as exc:
                print(f"\n{Fore.RED}  Agent error (scenario {scenario_id}, {model_name}): {exc}")
                agent_result = {
                    "final_email":     "",
                    "attempts":        0,
                    "quality_passed":  False,
                    "quality_issues":  [str(exc)],
                    "validation_error": str(exc),
                }

            final_email   = agent_result.get("final_email", "")
            attempts_made = agent_result.get("attempts", 0)
            quality_ok    = agent_result.get("quality_passed", False)

            # ── Evaluate the generated email ─────────────────────────────────
            if final_email.strip():
                try:
                    metrics = evaluate_email(
                        generated_email=final_email,
                        facts=facts,
                        tone=tone,
                    )
                except Exception as exc:
                    print(f"\n{Fore.YELLOW}  Eval error (scenario {scenario_id}, {model_name}): {exc}")
                    metrics = {
                        "frs": 0.0, "tas": 0.0, "tas_reason": str(exc),
                        "tas_raw": 0, "pqi": 0.0, "pqi_grammar": 0.0,
                        "pqi_readability": 0.0, "pqi_structure": 0.0,
                        "grammar_errors": 0, "flesch_score": 0.0, "composite": 0.0,
                    }
            else:
                metrics = {
                    "frs": 0.0, "tas": 0.0, "tas_reason": "No email generated.",
                    "tas_raw": 0, "pqi": 0.0, "pqi_grammar": 0.0,
                    "pqi_readability": 0.0, "pqi_structure": 0.0,
                    "grammar_errors": 0, "flesch_score": 0.0, "composite": 0.0,
                }

            rows.append({
                "scenario_id":       scenario_id,
                "intent":            intent,
                "tone":              tone,
                "model":             model_name,
                "attempts_needed":   attempts_made,
                "quality_passed":    quality_passed if (quality_passed := quality_ok) else False,
                "final_email":       final_email,
                "FRS":               metrics["frs"],
                "TAS":               metrics["tas"],
                "TAS_reason":        metrics["tas_reason"],
                "TAS_raw":           metrics["tas_raw"],
                "PQI":               metrics["pqi"],
                "PQI_grammar":       metrics["pqi_grammar"],
                "PQI_readability":   metrics["pqi_readability"],
                "PQI_structure":     metrics["pqi_structure"],
                "grammar_errors":    metrics["grammar_errors"],
                "flesch_score":      metrics["flesch_score"],
                "composite_score":   metrics["composite"],
            })

            progress_bar.update(1)

            # Pause between calls to respect free-tier rate limits
            if delay_seconds > 0:
                time.sleep(delay_seconds)

    progress_bar.close()

    # ── Write CSV ────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_path = Path(output_dir) / "evaluation_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n{Fore.GREEN}  Saved: {csv_path}")

    # ── Compute per-model averages ───────────────────────────────────────────
    model_summaries = {}
    for model_name in models:
        model_df = df[df["model"] == model_name]
        model_summaries[model_name] = {
            "name":                       model_name,
            "avg_FRS":                    round(float(model_df["FRS"].mean()), 4),
            "avg_TAS":                    round(float(model_df["TAS"].mean()), 4),
            "avg_PQI":                    round(float(model_df["PQI"].mean()), 4),
            "avg_composite":              round(float(model_df["composite_score"].mean()), 4),
            "avg_attempts":               round(float(model_df["attempts_needed"].mean()), 2),
            "scenarios_needing_refinement": int((model_df["attempts_needed"] > 1).sum()),
            "scenarios_quality_passed":   int(model_df["quality_passed"].sum()),
        }

    # Per-metric delta (Model A minus Model B)
    a = model_summaries[model_a]
    b = model_summaries[model_b]
    delta = {
        "FRS":       round(a["avg_FRS"]       - b["avg_FRS"],       4),
        "TAS":       round(a["avg_TAS"]       - b["avg_TAS"],       4),
        "PQI":       round(a["avg_PQI"]       - b["avg_PQI"],       4),
        "composite": round(a["avg_composite"] - b["avg_composite"], 4),
    }
    winner = model_a if a["avg_composite"] >= b["avg_composite"] else model_b

    # ── Write comparison report JSON ─────────────────────────────────────────
    report = {
        "generated_at":      datetime.now().isoformat(),
        "metric_definitions": METRIC_DEFINITIONS,
        "model_a":            a,
        "model_b":            b,
        "delta_A_minus_B":    delta,
        "winner":             winner,
        "note": (
            "Delta = Model A score minus Model B score. "
            "Positive delta means Model A performed better on that metric."
        ),
    }
    json_path = Path(output_dir) / "comparison_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"{Fore.GREEN}  Saved: {json_path}")

    # ── Print summary to console ─────────────────────────────────────────────
    _print_summary(a, b, delta, winner)


def _print_summary(a: dict, b: dict, delta: dict, winner: str) -> None:
    """Print a formatted summary table to the console."""
    print(f"\n{Fore.CYAN}{'='*65}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'='*65}{Style.RESET_ALL}")
    print(f"  {'Metric':<22} {'Model A':>12} {'Model B':>12} {'Delta (A-B)':>12}")
    print(f"  {'-'*58}")

    for metric, key_a, key_b in [
        ("FRS (Fact Recall)",   "avg_FRS", "avg_FRS"),
        ("TAS (Tone Align.)",   "avg_TAS", "avg_TAS"),
        ("PQI (Prof. Quality)", "avg_PQI", "avg_PQI"),
        ("Composite Score",     "avg_composite", "avg_composite"),
    ]:
        va  = a[key_a]
        vb  = b[key_b]
        d   = round(va - vb, 4)
        col = Fore.GREEN if d > 0 else (Fore.RED if d < 0 else Fore.WHITE)
        print(f"  {metric:<22} {va:>12.4f} {vb:>12.4f} {col}{d:>+12.4f}{Style.RESET_ALL}")

    print(f"\n  Avg attempts needed  : {a['avg_attempts']:>6.2f} (A)   {b['avg_attempts']:>6.2f} (B)")
    print(f"  Scenarios refined    : {a['scenarios_needing_refinement']:>6}     {b['scenarios_needing_refinement']:>6}")
    print(f"\n{Fore.YELLOW}  Recommended model   : {winner}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*65}{Style.RESET_ALL}\n")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        scenarios_file=args.scenarios_file,
        output_dir=args.output_dir,
        model_a=args.model_a,
        model_b=args.model_b,
        delay_seconds=args.delay,
    )
