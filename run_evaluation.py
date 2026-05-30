"""
run_evaluation.py — CLI runner: execute the full evaluation pipeline and
generate a human-readable scorecard + JSON results file.

Usage:
    python part2/run_evaluation.py [--skip-judge] [--test-ids TC001,TC003]

Flags:
    --skip-judge   Run only automated metrics (no Groq API calls for judge)
    --test-ids     Comma-separated list of test IDs to run (default: all)

Outputs:
    part2/results/eval_results.json   — full per-case results
    part2/results/scorecard.csv       — summary table for easy review
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Allow running from repo root or from part2/
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from agent import RetentionAgent, AgentResponse
from evaluator import AutomatedMetrics, LLMJudge, EvaluationResult

RESULTS_DIR = _HERE / "results"
SUITE_FILE  = _HERE / "test_suite.json"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _banner(text: str, width: int = 65, char: str = "─") -> str:
    return f"\n{char * width}\n{text}\n{char * width}"


def _pct(num: float, denom: int) -> str:
    if denom == 0:
        return "N/A"
    return f"{num/denom*100:.1f}%"


def _avg(values: list) -> float:
    vs = [v for v in values if v is not None]
    return round(sum(vs) / len(vs), 2) if vs else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Run TeleConnect Retention Agent Evaluation")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip the LLM-as-judge step (automated metrics only)")
    parser.add_argument("--test-ids", type=str, default="",
                        help="Comma-separated test IDs to run (e.g. TC001,TC003)")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("\n[ERROR] GROQ_API_KEY environment variable not set.")
        print("        Get a free key at https://console.groq.com and set it:\n")
        print("        Windows:  $env:GROQ_API_KEY='gsk_...'\n")
        sys.exit(1)

    # ── Load test suite ───────────────────────────────────────────────────────
    with open(SUITE_FILE, encoding="utf-8") as f:
        test_suite = json.load(f)

    if args.test_ids:
        wanted = set(args.test_ids.split(","))
        test_suite = [t for t in test_suite if t["id"] in wanted]
        if not test_suite:
            print(f"[ERROR] No matching test IDs found: {args.test_ids}")
            sys.exit(1)

    print(_banner(
        f"TeleConnect Retention Agent — Evaluation  ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
        char="═"
    ))
    print(f"  Test suite : {SUITE_FILE.name}  ({len(test_suite)} cases)")
    print(f"  Judge      : {'DISABLED (--skip-judge)' if args.skip_judge else 'llama-3.1-8b-instant (Groq)'}")
    print(f"  Agent model: llama-3.3-70b-versatile (Groq)")

    # ── Initialise agent (and optional judge) ─────────────────────────────────
    agent = RetentionAgent(api_key=api_key)
    judge = None if args.skip_judge else LLMJudge(api_key=api_key)

    # ── Run evaluations ───────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    all_results: list[EvaluationResult] = []

    for i, tc in enumerate(test_suite, 1):
        print(f"\n  [{i:02d}/{len(test_suite)}] {tc['id']}  ({tc['category']})")
        print(f"         Input: {tc['input'][:80]}{'...' if len(tc['input'])>80 else ''}")

        # Run agent
        try:
            ar: AgentResponse = agent.run(tc["input"])
        except Exception as e:
            # If the agent throws (e.g. rate limit), create an empty response
            from agent import AgentResponse
            ar = AgentResponse(
                content=f"[Agent error: {e}]",
                tool_trace=[],
                total_latency_ms=0,
                model=agent.model,
                error=str(e),
            )

        # Automated metrics
        auto = AutomatedMetrics.compute(tc, ar)
        print(f"         Tools called : {auto.tools_called}")
        print(f"         Tool recall  : {auto.tool_selection_recall:.0%}  "
              f"| Escalation: {'✓' if auto.escalation_correct else '✗'}  "
              f"| Latency: {auto.latency_ms:.0f}ms  "
              f"| Auto pass: {'✓' if auto.pass_automated else '✗'}")

        # LLM judge
        judge_result = None
        if judge:
            try:
                judge_result = judge.score(tc, ar)
                print(f"         Judge scores : "
                      f"FC={judge_result.factual_correctness} "
                      f"TU={judge_result.tool_use_appropriateness} "
                      f"Act={judge_result.actionability} "
                      f"Hal={judge_result.hallucination} "
                      f"→ {judge_result.composite_score:.1f}/5  "
                      f"{'✓' if judge_result.overall_pass else '✗'}")
                time.sleep(0.5)  # respect Groq rate limits
            except Exception as e:
                print(f"         [Judge error] {e}")

        all_results.append(EvaluationResult(
            test_case=tc,
            automated=auto,
            judge=judge_result,
            agent_response_dict=ar.to_dict(),
        ))

    # ── Compute aggregate scores ──────────────────────────────────────────────
    n = len(all_results)
    auto_pass  = sum(1 for r in all_results if r.automated.pass_automated)
    esc_correct = sum(1 for r in all_results if r.automated.escalation_correct)

    avg_recall       = _avg([r.automated.tool_selection_recall for r in all_results])
    avg_order        = _avg([r.automated.tool_order_accuracy for r in all_results])
    avg_completeness = _avg([r.automated.response_completeness for r in all_results])
    avg_latency      = _avg([r.automated.latency_ms for r in all_results])

    # Per-category
    categories: dict[str, dict] = {}
    for r in all_results:
        cat = r.automated.category
        if cat not in categories:
            categories[cat] = {"total": 0, "auto_pass": 0, "judge_scores": []}
        categories[cat]["total"] += 1
        if r.automated.pass_automated:
            categories[cat]["auto_pass"] += 1
        if r.judge and r.judge.composite_score:
            categories[cat]["judge_scores"].append(r.judge.composite_score)

    # Judge aggregates
    judge_results = [r.judge for r in all_results if r.judge and not r.judge.judge_error]
    avg_fc  = _avg([j.factual_correctness       for j in judge_results])
    avg_tu  = _avg([j.tool_use_appropriateness  for j in judge_results])
    avg_act = _avg([j.actionability             for j in judge_results])
    avg_hal = _avg([j.hallucination             for j in judge_results])
    avg_comp = _avg([j.composite_score          for j in judge_results])
    judge_pass = sum(1 for j in judge_results if j.overall_pass)

    # ── Print scorecard ───────────────────────────────────────────────────────
    print(_banner("EVALUATION SCORECARD", char="═"))

    print(f"\n  Dataset: {n} test cases  |  "
          f"Date: {datetime.now().strftime('%Y-%m-%d')}")

    print(_banner("AUTOMATED METRICS"))
    print(f"  Tool Selection Recall    : {avg_recall:.0%}  (avg across cases)")
    print(f"  Tool Order Accuracy      : {avg_order:.0%}")
    print(f"  Escalation Accuracy      : {esc_correct}/{n}  ({_pct(esc_correct, n)})")
    print(f"  Response Completeness    : {avg_completeness:.0%}  (keyword heuristic)")
    print(f"  Avg Latency              : {avg_latency:.0f}ms")
    print(f"  Overall Automated Pass   : {auto_pass}/{n}  ({_pct(auto_pass, n)})")

    if judge_results:
        print(_banner(f"LLM JUDGE SCORES  (n={len(judge_results)}, llama-3.1-8b-instant)"))
        print(f"  Factual Correctness      : {avg_fc:.2f} / 5")
        print(f"  Tool Use Appropriateness : {avg_tu:.2f} / 5")
        print(f"  Actionability for Rep    : {avg_act:.2f} / 5")
        print(f"  Hallucination            : {avg_hal:.2f} / 5  (5 = no hallucination)")
        print(f"  {'─'*40}")
        print(f"  Composite Score          : {avg_comp:.2f} / 5")
        print(f"  Judge Pass Rate          : {judge_pass}/{len(judge_results)}  "
              f"({_pct(judge_pass, len(judge_results))})")

    print(_banner("PER-CATEGORY RESULTS"))
    header = f"  {'Category':<32} {'Auto':>6} {'Judge avg':>9}"
    print(header)
    print(f"  {'─'*50}")
    for cat, data in sorted(categories.items()):
        auto_str = f"{data['auto_pass']}/{data['total']}"
        jscores  = data["judge_scores"]
        j_str    = f"{_avg(jscores):.1f}/5" if jscores else "N/A"
        mark     = "✓" if data["auto_pass"] == data["total"] else "✗"
        print(f"  {cat:<32} {auto_str:>4} {mark}   {j_str:>8}")

    # ── Narrative analysis ────────────────────────────────────────────────────
    print(_banner("CASE ANALYSIS"))

    # Pick 2 best-performing cases
    successes = sorted(
        all_results,
        key=lambda r: (r.automated.pass_automated,
                       r.judge.composite_score if r.judge else 0),
        reverse=True
    )[:2]

    # Pick 2 worst-performing cases
    failures = sorted(
        all_results,
        key=lambda r: (r.automated.pass_automated,
                       r.judge.composite_score if r.judge else 5),
    )[:2]

    print("\n  SUCCESS CASES:")
    for r in successes:
        jc = f" | Judge: {r.judge.composite_score:.1f}/5" if r.judge else ""
        print(f"\n  [{r.test_case['id']}]  {r.test_case['description'][:70]}")
        print(f"    Tools: {r.automated.tools_called}  | Recall: {r.automated.tool_selection_recall:.0%}{jc}")
        if r.judge and r.judge.justifications:
            jt = r.judge.justifications
            print(f"    Action justification: {jt.get('actionability', '')[:100]}")

    print("\n  FAILURE CASES + ROOT CAUSE:")
    for r in failures:
        if r.automated.pass_automated and (not r.judge or r.judge.overall_pass):
            continue  # skip if actually passed
        jc = f" | Judge: {r.judge.composite_score:.1f}/5" if r.judge else ""
        print(f"\n  [{r.test_case['id']}]  {r.test_case['description'][:70]}")
        print(f"    Called  : {r.automated.tools_called}")
        print(f"    Expected: {r.automated.tools_expected}")
        print(f"    Recall  : {r.automated.tool_selection_recall:.0%}  "
              f"| Escalation ✓: {r.automated.escalation_correct}{jc}")
        if r.judge and r.judge.judge_error:
            print(f"    Judge error: {r.judge.judge_error}")
        elif r.judge:
            worst_dim = min(
                {"FC": r.judge.factual_correctness, "TU": r.judge.tool_use_appropriateness,
                 "Act": r.judge.actionability, "Hal": r.judge.hallucination}.items(),
                key=lambda x: x[1]
            )
            print(f"    Weakest dimension: {worst_dim[0]} = {worst_dim[1]}/5")
            just = r.judge.justifications.get(
                {"FC": "factual_correctness", "TU": "tool_use_appropriateness",
                 "Act": "actionability", "Hal": "hallucination"}[worst_dim[0]], "")
            if just:
                print(f"    Reason: {just[:110]}")

    # ── Production Roadmap ────────────────────────────────────────────────────
    print(_banner("PRODUCTION ROADMAP"))
    print("""
  To productionise this evaluation pipeline at scale:

  1. CI INTEGRATION — add this script as a GitHub Actions step triggered on
     every PR. Gate merges when auto-pass rate drops below 80% or composite
     judge score falls below 3.5, preventing regressions in agent behaviour.

  2. TEST SUITE GROWTH — use production conversation logs to auto-generate new
     test cases (customer IDs → LLM-synthesised scenarios). Target 100 cases
     spanning all edge categories within 3 months.

  3. HUMAN CALIBRATION — label 30 cases by hand and compute Spearman correlation
     against the LLM judge. Replace judge with a fine-tuned evaluator if
     correlation < 0.70 on adversarial or escalation cases.

  4. LATENCY MONITORING — add p95 latency as a hard gate (< 8 s). Alert when
     avg latency climbs more than 30% week-over-week (model degradation signal).

  5. OUTCOME-BASED GROUND TRUTH — link agent recommendations to actual CRM
     retention outcomes (retained/churned 30 days later). Use this as the
     ultimate measure of offer quality; feed signal back to refine the offer
     catalog and system prompt.

  6. MULTI-JUDGE CONSENSUS — run two judge models independently (8b + 70b) and
     flag cases where they disagree by >= 2 points for human review. This builds
     a calibration dataset and surfaces systematic judge biases.
    """)

    # ── Save outputs ──────────────────────────────────────────────────────────
    results_json = [r.to_dict() for r in all_results]
    results_path = RESULTS_DIR / "eval_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"run_date": datetime.now().isoformat(),
                   "n_cases": n, "results": results_json}, f, indent=2, default=str)

    csv_path = RESULTS_DIR / "scorecard.csv"
    fieldnames = [
        "test_id", "category", "tool_recall", "tool_order", "escalation_correct",
        "completeness", "latency_ms", "auto_pass",
        "judge_fc", "judge_tu", "judge_act", "judge_hal", "judge_composite", "judge_pass",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            j = r.judge
            writer.writerow({
                "test_id": r.test_case["id"],
                "category": r.test_case["category"],
                "tool_recall": r.automated.tool_selection_recall,
                "tool_order": r.automated.tool_order_accuracy,
                "escalation_correct": r.automated.escalation_correct,
                "completeness": r.automated.response_completeness,
                "latency_ms": r.automated.latency_ms,
                "auto_pass": r.automated.pass_automated,
                "judge_fc":  j.factual_correctness       if j else "",
                "judge_tu":  j.tool_use_appropriateness  if j else "",
                "judge_act": j.actionability             if j else "",
                "judge_hal": j.hallucination             if j else "",
                "judge_composite": j.composite_score     if j else "",
                "judge_pass": j.overall_pass             if j else "",
            })

    print(f"\n  Results saved to:")
    print(f"    {results_path}")
    print(f"    {csv_path}")
    print()


if __name__ == "__main__":
    main()
