"""
evaluator.py — Automated metrics + LLM-as-judge evaluation pipeline.

Architecture
────────────
AutomatedMetrics (rule-based, no API required):
  • tool_selection_recall   — which expected tools were actually called?
  • tool_order_accuracy     — were tools called in the right order?
  • escalation_accuracy     — correct escalation decision?
  • response_completeness   — do key entities from tools appear in the response?
  • latency_ms              — wall-clock time for the agent response

LLMJudge (Groq llama-3.1-8b-instant):
  • factual_correctness      (1-5)  — does response accurately reflect tool data?
  • tool_use_appropriateness (1-5)  — right tools, right order, no waste?
  • actionability            (1-5)  — can the rep act on this immediately?
  • hallucination            (1-5)  — fully grounded (5) vs fabricated (1)?

Judge reliability discussion is in the module docstring below AutomatedMetrics.
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from groq import Groq


# ─────────────────────────────────────────────────────────────────────────────
# Automated Metrics
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AutomatedResult:
    test_id: str
    category: str
    tool_selection_recall: float          # 0-1
    tool_order_accuracy: float            # 0 or 1
    escalation_correct: bool
    response_completeness: float          # 0-1 heuristic
    latency_ms: float
    tools_called: list[str]
    tools_expected: list[str]
    pass_automated: bool


class AutomatedMetrics:
    """
    Pure rule-based metrics — no LLM required.

    Tool selection recall: |intersection(expected, called)| / |expected|
      Uses recall (not precision) because the agent may legitimately call
      extra tools (e.g. lookup first even when test case only expects predict).

    Tool order accuracy: 1 if every expected tool appears in the correct
      relative order in the called tool list, else 0.

    Escalation accuracy: True when expected_escalation matches whether
      escalate_to_supervisor appeared in the tool trace.

    Response completeness: a simple keyword heuristic — checks whether
      key_terms extracted from tool outputs appear in the response.
      Not a semantic check, but a fast baseline signal.

    Pass criterion: tool_selection_recall >= 0.75 AND escalation_correct.
    """

    @staticmethod
    def compute(
        test_case: dict,
        agent_response: "AgentResponse",  # type: ignore
    ) -> AutomatedResult:
        tools_called = [t.tool_name for t in agent_response.tool_trace]
        tools_expected = test_case.get("expected_tools", [])
        expected_order = test_case.get("expected_tool_order", [])
        expected_escalation = test_case.get("expected_escalation", False)

        # ── Tool selection recall ─────────────────────────────────────────────
        if tools_expected:
            matched = sum(1 for t in tools_expected if t in tools_called)
            recall = matched / len(tools_expected)
        else:
            # No tools expected — perfect if no tools were called
            recall = 1.0 if not tools_called else 0.7

        # ── Tool order accuracy ───────────────────────────────────────────────
        if expected_order:
            filtered = [t for t in tools_called if t in expected_order]
            order_correct = filtered == expected_order
        else:
            order_correct = not tools_called   # expect empty → order trivially correct
        order_acc = 1.0 if order_correct else 0.0

        # ── Escalation accuracy ───────────────────────────────────────────────
        actually_escalated = "escalate_to_supervisor" in tools_called
        escalation_correct = (actually_escalated == expected_escalation)

        # ── Response completeness (keyword heuristic) ─────────────────────────
        response_lower = (agent_response.content or "").lower()
        key_terms = AutomatedMetrics._extract_key_terms(agent_response.tool_trace)
        if key_terms:
            found = sum(1 for kw in key_terms if kw in response_lower)
            completeness = found / len(key_terms)
        else:
            completeness = 1.0  # no tool output → can't fail

        # ── Pass flag ─────────────────────────────────────────────────────────
        passed = (recall >= 0.75) and escalation_correct

        return AutomatedResult(
            test_id=test_case["id"],
            category=test_case["category"],
            tool_selection_recall=round(recall, 3),
            tool_order_accuracy=round(order_acc, 3),
            escalation_correct=escalation_correct,
            response_completeness=round(completeness, 3),
            latency_ms=round(agent_response.total_latency_ms, 1),
            tools_called=tools_called,
            tools_expected=tools_expected,
            pass_automated=passed,
        )

    @staticmethod
    def _extract_key_terms(tool_trace) -> list[str]:
        """
        Pull simple keyword signals from tool outputs:
        - risk_tier (high/medium/low)
        - offer names (lowercased)
        - customer_id
        Threshold: 50% of key terms must appear in the response.
        """
        terms = []
        for t in tool_trace:
            out = t.output or {}
            if isinstance(out, dict):
                tier = out.get("risk_tier", "")
                if tier:
                    terms.append(tier.lower())
                cid = out.get("customer_id", "")
                if cid:
                    terms.append(cid.lower().replace("tc-", "tc-"))
                offers = out.get("offers", [])
                if isinstance(offers, list):
                    for o in offers[:1]:  # just first offer name
                        name = o.get("name", "")
                        if name:
                            terms.append(name.lower().split()[0])  # first word
        return list(set(terms))


# ─────────────────────────────────────────────────────────────────────────────
# LLM-as-Judge
# ─────────────────────────────────────────────────────────────────────────────
JUDGE_MODEL = "llama-3.1-8b-instant"

# Deliberately separated from the main agent model to reduce self-evaluation bias.
# Using a smaller model (8b) as judge limits cost while remaining adequate for
# binary-flavoured rubric scoring.

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of AI-powered customer retention agents.
Score the agent's response on four dimensions using the rubrics below.

FACTUAL CORRECTNESS (1-5):
1 = Contains factual errors about the customer (wrong churn score, wrong contract type)
2 = One key fact incorrect or inconsistent with tool outputs
3 = Facts are correct but relevant tool data is omitted
4 = Correct and references most key data from tool outputs
5 = Perfectly accurate; every claim traceable to a tool output

TOOL USE APPROPRIATENESS (1-5):
1 = Critical tool skipped (e.g. no lookup before prediction) or wrong tool called
2 = Right tools but wrong order, or clearly unnecessary extra calls
3 = Mostly appropriate with one minor inefficiency
4 = Correct selection and order, no wasted calls
5 = Optimal chain; parameters extracted precisely from context

ACTIONABILITY FOR REP (1-5):
1 = Generic advice not tailored to this customer
2 = Mentions customer but no specific offer or next step
3 = Customer-specific recommendation but missing a clear rep script
4 = Clear offer + how to present it to the customer
5 = Prioritised offer, rep opening script, and a fallback offer included

HALLUCINATION (1-5 — higher = less hallucination):
1 = Significant invented data not from any tool output
2 = Multiple unsupported claims or extrapolations
3 = One minor unsupported claim
4 = Fully grounded with a minor imprecision
5 = Every statement traceable to a tool output or general domain knowledge

Return ONLY a JSON object with this exact structure:
{
  "factual_correctness": <int 1-5>,
  "tool_use_appropriateness": <int 1-5>,
  "actionability": <int 1-5>,
  "hallucination": <int 1-5>,
  "composite_score": <float, avg of above>,
  "justifications": {
    "factual_correctness": "<one sentence>",
    "tool_use_appropriateness": "<one sentence>",
    "actionability": "<one sentence>",
    "hallucination": "<one sentence>"
  },
  "overall_pass": <bool — true if composite_score >= 3.0>
}"""


@dataclass
class JudgeResult:
    test_id: str
    factual_correctness: int = 0
    tool_use_appropriateness: int = 0
    actionability: int = 0
    hallucination: int = 0
    composite_score: float = 0.0
    justifications: dict = field(default_factory=dict)
    overall_pass: bool = False
    raw_judge_response: str = ""
    judge_error: Optional[str] = None


class LLMJudge:
    """
    Uses llama-3.1-8b-instant (via Groq) to score agent responses.

    Reliability discussion:
    ──────────────────────
    • Inter-rater consistency: Small models exhibit higher variance than larger
      ones. Running the same case twice can produce ±1 on ordinal scores.
      Mitigation: use temperature=0 for the judge; report avg across 2 runs for
      high-stakes cases.

    • Positivity bias: LLMs tend to score generously (scores cluster at 4-5).
      The anchored rubric with explicit anchors at 1 and 2 partially counteracts
      this. Calibration against 5 gold-standard cases (2 obvious fails, 3 passes)
      is recommended before trusting aggregate numbers.

    • Self-evaluation proximity: Using a *different* (smaller) model for judging
      reduces the risk of a model grading its own outputs charitably. However,
      the 8b model may struggle with nuanced reasoning about tool call correctness.
      Consider moving to a stronger judge (e.g. llama-3.1-70b) for production.

    • Calibration approach: compare judge scores against a human-labelled sample
      of 20 cases. A Spearman correlation >= 0.7 suggests acceptable agreement.
    """

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise ValueError("Groq API key required for LLM judge.")
        self.client = Groq(api_key=key)

    def score(
        self,
        test_case: dict,
        agent_response: "AgentResponse",  # type: ignore
    ) -> JudgeResult:
        """Run the judge on one test case."""
        tool_summary = json.dumps(
            [{"tool": t.tool_name, "output_keys": list((t.output or {}).keys())}
             for t in agent_response.tool_trace],
            indent=2,
        )

        user_content = f"""TEST CASE
ID: {test_case['id']}
Category: {test_case['category']}
Rep Input: {test_case['input']}
Expected tools: {test_case.get('expected_tools', [])}
Expected escalation: {test_case.get('expected_escalation', False)}

AGENT RESPONSE
{agent_response.content}

TOOLS CALLED
{tool_summary}

QUALITY CRITERIA
{test_case.get('quality_criteria', 'N/A')}

Now score on all four dimensions and return the JSON object as specified."""

        try:
            resp = self.client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            raw = resp.choices[0].message.content or ""
            # Extract JSON from response (model may wrap it in markdown)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in judge response")
            data = json.loads(match.group())
            composite = round(
                (data.get("factual_correctness", 3) +
                 data.get("tool_use_appropriateness", 3) +
                 data.get("actionability", 3) +
                 data.get("hallucination", 3)) / 4, 2
            )
            return JudgeResult(
                test_id=test_case["id"],
                factual_correctness=int(data.get("factual_correctness", 3)),
                tool_use_appropriateness=int(data.get("tool_use_appropriateness", 3)),
                actionability=int(data.get("actionability", 3)),
                hallucination=int(data.get("hallucination", 3)),
                composite_score=composite,
                justifications=data.get("justifications", {}),
                overall_pass=bool(data.get("overall_pass", composite >= 3.0)),
                raw_judge_response=raw,
            )
        except Exception as e:
            return JudgeResult(
                test_id=test_case["id"],
                composite_score=0.0,
                judge_error=str(e),
                raw_judge_response="",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Combined evaluation result
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EvaluationResult:
    test_case: dict
    automated: AutomatedResult
    judge: Optional[JudgeResult]
    agent_response_dict: dict

    def to_dict(self) -> dict:
        return {
            "test_id": self.test_case["id"],
            "category": self.test_case["category"],
            "input": self.test_case["input"],
            "automated": {
                "tool_selection_recall": self.automated.tool_selection_recall,
                "tool_order_accuracy": self.automated.tool_order_accuracy,
                "escalation_correct": self.automated.escalation_correct,
                "response_completeness": self.automated.response_completeness,
                "latency_ms": self.automated.latency_ms,
                "tools_called": self.automated.tools_called,
                "tools_expected": self.automated.tools_expected,
                "pass": self.automated.pass_automated,
            },
            "judge": {
                "factual_correctness": self.judge.factual_correctness if self.judge else None,
                "tool_use_appropriateness": self.judge.tool_use_appropriateness if self.judge else None,
                "actionability": self.judge.actionability if self.judge else None,
                "hallucination": self.judge.hallucination if self.judge else None,
                "composite_score": self.judge.composite_score if self.judge else None,
                "pass": self.judge.overall_pass if self.judge else None,
                "justifications": self.judge.justifications if self.judge else {},
                "error": self.judge.judge_error if self.judge else None,
            } if self.judge else None,
            "agent_response": self.agent_response_dict.get("content", ""),
        }
