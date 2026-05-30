"""
tools.py — All five tool implementations for the TeleConnect Retention Agent.

Design principle: every function is independently testable and returns a
consistent dict structure.  Adding a sixth tool means:
  1. Add the function here
  2. Add its JSON schema to TOOL_DEFINITIONS
  3. Add it to TOOL_REGISTRY
  — no changes to agent.py required.
"""

import json
import uuid
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Paths — resolve relative to this file so the tools work from any cwd
# ─────────────────────────────────────────────────────────────────────────────
_HERE    = Path(__file__).parent
_ROOT    = _HERE.parent
_DATA    = _ROOT / "results" / "cleaned_data.csv"
_LOG_FILE = _HERE / "results" / "interaction_log.jsonl"
_ESC_FILE = _HERE / "results" / "escalation_log.jsonl"

# ── Load customer dataset once at import time ─────────────────────────────────
try:
    _CUSTOMERS = pd.read_csv(_DATA, dtype={"customer_id": str})
    _CUSTOMERS["customer_id"] = _CUSTOMERS["customer_id"].str.strip()
    _CUSTOMER_COLS = [
        "customer_id", "age", "gender", "tenure_months", "contract_type",
        "monthly_charges", "total_charges", "internet_service", "phone_service",
        "avg_monthly_gb_used", "num_support_tickets", "avg_monthly_minutes",
        "satisfaction_score", "payment_method", "num_additional_services",
        "days_since_contact",
    ]
    _CUSTOMERS_CLEAN = _CUSTOMERS[[c for c in _CUSTOMER_COLS if c in _CUSTOMERS.columns]]
except Exception:
    _CUSTOMERS = None
    _CUSTOMERS_CLEAN = None

# ── Load predict_churn from Part 1 (graceful mock fallback) ──────────────────
_REAL_MODEL = False
try:
    sys.path.insert(0, str(_ROOT))
    from predict_churn import predict_churn as _real_predict_churn
    _REAL_MODEL = True
except Exception:
    def _real_predict_churn(features: dict) -> dict:
        """Simple rule-based mock when model artifact is unavailable."""
        score = 0.35
        sat = float(features.get("satisfaction_score", 5))
        tenure = float(features.get("tenure_months", 24))
        tickets = float(features.get("num_support_tickets", 1))
        contract = str(features.get("contract_type", "Month-to-month"))
        if sat < 4:            score += 0.22
        if contract == "Month-to-month": score += 0.15
        if tenure < 6:         score += 0.12
        if tickets > 3:        score += 0.10
        score = round(min(score, 0.97), 4)
        tier = "high" if score >= 0.65 else ("medium" if score >= 0.35 else "low")
        return {
            "churn_probability": score,
            "risk_tier": tier,
            "top_risk_factors": [
                {"feature": "satisfaction_score", "shap_value": round(-(sat - 5) * 0.15, 4),
                 "direction": "increases churn risk" if sat < 5 else "decreases churn risk"},
                {"feature": "contract_type", "shap_value": 0.15 if "month" in contract.lower() else -0.10,
                 "direction": "increases churn risk" if "month" in contract.lower() else "decreases churn risk"},
                {"feature": "tenure_months", "shap_value": round(-tenure * 0.003, 4),
                 "direction": "decreases churn risk"},
            ],
            "_mock": True,
        }

# ─────────────────────────────────────────────────────────────────────────────
# OFFER CATALOG
# ─────────────────────────────────────────────────────────────────────────────
_OFFERS = {
    ("high", "Month-to-month"): [
        {"offer_id": "R-H-M2M-01", "name": "Loyalty Lock-In Discount",
         "description": "Commit to a 1-year contract and receive 20% off monthly charges for 12 months",
         "key_benefit": "Locks in rate + saves ~$166/year at median charge",
         "rep_script": "I can freeze your current rate with a 20% discount — that's an immediate saving from next month if you sign a one-year agreement today.",
         "estimated_retention_uplift": "62%", "cost_tier": "medium"},
        {"offer_id": "R-H-M2M-02", "name": "Premium Bundle Upgrade",
         "description": "Add TV + Home Security package at 50% off for 6 months",
         "key_benefit": "Adds tangible value to offset price concerns",
         "rep_script": "We can add our TV + Security bundle to your account at half price for 6 months — that's $30/month of savings in added services.",
         "estimated_retention_uplift": "54%", "cost_tier": "medium"},
        {"offer_id": "R-H-M2M-03", "name": "Speed Upgrade + Bill Credit",
         "description": "Free internet speed tier upgrade plus $50 immediate bill credit",
         "key_benefit": "Instant $50 credit shows goodwill quickly",
         "rep_script": "I'd like to apply a $50 credit to your next invoice right now, and upgrade your internet speed — no charge.",
         "estimated_retention_uplift": "47%", "cost_tier": "low"},
    ],
    ("high", "One year"): [
        {"offer_id": "R-H-1Y-01", "name": "Two-Year Commitment Discount",
         "description": "Upgrade to a 2-year contract and lock in current rates + 15% discount",
         "key_benefit": "Price certainty for 2 years",
         "rep_script": "If you're happy with the service, locking in for two years protects you from any future price increases and saves you 15%.",
         "estimated_retention_uplift": "70%", "cost_tier": "medium"},
        {"offer_id": "R-H-1Y-02", "name": "Loyalty Service Credit",
         "description": "$75 bill credit + priority support tier for 12 months",
         "key_benefit": "Immediate financial relief + service quality improvement",
         "rep_script": "As a valued customer I want to apply a $75 credit today and move you to our priority support queue.",
         "estimated_retention_uplift": "55%", "cost_tier": "low"},
    ],
    ("high", "Two year"): [
        {"offer_id": "R-H-2Y-01", "name": "Early Renewal Discount",
         "description": "Renew contract early and receive 25% off for first 3 months of new term",
         "key_benefit": "Reward loyalty, remove uncertainty about renewal",
         "rep_script": "I can process an early renewal right now with a 3-month discount — no interruption in service.",
         "estimated_retention_uplift": "65%", "cost_tier": "medium"},
        {"offer_id": "R-H-2Y-02", "name": "Satisfaction Recovery Package",
         "description": "Free service review + $100 credit + dedicated account manager for 90 days",
         "key_benefit": "Addresses root cause if satisfaction is the driver",
         "rep_script": "I want to schedule a service review call, apply a $100 credit, and personally ensure the issues are resolved.",
         "estimated_retention_uplift": "60%", "cost_tier": "high"},
    ],
    ("medium", "Month-to-month"): [
        {"offer_id": "R-M-M2M-01", "name": "Annual Contract Incentive",
         "description": "Switch to annual plan with 10% discount for 6 months",
         "rep_script": "We're offering a 10% discount for the next 6 months if you'd like to move to an annual plan.",
         "estimated_retention_uplift": "45%", "cost_tier": "low"},
        {"offer_id": "R-M-M2M-02", "name": "Free Add-On Service",
         "description": "Add one additional service (e.g. cloud storage or security) free for 3 months",
         "rep_script": "I can add our cloud backup service to your account at no charge for 3 months as a thank-you for being with us.",
         "estimated_retention_uplift": "38%", "cost_tier": "low"},
    ],
    ("medium", "One year"): [
        {"offer_id": "R-M-1Y-01", "name": "Loyalty Upgrade",
         "description": "Free speed upgrade + $25 bill credit",
         "rep_script": "As a loyal customer I'd like to upgrade your internet speed and apply a small thank-you credit.",
         "estimated_retention_uplift": "42%", "cost_tier": "low"},
    ],
    ("medium", "Two year"): [
        {"offer_id": "R-M-2Y-01", "name": "Renewal Reminder + Rate Lock",
         "description": "Early renewal confirmation with current rate guaranteed",
         "rep_script": "I want to confirm your renewal is secure and your rates are locked — no surprises.",
         "estimated_retention_uplift": "50%", "cost_tier": "none"},
    ],
    ("low", "Month-to-month"): [
        {"offer_id": "R-L-M2M-01", "name": "Contract Stability Offer",
         "description": "5% discount for switching to annual contract",
         "rep_script": "Just to flag — moving to an annual plan would save you 5% and locks in your current rate.",
         "estimated_retention_uplift": "30%", "cost_tier": "none"},
    ],
    ("low", "One year"): [
        {"offer_id": "R-L-1Y-01", "name": "Loyalty Acknowledgment",
         "description": "Thank-you note + renewal confirmation",
         "rep_script": "I just want to confirm you're all set for renewal and thank you for staying with us.",
         "estimated_retention_uplift": "25%", "cost_tier": "none"},
    ],
    ("low", "Two year"): [
        {"offer_id": "R-L-2Y-01", "name": "Proactive Check-In",
         "description": "Satisfaction survey + minor service improvement if requested",
         "rep_script": "You're one of our most valued long-term customers. I'm just checking in — is there anything we can improve?",
         "estimated_retention_uplift": "20%", "cost_tier": "none"},
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────

def lookup_customer(customer_id: str) -> dict:
    """
    Retrieve a customer profile by ID.
    Returns demographics, contract, tenure, charges, satisfaction.
    """
    customer_id = str(customer_id).strip()
    if _CUSTOMERS_CLEAN is None:
        return {"status": "error", "message": "Customer database unavailable"}

    row = _CUSTOMERS_CLEAN[_CUSTOMERS_CLEAN["customer_id"] == customer_id]
    if row.empty:
        return {
            "status": "not_found",
            "customer_id": customer_id,
            "message": f"No customer found with ID '{customer_id}'. Please verify the ID.",
        }

    profile = row.iloc[0].to_dict()
    # Convert numpy types to native Python for JSON serialisation
    profile = {k: (float(v) if hasattr(v, "item") else v) for k, v in profile.items()}
    # Round floats for readability
    for k in ["age", "tenure_months", "monthly_charges", "total_charges",
              "avg_monthly_gb_used", "avg_monthly_minutes", "satisfaction_score"]:
        if k in profile and profile[k] is not None:
            profile[k] = round(float(profile[k]), 2)
    profile["num_support_tickets"] = int(profile.get("num_support_tickets", 0))
    profile["num_additional_services"] = int(profile.get("num_additional_services", 0))
    profile["status"] = "found"
    return profile


def predict_churn(customer_features: dict) -> dict:
    """
    Run the churn prediction model on customer features.
    Returns churn_probability (0-1), risk_tier, and top 3 risk factors.
    """
    if not customer_features:
        return {"status": "error", "message": "customer_features cannot be empty"}
    try:
        result = _real_predict_churn(customer_features)
        result["status"] = "ok"
        result["model"] = "churn_model_v1 (Logistic Regression)" if _REAL_MODEL else "mock_model"
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_retention_offers(risk_tier: str, contract_type: str) -> dict:
    """
    Return retention offers filtered by the customer's risk tier and contract type.
    """
    risk_tier = risk_tier.lower().strip()
    # Normalise contract_type
    ct_map = {
        "month-to-month": "Month-to-month",
        "month to month": "Month-to-month",
        "one year": "One year",
        "1 year": "One year",
        "two year": "Two year",
        "2 year": "Two year",
    }
    contract_type = ct_map.get(contract_type.lower().strip(), contract_type)
    key = (risk_tier, contract_type)
    # Fallback to month-to-month if exact key not found
    offers = _OFFERS.get(key) or _OFFERS.get((risk_tier, "Month-to-month")) or []
    return {
        "risk_tier": risk_tier,
        "contract_type": contract_type,
        "offers": offers,
        "total_offers": len(offers),
        "status": "ok",
    }


def log_interaction(
    customer_id: str,
    outcome: str,
    notes: str = "",
    offer_accepted: str = "",
) -> dict:
    """
    Record the outcome of a retention conversation.
    outcome: one of 'retained', 'churned', 'escalated', 'callback_scheduled', 'no_action'
    """
    VALID_OUTCOMES = {"retained", "churned", "escalated", "callback_scheduled", "no_action"}
    if outcome not in VALID_OUTCOMES:
        return {"status": "error",
                "message": f"Invalid outcome '{outcome}'. Must be one of: {VALID_OUTCOMES}"}

    record = {
        "log_id": f"LOG-{datetime.now().strftime('%Y%m%d%H%M%S')}-{customer_id}",
        "customer_id": customer_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "outcome": outcome,
        "offer_accepted": offer_accepted,
        "notes": notes,
        "agent_version": "v1.0",
    }
    _LOG_FILE.parent.mkdir(exist_ok=True)
    with open(_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return {"status": "logged", **record}


def escalate_to_supervisor(
    customer_id: str,
    reason: str,
    context_summary: str = "",
) -> dict:
    """
    Transfer the case to a human supervisor with context.
    Use for: legal threats, billing disputes > $200, abusive customers,
             regulatory complaints, or any situation outside agent scope.
    """
    PRIORITY_KEYWORDS = {"legal", "lawyer", "attorney", "sue", "lawsuit",
                         "regulator", "regulatory", "threat", "harassment"}
    priority = "urgent" if any(kw in reason.lower() for kw in PRIORITY_KEYWORDS) else "normal"

    record = {
        "escalation_id": f"ESC-{datetime.now().strftime('%Y%m%d%H%M%S')}-{customer_id}",
        "customer_id": customer_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "reason": reason,
        "priority": priority,
        "context_summary": context_summary,
        "status": "escalated",
        "assigned_to": "Supervisor Queue",
        "estimated_wait_minutes": "2-5" if priority == "urgent" else "10-15",
        "instructions_for_rep": (
            "Place customer on a brief hold and stay on the line. "
            "A supervisor will join within the estimated wait time. "
            "Do NOT make any commitments regarding resolution."
        ),
    }
    _ESC_FILE.parent.mkdir(exist_ok=True)
    with open(_ESC_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


# ─────────────────────────────────────────────────────────────────────────────
# TOOL REGISTRY — maps tool name → function
# ─────────────────────────────────────────────────────────────────────────────
TOOL_REGISTRY = {
    "lookup_customer":      lookup_customer,
    "predict_churn":        predict_churn,
    "get_retention_offers": get_retention_offers,
    "log_interaction":      log_interaction,
    "escalate_to_supervisor": escalate_to_supervisor,
}


def execute_tool(name: str, args: dict) -> dict:
    """Dispatch a tool call by name. Returns error dict on unknown tool."""
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return {"status": "error", "message": f"Unknown tool: '{name}'"}
    try:
        return fn(**args)
    except TypeError as e:
        return {"status": "error", "message": f"Invalid arguments for {name}: {e}"}
    except Exception as e:
        return {"status": "error", "message": f"Tool '{name}' raised: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL DEFINITIONS — JSON Schema for Groq / OpenAI function calling API
# ─────────────────────────────────────────────────────────────────────────────
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_customer",
            "description": (
                "Retrieve a customer profile by their TeleConnect customer ID. "
                "ALWAYS call this first before running churn prediction or retrieving offers. "
                "Returns demographics, contract type, charges, satisfaction, and service history."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "TeleConnect customer ID, format TC-XXXXXX (e.g. TC-004711)"
                    }
                },
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_churn",
            "description": (
                "Run the trained churn prediction model on a customer's features. "
                "Call this after lookup_customer to get churn_probability (0-1), "
                "risk_tier ('high'/'medium'/'low'), and the top 3 risk factors driving the score."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_features": {
                        "type": "object",
                        "description": (
                            "Dictionary of customer feature values from lookup_customer output. "
                            "Pass the entire profile dict from lookup_customer."
                        ),
                    }
                },
                "required": ["customer_features"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_retention_offers",
            "description": (
                "Return a list of retention offers filtered by the customer's churn risk tier "
                "and current contract type. Call this after predict_churn to get specific offers "
                "the rep can present."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "risk_tier": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Customer's churn risk tier from predict_churn"
                    },
                    "contract_type": {
                        "type": "string",
                        "enum": ["Month-to-month", "One year", "Two year"],
                        "description": "Customer's current contract type from lookup_customer"
                    },
                },
                "required": ["risk_tier", "contract_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_interaction",
            "description": (
                "Record the outcome of a retention conversation for CRM and reporting. "
                "Call this after the conversation is resolved or when the rep needs to log a callback."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "Customer ID"},
                    "outcome": {
                        "type": "string",
                        "enum": ["retained", "churned", "escalated",
                                 "callback_scheduled", "no_action"],
                        "description": "Result of the retention conversation"
                    },
                    "notes": {
                        "type": "string",
                        "description": "Free-text notes about the conversation"
                    },
                    "offer_accepted": {
                        "type": "string",
                        "description": "offer_id of the offer the customer accepted (if any)"
                    },
                },
                "required": ["customer_id", "outcome"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_supervisor",
            "description": (
                "Transfer the case to a human supervisor. Use this for: "
                "(1) legal threats or mentions of lawyers/lawsuits, "
                "(2) regulatory complaints, "
                "(3) billing disputes exceeding $200 that cannot be explained, "
                "(4) abusive or threatening customer behaviour, "
                "(5) customer explicitly requesting a supervisor. "
                "Do NOT attempt to resolve these situations without escalating."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "Customer ID"},
                    "reason": {
                        "type": "string",
                        "description": "Brief reason for escalation (1-2 sentences)"
                    },
                    "context_summary": {
                        "type": "string",
                        "description": "Summary of the conversation so far for the supervisor"
                    },
                },
                "required": ["customer_id", "reason"],
            },
        },
    },
]
