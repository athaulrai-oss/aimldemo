# TeleConnect Churn Prediction & Retention AI Agent

An end-to-end AI/ML solution for customer churn prediction and real-time retention assistance, built as part of the TeleConnect AI/ML Engineer assessment. The project is split into two connected parts: a machine learning pipeline (Part 1) and a live AI-powered agent for retention representatives (Part 2).

**Live Demo:** [aimldemo-m7iaziuqfeabbzrappgsbne.streamlit.app](https://aimldemo-m7iaziuqfeabbzrappgsbne.streamlit.app)

---

## Repository Structure

```
aimldemo/
├── app.py                    # Streamlit UI — chat panel + tool execution trace
├── agent.py                  # RetentionAgent — Groq LLM + raw tool-calling loop
├── tools.py                  # All 5 tool implementations + JSON schemas + registry
├── predict_churn.py          # Inference wrapper — loads joblib artifact, returns risk tier
├── evaluator.py              # LLM-as-judge + automated evaluation metrics
├── run_evaluation.py         # Evaluation runner — executes test suite, writes scorecard
├── smoke_agent.py            # Quick smoke test for the agent (no test framework needed)
├── test_suite.json           # 14 structured test cases across 7 categories
├── requirements.txt          # Python dependencies
├── part1_churn_model.ipynb   # Part 1 — full model training notebook with narrative
└── results/
    ├── cleaned_data.csv              # Cleaned customer dataset (5,050 rows)
    └── models/
        └── churn_model_v1.joblib     # Trained model artifact (Random Forest + preprocessor)
```

---

## Part 1 — Churn Prediction Model

### Dataset

| Property | Value |
|---|---|
| Source | Synthetic TeleConnect customer export |
| Rows | 5,050 |
| Features | 17 raw columns |
| Target | `churned` (0 = stayed, 1 = churned) |
| Class split | ~64% stayed / 36% churned |

### Data Quality Issues Found & Resolved

The dataset was exported from multiple legacy systems and required systematic cleaning before any modelling. Issues identified:

| Issue | Column(s) | Fix Applied |
|---|---|---|
| Inconsistent casing (`Month-to-month` vs `MONTH TO MONTH`) | `contract_type`, `internet_service`, `gender` | `.str.strip().str.title()` normalisation |
| Sentinel / placeholder values (`-1`, `999`) | `satisfaction_score`, `num_support_tickets` | Replaced with `NaN`, then median-imputed |
| Impossible numeric values (age > 100, negative charges) | `age`, `monthly_charges` | Clipped to domain-valid bounds |
| `total_charges` stored as string with whitespace | `total_charges` | Coerced to float; blank strings → `NaN` |
| Missing values across multiple columns | Various | Median for numerics, mode for categoricals |

### Feature Engineering

Four interaction features were added on top of the raw columns:

| Feature | Formula | Rationale |
|---|---|---|
| `charge_tenure_ratio` | `monthly_charges / (tenure_months + 1)` | Customers paying a lot relative to their tenure are disproportionately price-sensitive |
| `support_burden` | `num_support_tickets / (tenure_months + 1)` | Normalises ticket volume by tenure to catch early trouble signals |
| `sat_x_tickets` | `(10 - satisfaction_score) × num_support_tickets` | Multiplicative interaction — low satisfaction combined with high tickets is non-linear in risk |
| `longterm_mtm` | `1 if contract == "Month-to-month" and tenure > 24` | Loyal customers who remain uncommitted are a distinct churn risk profile |

### Models Trained

Two models from different families were trained and compared.

#### Logistic Regression (`scikit-learn`)

Linear baseline with a full preprocessing pipeline (median imputation → StandardScaler for numerics; mode imputation → OneHotEncoder for categoricals), composed via `ColumnTransformer`. Used to establish a performance floor and validate that feature relationships are not purely linear.

#### Random Forest + SMOTE (`scikit-learn` + `imbalanced-learn`)

SMOTE (Synthetic Minority Over-Sampling Technique) is applied exclusively on the training set after the 80/20 split to prevent data leakage. For each minority class sample, 5 nearest neighbours are located in scaled feature space and a synthetic point is interpolated:

```
x_synthetic = x_i + λ × (x_nn − x_i),    λ ~ Uniform(0, 1)
```

This balanced the training set to ~2,570 samples per class without duplicating real records.

### Evaluation Results

**Primary metric: PR-AUC (Precision-Recall AUC)**

ROC-AUC is optimistic under class imbalance because it incorporates true negatives — a metric that is easy to inflate when the majority class is large. PR-AUC focuses exclusively on the minority (churned) class and better reflects the business cost of a missed churner.

| Model | ROC-AUC | PR-AUC | F1 | Recall |
|---|---|---|---|---|
| **Random Forest + SMOTE** ✅ | **0.7155** | **0.5687** | 0.5530 | 0.5831 |
| Logistic Regression + SMOTE | 0.7105 | 0.5495 | 0.5989 | 0.7466 |
| Random Forest (no SMOTE, baseline) | 0.7150 | 0.5621 | 0.5491 | 0.5640 |

The Random Forest + SMOTE was selected as the production artifact. While Logistic Regression achieves higher recall, it does so at lower precision — generating more false positives, which in a retention context means unnecessary discount offers to customers who were not actually at risk.

### Top Risk Factors (Feature Importance)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `tenure_months` | 0.1074 |
| 2 | `total_charges` | 0.0982 |
| 3 | `avg_monthly_gb_used` | 0.0962 |
| 4 | `avg_monthly_minutes` | 0.0941 |
| 5 | `monthly_charges` | 0.0923 |

Per-prediction explanations use SHAP `TreeExplainer` for the Random Forest (exact values) and `coef × standardised_feature` for the Logistic Regression fallback.

### Inference Function

```python
from predict_churn import predict_churn

result = predict_churn({
    "tenure_months": 3,
    "satisfaction_score": 2.1,
    "contract_type": "Month-to-month",
    "num_support_tickets": 5,
    "monthly_charges": 85.00,
    "internet_service": "Fiber optic"
})

# Returns:
# {
#     "churn_probability": 0.8134,
#     "risk_tier": "high",          # high >= 0.65 | medium >= 0.35 | low < 0.35
#     "top_risk_factors": [
#         {"feature": "sat_x_tickets",  "shap_value": 0.2341, "direction": "increases churn risk"},
#         {"feature": "tenure_months",  "shap_value": 0.1876, "direction": "increases churn risk"},
#         {"feature": "support_burden", "shap_value": 0.1203, "direction": "increases churn risk"}
#     ]
# }
```

---

## Part 2 — Retention AI Agent

### Architecture

```
User (Streamlit chat)
        │
        ▼
  RetentionAgent (agent.py)
        │  ← Groq API  [Llama 3.3 70B Versatile]
        │
        ├── lookup_customer()         CSV lookup by customer ID → profile dict
        ├── predict_churn()           Calls predict_churn.py → risk tier + SHAP factors
        ├── get_retention_offers()    Offer catalog filtered by risk tier + contract type
        ├── escalate_to_supervisor()  Triggered on legal/regulatory/dispute signals
        └── log_interaction()         CRM-style outcome record (JSONL append)
```

The agent is fully stateless — conversation history is managed by the Streamlit app and passed in per call. Adding a new tool requires changes only in `tools.py` (function + JSON schema + one registry entry). `agent.py` is never modified.

### LLM: Llama 3.3 70B via Groq

| Property | Value |
|---|---|
| Model | `llama-3.3-70b-versatile` |
| Provider | Meta (open-weight, Llama 3.3 licence) |
| Inference | Groq API (LPU-based, ~10–20× faster than GPU providers) |
| Temperature | 0.1 (consistent tool calling) |
| Max tokens | 2,048 per response |

Two alternative models are selectable in the UI sidebar:
- `llama-3.1-70b-versatile` — Meta, open-weight
- `mixtral-8x7b-32768` — Mistral AI, open-weight, Apache 2.0

### Standard Tool-Calling Flow

**Input from rep:**
```
I have customer TC-004711 on the line and they're thinking about cancelling.
```

**Step 1 — `lookup_customer("TC-004711")`**
```json
{
  "status": "found",
  "customer_id": "TC-004711",
  "tenure_months": 3.0,
  "contract_type": "Month-to-month",
  "monthly_charges": 85.5,
  "satisfaction_score": 2.1,
  "num_support_tickets": 5
}
```

**Step 2 — `predict_churn(customer_features)`**
```json
{
  "churn_probability": 0.8134,
  "risk_tier": "high",
  "top_risk_factors": [
    {"feature": "sat_x_tickets",  "direction": "increases churn risk"},
    {"feature": "tenure_months",  "direction": "increases churn risk"},
    {"feature": "support_burden", "direction": "increases churn risk"}
  ]
}
```

**Step 3 — `get_retention_offers("high", "Month-to-month")`**
```json
{
  "offers": [{
    "offer_id": "R-H-M2M-01",
    "name": "Loyalty Lock-In Discount",
    "description": "Commit to 1-year contract, receive 20% off for 12 months",
    "rep_script": "I can freeze your current rate with a 20% discount — that's an immediate saving from next month if you sign a one-year agreement today.",
    "estimated_retention_uplift": "62%"
  }]
}
```

**Agent final response (natural language):**
> TC-004711 is HIGH risk — 81% churn probability. Key drivers: very low satisfaction (2.1/10), 5 support tickets in only 3 months tenure, and a high support burden ratio. Lead with the **Loyalty Lock-In Discount** — 20% off for 12 months in exchange for a 1-year commitment. Script: *"I can freeze your current rate with a 20% discount — that's an immediate saving from next month if you sign today."* Fallback: Speed Upgrade + $50 Bill Credit (R-H-M2M-03).

### Escalation Handling

The agent escalates immediately when legal, regulatory, or abusive signals are detected — without attempting to negotiate or offer discounts.

**Trigger keywords:** `lawyer`, `sue`, `lawsuit`, `legal action`, `regulator`, `regulatory`, `harassment`, `threatening`

**Example:**
```
Customer TC-004711 says they'll sue us and have already contacted their lawyer.
```
Tool call: `lookup_customer` → `escalate_to_supervisor` (priority: `urgent`)

The agent instructs the rep to place the customer on hold, does not make any commitments, and logs the escalation with an estimated 2–5 minute supervisor wait time.

### Ambiguity Handling

| Scenario | Agent Behaviour |
|---|---|
| No customer ID provided | Asks for the TC-XXXXXX ID before calling any tools |
| Name given but no ID | Explains the system requires an ID, not a name |
| ID not found in dataset | Returns `not_found` clearly — does not hallucinate a profile |
| Out-of-scope request | Declines politely, explains what the agent can help with |

### Evaluation Framework

**Automated metrics (`evaluator.py`):**

- **Tool Selection Accuracy** — did the agent call the correct tools?
- **Tool Order Accuracy** — were tools called in the correct sequence?
- **Escalation Detection** — did the agent escalate when required, and not escalate when it shouldn't?
- **Hallucination Check** — does the response contain figures or IDs not present in any tool output?

**LLM-as-Judge rubric (anchored 1–5 per dimension):**

| Dimension | Score 1 | Score 3 | Score 5 |
|---|---|---|---|
| Factual correctness | Contradicts tool output | Mostly accurate, minor errors | All facts match tool outputs exactly |
| Tool use appropriateness | Wrong tools or wrong order | Correct tools, suboptimal order | Correct tools, correct order, no redundant calls |
| Actionability for rep | Raw JSON or vague advice | Some actionable elements | Risk summary + specific offer + opening script |
| Hallucination | Invents customer data | Occasional unsupported inference | All claims traceable to a tool output |

**Test suite: 14 cases across 7 categories**

| Category | Test IDs | Description |
|---|---|---|
| Happy path — single tool | TC001, TC002 | Customer lookup, direct feature prediction |
| Multi-step chaining | TC003, TC004, TC005, TC011, TC014 | Full lookup → predict → offers → log flows |
| Ambiguous input | TC006, TC007 | No ID provided, name instead of ID |
| Out-of-scope | TC008 | Marketing request — agent must decline |
| Escalation triggers | TC009, TC010 | Legal threat, regulatory complaint |
| Adversarial / edge cases | TC012, TC013 | Prompt injection, nonexistent customer ID |

### Demo Customer IDs

| Customer ID | Profile |
|---|---|
| `TC-004711` | High risk, month-to-month, low satisfaction |
| `TC-000066` | Low risk, two-year plan, long tenure |
| `TC-003427` | Long tenure, month-to-month, price complaint |
| `TC-000692` | Short tenure, potential escalation scenario |
| `TC-000829` | Model disagreement candidate (medium model score, high qualitative risk) |

---

## Setup & Local Run

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com) — no credit card required

### Install

```bash
git clone https://github.com/athaulrai/aimldemo.git
cd aimldemo
pip install -r requirements.txt
```

### Run the Streamlit app

```bash
export GROQ_API_KEY="gsk_..."
streamlit run app.py
```

### Run the smoke test

```bash
export GROQ_API_KEY="gsk_..."
python smoke_agent.py
```

### Run the evaluation suite

```bash
export GROQ_API_KEY="gsk_..."
python run_evaluation.py
```

### Run inference only (no agent)

```bash
python predict_churn.py
# Runs the sample in __main__ and prints JSON output
```

---

## Open-Source Stack

| Component | Library / Model | Licence |
|---|---|---|
| LLM (primary) | Llama 3.3 70B via Groq | Meta Llama 3.3 (open-weight) |
| LLM (alt 1) | Llama 3.1 70B via Groq | Meta Llama 3.1 (open-weight) |
| LLM (alt 2) | Mixtral 8x7B via Groq | Apache 2.0 |
| ML model | scikit-learn `RandomForestClassifier` | BSD |
| Imbalance handling | imbalanced-learn `SMOTE` | MIT |
| Explainability | `shap` TreeExplainer | MIT |
| Data handling | pandas, numpy | BSD |
| Model serialisation | joblib | BSD |
| Gradient boosting (evaluated) | xgboost | Apache 2.0 |
| Web UI | Streamlit | Apache 2.0 |
| LLM client | groq Python SDK | Apache 2.0 |

No proprietary model APIs (OpenAI, Anthropic, Azure OpenAI) are used anywhere in the stack.

---

## Deployment

Deployed on **Streamlit Community Cloud**.

- **Main file:** `app.py` (flat repo — all files at root level)
- **Secrets:** Groq API key injected via `st.secrets["GROQ_API_KEY"]`
- **Model artifact:** `results/models/churn_model_v1.joblib` committed to the repo, loaded at startup
- **Customer data:** `results/cleaned_data.csv` committed to the repo, loaded once at import time

To redeploy: push to `main`. Streamlit Cloud picks up the change automatically.

---

## Design Decisions

**Why Groq over OpenAI/Anthropic?**
Groq serves open-weight models on custom LPU hardware at speeds typically 10–20× faster than GPU-based providers. For a retention agent where the rep has a live customer on the line, latency directly affects the product experience. Groq's free tier is also sufficient for evaluation and demo purposes without incurring API costs.

**Why no LangChain?**
The agent orchestration loop is ~80 lines of raw Python. Using a framework for a 5-tool agent adds abstraction without benefit and makes the tool-calling logic harder to inspect, test, and extend. The design principle is: adding a sixth tool touches only `tools.py`.

**Why PR-AUC over ROC-AUC?**
ROC-AUC includes true negatives in its calculation, which are trivially easy to accumulate when the majority class is large. For a churn use case the business cost is asymmetric — a missed churner (false negative) is more expensive than a false alarm (false positive). PR-AUC focuses on the minority class and reflects this cost more honestly.

**Why Random Forest over XGBoost?**
XGBoost was evaluated but did not outperform Random Forest on this dataset at the given scale. Random Forest + SMOTE achieved the highest PR-AUC (0.5687) and is simpler to deploy (no special serialisation concerns, no booster version pinning).

---

*All customer data in this project is synthetic.*
