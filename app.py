"""
app.py — Streamlit demo for the TeleConnect Retention AI Agent.

Layout:
  ┌─────────────────────────────────────────────────────────────────┐
  │  Sidebar: API key, model selector, demo customer IDs           │
  ├───────────────────────────┬─────────────────────────────────────┤
  │  Left (60%): Chat UI      │  Right (40%): Tool Execution Trace │
  └───────────────────────────┴─────────────────────────────────────┘

Deployment (Streamlit Community Cloud):
  1. Push repo to GitHub (root contains predict_churn.py & results/models/)
  2. Create app at share.streamlit.io → set Main file: part2/app.py
  3. Add secret: GROQ_API_KEY = "gsk_..."
"""

import json
import os
import sys
from pathlib import Path

import streamlit as st

# ── Path setup — works from repo root or from part2/ ──────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TeleConnect Retention Agent",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.risk-high   { color: #e53e3e; font-weight: 700; font-size: 1.1em; }
.risk-medium { color: #dd6b20; font-weight: 700; font-size: 1.1em; }
.risk-low    { color: #38a169; font-weight: 700; font-size: 1.1em; }
.tool-header { font-size: 0.85em; font-weight: 600; color: #4a5568; }
.tool-badge  {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 0.75em; font-weight: 600; background: #ebf4ff; color: #2b6cb0;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DEMO_CUSTOMERS = {
    "TC-004711": "High-risk, month-to-month",
    "TC-000066": "Low-risk, two-year plan",
    "TC-003427": "Long-tenure, month-to-month",
    "TC-000692": "Short-tenure, potential escalation",
    "TC-000829": "Model disagreement candidate",
}

AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
]

# ─────────────────────────────────────────────────────────────────────────────
# API key resolution  (secrets > env var > sidebar input)
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_api_key(user_input: str) -> str:
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "") or user_input


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []       # conversation history for multi-turn
if "tool_traces" not in st.session_state:
    st.session_state.tool_traces = []    # list of per-turn tool trace lists
if "agent" not in st.session_state:
    st.session_state.agent = None
if "last_risk" not in st.session_state:
    st.session_state.last_risk = None    # "high" | "medium" | "low" | None


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    api_input = st.text_input(
        "Groq API Key",
        type="password",
        help="Free key at console.groq.com — no credit card required.",
        placeholder="gsk_...",
    )
    api_key = _resolve_api_key(api_input)

    selected_model = st.selectbox("Model", AVAILABLE_MODELS, index=0,
                                  help="All models are open-weight (Llama / Mixtral) served via Groq.")

    if api_key and (st.session_state.agent is None or
                    getattr(st.session_state.agent, "model", "") != selected_model):
        try:
            from agent import RetentionAgent
            st.session_state.agent = RetentionAgent(
                api_key=api_key, model=selected_model
            )
            st.success("Agent ready", icon="✅")
        except Exception as e:
            st.error(f"Failed to initialise agent: {e}")
    elif not api_key:
        st.warning("Enter a Groq API key to start.", icon="🔑")

    st.divider()
    st.markdown("**Demo Customer IDs**")
    st.markdown("*Click to copy, paste into chat*")
    for cid, desc in DEMO_CUSTOMERS.items():
        st.code(cid)
        st.caption(desc)

    st.divider()
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.tool_traces = []
        st.session_state.last_risk = None
        st.rerun()

    st.divider()
    st.caption(
        "**TeleConnect Retention Agent** · Part 2\n\n"
        "Powered by [Groq](https://groq.com) + Llama 3.3 70B (open-weight)\n\n"
        "_All customer data is synthetic._"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("📞 TeleConnect Retention AI Agent")
st.caption(
    "An AI copilot for retention representatives — lookup customer risk, "
    "get tailored offers, and log outcomes. All tool calls are visible on the right."
)

# Risk badge at top (updates after each prediction)
if st.session_state.last_risk:
    risk = st.session_state.last_risk
    colour = {"high": "red", "medium": "orange", "low": "green"}.get(risk, "gray")
    st.markdown(
        f"**Last assessed risk:** "
        f"<span class='risk-{risk}'>{risk.upper()}</span>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# Two-column layout
# ─────────────────────────────────────────────────────────────────────────────
col_chat, col_trace = st.columns([1.4, 1.0])


# ── LEFT: Chat ────────────────────────────────────────────────────────────────
with col_chat:
    st.subheader("💬 Chat")

    # Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    placeholder = (
        "e.g. 'I have customer TC-004711 on the line thinking about cancelling'"
        if api_key else "Enter Groq API key in the sidebar to start"
    )
    user_input = st.chat_input(placeholder, disabled=not bool(api_key))

    if user_input and st.session_state.agent:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Run agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    ar = st.session_state.agent.run(
                        user_input,
                        conversation_history=st.session_state.messages[:-1],
                    )
                    response_text = ar.content or "_No response generated._"
                    st.markdown(response_text)

                    # Store tool trace for the right panel
                    st.session_state.tool_traces.append(ar.tool_trace)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response_text}
                    )

                    # Update risk badge
                    for t in ar.tool_trace:
                        if t.tool_name == "predict_churn":
                            tier = (t.output or {}).get("risk_tier", "")
                            if tier:
                                st.session_state.last_risk = tier

                    # Latency info
                    st.caption(f"⏱ {ar.total_latency_ms / 1000:.1f}s · {len(ar.tool_trace)} tool call(s)")

                except Exception as e:
                    st.error(f"Agent error: {e}")
                    st.session_state.tool_traces.append([])

        st.rerun()


# ── RIGHT: Tool Trace ─────────────────────────────────────────────────────────
with col_trace:
    st.subheader("🔧 Tool Execution Trace")

    if not st.session_state.tool_traces:
        st.info("Tool calls will appear here as the agent works.")
    else:
        total_turns = len(st.session_state.tool_traces)
        for turn_idx, trace in enumerate(reversed(st.session_state.tool_traces), 1):
            turn_num = total_turns - turn_idx + 1
            if not trace:
                with st.expander(f"Turn {turn_num} — no tool calls", expanded=False):
                    st.caption("Agent responded without calling any tools.")
                continue

            with st.expander(
                f"Turn {turn_num} — {len(trace)} tool call(s)",
                expanded=(turn_idx == 1)
            ):
                for t in trace:
                    # Tool badge
                    st.markdown(
                        f"<span class='tool-badge'>#{t.sequence} {t.tool_name}</span>",
                        unsafe_allow_html=True
                    )

                    in_tab, out_tab = st.tabs(["📥 Input", "📤 Output"])
                    with in_tab:
                        st.json(t.input_args)
                    with out_tab:
                        out = t.output or {}
                        # Special rendering for key fields
                        if "risk_tier" in out:
                            tier = out["risk_tier"]
                            prob = out.get("churn_probability")
                            prob_str = f"({prob:.0%})" if isinstance(prob, (int, float)) else ""
                            st.markdown(
                                f"Risk: <span class='risk-{tier}'>{tier.upper()}</span> "
                                f"{prob_str}",
                                unsafe_allow_html=True
                            )
                        if "offers" in out and out["offers"]:
                            st.markdown("**Top offer:**")
                            first = out["offers"][0]
                            st.markdown(f"- **{first.get('name')}**: {first.get('description')}")
                        # Always show full JSON for transparency
                        with st.expander("Full JSON"):
                            st.json(out)

                    st.caption(f"⏱ {t.duration_ms:.0f}ms")
                    st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Architecture: Groq API (open-weight Llama 3.3 70B) · "
    "Tool calling: `lookup_customer` → `predict_churn` → `get_retention_offers` · "
    "Escalation: `escalate_to_supervisor` · "
    "Logging: `log_interaction`"
)
