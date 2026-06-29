"""
Streamlit UI for the Agentic Payment-Policy Auditor.

Run:
  streamlit run app.py

Use the sidebar to switch between mock mode (no key needed) and live Claude mode.
The UI is intentionally simple so the agent pipeline, not the front end, is the star.
"""

import json
import os

import streamlit as st

from src.llm import LLMClient
from src.pipeline import AuditPipeline

ROOT = os.path.dirname(os.path.abspath(__file__))

BADGE = {
    "PASS": ("#1a7f37", "PASS"),
    "FAIL": ("#cf222e", "FAIL"),
    "NEEDS_REVIEW": ("#9a6700", "NEEDS REVIEW"),
}


def load_text(path):
    with open(path) as f:
        return f.read()


def badge(decision):
    color, label = BADGE.get(decision, ("#555", decision))
    return (
        f"<span style='background:{color};color:#fff;padding:3px 10px;"
        f"border-radius:12px;font-weight:600;font-size:0.85rem'>{label}</span>"
    )


st.set_page_config(page_title="Agentic Payment-Policy Auditor", layout="wide")
st.title("Agentic Payment-Policy Auditor")
st.caption(
    "Compiles written CMS/NCCI payment policy into executable rules, audits claims "
    "against them, and independently verifies each decision before it is finalized."
)

with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "LLM mode",
        ["mock", "anthropic"],
        help="mock runs offline with no API key. anthropic uses live Claude calls.",
    )
    if mode == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("Set ANTHROPIC_API_KEY in your environment for live mode.")
    st.markdown("---")
    st.markdown("**Pipeline**")
    st.markdown(
        "1. PolicyCompilerAgent\n2. AuditorAgent\n3. CriticAgent (grounding + consistency)"
    )

policy_text = load_text(os.path.join(ROOT, "data", "policies", "ncci_ptp.md"))
claims = json.loads(load_text(os.path.join(ROOT, "data", "claims.json")))

llm = LLMClient(mode=mode)
pipe = AuditPipeline(llm)

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1 - Written policy (input)")
    with st.expander("View NCCI PTP policy excerpt", expanded=False):
        st.markdown(policy_text)

    st.subheader("2 - Compiled rules")
    if st.button("Compile policy into rules", type="primary"):
        st.session_state["rules"] = pipe.load_policy(policy_text)
    if "rules" in st.session_state:
        st.dataframe(
            [
                {
                    "Column One": r["column_one"],
                    "Column Two": r["column_two"],
                    "Modifier Indicator": r["modifier_indicator"],
                }
                for r in st.session_state["rules"]
            ],
            use_container_width=True,
            hide_index=True,
        )

with col_right:
    st.subheader("3 - Audit claims")
    if "rules" not in st.session_state:
        st.info("Compile the policy first, then run the audit.")
    else:
        pipe.rules = st.session_state["rules"]
        pipe.policy_text = policy_text
        if st.button("Run audit on sample claims", type="primary"):
            st.session_state["results"] = [pipe.run_claim(c) for c in claims]

    for res in st.session_state.get("results", []):
        with st.container(border=True):
            top = st.columns([2, 1])
            top[0].markdown(f"**Claim {res.claim_id}**")
            top[1].markdown(badge(res.final_decision), unsafe_allow_html=True)
            st.write(res.rationale)
            meta = st.columns(3)
            meta[0].metric("Citation grounded", "Yes" if res.grounded else "No")
            meta[1].metric("Critic agrees", "Yes" if res.critic_agrees else "No")
            meta[2].metric("Confidence", f"{res.confidence:.2f}")
            if res.citation:
                with st.expander("Cited policy span"):
                    st.code(res.citation)
            if res.critic_issues:
                for issue in res.critic_issues:
                    st.warning(issue)
