# Agentic Payment-Policy Auditor

A proof-of-concept agentic system that reads **written healthcare payment policy**,
compiles it into **executable rules**, audits claims against those rules, and
**independently verifies every decision** before finalizing it.

Built as a focused demonstrator for Cotiviti's core problem space — payment
integrity and payment policy management — where written CMS / payer policy must be
turned into rules that catch incorrect claims.

> Scope note: this is deliberately a small, satisficing POC. It demonstrates the
> agentic pattern and a working anti-hallucination guardrail end to end, without
> over-engineering. Production hardening notes are at the bottom.

---

## What it does

```
Written policy (NCCI PTP excerpt)
        │
        ▼
[1] PolicyCompilerAgent   ── compiles policy text into structured, executable rules
        │
        ▼
[2] AuditorAgent          ── audits each claim → PASS / FAIL / NEEDS_REVIEW + citation
        │
        ▼
[3] CriticAgent           ── independent verification:
                                • grounding check (citation must be verbatim in source)
                                • consistency check (decision must follow from the rule)
        │
        ▼
Orchestrator              ── escalate-on-doubt: uncertain cases go to a human reviewer,
                              never silently auto-denied
```

This mirrors a **RAG + critic** two-layer architecture: the first agent reasons over
retrieved policy, and a second agent independently verifies before the decision is
trusted.

## Why this design

- **Policy → executable rules** is exactly the "conversion of written policy into
  rules/models" task that drives payment integrity.
- The **critic / grounding guardrail** is the difference between a chatbot that
  *describes* and an agent that can be *trusted to act*. Every citation is checked
  against the source policy verbatim, so fabricated citations are caught even when
  the language model is wrong.
- **Human-in-the-loop by default.** In healthcare, the responsible position is to
  flag uncertain claims for a reviewer, not to auto-deny. The orchestrator enforces
  this.

## Quick start

Mock mode runs fully offline with **no API key** — clone and go:

```bash
pip install -r requirements.txt

# CLI trace of the whole pipeline
python -m src.cli

# Interactive UI for the demo / screenshare
streamlit run app.py
```

## Sample outcomes

| Claim | Codes | Why | Final |
|-------|-------|-----|-------|
| CLM-1001 | 80053 + 80048 | Unbundling, Modifier Indicator 0 — no bypass possible | FAIL |
| CLM-1002 | 93000 + 93005 (mod 59) | Indicator 1, modifier present — needs documentation | NEEDS_REVIEW |
| CLM-1003 | 93000 + 93005 | Indicator 1, no bypass modifier | FAIL |
| CLM-1004 | 80053 + 99213 | No active edit pair triggered | PASS |

## Project layout

```
data/policies/ncci_ptp.md   real public NCCI PTP policy excerpt (the written policy)
data/claims.json            sample claims
src/llm.py                  LLM client (live Anthropic + deterministic mock)
src/agents.py               PolicyCompilerAgent, AuditorAgent, CriticAgent
src/pipeline.py             orchestrator with escalate-on-doubt guardrail
src/cli.py                  command-line demo
app.py                      Streamlit UI
```

## How a production version would extend this

- Swap the bundled excerpt for **RAG over the full NCCI / LCD / NCD corpus** with
  embeddings + retrieval instead of a single policy file.
- Replace the LLM consistency check with a **fine-tuned NLI cross-encoder**
  (e.g. DeBERTa-V3) for contradiction detection, keeping the deterministic
  grounding check as a hard guardrail.
- Add **human-review queue integration** and audit logging for every decision.
- Expand rule types beyond PTP edits (medical-necessity, frequency, MUEs).

## Source

NCCI policy text is adapted from the publicly available CMS National Correct Coding
Initiative Policy Manual (cms.gov) and is lightly simplified for demonstration.
