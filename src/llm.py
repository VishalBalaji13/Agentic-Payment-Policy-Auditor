"""
LLM client for the Agentic Payment-Policy Auditor.

Two modes:
  - "anthropic": real calls to the Claude API (requires ANTHROPIC_API_KEY)
  - "mock":      deterministic, offline responses so the demo runs with zero setup

The mock mode is what makes this a true "satisficing" POC: a reviewer can clone
the repo and watch the full agent pipeline run without an API key, while the
anthropic mode shows the production path. Both modes exercise the same agent
orchestration and the same deterministic verification guardrail.
"""

import json
import os
import re

DEFAULT_MODEL = os.environ.get("AUDITOR_MODEL", "claude-sonnet-4-6")


class LLMClient:
    def __init__(self, mode: str = "mock", model: str = DEFAULT_MODEL):
        self.mode = mode
        self.model = model
        self._client = None
        if mode == "anthropic":
            try:
                import anthropic  # imported lazily so mock mode needs no deps
            except ImportError as exc:
                raise RuntimeError(
                    "anthropic package not installed. Run `pip install anthropic` "
                    "or use mode='mock'."
                ) from exc
            self._client = anthropic.Anthropic()

    # -- public API ---------------------------------------------------------

    def complete(self, task: str, payload: dict) -> dict:
        """Run one agent step. `task` selects the agent; `payload` is its input.

        Returns a parsed dict. In anthropic mode the model is prompted to return
        strict JSON, which we parse; in mock mode we compute the response directly.
        """
        if self.mode == "mock":
            return _mock_complete(task, payload)
        return self._anthropic_complete(task, payload)

    # -- anthropic backend --------------------------------------------------

    def _anthropic_complete(self, task: str, payload: dict) -> dict:
        system, prompt = _build_prompt(task, payload)
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return _parse_json(text)


# -- prompt construction (anthropic mode) -----------------------------------

def _build_prompt(task: str, payload: dict):
    if task == "extract":
        system = (
            "You are a healthcare payment-policy compiler. You read written payer "
            "policy and convert it into structured, machine-checkable rules. "
            "Return ONLY valid JSON, no prose, no markdown fences."
        )
        prompt = (
            "Read the policy below and extract every active Procedure-to-Procedure "
            "edit pair as a JSON object of the form:\n"
            '{"rules": [{"column_one": "CODE", "column_two": "CODE", '
            '"modifier_indicator": 0, "rationale": "short reason", '
            '"source_span": "verbatim sentence from the policy supporting this rule"}]}\n\n'
            f"POLICY:\n{payload['policy_text']}"
        )
        return system, prompt

    if task == "audit":
        system = (
            "You are a claims auditor. Given a claim and the compiled policy rules, "
            "decide PASS, FAIL, or NEEDS_REVIEW for each triggered rule. Cite the "
            "policy source_span you relied on. Return ONLY valid JSON."
        )
        prompt = (
            "Decide the outcome for this claim. JSON form:\n"
            '{"decision": "PASS|FAIL|NEEDS_REVIEW", "rationale": "...", '
            '"citation": "verbatim policy span", "triggered_pair": ["CODE","CODE"]}\n\n'
            f"CLAIM:\n{json.dumps(payload['claim'])}\n\n"
            f"RULES:\n{json.dumps(payload['rules'])}"
        )
        return system, prompt

    if task == "critique":
        system = (
            "You are a verification agent. Independently check whether the auditor's "
            "decision is consistent with the cited policy and the compiled rule, in "
            "the spirit of natural-language-inference contradiction detection. "
            "Return ONLY valid JSON."
        )
        prompt = (
            "Check the decision for contradictions or unsupported claims. JSON form:\n"
            '{"agrees": true|false, "issues": ["..."], "confidence": 0.0-1.0}\n\n'
            f"CLAIM:\n{json.dumps(payload['claim'])}\n\n"
            f"RULE:\n{json.dumps(payload['rule'])}\n\n"
            f"DECISION:\n{json.dumps(payload['decision'])}"
        )
        return system, prompt

    raise ValueError(f"unknown task: {task}")


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    return json.loads(text)


# -- mock backend (deterministic, offline) ----------------------------------

# Pre-computed extraction for the bundled NCCI excerpt. In anthropic mode the
# model derives this from the policy text; here we return it directly so the
# pipeline runs with no network and no key.
_MOCK_RULES = [
    {
        "column_one": "80053",
        "column_two": "80048",
        "modifier_indicator": 0,
        "rationale": "Basic metabolic panel is a component of the comprehensive metabolic panel and is not separately payable.",
        "source_span": "Column One 80053 (Comprehensive metabolic panel) / Column Two 80048 (Basic\n   metabolic panel) — Modifier Indicator 0.",
    },
    {
        "column_one": "93000",
        "column_two": "93005",
        "modifier_indicator": 1,
        "rationale": "A clinically appropriate modifier may bypass the edit when documentation supports a distinct service.",
        "source_span": "Column One 93000 (Electrocardiogram, complete, with interpretation and report) /\n   Column Two 93005 (Electrocardiogram, tracing only, without interpretation) —\n   Modifier Indicator 1.",
    },
    {
        "column_one": "36415",
        "column_two": "36416",
        "modifier_indicator": 1,
        "rationale": "A clinically appropriate modifier may bypass the edit when the collections are independent.",
        "source_span": "Column One 36415 (Collection of venous blood by venipuncture) / Column Two 36416\n   (Collection of capillary blood specimen) — Modifier Indicator 1.",
    },
]


def _mock_complete(task: str, payload: dict) -> dict:
    if task == "extract":
        return {"rules": _MOCK_RULES}

    if task == "audit":
        return _mock_audit(payload["claim"], payload["rules"])

    if task == "critique":
        return _mock_critique(payload["rule"], payload["decision"])

    raise ValueError(f"unknown task: {task}")


def _mock_audit(claim: dict, rules: list) -> dict:
    codes = {line["cpt"]: line for line in claim["lines"]}
    for rule in rules:
        c1, c2 = rule["column_one"], rule["column_two"]
        if c1 in codes and c2 in codes:
            col_two_modifiers = codes[c2].get("modifiers", [])
            indicator = rule["modifier_indicator"]
            if indicator == 0:
                decision = "FAIL"
                rationale = (
                    f"Codes {c1} and {c2} form a PTP edit pair with Modifier "
                    f"Indicator 0. No modifier can bypass this edit, so {c2} is not "
                    f"separately payable. {rule['rationale']}"
                )
            elif indicator == 1 and col_two_modifiers:
                decision = "NEEDS_REVIEW"
                rationale = (
                    f"Codes {c1} and {c2} form a PTP edit pair with Modifier "
                    f"Indicator 1. Modifier(s) {col_two_modifiers} were submitted on "
                    f"{c2}, which may bypass the edit if documentation supports a "
                    f"distinct service. Route to a human reviewer to confirm "
                    f"supporting documentation."
                )
            else:
                decision = "FAIL"
                rationale = (
                    f"Codes {c1} and {c2} form a PTP edit pair with Modifier "
                    f"Indicator 1, but no bypass modifier was submitted on {c2}, so "
                    f"{c2} is denied."
                )
            return {
                "decision": decision,
                "rationale": rationale,
                "citation": rule["source_span"],
                "triggered_pair": [c1, c2],
            }
    return {
        "decision": "PASS",
        "rationale": "No active PTP edit pair was triggered by the codes on this claim.",
        "citation": "",
        "triggered_pair": [],
    }


def _mock_critique(rule: dict, decision: dict) -> dict:
    # The deterministic grounding check in agents.py does the heavy lifting; the
    # critic here adds a consistency assessment between rule and decision.
    issues = []
    if not rule:
        return {"agrees": True, "issues": [], "confidence": 0.95}

    indicator = rule.get("modifier_indicator")
    d = decision.get("decision")
    if indicator == 0 and d != "FAIL":
        issues.append(
            "Rule has Modifier Indicator 0 (no bypass possible) but decision is not FAIL."
        )
    if indicator == 1 and d == "PASS":
        issues.append(
            "Rule was triggered (Indicator 1) but decision is PASS without addressing the edit."
        )
    agrees = len(issues) == 0
    confidence = 0.9 if agrees else 0.4
    return {"agrees": agrees, "issues": issues, "confidence": confidence}
