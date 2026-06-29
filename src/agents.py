"""
The three agents of the pipeline.

  PolicyCompilerAgent  : written policy text  -> structured, executable rules
  AuditorAgent         : claim + rules        -> PASS / FAIL / NEEDS_REVIEW + citation
  CriticAgent          : decision + rule      -> contradiction / grounding check

The CriticAgent combines two checks:
  1. A deterministic GROUNDING check: every citation the auditor produces must be
     a verbatim span of the source policy. This catches fabricated citations even
     when the language model is wrong, and runs identically in mock and live modes.
  2. An LLM (or mock) consistency check in the spirit of NLI contradiction
     detection: does the decision actually follow from the rule?

This two-layer design mirrors a RAG + critic architecture: the first agent reasons,
the second agent independently verifies before anything is surfaced as final.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


def _normalize(text: str) -> str:
    return " ".join(text.split())


@dataclass
class AuditResult:
    claim_id: str
    decision: str
    rationale: str
    citation: str
    triggered_pair: list
    grounded: bool
    critic_agrees: bool
    critic_issues: list = field(default_factory=list)
    confidence: float = 0.0
    final_decision: str = ""

    def to_dict(self):
        return asdict(self)


class PolicyCompilerAgent:
    """Compiles written policy into structured rules."""

    def __init__(self, llm):
        self.llm = llm

    def compile(self, policy_text: str) -> list:
        out = self.llm.complete("extract", {"policy_text": policy_text})
        return out["rules"]


class AuditorAgent:
    """Audits a claim against the compiled rules."""

    def __init__(self, llm):
        self.llm = llm

    def audit(self, claim: dict, rules: list) -> dict:
        return self.llm.complete("audit", {"claim": claim, "rules": rules})


class CriticAgent:
    """Independently verifies the auditor's decision before it is finalized."""

    def __init__(self, llm):
        self.llm = llm

    def grounding_check(self, citation: str, policy_text: str) -> bool:
        """A citation is grounded only if it appears verbatim in the source."""
        if not citation:
            return True  # PASS decisions legitimately carry no citation
        return _normalize(citation) in _normalize(policy_text)

    def review(self, decision: dict, rule: Optional[dict], policy_text: str) -> dict:
        grounded = self.grounding_check(decision.get("citation", ""), policy_text)
        consistency = self.llm.complete(
            "critique", {"claim": {}, "rule": rule or {}, "decision": decision}
        )
        issues = list(consistency.get("issues", []))
        if not grounded:
            issues.append(
                "Citation is not found verbatim in the source policy (possible hallucination)."
            )
        agrees = consistency.get("agrees", True) and grounded
        confidence = consistency.get("confidence", 0.0)
        if not grounded:
            confidence = min(confidence, 0.2)
        return {
            "grounded": grounded,
            "agrees": agrees,
            "issues": issues,
            "confidence": confidence,
        }
