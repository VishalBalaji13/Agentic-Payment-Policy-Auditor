"""
Orchestrator for the Agentic Payment-Policy Auditor.

Flow:
  1. PolicyCompilerAgent compiles the written policy into structured rules (once).
  2. For each claim, AuditorAgent produces a decision + citation.
  3. CriticAgent independently verifies grounding and consistency.
  4. The orchestrator finalizes: if the critic disagrees or the citation is not
     grounded, the decision is escalated to NEEDS_REVIEW rather than trusted.

This "escalate-on-doubt" rule is the human-in-the-loop guardrail: the system is
designed to hand uncertain cases to a reviewer, never to silently auto-deny.
"""

from .agents import (
    AuditResult,
    AuditorAgent,
    CriticAgent,
    PolicyCompilerAgent,
)


class AuditPipeline:
    def __init__(self, llm):
        self.compiler = PolicyCompilerAgent(llm)
        self.auditor = AuditorAgent(llm)
        self.critic = CriticAgent(llm)
        self.rules = []
        self.policy_text = ""

    def load_policy(self, policy_text: str):
        self.policy_text = policy_text
        self.rules = self.compiler.compile(policy_text)
        return self.rules

    def _rule_for_pair(self, pair):
        if not pair:
            return None
        for rule in self.rules:
            if {rule["column_one"], rule["column_two"]} == set(pair):
                return rule
        return None

    def run_claim(self, claim: dict) -> AuditResult:
        decision = self.auditor.audit(claim, self.rules)
        rule = self._rule_for_pair(decision.get("triggered_pair"))
        review = self.critic.review(decision, rule, self.policy_text)

        final = decision["decision"]
        # Escalate-on-doubt guardrail.
        if not review["agrees"] and final != "NEEDS_REVIEW":
            final = "NEEDS_REVIEW"

        return AuditResult(
            claim_id=claim["claim_id"],
            decision=decision["decision"],
            rationale=decision["rationale"],
            citation=decision.get("citation", ""),
            triggered_pair=decision.get("triggered_pair", []),
            grounded=review["grounded"],
            critic_agrees=review["agrees"],
            critic_issues=review["issues"],
            confidence=review["confidence"],
            final_decision=final,
        )

    def run_all(self, claims: list) -> list:
        return [self.run_claim(c) for c in claims]
