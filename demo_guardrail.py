"""
Guardrail demo — for the video walkthrough.

Runs fully offline (mock mode, no API key). It shows two things back to back:

  1. A normal decision the system trusts (citation is grounded in the policy).
  2. A deliberately fabricated citation that the grounding guardrail catches:
     grounded = False, confidence collapses, and the case is escalated to a human
     instead of being trusted.

Run:
    python demo_guardrail.py
"""

import os
import time

from src.llm import LLMClient
from src.pipeline import AuditPipeline

ROOT = os.path.dirname(os.path.abspath(__file__))
PAUSE = 0.6  # small pauses so the trace is easy to follow on camera

BAR = "=" * 68
LINE = "-" * 68


def load_policy():
    with open(os.path.join(ROOT, "data", "policies", "ncci_ptp.md")) as f:
        return f.read()


def show(label, value):
    print(f"    {label:<20} {value}")


def main():
    policy_text = load_policy()
    pipe = AuditPipeline(LLMClient(mode="mock"))

    print(f"\n{BAR}")
    print("  GROUNDING GUARDRAIL DEMO  (offline mock mode)")
    print(BAR)
    print("  Compiling written NCCI policy into executable rules ...")
    rules = pipe.load_policy(policy_text)
    time.sleep(PAUSE)
    print(f"  Compiled {len(rules)} rules.\n")

    # ---- Case 1: a normal, trusted decision -------------------------------
    print(LINE)
    print("  CASE 1 - A normal claim the system trusts")
    print(LINE)
    claim = {
        "claim_id": "CLM-1001",
        "date_of_service": "2026-03-14",
        "provider_npi": "1234567890",
        "lines": [
            {"cpt": "80053", "modifiers": [], "units": 1},
            {"cpt": "80048", "modifiers": [], "units": 1},
        ],
    }
    time.sleep(PAUSE)
    res = pipe.run_claim(claim)
    show("Auditor decision:", res.decision)
    show("Citation grounded:", "YES  (found verbatim in the source policy)" if res.grounded else "NO")
    show("Critic agrees:", "YES" if res.critic_agrees else "NO")
    show("Confidence:", f"{res.confidence:.2f}")
    show("FINAL:", f"{res.final_decision}   <- trusted")
    print()
    time.sleep(PAUSE)

    # ---- Case 2: a fabricated citation the guardrail catches --------------
    print(LINE)
    print("  CASE 2 - The AI hallucinates a citation (we inject a fake one)")
    print(LINE)
    tampered = {
        "decision": "FAIL",
        "rationale": "Claim denied based on policy (fabricated reasoning).",
        "citation": "Column One 11111 / Column Two 22222 - Modifier Indicator 0. "
                    "(this policy text does not exist anywhere in the source)",
        "triggered_pair": ["80053", "80048"],
    }
    print("    Injected citation:")
    print(f"      \"{tampered['citation'][:60]}...\"")
    time.sleep(PAUSE)
    rule = pipe._rule_for_pair(tampered["triggered_pair"])
    review = pipe.critic.review(tampered, rule, pipe.policy_text)
    print()
    show("Auditor decision:", tampered["decision"])
    show("Citation grounded:", "YES" if review["grounded"] else "NO   <- guardrail caught it")
    show("Critic agrees:", "YES" if review["agrees"] else "NO")
    show("Confidence:", f"{review['confidence']:.2f}   <- collapsed")
    for issue in review["issues"]:
        show("Issue flagged:", issue)
    final = "NEEDS_REVIEW" if not review["agrees"] else tampered["decision"]
    show("FINAL:", f"{final}   <- escalated to a human, NOT trusted")
    print()

    print(BAR)
    print("  Takeaway: even when the model invents a citation, the system")
    print("  refuses to act on it and routes the case to a human reviewer.")
    print(f"{BAR}\n")


if __name__ == "__main__":
    main()