"""
Command-line demo runner.

Usage:
  python -m src.cli                # mock mode, all bundled claims
  python -m src.cli --mode anthropic   # live Claude calls (needs ANTHROPIC_API_KEY)

This prints a readable trace of every agent step so the screenshare clearly shows
the compile -> audit -> verify flow.
"""

import argparse
import json
import os

from .llm import LLMClient
from .pipeline import AuditPipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ICON = {"PASS": "\u2705", "FAIL": "\u274c", "NEEDS_REVIEW": "\u26a0\ufe0f"}


def load(path):
    with open(path) as f:
        return f.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mock", "anthropic"], default="mock")
    args = ap.parse_args()

    policy_text = load(os.path.join(ROOT, "data", "policies", "ncci_ptp.md"))
    claims = json.loads(load(os.path.join(ROOT, "data", "claims.json")))

    llm = LLMClient(mode=args.mode)
    pipe = AuditPipeline(llm)

    print(f"\n=== Agentic Payment-Policy Auditor  (mode: {args.mode}) ===\n")
    print("[1] PolicyCompilerAgent: compiling written policy into rules ...")
    rules = pipe.load_policy(policy_text)
    print(f"    Compiled {len(rules)} executable rule(s):")
    for r in rules:
        print(
            f"      - {r['column_one']} / {r['column_two']}  "
            f"(modifier_indicator={r['modifier_indicator']})"
        )
    print()

    for claim in claims:
        res = pipe.run_claim(claim)
        icon = ICON.get(res.final_decision, "")
        print(f"[2] AuditorAgent  -> claim {res.claim_id}")
        print(f"    auditor decision : {res.decision}")
        print(f"    triggered pair   : {res.triggered_pair or '(none)'}")
        print(f"[3] CriticAgent   -> grounded={res.grounded}  "
              f"agrees={res.critic_agrees}  confidence={res.confidence:.2f}")
        if res.critic_issues:
            for issue in res.critic_issues:
                print(f"    ! {issue}")
        print(f"    FINAL: {icon} {res.final_decision}")
        print(f"    why  : {res.rationale}")
        print("-" * 72)


if __name__ == "__main__":
    main()
