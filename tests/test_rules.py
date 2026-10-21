"""Sanity checks: the rules baseline must get the guideline examples right."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))

from rules import classify, classify_complexity, classify_task  # noqa: E402


class GuidelineExamples(unittest.TestCase):
    def test_task_type_examples(self):
        cases = {
            "Write a Python function that retries an HTTP call with exponential backoff.": "code_generation",
            "My Terraform apply fails with 'cycle detected', here's the plan output.": "debugging",
            "Review this PR diff for security issues.": "code_explanation_review",
            "Turn these bullet points into a runbook for on-call.": "writing_docs",
            "Here's a CSV of p95 latency by region, what changed last week?": "data_analysis",
            "What's the difference between a Kubernetes Deployment and a StatefulSet?": "research_qa",
        }
        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                self.assertEqual(classify_task(prompt), expected)

    def test_main_rule_fix_wins_over_explain(self):
        prompt = "Here's my stack trace, explain why it fails and write a fix."
        self.assertEqual(classify_task(prompt), "debugging")

    def test_short_design_prompt_is_high(self):
        self.assertEqual(classify_complexity("Design a multi-region failover for our Postgres."), "high")

    def test_simple_ask_is_low(self):
        self.assertEqual(classify_complexity("What does line 3 do?"), "low")

    def test_output_shape(self):
        self.assertEqual(set(classify("hello")), {"task_type", "complexity"})


if __name__ == "__main__":
    unittest.main()
