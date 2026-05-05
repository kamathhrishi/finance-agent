"""FinanceBench benchmark for agent.

Workflow:
  1. `download.py`  Ensure every filing referenced by a FinanceBench question
                    is present in the local corpus (downloads any missing ones
                    via agent.ingest).
  2. `runner.py`    Run the agent on each question, save (question, expected,
                    predicted, citations, latency) to a JSONL.
  3. `judge.py`     LLM judge scores each (expected, predicted) pair 1-10.
                    Score >= 7 is counted as correct; accuracy = #>=7 / #total.
  4. `cli.py`       End-to-end entrypoint.
"""
