"""
LLM-as-judge for agent benchmark answers.

Scoring:
    1-10 integer; >= 7 means the predicted answer is materially correct.
    Accuracy = (# of >=7 scores) / (total questions judged).

Design notes:
    - The judge MUST tolerate verbose answers. The agent often
      writes 200-500 word analyst-style responses with surrounding narrative
      and citations. As long as the *core fact* the question asks for matches
      the expected answer, it should score 7 or higher. Extra context is good,
      not bad.
    - The judge MUST penalize wrong numbers, missing core facts, and answers
      that say "I couldn't find this" when the expected answer is concrete.
    - Numerical equivalence is fact-equivalence: "$1,577 million" == "$1.577B"
      == "1577 million USD" — none of these should be downscored for unit form.
    - The justification field from FinanceBench (where the answer comes from
      in the filing) is fed to the judge as additional ground-truth context.
    - Output is strict JSON: {"score": int 1-10, "reasoning": str}.
      No markdown, no preamble.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

import openai

logger = logging.getLogger("agent.benchmarks.judge")

DEFAULT_JUDGE_MODEL = "gpt-5.4-mini-2026-03-17"
PASS_THRESHOLD = 7  # scores >= this are "correct"


_JUDGE_SYSTEM = """\
You are a strict but fair judge of financial-analyst answers. You compare a
PREDICTED answer (from an AI agent) against the EXPECTED answer (the gold
truth from the FinanceBench dataset). You return a single integer score
1-10 plus a short reasoning string.

## Rubric

- **9-10**: Predicted answer matches expected on the core fact(s). Numbers
  agree (allowing for unit-form differences like "$1,577M" vs "$1.577B" vs
  "1577 million dollars"). Direction (up/down, increase/decrease) matches.
  Verbose / well-explained answers that still get the core fact right ALWAYS
  belong here — do NOT downscore for length, extra context, citations, or
  caveats.

- **7-8**: Core fact is correct but slightly imprecise (e.g. answer rounds to
  $1.6B vs expected $1,577M — same number, different precision). Or the
  answer says the right thing in a roundabout way. Still considered correct.

- **5-6**: Partially correct. The agent identified the right metric / time
  period but got the number wrong by a noticeable amount, or got the number
  right but the wrong period. Mixed signals.

- **3-4**: Mostly wrong. Agent answered the wrong question, cited wrong
  numbers, or missed the main fact entirely.

- **1-2**: Entirely wrong, hallucinated, or non-responsive ("I don't have
  access to that"). Reserve 1 for refusals when the expected answer is
  clearly stated and verifiable.

## Critical rules

- A verbose, well-cited answer that contains the correct core fact MUST score
  >= 7. Length is never a reason to downscore.
- An answer that gives the WRONG number, even if surrounded by good context,
  CANNOT score above 5.
- An answer that punts ("I couldn't find this in the filings") when the
  expected answer is a specific number CANNOT score above 3.
- For yes/no questions, getting the direction right earns >= 7; the supporting
  reasoning details are nice but not required for full credit.
- Treat numerical equivalence generously. $1.577B == $1,577 million ==
  $1,577.00M == 1577.0 million USD. Don't fixate on currency notation.

## Output

Respond ONLY with valid JSON: {"score": <int 1-10>, "reasoning": "<one or two sentences>"}.
No markdown fences, no preamble, no trailing text.
"""


@dataclass
class JudgeResult:
    score: int
    reasoning: str
    raw: str             # raw LLM output, kept for audit
    passed: bool         # convenience: score >= PASS_THRESHOLD

    def to_dict(self) -> dict:
        return {"score": self.score, "reasoning": self.reasoning, "passed": self.passed}


def _build_user_prompt(question: str, expected: str, predicted: str, justification: str = "") -> str:
    just_block = ""
    if justification:
        just_block = f"\n## Ground-truth justification (from FinanceBench)\n{justification}\n"
    return (
        f"## Question\n{question}\n\n"
        f"## Expected answer (gold)\n{expected}\n"
        f"{just_block}"
        f"\n## Predicted answer (from the AI agent)\n{predicted}\n"
        f"\n## Your job\nScore the predicted answer 1-10 against the expected answer using the rubric. Return JSON only."
    )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_output(text: str) -> tuple[int, str]:
    """Tolerant JSON parse — extract the first {...} block, fall back to defaults."""
    if not text:
        return 1, "empty judge output"
    m = _JSON_RE.search(text)
    if not m:
        return 1, f"no JSON in judge output: {text[:200]!r}"
    try:
        obj = json.loads(m.group(0))
        score = int(obj.get("score", 0))
        reasoning = str(obj.get("reasoning", "")).strip()
        score = max(1, min(10, score))
        return score, reasoning or "(no reasoning provided)"
    except Exception as e:
        return 1, f"judge JSON parse error: {e}; raw={text[:200]!r}"


class Judge:
    """Single-shot LLM judge. Each call is independent (no chat history)."""

    def __init__(
        self,
        model: str = DEFAULT_JUDGE_MODEL,
        api_key: Optional[str] = None,
        max_completion_tokens: int = 600,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY required for Judge")
        self._client = openai.AsyncOpenAI(api_key=self.api_key)
        self.max_completion_tokens = max_completion_tokens

    async def judge(
        self,
        *,
        question: str,
        expected: str,
        predicted: str,
        justification: str = "",
    ) -> JudgeResult:
        user = _build_user_prompt(question, expected, predicted, justification)
        raw = ""
        for _attempt in range(5):
            try:
                resp = await self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _JUDGE_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    temperature=0,
                    max_completion_tokens=self.max_completion_tokens,
                )
                raw = resp.choices[0].message.content or ""
                break
            except openai.RateLimitError as e:
                wait = float(getattr(e, "retry_after", None) or (2 ** _attempt))
                wait = max(0.5, min(wait, 30.0))
                logger.warning(f"judge 429; sleeping {wait:.1f}s")
                await asyncio.sleep(wait)
                continue
            except (openai.APITimeoutError, openai.APIConnectionError):
                await asyncio.sleep(min(2 ** _attempt, 15))
                continue
            except Exception as e:
                logger.exception("Judge LLM call failed (non-retryable)")
                return JudgeResult(score=1, reasoning=f"judge call failed: {e}", raw="", passed=False)
        if not raw:
            return JudgeResult(score=1, reasoning="judge rate-limited after 5 retries", raw="", passed=False)

        score, reasoning = _parse_judge_output(raw)
        return JudgeResult(score=score, reasoning=reasoning, raw=raw, passed=score >= PASS_THRESHOLD)
