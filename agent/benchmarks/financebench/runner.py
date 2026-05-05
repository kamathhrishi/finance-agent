"""
Runner: drive the agent through every FinanceBench question,
then judge each (expected, predicted) pair with the LLM judge.

Output: results/<run_name>/results.jsonl  (one JSON object per question)
        results/<run_name>/summary.json   (aggregate stats)

Resume: if results.jsonl exists, every question.id already in it is skipped.

Concurrency: bounded asyncio.Semaphore — N agent runs in flight at once.
The agent itself does its own LLM/file-IO so this is mostly I/O bound.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.agent import FilesystemResearchAgent
from .dataset import FinanceBenchQuestion, load_questions
from .download import BENCHMARK_DATA_ROOT
from .judge import Judge, JudgeResult, PASS_THRESHOLD

logger = logging.getLogger("agent.benchmarks.fb.runner")

RESULTS_ROOT = Path(__file__).resolve().parent / "results"


# ──────────────────────────────────────────────────────────────────────────────
# Per-question record
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class QuestionResult:
    id: str
    company: str
    ticker: str
    form: Optional[str]
    year: Optional[int]
    quarter: Optional[str]
    filing_date: Optional[str]
    question: str
    expected: str
    predicted: str
    justification: str
    judge_score: int
    judge_reasoning: str
    judge_passed: bool
    tool_calls: int
    llm_calls: int
    elapsed_s: float
    error: Optional[str] = None

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _load_existing_ids(jsonl_path: Path) -> set[str]:
    """Resume support — skip ids already present in results.jsonl."""
    if not jsonl_path.is_file():
        return set()
    out: set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("id"):
                    out.add(str(obj["id"]))
            except Exception:
                continue
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Single-question driver
# ──────────────────────────────────────────────────────────────────────────────


async def _run_one(
    q: FinanceBenchQuestion,
    *,
    agent: FilesystemResearchAgent,
    judge: Judge,
    max_tool_calls: int,
    trace_dir: Optional[Path] = None,
) -> QuestionResult:
    """Run the agent on one question, then judge. All exceptions captured.

    If `trace_dir` is given, every event the agent emits (tool calls + full
    tool results + intermediate text) is appended to
    `<trace_dir>/<question_id>.jsonl` so later forensic analysis can replay
    exactly what the LLM saw and decided.
    """
    t0 = time.time()
    final_answer = ""
    tool_calls = 0
    llm_calls = 0
    err: Optional[str] = None

    trace_path: Optional[Path] = None
    trace_f = None
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / f"{q.id}.jsonl"
        # Truncate any prior trace for this question (resume keeps prior result
        # but a fresh trace is cleaner than appending into a half-trace)
        trace_f = trace_path.open("w", encoding="utf-8")
        trace_f.write(json.dumps({
            "type": "question",
            "id": q.id, "company": q.company, "ticker": q.ticker, "form": q.form,
            "year": q.year, "quarter": q.quarter, "filing_date": q.filing_date,
            "question": q.question, "expected": q.expected,
            "justification": q.justification, "doc_name": q.doc_name,
        }, ensure_ascii=False) + "\n")

    # FinanceBench questions implicitly assume a specific filing context (the
    # `doc_name`, e.g. `PFIZER_2019_10K`). Without that hint the agent guesses
    # the latest 10-K — wrong for questions like "What 3 acquisitions are
    # mentioned in this 10-K?" where the gold expects FY2019-era acquisitions.
    # Append a one-line context hint that names the filing the question refers
    # to, without giving away the answer.
    period_hint = ""
    if q.form and q.year:
        if q.form == "10-K":
            period_hint = f"{q.ticker} {q.form} for fiscal year {q.year}"
        elif q.form == "10-Q" and q.quarter:
            period_hint = f"{q.ticker} {q.form} for {q.quarter} of fiscal year {q.year}"
        elif q.form == "8-K" and q.filing_date:
            period_hint = f"{q.ticker} {q.form} filed {q.filing_date}"
        else:
            period_hint = f"{q.ticker} {q.form} for fiscal year {q.year}"
    augmented_question = q.question
    if period_hint:
        augmented_question = (
            f"{q.question}\n\n"
            f"_(For grounding: this question refers to **{q.company}**'s "
            f"{period_hint}. Use the corresponding folder in the corpus.)_"
        )

    # Per-question budget override — keep agent honest at this question
    saved_budget = agent.max_tool_calls
    agent.max_tool_calls = max_tool_calls
    try:
        async for ev in agent.run(augmented_question):
            et = ev.get("type")
            if trace_f is not None and et != "token":  # tokens are noise in the trace
                trace_f.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
                trace_f.flush()
            if et == "tool_end":
                tool_calls = max(tool_calls, int(ev.get("call_num", tool_calls)))
            elif et == "llm_call":
                llm_calls += 1
            elif et == "result":
                final_answer = ev.get("answer", "") or ""
            elif et == "error":
                err = ev.get("message", "agent error")
    except Exception as e:
        logger.exception(f"agent run failed for {q.id}")
        err = f"{type(e).__name__}: {e}"
    finally:
        agent.max_tool_calls = saved_budget
        if trace_f is not None:
            trace_f.close()

    elapsed = time.time() - t0

    # Judge — even on agent error, judge whatever (possibly empty) answer we got
    if err and not final_answer:
        # No predicted answer at all → score 1 without spending a judge call
        jres = JudgeResult(score=1, reasoning=f"agent failed: {err}", raw="", passed=False)
    else:
        jres = await judge.judge(
            question=q.question,
            expected=q.expected,
            predicted=final_answer,
            justification=q.justification,
        )

    return QuestionResult(
        id=q.id,
        company=q.company,
        ticker=q.ticker or "",
        form=q.form,
        year=q.year,
        quarter=q.quarter,
        filing_date=q.filing_date,
        question=q.question,
        expected=q.expected,
        predicted=final_answer,
        justification=q.justification,
        judge_score=jres.score,
        judge_reasoning=jres.reasoning,
        judge_passed=jres.passed,
        tool_calls=tool_calls,
        llm_calls=llm_calls,
        elapsed_s=round(elapsed, 1),
        error=err,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Batch driver
# ──────────────────────────────────────────────────────────────────────────────


async def run_benchmark(
    *,
    run_name: str,
    questions: Optional[List[FinanceBenchQuestion]] = None,
    only_companies: Optional[List[str]] = None,
    limit: Optional[int] = None,
    concurrency: int = 4,
    max_tool_calls: int = 25,
    agent_model: Optional[str] = None,
    judge_model: Optional[str] = None,
    resume: bool = True,
) -> Dict[str, Any]:
    """End-to-end: load questions → run agent → judge → write JSONL + summary."""
    qs = questions if questions is not None else load_questions(only_companies=only_companies)
    if limit:
        qs = qs[:limit]

    out_dir = RESULTS_ROOT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"
    trace_dir = out_dir / "traces"

    seen = _load_existing_ids(jsonl_path) if resume else set()
    pending = [q for q in qs if q.id not in seen]
    logger.info(f"FinanceBench: {len(qs)} total, {len(seen)} already done, {len(pending)} to run")

    # Single agent + single judge instance shared across all tasks (both clients are async-safe).
    # Agent is sandboxed to the BENCHMARK corpus, NOT the main fs_research corpus —
    # this isolates the benchmark from any unrelated tech-batch filings.
    if not BENCHMARK_DATA_ROOT.is_dir():
        raise RuntimeError(
            f"Benchmark corpus not found at {BENCHMARK_DATA_ROOT}. "
            f"Run `python -m agent.benchmarks.financebench.cli download` first."
        )
    agent_kwargs: Dict[str, Any] = {"data_root": BENCHMARK_DATA_ROOT}
    if agent_model:
        agent_kwargs["model"] = agent_model
    agent = FilesystemResearchAgent(**agent_kwargs)
    logger.info(f"Agent sandboxed to benchmark corpus: {BENCHMARK_DATA_ROOT}")
    judge = Judge(model=judge_model) if judge_model else Judge()

    sem = asyncio.Semaphore(concurrency)
    completed = 0
    pass_count = 0

    async def _bounded(q: FinanceBenchQuestion) -> QuestionResult:
        async with sem:
            return await _run_one(
                q, agent=agent, judge=judge, max_tool_calls=max_tool_calls,
                trace_dir=trace_dir,
            )

    # Stream results to disk as each task finishes (so a crash doesn't lose progress)
    with jsonl_path.open("a", encoding="utf-8") as out_f:
        tasks = [asyncio.create_task(_bounded(q)) for q in pending]
        for fut in asyncio.as_completed(tasks):
            try:
                res = await fut
            except Exception as e:
                logger.exception("task crashed")
                continue
            out_f.write(res.to_jsonl() + "\n")
            out_f.flush()
            completed += 1
            if res.judge_passed:
                pass_count += 1
            tag = "✓" if res.judge_passed else "✗"
            print(
                f"  [{completed}/{len(pending)}]  {tag}  {res.judge_score:>2}/10  "
                f"{res.id:<22}  {res.ticker:<5}  {res.form or '-':<4}  "
                f"({res.tool_calls}t, {res.elapsed_s:.0f}s)  → {res.judge_reasoning[:80]}"
            )

    # Aggregate stats from full JSONL (includes resumed rows)
    summary = _summarize(jsonl_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print("━━━ SUMMARY ━━━")
    print(f"  total judged: {summary['total']}")
    print(f"  passed (≥{PASS_THRESHOLD}/10): {summary['passed']}")
    print(f"  accuracy:    {summary['accuracy']:.1%}")
    print(f"  avg score:   {summary['avg_score']:.2f}/10")
    print(f"  output:      {jsonl_path}")
    return summary


def _summarize(jsonl_path: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    if jsonl_path.is_file():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    total = len(rows)
    if total == 0:
        return {"total": 0, "passed": 0, "accuracy": 0.0, "avg_score": 0.0, "by_form": {}, "by_company": {}}

    passed = sum(1 for r in rows if r.get("judge_passed"))
    avg_score = sum(r.get("judge_score", 0) for r in rows) / total

    def _bucket(field: str) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            k = r.get(field) or "_unknown"
            b = out.setdefault(k, {"total": 0, "passed": 0, "score_sum": 0})
            b["total"] += 1
            b["score_sum"] += int(r.get("judge_score", 0))
            if r.get("judge_passed"):
                b["passed"] += 1
        for k, b in out.items():
            b["accuracy"] = b["passed"] / b["total"]
            b["avg_score"] = b["score_sum"] / b["total"]
        return out

    return {
        "total": total,
        "passed": passed,
        "accuracy": passed / total,
        "avg_score": avg_score,
        "pass_threshold": PASS_THRESHOLD,
        "by_form": _bucket("form"),
        "by_company": _bucket("company"),
    }
