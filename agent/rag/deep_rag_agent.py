#!/usr/bin/env python3
"""
DeepRAGAgent - LangGraph ReAct agent replacing the hand-coded RAG pipeline.

Architecture:
  - Three specialized sub-services (SEC, transcripts, Tavily) run unchanged,
    still using Cerebras internally for fast inference.
  - The orchestrating ReAct agent uses gpt-5.4-nano-2026-03-17 (OpenAI).
  - LangGraph streams tool calls and results; we translate these to the same
    SSE event format the frontend already consumes — zero frontend changes needed.

Drop-in replacement for RAGAgent:
  - Same __init__ signature
  - Same execute_rag_flow() signature and yielded event shapes
  - Same set_database_connection() method
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False
    logfire = None

from .config import Config
from .database_manager import DatabaseManager, DatabaseConnectionError
from .deep_agent_tools import CitationAccumulator, make_tools, make_supervisor_tools
from .earnings_transcript_service import EarningsTranscriptService
from .question_analyzer import QuestionAnalyzer
from .rag_utils import deduplicate_citations_and_chunks
from .search_engine import SearchEngine
from .sec_filings_service_smart_parallel import SmartParallelSECFilingsService as SECFilingsService
from .tavily_service import TavilyService

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=env_path, override=True)

logger = logging.getLogger(__name__)
rag_logger = logging.getLogger('rag_system')

# ─────────────────────────────────────────────────────────────────────────────
# System prompt for the DeepAgent orchestrator
# ─────────────────────────────────────────────────────────────────────────────

_DEEP_AGENT_SYSTEM_PROMPT_BASE = """You are a senior financial research orchestrator. You coordinate specialized research subagents and synthesize their findings into a comprehensive final answer.

Your tools:
1. **research_company(ticker, question)** — deep-dives one company (SEC filings + earnings transcripts)
2. **research_companies(tickers, question)** — same but multiple companies concurrently (always prefer this for comparisons)
3. **search_news(query)** — real-time news via Tavily
4. **write/read_research_notes** — scratch pad for multi-step synthesis

{data_availability}

## Research guidelines

**Cite every fact.** Use citation markers exactly as returned by tools:
- [C-N] for SEC filing passages
- [TC-N] for earnings transcript passages
- [N-N] for news articles

**Be methodical for complex questions:**
1. Use `write_research_notes` to save intermediate findings so you don't lose context
2. For multi-company comparisons, search each company separately, save results, then synthesize
3. For multi-year trends, search each year separately

**Choose the right tool:**
- One company: deep financial/filing/transcript research → `research_company(ticker, question)`
- Multiple companies or comparisons → `research_companies(tickers, question)` — runs subagents in parallel; always prefer this over calling `research_company` in a loop
- Recent news or current events → `search_news`

**Be concise in tool calls.** Pass focused, specific questions — not the user's raw question verbatim.

**Never give up or ask the user for clarification.** If one source returns no results, synthesize the best possible answer from the sources that did return data. You have rich 10-K filings and news — use them. Do not ask the user to re-phrase or specify; just answer with what you have and note any gaps inline.

**Do not repeat the same search twice.** If a tool call returns no results, try a different query or a different source — never call the same tool with the same parameters again.

## Response format

After completing research, write a **detailed, comprehensive answer** that:
- Covers all sub-questions in the user's request
- Includes specific numbers, percentages, and figures from the sources
- Organises multi-company comparisons with clear structure (e.g. headers or tables)
- Cites every fact with inline citation markers [C-N], [TC-N], or [N-N]
- Is as long as needed to fully answer the question — do NOT truncate or summarise away important data
- **Never asks the user for clarification at the end** — answer fully with available data

End your response with exactly this section (no extra text after it):

### Suggested follow-up questions
- [question 1]
- [question 2]
- [question 3]
"""

# ─────────────────────────────────────────────────────────────────────────────
# Event type sets (mirrors chat.py categorisation)
# ─────────────────────────────────────────────────────────────────────────────

_REASONING_TYPES = frozenset({
    'reasoning', 'progress', 'analysis', 'search', 'news_search',
    '10k_search', 'iteration_start', 'iteration_search',
    'iteration_transcript_search', 'iteration_news_search',
    'iteration_followup', 'iteration_complete', 'iteration_final',
    'agent_decision', 'planning_start', 'planning_complete',
    'retrieval_complete', 'evaluation_complete', 'search_complete',
    '10k_planning', '10k_retrieval', '10k_evaluation', 'api_retry',
})

# ─────────────────────────────────────────────────────────────────────────────
# Follow-up question parser
# ─────────────────────────────────────────────────────────────────────────────

_FOLLOWUP_RE = re.compile(
    r'###\s*Suggested follow-up questions\s*\n(.*?)(?:\Z)',
    re.DOTALL | re.IGNORECASE,
)

def _extract_follow_up_questions(text: str) -> tuple[str, List[str]]:
    """
    Split the final answer into (answer_body, follow_up_questions).
    The follow-up block is removed from the returned answer_body.
    """
    m = _FOLLOWUP_RE.search(text)
    if not m:
        return text.strip(), []

    answer_body = text[:m.start()].strip()
    block = m.group(1)
    questions = []
    for line in block.splitlines():
        line = line.strip().lstrip('-•*').strip()
        if line:
            questions.append(line)
    return answer_body, questions[:5]  # cap at 5


# ─────────────────────────────────────────────────────────────────────────────
# DeepRAGAgent
# ─────────────────────────────────────────────────────────────────────────────

class DeepRAGAgent:
    """
    Drop-in replacement for RAGAgent.

    The orchestrating LLM is gpt-5.4-nano-2026-03-17.
    Sub-services (SEC, transcripts) still use Cerebras internally.
    """

    # Tool-call budget: stop if the agent spends too many calls
    MAX_TOOL_CALLS = 20

    def __init__(self, openai_api_key: Optional[str] = None):
        self.instance_id = f"DeepRAGAgent_{int(time.time() * 1000)}"
        logger.info(f"🚀 Creating DeepRAGAgent: {self.instance_id}")

        self.config = Config()
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")

        try:
            self.config.fetch_available_quarters_from_db()
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch quarters from DB: {e}")

        self.database_manager = DatabaseManager(self.config)
        self.question_analyzer = QuestionAnalyzer(None, self.config, self.database_manager)
        self.search_engine = SearchEngine(self.config, self.database_manager)
        self.tavily_service = TavilyService()
        self.sec_service = SECFilingsService(self.database_manager, self.config)
        self.transcript_service = EarningsTranscriptService(self.search_engine, self.config)

        # LLM for the orchestrating agent (NOT the sub-services)
        _model = os.getenv("DEEP_AGENT_MODEL", "gpt-5.4-nano-2026-03-17")
        _max_completion_tokens = int(os.getenv("DEEP_AGENT_MAX_TOKENS", "32000"))
        self._llm = ChatOpenAI(
            model=_model,
            api_key=self.openai_api_key,
            temperature=0,
            streaming=False,  # We handle streaming ourselves via LangGraph
            model_kwargs={
                "max_completion_tokens": _max_completion_tokens,
            },
        )

        logger.info(
            f"✅ DeepRAGAgent ready (orchestrator: {_model}, max_tokens={_max_completion_tokens})"
        )

    # ── Public ────────────────────────────────────────────────────────────────

    def set_database_connection(self, db_connection):
        """Pass a request-scoped DB connection to ConversationMemory."""
        self.question_analyzer.conversation_memory.set_database_connection(db_connection)

    async def execute_rag_flow(
        self,
        question: str,
        show_details: bool = False,
        comprehensive: bool = True,
        stream_callback=None,          # kept for API compat, not used
        max_iterations: Optional[int] = None,
        conversation_id: Optional[str] = None,
        stream: bool = True,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute the RAG flow as a LangGraph ReAct agent.

        Yields SSE event dicts identical to RAGAgent.execute_rag_flow().
        Final event is always {type: 'result', data: {...}} or {type: 'error', ...}.
        """
        start_time = time.time()
        request_id = f"{self.instance_id}_{int(start_time * 1000)}"

        rag_logger.info("=" * 80)
        rag_logger.info("[DeepAgent] ▶ NEW REQUEST")
        rag_logger.info(f"[DeepAgent]   request_id : {request_id}")
        rag_logger.info(f"[DeepAgent]   question   : {question[:120]}")
        rag_logger.info(f"[DeepAgent]   conv_id    : {conversation_id}")
        rag_logger.info(f"[DeepAgent]   stream     : {stream}")
        rag_logger.info(f"[DeepAgent]   max_iter   : {max_iterations}")
        rag_logger.info("=" * 80)

        if stream:
            yield {'type': 'progress', 'message': 'Starting analysis...', 'step': 'init', 'data': {}}

        # ── Citation accumulator and event queue (per-request) ────────────────
        citation_acc = CitationAccumulator()
        event_queue: asyncio.Queue = asyncio.Queue()

        # ── Build supervisor tools (each research_company/companies call spawns subagents) ──
        tools = make_supervisor_tools(
            llm=self._llm,
            sec_service=self.sec_service,
            transcript_service=self.transcript_service,
            tavily_service=self.tavily_service,
            search_engine=self.search_engine,
            event_queue=event_queue,
            citation_accumulator=citation_acc,
            request_id=request_id,
            question_analysis=None,
            config=self.config,
        )

        # ── Build LangGraph supervisor agent ─────────────────────────────────
        recursion_limit = min((max_iterations or 10) * 4 + 10, 80)
        rag_logger.info(f"[DeepAgent] Building LangGraph supervisor agent (recursion_limit={recursion_limit})")
        agent = create_react_agent(self._llm, tools)

        # ── Build conversation history ────────────────────────────────────────
        messages = await self._build_messages(question, conversation_id)
        rag_logger.info(f"[DeepAgent] Message history: {len(messages)} messages (including system + history)")

        # ── Run agent in background, drain events from queue ─────────────────
        sentinel = object()

        async def _run_agent():
            try:
                async for chunk in agent.astream(
                    {"messages": messages},
                    config={"recursion_limit": recursion_limit},
                    stream_mode="updates",
                ):
                    await event_queue.put(("langgraph", chunk))
            except Exception as e:
                await event_queue.put(("error", str(e)))
            finally:
                await event_queue.put(sentinel)

        rag_logger.info("[DeepAgent] Launching agent background task...")
        agent_task = asyncio.create_task(_run_agent())

        # Track state for final result assembly
        final_answer: Optional[str] = None
        tool_call_count = 0
        last_ai_content: Optional[str] = None

        try:
            while True:
                item = await event_queue.get()

                # ── Sentinel: agent finished ──────────────────────────────────
                if item is sentinel:
                    rag_logger.info(f"[DeepAgent] Agent task finished (tool_calls={tool_call_count})")
                    break

                # ── Error from agent task ─────────────────────────────────────
                if isinstance(item, tuple) and item[0] == "error":
                    err_msg = item[1]
                    rag_logger.error(f"[DeepAgent] Agent error: {err_msg}")
                    yield {
                        'type': 'error',
                        'message': f"Agent error: {err_msg}",
                        'step': 'generation',
                        'data': {'error': err_msg},
                    }
                    await agent_task
                    return

                # ── Tool intermediate events (put by tool closures) ───────────
                if isinstance(item, dict):
                    if stream:
                        yield item
                    continue

                # ── LangGraph update chunks ───────────────────────────────────
                if isinstance(item, tuple) and item[0] == "langgraph":
                    chunk = item[1]
                    async for event in self._translate_langgraph_chunk(
                        chunk, stream, tool_call_count
                    ):
                        # Count tool calls for budget enforcement
                        if event.get('_is_tool_call'):
                            tool_call_count += 1
                            tool_name = event.get('data', {}).get('tool', '?')
                            rag_logger.info(
                                f"[DeepAgent] ── Tool call #{tool_call_count}: {tool_name} "
                                f"| args: { {k: v for k, v in event.get('data', {}).items() if k != 'tool'} }"
                            )
                            event.pop('_is_tool_call', None)
                            if tool_call_count > self.MAX_TOOL_CALLS:
                                rag_logger.warning(
                                    f"[DeepAgent] ⚠ Tool call budget exceeded ({tool_call_count}/{self.MAX_TOOL_CALLS}), stopping further calls"
                                )
                                break
                        if event.get('_final_answer') is not None:
                            last_ai_content = event.pop('_final_answer')
                            rag_logger.info(
                                f"[DeepAgent] ✅ Final answer received ({len(last_ai_content)} chars)"
                            )
                        elif stream and event:
                            yield event

        finally:
            await agent_task

        # ── Assemble final result ─────────────────────────────────────────────
        if last_ai_content is None:
            yield {
                'type': 'error',
                'message': 'Agent did not produce a final answer.',
                'step': 'generation',
                'data': {},
            }
            return

        answer_body, follow_up_questions = _extract_follow_up_questions(last_ai_content)
        rag_logger.info(f"[DeepAgent] Answer body: {len(answer_body)} chars | follow-up questions: {len(follow_up_questions)}")
        if follow_up_questions:
            rag_logger.info(f"[DeepAgent] Follow-up questions: {follow_up_questions}")

        # Stream the final answer token-by-token (same as old pipeline)
        if stream:
            yield {'type': 'progress', 'message': 'Generating response...', 'step': 'generation', 'data': {}}
            words = answer_body.split(' ')
            for i, word in enumerate(words):
                sep = ' ' if i < len(words) - 1 else ''
                yield {'type': 'token', 'content': word + sep, 'step': 'generation', 'data': {}}
                if i % 8 == 0:
                    await asyncio.sleep(0)  # yield to event loop every 8 words

        # Deduplicate citations
        all_cits = citation_acc.all_citations()
        rag_logger.info(
            f"[DeepAgent] Citations before dedup: {len(all_cits)} "
            f"(sec={len(citation_acc.sec_citations)}, "
            f"transcript={len(citation_acc.transcript_citations)}, "
            f"news={len(citation_acc.news_citations)})"
        )
        unique_citations, _ = deduplicate_citations_and_chunks(all_cits, [], rag_logger)
        rag_logger.info(f"[DeepAgent] Citations after dedup: {len(unique_citations)}")

        total_time = time.time() - start_time

        # Update conversation memory
        if conversation_id:
            self.question_analyzer.add_to_conversation_memory(conversation_id, question, "user")
            self.question_analyzer.add_to_conversation_memory(conversation_id, answer_body, "assistant")

        rag_logger.info("=" * 80)
        rag_logger.info("[DeepAgent] ✅ COMPLETE")
        rag_logger.info(f"[DeepAgent]   total_time  : {total_time:.2f}s")
        rag_logger.info(f"[DeepAgent]   tool_calls  : {tool_call_count}")
        rag_logger.info(f"[DeepAgent]   citations   : {len(unique_citations)}")
        rag_logger.info(f"[DeepAgent]   answer_len  : {len(answer_body)} chars")
        rag_logger.info("=" * 80)

        if LOGFIRE_AVAILABLE and logfire:
            logfire.info(
                "deep_rag.flow.complete",
                total_time_ms=int(total_time * 1000),
                tool_calls=tool_call_count,
                citations=len(unique_citations),
                answer_length=len(answer_body),
            )

        final_result = {
            'success': True,
            'response': {
                'answer': answer_body,
                'confidence': 0.9,
                'citations': unique_citations,
                'context_chunks': [],
                'iterations': [],
                'total_iterations': tool_call_count,
                'follow_up_questions_asked': follow_up_questions,
                'accumulated_chunks_count': 0,
                'answer_mode': 'standard',
            },
            'chunks': [],
            'analysis': {},
            'timing': {
                'analysis': 0.0,
                'search': 0.0,
                'generation': total_time,
                'total': total_time,
            },
        }

        yield {
            'type': 'result',
            'message': 'Response generated successfully',
            'step': 'complete',
            'data': final_result,
        }

    async def execute_rag_flow_async(self, question: str, **kwargs) -> Dict[str, Any]:
        """Non-streaming wrapper — returns final result dict."""
        kwargs['stream'] = False
        final_result = None
        async for event in self.execute_rag_flow(question, **kwargs):
            if event.get('type') == 'result':
                final_result = event.get('data')
        return final_result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        """Build system prompt dynamically, injecting live data availability from config."""
        from datetime import datetime
        try:
            data_availability = self.config.get_quarter_context_for_llm()
        except Exception:
            data_availability = "Data availability information not loaded."
        current_date = datetime.now().strftime("%B %d, %Y")
        return _DEEP_AGENT_SYSTEM_PROMPT_BASE.format(
            data_availability=f"Today's date: {current_date}\n\n{data_availability}"
        )

    async def _build_messages(
        self, question: str, conversation_id: Optional[str]
    ) -> List[BaseMessage]:
        """Build LangChain message list including conversation history."""
        system_prompt = self._build_system_prompt()
        messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]

        if conversation_id:
            try:
                history_str = await self.question_analyzer.conversation_memory.format_context(
                    conversation_id
                )
                if history_str:
                    # Convert the formatted string back into alternating HumanMessage/AIMessage
                    for line in history_str.strip().splitlines():
                        # format_context produces lines like: "👤 User: ..." / "🤖 Assistant: ..."
                        if line.startswith("👤 User:"):
                            content = line[len("👤 User:"):].strip()
                            messages.append(HumanMessage(content=content))
                        elif line.startswith("🤖 Assistant:"):
                            content = line[len("🤖 Assistant:"):].strip()
                            messages.append(AIMessage(content=content))
            except Exception as e:
                logger.warning(f"⚠️ Could not load conversation history: {e}")

        messages.append(HumanMessage(content=question))
        rag_logger.info(f"[DeepAgent] Built {len(messages)} input messages (1 system + {len(messages)-2} history + 1 user)")
        return messages

    async def _translate_langgraph_chunk(
        self,
        chunk: Dict[str, Any],
        stream: bool,
        current_tool_calls: int,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Translate a single LangGraph stream update into SSE events.

        LangGraph yields {"agent": {...}} or {"tools": {...}}.
        We translate:
          agent node  → tool call announcements OR final answer marker
          tools node  → (already handled via event_queue by tool closures)
        """
        for node_name, update in chunk.items():
            if not isinstance(update, dict):
                continue
            messages = update.get("messages", [])
            if not messages:
                continue

            rag_logger.debug(f"[DeepAgent] LangGraph node='{node_name}' yielded {len(messages) if isinstance(messages, list) else 1} message(s)")

            for msg in (messages if isinstance(messages, list) else [messages]):
                if not isinstance(msg, BaseMessage):
                    continue

                # ── Agent node: LLM decided to call tools ──────────────────
                if isinstance(msg, AIMessage):
                    tool_calls = getattr(msg, 'tool_calls', None) or []

                    if tool_calls:
                        rag_logger.info(f"[DeepAgent] LLM → {len(tool_calls)} tool call(s): {[tc.get('name') for tc in tool_calls]}")
                        for tc in tool_calls:
                            tool_name = tc.get('name', '?')
                            tool_args = tc.get('args', {})
                            announce = self._announce_tool_call(tool_name, tool_args)
                            if stream and announce:
                                yield {**announce, '_is_tool_call': True}

                    else:
                        # No tool calls → this is the final answer
                        rag_logger.info("[DeepAgent] LLM → no tool calls, treating as final answer")
                        content = msg.content
                        if isinstance(content, list):
                            # Handle structured content blocks
                            text_parts = [
                                b.get('text', '') for b in content
                                if isinstance(b, dict) and b.get('type') == 'text'
                            ]
                            content = ' '.join(text_parts)
                        if isinstance(content, str) and content.strip():
                            yield {'_final_answer': content}

                # ── Tools node: raw tool messages already handled by closures ──
                # (tool closures put events into event_queue before returning)
                # We don't need to emit extra events here, but log for debugging.
                elif isinstance(msg, ToolMessage):
                    rag_logger.info(
                        f"[DeepAgent] ToolMessage from '{msg.name}': "
                        f"{len(str(msg.content))} chars returned to LLM"
                    )

    def _announce_tool_call(
        self, tool_name: str, tool_args: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Return a 'reasoning' event for an upcoming tool call (shown before search starts)."""
        ticker = tool_args.get('ticker', '').upper()
        question_snippet = str(tool_args.get('question', tool_args.get('query', '')))[:80]
        fiscal_year = tool_args.get('fiscal_year')

        if tool_name == 'search_sec_filings':
            fy_str = f" FY{fiscal_year}" if fiscal_year else ""
            return {
                'type': 'reasoning',
                'message': f"Looking up {ticker}{fy_str} in SEC annual filings...",
                'step': '10k_planning',
                'event_name': 'agent_tool_call',
                'data': {'tool': tool_name, 'ticker': ticker, 'fiscal_year': fiscal_year},
            }
        elif tool_name == 'search_earnings_transcripts':
            quarters = tool_args.get('quarters') or []
            q_str = f" ({', '.join(quarters[:2])})" if quarters else ""
            return {
                'type': 'reasoning',
                'message': f"Searching {ticker} earnings call transcripts{q_str}...",
                'step': 'search_planning',
                'event_name': 'agent_tool_call',
                'data': {'tool': tool_name, 'ticker': ticker, 'quarters': quarters},
            }
        elif tool_name == 'research_company':
            return {
                'type': 'reasoning',
                'message': f"Researching {ticker} (SEC filings + transcripts)...",
                'step': '10k_planning',
                'event_name': 'agent_tool_call',
                'data': {'tool': tool_name, 'ticker': ticker},
            }
        elif tool_name == 'research_companies':
            tickers_list = tool_args.get('tickers', [])
            tickers_str = ', '.join(t.upper() for t in tickers_list)
            return {
                'type': 'reasoning',
                'message': f"Researching {tickers_str} in parallel...",
                'step': '10k_planning',
                'event_name': 'agent_tool_call',
                'data': {'tool': tool_name, 'tickers': tickers_list},
            }
        elif tool_name == 'search_news':
            return {
                'type': 'reasoning',
                'message': f"Searching news: \"{question_snippet}\"",
                'step': 'news_search',
                'event_name': 'agent_tool_call',
                'data': {'tool': tool_name, 'query': question_snippet},
            }
        elif tool_name == 'write_research_notes':
            key = tool_args.get('key', '')
            return {
                'type': 'reasoning',
                'message': f"Saving intermediate findings ({key})...",
                'step': 'analysis',
                'event_name': 'agent_tool_call',
                'data': {'tool': tool_name, 'key': key},
            }
        elif tool_name == 'read_research_notes':
            key = tool_args.get('key', '')
            return {
                'type': 'reasoning',
                'message': f"Retrieving saved notes ({key})...",
                'step': 'analysis',
                'event_name': 'agent_tool_call',
                'data': {'tool': tool_name, 'key': key},
            }
        return None
