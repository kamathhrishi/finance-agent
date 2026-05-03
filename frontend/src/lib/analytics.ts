/**
 * Thin typed wrapper around PostHog event capture.
 *
 * Why a wrapper:
 *   - Single source of truth for event names and payload shapes (TypeScript
 *     enforces that callers pass the right props for each event)
 *   - One place to mock/silence in tests
 *   - One place to add per-event sampling, redaction, or env-based gating
 *
 * Pageviews and clicks are autocaptured by PostHog's React SDK — don't
 * fire those manually. Only the events below need explicit calls because
 * they describe domain semantics PostHog couldn't infer (e.g. "the user
 * pinned NVDA's FY2025 10-K to chat scope").
 *
 * If `VITE_PUBLIC_POSTHOG_KEY` is unset, every call is a silent no-op —
 * dev workflows don't pollute the production project.
 */
import posthog from 'posthog-js'

// ─── Event catalogue ───────────────────────────────────────────────────────
//
// Naming convention: `<noun>_<verb_past>` (e.g. `chat_message_sent`).
// Properties are flat where possible (PostHog's UI groups + filters by them).

export type AnalyticsEvent =
  | { name: 'chat_message_sent'; props: { model: string; chars: number; pinned_count: number; conversation_existing: boolean } }
  | { name: 'chat_response_received'; props: { model: string; latency_ms: number; tool_calls?: number; citation_count?: number; conversation_id?: string } }
  | { name: 'chat_response_error'; props: { model: string; reason?: string } }
  | { name: 'filing_pinned'; props: { ticker: string; form: string; period: string; total_pinned_after: number } }
  | { name: 'filing_unpinned'; props: { ticker: string; form: string; period: string; total_pinned_after: number } }
  | { name: 'filing_viewed'; props: { ticker: string; form: string; period: string; source: 'companies' | 'latest' | 'drilldown' | 'chat' | 'unknown' } }
  | { name: 'filing_edgar_opened'; props: { ticker: string; form: string; period: string } }
  | { name: 'model_changed'; props: { from: string; to: string } }
  | { name: 'coverage_modal_opened'; props: Record<string, never> }
  | { name: 'example_query_clicked'; props: { surface: 'landing' | 'chat'; query: string } }
  | { name: 'tab_visited'; props: { tab: 'companies' | 'latest_filings' | 'chat' | 'company_detail' | 'filing_viewer' } }
  | { name: 'scope_cleared'; props: { count_before: number } }

/**
 * Fire a custom event. Type-safe: TS will reject unknown event names or
 * mismatched property shapes.
 */
export function track<E extends AnalyticsEvent>(event: E): void {
  if (typeof window === 'undefined') return
  // posthog-js no-ops cleanly when not initialised, but guard explicitly to
  // avoid throwing in dev if the user hasn't set the env vars at all.
  try {
    if (!import.meta.env.VITE_PUBLIC_POSTHOG_KEY) return
    posthog.capture(event.name, event.props as Record<string, unknown>)
  } catch (e) {
    // Analytics failures must NEVER break the user-facing flow.
    console.warn('[analytics] capture failed:', e)
  }
}

/**
 * Identify the current user once they're signed in. Call from a useEffect
 * that depends on Clerk's userId.
 */
export function identify(userId: string, traits?: Record<string, unknown>): void {
  if (typeof window === 'undefined') return
  try {
    if (!import.meta.env.VITE_PUBLIC_POSTHOG_KEY) return
    posthog.identify(userId, traits)
  } catch (e) {
    console.warn('[analytics] identify failed:', e)
  }
}

/** Reset identity on sign-out so a future visitor isn't bucketed as the prior user. */
export function reset(): void {
  if (typeof window === 'undefined') return
  try {
    if (!import.meta.env.VITE_PUBLIC_POSTHOG_KEY) return
    posthog.reset()
  } catch (e) {
    console.warn('[analytics] reset failed:', e)
  }
}
