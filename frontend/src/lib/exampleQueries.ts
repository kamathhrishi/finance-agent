/**
 * Single source of truth for the example queries shown on the landing page
 * and the chat page's empty state. Both surfaces import from here so they
 * never drift.
 *
 * Curation principles:
 *   - Show analyst value, not toy use cases (avoid "what's NVDA's gross
 *     margin" — gross margin is on the 10-K cover page, no agent needed)
 *   - Pick HOT topics — things actually being debated right now (OpenAI
 *     partnerships, China export controls, SBC dilution, hyperscaler M&A)
 *   - Each query should require multi-section / multi-filing / multi-ticker
 *     reading — that's the corpus's edge over a one-off Google search
 *   - Two visual buckets so users see both shapes:
 *       single  — deep dive on one company
 *       across  — pattern across many companies
 */

export interface ExampleQuery {
  text: string
  /** Loose tag for analytics; the bucket the query renders under in the UI. */
  bucket: 'single' | 'across'
}

export const EXAMPLE_QUERIES: ExampleQuery[] = [
  // ── Single company — deep-dive on a specific business question ──────────
  {
    bucket: 'single',
    text:
      "What has $NVDA disclosed about customer concentration in its latest 10-K and 10-Qs (named customers, % of revenue), and how have its long-term supply purchase obligations and prepayments to TSMC and other suppliers evolved over the last two fiscal years?",
  },
  {
    bucket: 'single',
    text:
      "Analyze $PLTR's last two 10-Ks and explain why growth has been high but operating margins have stayed thin — what's eating the operating leverage?",
  },
  {
    bucket: 'single',
    text:
      "Trace $MSFT's disclosures about its OpenAI partnership and equity investment across the latest 10-K and recent 10-Qs — economics, accounting treatment, and how the language has shifted",
  },

  // ── Across companies — pattern detection across the coverage ────────────
  {
    bucket: 'across',
    text:
      "Track how $NVDA, $AMD, and $AVGO have modified their 10-K risk factors over the last 2–3 annual filings around US export controls on China, AI chip restrictions, and concentration in AI customers — what new language has appeared and what has been removed?",
  },
  {
    bucket: 'across',
    text:
      "What strategic narratives are the Mag 7 ($MSFT, $GOOGL, $AMZN, $META, $NVDA, $AAPL, $TSLA) emphasizing in their latest 10-K business overviews — where do they overlap, and where do they diverge most sharply?",
  },
  {
    bucket: 'across',
    text:
      "Compare stock-based compensation as a percentage of revenue for $PLTR, $SNOW, $CRWD, $DDOG, and $NET in their latest 10-Ks — which are most and least dilutive, and how is each company framing it?",
  },
]

/** Convenience accessors for the two buckets. */
export const SINGLE_COMPANY_EXAMPLES = EXAMPLE_QUERIES.filter((q) => q.bucket === 'single')
export const ACROSS_COMPANIES_EXAMPLES = EXAMPLE_QUERIES.filter((q) => q.bucket === 'across')
