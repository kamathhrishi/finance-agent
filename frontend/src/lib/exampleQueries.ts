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
 */

export const EXAMPLE_QUERIES: string[] = [
  "What has $NVDA disclosed about customer concentration in its latest 10-K and 10-Qs (named customers, % of revenue), and how have its long-term supply purchase obligations and prepayments to TSMC and other suppliers evolved over the last two fiscal years?",
  "Analyze $PLTR's last two 10-Ks and explain why growth has been high but operating margins have stayed thin. What's eating the operating leverage?",
  "Trace $MSFT's disclosures about its OpenAI partnership and equity investment across the latest 10-K and recent 10-Qs: economics, accounting treatment, and how the language has shifted.",
  "Track how $NVDA, $AMD, and $AVGO have modified their 10-K risk factors over the last 2-3 annual filings around US export controls on China, AI chip restrictions, and concentration in AI customers. What new language has appeared and what has been removed?",
  "What strategic narratives are the Mag 7 ($MSFT, $GOOGL, $AMZN, $META, $NVDA, $AAPL, $TSLA) emphasizing in their latest 10-K business overviews? Where do they overlap, and where do they diverge most sharply?",

  // ── Bench: queries we tried and pulled but worth keeping for reference ──
  // Restore by uncommenting and putting back into the array above.
  //
  // "Compare stock-based compensation as a percentage of revenue for $PLTR,
  //  $SNOW, $CRWD, $DDOG, and $NET in their latest 10-Ks. Which are most and
  //  least dilutive, and how is each company framing it?"
  //   — pulled because it overlapped tonally with other multi-ticker compares
  //     and the SBC angle felt narrower than the other five.
]
