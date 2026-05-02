import { useState, useEffect, useCallback } from 'react'
import { Newspaper, RefreshCw } from 'lucide-react'
import Sidebar from '../components/Sidebar'
import FilingTable from '../components/FilingTable'
import Pagination from '../components/Pagination'
import {
  fetchLatestFilings,
  fetchCoverageStatus,
  relativeTime,
  type Filing,
  type CoverageStatus,
} from '../lib/coverageApi'

const FORMS = ['All', '10-K', '10-Q', '8-K'] as const
type FormFilter = (typeof FORMS)[number]
const PAGE_SIZE = 25

/**
 * Latest Filings tab — newest-first feed across the whole coverage universe.
 *
 * Mystery rule: don't expose the overall filing total. The header shows the
 * number of companies covered and the last refresh time, nothing more. The
 * pagination footer DOES show the filtered total (so users can navigate),
 * which is honest and necessary for usability.
 */
export default function LatestFilingsPage() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [formFilter, setFormFilter] = useState<FormFilter>('All')
  const [tickerFilter, setTickerFilter] = useState('')
  const [filings, setFilings] = useState<Filing[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [status, setStatus] = useState<CoverageStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const loadPage = useCallback(
    async (p: number) => {
      try {
        setLoading(true)
        setError(null)
        const resp = await fetchLatestFilings({
          limit: PAGE_SIZE,
          offset: (p - 1) * PAGE_SIZE,
          form: formFilter === 'All' ? undefined : (formFilter as '10-K' | '10-Q' | '8-K'),
          ticker: tickerFilter.trim() || undefined,
        })
        setFilings(resp.items)
        setTotal(resp.total)
      } catch (e) {
        console.error('coverage/latest load failed:', e)
        setError("We couldn't load the latest filings right now. Please try again in a moment.")
      } finally {
        setLoading(false)
      }
    },
    [formFilter, tickerFilter],
  )

  // Reset to page 1 + reload when filters change
  useEffect(() => {
    setPage(1)
    loadPage(1)
  }, [formFilter, tickerFilter, loadPage])

  // Load page on page change (skip when filters trigger reload to page 1)
  useEffect(() => {
    if (page !== 1) loadPage(page)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page])

  // Status badge
  useEffect(() => {
    fetchCoverageStatus().then(setStatus).catch(() => {})
  }, [])

  const pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE))

  return (
    <div className="min-h-screen bg-slate-50">
      <Sidebar
        isCollapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed((c) => !c)}
      />

      <main
        className="min-h-screen transition-all duration-300"
        style={{ paddingLeft: sidebarCollapsed ? 60 : 220 }}
      >
        {/* Header */}
        <div className="border-b border-slate-200 bg-white">
          <div className="max-w-6xl mx-auto px-6 py-6">
            <div className="flex items-center gap-3 mb-1">
              <div className="w-9 h-9 rounded-lg bg-[#0a1628] flex items-center justify-center">
                <Newspaper className="w-5 h-5 text-white" />
              </div>
              <h1 className="text-2xl font-semibold text-[#0a1628] tracking-tight">
                Latest filings
              </h1>
            </div>
            <p className="text-sm text-slate-600 ml-12 flex items-center gap-3 flex-wrap">
              {status ? (
                <>
                  <span>
                    <span className="font-semibold text-slate-900">
                      {status.company_count}
                    </span>{' '}
                    tech companies in coverage
                  </span>
                  <span className="text-slate-300">·</span>
                  <span className="inline-flex items-center gap-1.5 text-slate-500">
                    <RefreshCw className="w-3 h-3" />
                    Updated {relativeTime(status.generated_at)}
                  </span>
                </>
              ) : (
                'Loading…'
              )}
            </p>

            {/* Filter row */}
            <div className="mt-5 flex items-center gap-3 flex-wrap">
              <div className="flex items-center gap-1.5">
                {FORMS.map((f) => (
                  <button
                    key={f}
                    onClick={() => setFormFilter(f)}
                    className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                      formFilter === f
                        ? 'bg-[#0a1628] text-white'
                        : 'bg-white border border-slate-200 text-slate-600 hover:bg-slate-50'
                    }`}
                  >
                    {f}
                  </button>
                ))}
              </div>
              <input
                value={tickerFilter}
                onChange={(e) => setTickerFilter(e.target.value.toUpperCase())}
                placeholder="Filter by ticker (e.g. NVDA)"
                className="px-3 py-1.5 bg-white border border-slate-200 rounded-md text-xs placeholder:text-slate-400 focus:outline-none focus:border-[#0a1628] w-52"
              />
            </div>
          </div>
        </div>

        {/* Body */}
        <div className="max-w-6xl mx-auto px-6 py-6">
          {error && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 mb-4">
              {error}
            </div>
          )}

          {loading && filings.length === 0 && (
            <div className="space-y-2">
              {Array.from({ length: 12 }).map((_, i) => (
                <div
                  key={i}
                  className="h-11 rounded-lg border border-slate-200 bg-white animate-pulse"
                />
              ))}
            </div>
          )}

          {!loading && filings.length === 0 && (
            <p className="text-sm text-slate-500 text-center py-12">
              No filings match the current filters.
            </p>
          )}

          {filings.length > 0 && (
            <>
              <FilingTable filings={filings} />
              <div className="mt-4">
                <Pagination
                  page={page}
                  pageCount={pageCount}
                  onPageChange={setPage}
                  showingFrom={(page - 1) * PAGE_SIZE + 1}
                  showingTo={Math.min(page * PAGE_SIZE, total)}
                  total={total}
                />
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  )
}
