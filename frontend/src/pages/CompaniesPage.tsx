import { useState, useEffect, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { Search, Building2, Calendar, ChevronRight } from 'lucide-react'
import Sidebar from '../components/Sidebar'
import Pagination from '../components/Pagination'
import {
  fetchCoverageCompanies,
  fetchCoverageStatus,
  formatFilingDate,
  formatCoverageCount,
  type CompanySummary,
  type CoverageStatus,
} from '../lib/coverageApi'

/**
 * Companies tab — covered SEC-filing universe.
 *
 * Mystery rule: do NOT show per-form filing counts or any overall filing total.
 * What we DO show: number of companies, per-company total filings, latest
 * filing date.
 */
const PAGE_SIZE = 24

export default function CompaniesPage() {
  const navigate = useNavigate()
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [query, setQuery] = useState('')
  const [companies, setCompanies] = useState<CompanySummary[] | null>(null)
  const [status, setStatus] = useState<CoverageStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [page, setPage] = useState(1)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    Promise.all([fetchCoverageCompanies(), fetchCoverageStatus()])
      .then(([rows, st]) => {
        if (cancelled) return
        setCompanies(rows)
        setStatus(st)
      })
      .catch((e) => {
        if (cancelled) return
        // Log details for the developer console; show the user something vague.
        // Internal exception text never reaches the page.
        console.error('coverage/companies load failed:', e)
        setError("We couldn't load the companies right now. Please try again in a moment.")
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  // Reset to page 1 whenever the query changes
  useEffect(() => setPage(1), [query])

  const filtered = useMemo(() => {
    if (!companies) return []
    const q = query.trim().toLowerCase()
    if (!q) return companies
    return companies.filter(
      (c) =>
        c.ticker.toLowerCase().includes(q) ||
        (c.company_name || '').toLowerCase().includes(q),
    )
  }, [companies, query])

  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))
  const pageSlice = useMemo(
    () => filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE),
    [filtered, page],
  )

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
                <Building2 className="w-5 h-5 text-white" />
              </div>
              <h1 className="text-2xl font-semibold text-[#0a1628] tracking-tight">
                Companies
              </h1>
            </div>
            <p className="text-sm text-slate-600 ml-12">
              {status ? (
                <>
                  <span className="font-semibold text-slate-900">
                    {formatCoverageCount(status.company_count)}
                  </span>{' '}
                  tech companies in coverage. Click any company to see its SEC filings.
                </>
              ) : (
                'Loading coverage…'
              )}
            </p>

            {/* Search bar */}
            <div className="mt-5 relative max-w-md">
              <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search by ticker or company name…"
                className="w-full pl-10 pr-3 py-2.5 bg-slate-50 border border-slate-200 rounded-lg text-sm placeholder:text-slate-400 focus:outline-none focus:border-[#0a1628] focus:bg-white transition-colors"
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

          {loading && !companies && (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {Array.from({ length: 12 }).map((_, i) => (
                <div
                  key={i}
                  className="h-[88px] rounded-lg border border-slate-200 bg-white animate-pulse"
                />
              ))}
            </div>
          )}

          {!loading && filtered.length === 0 && companies && (
            <p className="text-sm text-slate-500 text-center py-12">
              No companies match "{query}".
            </p>
          )}

          {pageSlice.length > 0 && (
            <>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                {pageSlice.map((c) => (
                  <CompanyCard key={c.ticker} company={c} onClick={() => navigate(`/companies/${c.ticker}`)} />
                ))}
              </div>

              <div className="mt-6">
                <Pagination
                  page={page}
                  pageCount={pageCount}
                  onPageChange={setPage}
                  showingFrom={(page - 1) * PAGE_SIZE + 1}
                  showingTo={Math.min(page * PAGE_SIZE, filtered.length)}
                  total={filtered.length}
                />
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  )
}

function CompanyCard({
  company,
  onClick,
}: {
  company: CompanySummary
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className="group text-left rounded-lg border border-slate-200 bg-white px-4 py-3.5 hover:border-[#0a1628] hover:shadow-sm hover:-translate-y-px transition-all"
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-base font-bold text-[#0a1628] tracking-tight">
          {company.ticker}
        </span>
        <ChevronRight className="w-4 h-4 text-slate-300 group-hover:text-slate-500 transition-colors" />
      </div>
      <div
        className="text-sm text-slate-700 mt-0.5 truncate"
        title={company.company_name}
      >
        {company.company_name || '—'}
      </div>
      <div className="flex items-center gap-2 mt-2 text-[11px] text-slate-500">
        <span className="font-medium tabular-nums">
          {company.total > 0 ? `${company.total} filings` : 'no filings yet'}
        </span>
        {company.latest_filing_date && (
          <>
            <span className="text-slate-300">·</span>
            <span className="inline-flex items-center gap-1">
              <Calendar className="w-3 h-3" />
              {formatFilingDate(company.latest_filing_date)}
            </span>
          </>
        )}
      </div>
    </button>
  )
}
