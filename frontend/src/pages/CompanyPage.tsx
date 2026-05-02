import { useState, useEffect, useMemo } from 'react'
import { useParams, Link } from 'react-router-dom'
import { ArrowLeft, Building2, Calendar } from 'lucide-react'
import Sidebar from '../components/Sidebar'
import FilingTable from '../components/FilingTable'
import Pagination from '../components/Pagination'
import {
  fetchCoverageCompany,
  formatFilingDate,
  type CompanyDetail,
} from '../lib/coverageApi'

const FORMS = ['All', '10-K', '10-Q', '8-K'] as const
type FormFilter = (typeof FORMS)[number]
const PAGE_SIZE = 25

export default function CompanyPage() {
  const { symbol } = useParams<{ symbol: string }>()
  const ticker = (symbol || '').trim().toUpperCase()
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [detail, setDetail] = useState<CompanyDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [formFilter, setFormFilter] = useState<FormFilter>('All')
  const [page, setPage] = useState(1)

  useEffect(() => {
    if (!ticker) return
    let cancelled = false
    setLoading(true)
    setError(null)
    setDetail(null)
    fetchCoverageCompany(ticker)
      .then((d) => {
        if (!cancelled) setDetail(d)
      })
      .catch((e) => {
        if (cancelled) return
        console.error('coverage/company load failed:', e)
        // Distinguish 404 (ticker not in coverage) from generic failures so
        // the user sees something actionable, not a vague message for both.
        const is404 = String(e?.message || '').includes('404')
        setError(
          is404
            ? `${ticker} isn't in the coverage universe.`
            : "We couldn't load this company's filings right now. Please try again in a moment.",
        )
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [ticker])

  // Reset page when filter changes
  useEffect(() => setPage(1), [formFilter])

  const visibleFilings = useMemo(() => {
    if (!detail) return []
    if (formFilter === 'All') return detail.filings
    return detail.filings.filter((f) => f.form === formFilter)
  }, [detail, formFilter])

  const pageCount = Math.max(1, Math.ceil(visibleFilings.length / PAGE_SIZE))
  const pageSlice = useMemo(
    () => visibleFilings.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE),
    [visibleFilings, page],
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
          <div className="max-w-5xl mx-auto px-6 py-5">
            <Link
              to="/companies"
              className="inline-flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-900 mb-3 transition-colors"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              All companies
            </Link>

            <div className="flex items-start gap-3">
              <div className="w-11 h-11 rounded-lg bg-[#0a1628] flex items-center justify-center flex-shrink-0">
                <Building2 className="w-5 h-5 text-white" />
              </div>
              <div className="min-w-0">
                <div className="flex items-baseline gap-2 flex-wrap">
                  <h1 className="text-2xl font-bold text-[#0a1628] tracking-tight">
                    {ticker}
                  </h1>
                  {detail && (
                    <span className="text-base text-slate-700 truncate">
                      · {detail.company_name}
                    </span>
                  )}
                </div>
                {detail && (
                  <div className="flex items-center gap-3 text-xs text-slate-500 mt-1">
                    <span>{detail.total} filings on file</span>
                    {(() => {
                      const latest = detail.filings[0]?.filing_date
                      return latest ? (
                        <>
                          <span className="text-slate-300">·</span>
                          <span className="inline-flex items-center gap-1">
                            <Calendar className="w-3 h-3" />
                            Latest filed {formatFilingDate(latest)}
                          </span>
                        </>
                      ) : null
                    })()}
                  </div>
                )}
              </div>
            </div>

            {/* Form filter pills */}
            {detail && detail.filings.length > 0 && (
              <div className="mt-5 flex items-center gap-1.5">
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
            )}
          </div>
        </div>

        {/* Body */}
        <div className="max-w-5xl mx-auto px-6 py-6">
          {error && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 mb-4">
              {error}
            </div>
          )}

          {loading && (
            <div className="space-y-2">
              {Array.from({ length: 8 }).map((_, i) => (
                <div
                  key={i}
                  className="h-11 rounded-lg border border-slate-200 bg-white animate-pulse"
                />
              ))}
            </div>
          )}

          {!loading && detail && visibleFilings.length === 0 && (
            <p className="text-sm text-slate-500 text-center py-12">
              {formFilter === 'All'
                ? 'No filings yet for this company.'
                : `No ${formFilter} filings yet for this company.`}
            </p>
          )}

          {!loading && pageSlice.length > 0 && (
            <>
              <FilingTable filings={pageSlice} hideTickerColumn />
              <div className="mt-4">
                <Pagination
                  page={page}
                  pageCount={pageCount}
                  onPageChange={setPage}
                  showingFrom={(page - 1) * PAGE_SIZE + 1}
                  showingTo={Math.min(page * PAGE_SIZE, visibleFilings.length)}
                  total={visibleFilings.length}
                />
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  )
}
