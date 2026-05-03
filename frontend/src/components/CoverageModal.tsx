import { motion, AnimatePresence } from 'framer-motion'
import { X, Search, FileText } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { fetchCoverageCompanies, type CompanySummary } from '../lib/coverageApi'

interface CoverageModalProps {
  isOpen: boolean
  onClose: () => void
}

/**
 * Coverage modal — dynamic, fetches the live list from `/coverage/companies`
 * so it always matches what the deployed corpus actually has. The previous
 * version hardcoded a 138-ticker array in this file; the universe doubled
 * shortly after and the modal silently lied. Don't repeat that.
 *
 * On modal open we fire a single fetch (cached for the modal's lifetime).
 * Showing a small loading state while the fetch is in flight is fine —
 * the modal is opened by an explicit user action so a 200ms shimmer is OK.
 */
export default function CoverageModal({ isOpen, onClose }: CoverageModalProps) {
  const [query, setQuery] = useState('')
  const [companies, setCompanies] = useState<CompanySummary[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const onEscape = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    if (isOpen) {
      document.addEventListener('keydown', onEscape)
      document.body.style.overflow = 'hidden'
      setQuery('')

      // Fetch fresh on every open — the universe grows, so a 30-min-old
      // snapshot kept across opens isn't worth a stale-by-default UX.
      let cancelled = false
      setError(null)
      fetchCoverageCompanies()
        .then((rows) => {
          if (!cancelled) setCompanies(rows)
        })
        .catch((e) => {
          if (cancelled) return
          console.error('coverage modal load failed:', e)
          setError("We couldn't load the coverage list right now. Please try again in a moment.")
        })
      return () => {
        cancelled = true
        document.removeEventListener('keydown', onEscape)
        document.body.style.overflow = ''
      }
    }
    return () => {
      document.removeEventListener('keydown', onEscape)
      document.body.style.overflow = ''
    }
  }, [isOpen, onClose])

  const total = companies?.length ?? 0

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

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-[9999] flex items-center justify-center p-4"
          onClick={onClose}
        >
          {/* Backdrop */}
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />

          {/* Modal */}
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="relative w-full max-w-3xl max-h-[85vh] bg-white rounded-2xl shadow-2xl flex flex-col overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 bg-gradient-to-r from-slate-50 to-white shrink-0">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-[#0a1628] rounded-xl flex items-center justify-center shadow-sm">
                  <FileText className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h2 className="text-lg font-semibold text-slate-900">Coverage</h2>
                  <p className="text-xs text-slate-500 mt-0.5">
                    {companies
                      ? `${total} tech companies — full 10-K, 10-Q, 8-K SEC filings`
                      : 'Loading coverage…'}
                  </p>
                </div>
              </div>
              <button
                onClick={onClose}
                className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-slate-100 text-slate-500 hover:text-slate-900 transition-colors"
                aria-label="Close coverage"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Coverage stats strip */}
            <div className="grid grid-cols-3 gap-px bg-slate-200 border-b border-slate-200 shrink-0">
              <div className="bg-white px-5 py-3">
                <div className="text-2xl font-semibold text-[#0a1628]">
                  {companies ? total : '—'}
                </div>
                <div className="text-xs text-slate-500 mt-0.5">Tech Companies</div>
              </div>
              <div className="bg-white px-5 py-3">
                <div className="text-2xl font-semibold text-[#0a1628]">3+</div>
                <div className="text-xs text-slate-500 mt-0.5">Years Coverage</div>
              </div>
              <div className="bg-white px-5 py-3">
                <div className="text-2xl font-semibold text-[#0a1628]">10-K · 10-Q · 8-K</div>
                <div className="text-xs text-slate-500 mt-0.5">Filing Types</div>
              </div>
            </div>

            {/* Search */}
            <div className="px-6 py-3 border-b border-slate-200 shrink-0">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                <input
                  type="text"
                  autoFocus
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Filter by ticker or company name..."
                  className="w-full pl-9 pr-3 py-2 text-sm bg-slate-50 border border-slate-200 rounded-lg placeholder:text-slate-400 focus:outline-none focus:border-[#0a1628] focus:ring-1 focus:ring-[#0a1628]"
                />
              </div>
              {query && companies && (
                <p className="text-xs text-slate-400 mt-2">
                  {filtered.length} of {total} match
                </p>
              )}
            </div>

            {/* Ticker grid */}
            <div className="flex-1 overflow-y-auto px-6 py-4">
              {error && (
                <p className="text-sm text-red-600 text-center py-12">{error}</p>
              )}
              {!error && !companies && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
                  {Array.from({ length: 12 }).map((_, i) => (
                    <div key={i} className="h-8 rounded-lg bg-slate-100 animate-pulse" />
                  ))}
                </div>
              )}
              {!error && companies && filtered.length === 0 && (
                <p className="text-sm text-slate-500 text-center py-12">
                  No matches. We currently cover {total} tech companies — request additions via the chat.
                </p>
              )}
              {!error && companies && filtered.length > 0 && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
                  {filtered.map((c) => (
                    <div
                      key={c.ticker}
                      className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-slate-50 transition-colors"
                    >
                      <span className="text-xs font-mono font-semibold text-[#0a1628] bg-slate-100 px-2 py-0.5 rounded min-w-[3.25rem] text-center">
                        {c.ticker}
                      </span>
                      <span className="text-sm text-slate-700 truncate">{c.company_name || '—'}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Footer */}
            <div className="px-6 py-3 border-t border-slate-200 bg-slate-50 text-xs text-slate-500 flex items-center justify-between shrink-0">
              <span>Updated continuously from SEC EDGAR.</span>
              <span className="text-slate-400">Don't see your company? Ask in chat.</span>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
