import { ChevronLeft, ChevronRight } from 'lucide-react'

interface PaginationProps {
  page: number // 1-indexed
  pageCount: number
  onPageChange: (page: number) => void
  /** Items shown on current page (for the "Showing 1–24 of 138" counter) */
  showingFrom?: number
  showingTo?: number
  total?: number
  className?: string
}

/**
 * Compact numbered pagination (Prev · 1 2 … 9 10 · Next) — capped to 7 visible
 * page buttons. Auto-collapses with ellipses for large page counts.
 */
export default function Pagination({
  page,
  pageCount,
  onPageChange,
  showingFrom,
  showingTo,
  total,
  className,
}: PaginationProps) {
  if (pageCount <= 1) {
    if (total != null && total > 0 && showingFrom != null && showingTo != null) {
      return (
        <div
          className={`flex items-center justify-between gap-3 px-4 py-3 bg-white border border-slate-200 rounded-lg text-sm text-slate-600 ${
            className || ''
          }`}
        >
          <span>
            Showing <span className="font-semibold text-slate-900">{showingFrom}–{showingTo}</span> of{' '}
            <span className="font-semibold text-slate-900">{total}</span>
          </span>
          <span />
        </div>
      )
    }
    return null
  }

  const pages = visiblePages(page, pageCount)

  return (
    <div
      className={`flex items-center justify-between gap-3 px-4 py-3 bg-white border border-slate-200 rounded-lg ${
        className || ''
      }`}
    >
      {showingFrom != null && showingTo != null && total != null ? (
        <span className="text-sm text-slate-600">
          Showing <span className="font-semibold text-slate-900">{showingFrom}–{showingTo}</span> of{' '}
          <span className="font-semibold text-slate-900">{total}</span>
        </span>
      ) : (
        <span />
      )}
      <nav className="flex items-center gap-1" aria-label="Pagination">
        <button
          onClick={() => onPageChange(Math.max(1, page - 1))}
          disabled={page === 1}
          className="inline-flex items-center gap-1 px-3 h-9 rounded-md border border-slate-200 text-slate-700 hover:bg-slate-50 hover:text-slate-900 disabled:opacity-40 disabled:cursor-not-allowed transition-colors text-sm font-medium"
          aria-label="Previous page"
        >
          <ChevronLeft className="w-4 h-4" />
          <span className="hidden sm:inline">Prev</span>
        </button>

        {pages.map((p, i) =>
          p === 'ellipsis' ? (
            <span
              key={`e-${i}`}
              className="inline-flex items-center justify-center w-9 h-9 text-sm text-slate-400"
            >
              …
            </span>
          ) : (
            <button
              key={p}
              onClick={() => onPageChange(p)}
              aria-current={p === page ? 'page' : undefined}
              className={`inline-flex items-center justify-center min-w-[2.25rem] h-9 px-3 rounded-md text-sm font-medium transition-colors ${
                p === page
                  ? 'bg-[#0a1628] text-white shadow-sm'
                  : 'border border-slate-200 text-slate-700 hover:bg-slate-50 hover:text-slate-900'
              }`}
            >
              {p}
            </button>
          ),
        )}

        <button
          onClick={() => onPageChange(Math.min(pageCount, page + 1))}
          disabled={page === pageCount}
          className="inline-flex items-center gap-1 px-3 h-9 rounded-md border border-slate-200 text-slate-700 hover:bg-slate-50 hover:text-slate-900 disabled:opacity-40 disabled:cursor-not-allowed transition-colors text-sm font-medium"
          aria-label="Next page"
        >
          <span className="hidden sm:inline">Next</span>
          <ChevronRight className="w-4 h-4" />
        </button>
      </nav>
    </div>
  )
}

/** Choose up to 7 page numbers to render, with 'ellipsis' fillers. */
function visiblePages(current: number, total: number): (number | 'ellipsis')[] {
  if (total <= 7) return Array.from({ length: total }, (_, i) => i + 1)

  // Always show first, last, current, and immediate neighbors
  const set = new Set<number>([1, total, current, current - 1, current + 1])
  if (current <= 3) [2, 3, 4].forEach((p) => set.add(p))
  if (current >= total - 2) [total - 1, total - 2, total - 3].forEach((p) => set.add(p))

  const sorted = Array.from(set)
    .filter((p) => p >= 1 && p <= total)
    .sort((a, b) => a - b)

  const out: (number | 'ellipsis')[] = []
  let prev = 0
  for (const p of sorted) {
    if (p - prev > 1) out.push('ellipsis')
    out.push(p)
    prev = p
  }
  return out
}
