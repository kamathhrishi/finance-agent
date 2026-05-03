import { useState } from 'react'
import { Eye, ExternalLink, Check, MessageSquarePlus, ChevronDown, FileText, Layers, Hash, Calendar } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import {
  buildEdgarUrl,
  formatFilingDate,
  formatAccession,
  formatBytes,
  type Filing,
} from '../lib/coverageApi'
import { showToast } from './Toast'
import { useScope } from '../lib/scopeStore'
import { track } from '../lib/analytics'

interface FilingTableProps {
  filings: Filing[]
  /** Hides the Ticker column (used inside a single-company drilldown). */
  hideTickerColumn?: boolean
}

/**
 * Standardized filings table.
 *
 * Click anywhere on a row (except an action button) to expand a metadata
 * panel showing accession, sections, exhibits, size — actionable context
 * without taking the user to a separate page.
 *
 * Pin button uses scopeStore directly + fires toasts.
 */
export default function FilingTable({ filings, hideTickerColumn = false }: FilingTableProps) {
  const navigate = useNavigate()
  const { isInScope, add, remove, count, max } = useScope()
  const [expanded, setExpanded] = useState<Set<string>>(new Set())

  const toggleExpand = (path: string) => {
    setExpanded((prev) => {
      const next = new Set(prev)
      if (next.has(path)) next.delete(path)
      else next.add(path)
      return next
    })
  }

  const handleTogglePin = (f: Filing) => {
    if (isInScope(f.path)) {
      remove(f.path)
      track({
        name: 'filing_unpinned',
        props: { ticker: f.ticker, form: f.form, period: f.period_label, total_pinned_after: count - 1 },
      })
      showToast({
        title: 'Removed from chat scope',
        body: `${f.ticker} ${f.form} ${f.period_label}`,
        variant: 'info',
        durationMs: 2500,
      })
      return
    }
    if (count >= max) {
      showToast({
        title: `Scope limit reached (${max})`,
        body: 'Remove a pinned filing before adding another.',
        variant: 'error',
        durationMs: 4000,
      })
      return
    }
    add(f)
    track({
      name: 'filing_pinned',
      props: { ticker: f.ticker, form: f.form, period: f.period_label, total_pinned_after: count + 1 },
    })
    showToast({
      title: 'Added to chat scope',
      body: `${f.ticker} ${f.form} · ${f.period_label}`,
      variant: 'success',
      action: { label: 'Open chat', href: '/chat' },
      durationMs: 4000,
    })
  }

  return (
    <div className="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
      <table className="w-full text-sm">
        <colgroup>
          <col className="w-8" />
          <col className="w-32" />
          {!hideTickerColumn && <col className="w-20" />}
          <col className="w-20" />
          <col />
          <col className="w-64" />
        </colgroup>
        <thead>
          <tr className="bg-slate-50 border-b border-slate-200 text-left text-[10px] font-semibold uppercase tracking-wider text-slate-500">
            <th className="px-2 py-2.5"></th>
            <th className="px-3 py-2.5">Filed</th>
            {!hideTickerColumn && <th className="px-3 py-2.5">Ticker</th>}
            <th className="px-3 py-2.5">Form</th>
            <th className="px-3 py-2.5">Period</th>
            <th className="px-3 py-2.5 text-right pr-4">Actions</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {filings.map((f) => {
            const inScope = isInScope(f.path)
            const edgarUrl = buildEdgarUrl(f.accession)
            const isExpanded = expanded.has(f.path)
            return (
              <>
                <tr
                  key={f.path}
                  onClick={() => toggleExpand(f.path)}
                  className="hover:bg-slate-50/70 transition-colors cursor-pointer"
                >
                  <td className="px-2 py-3">
                    <ChevronDown
                      className={`w-3.5 h-3.5 text-slate-400 transition-transform ${
                        isExpanded ? '' : '-rotate-90'
                      }`}
                    />
                  </td>
                  <td className="px-3 py-3 whitespace-nowrap text-slate-700 font-mono text-xs tabular-nums">
                    {formatFilingDate(f.filing_date)}
                  </td>

                  {!hideTickerColumn && (
                    <td className="px-3 py-3 whitespace-nowrap">
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          navigate(`/companies/${f.ticker}`)
                        }}
                        className="font-bold text-[#0a1628] hover:underline text-sm"
                      >
                        {f.ticker}
                      </button>
                    </td>
                  )}

                  <td className="px-3 py-3 whitespace-nowrap">
                    <FormBadge form={f.form} />
                  </td>

                  <td className="px-3 py-3 text-slate-600 text-sm">{f.period_label}</td>

                  <td
                    className="px-3 py-3 whitespace-nowrap text-right pr-4"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <div className="inline-flex items-center gap-1.5">
                      <button
                        onClick={() => handleTogglePin(f)}
                        className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-[11px] font-semibold transition-colors ${
                          inScope
                            ? 'bg-emerald-600 text-white hover:bg-emerald-700'
                            : 'bg-[#0a1628] text-white hover:bg-[#1a2d4a]'
                        }`}
                        title={
                          inScope
                            ? 'Remove from chat scope'
                            : 'Pin this filing as context for chat'
                        }
                      >
                        {inScope ? (
                          <>
                            <Check className="w-3.5 h-3.5" />
                            <span>In chat scope</span>
                          </>
                        ) : (
                          <>
                            <MessageSquarePlus className="w-3.5 h-3.5" />
                            <span>Add to chat</span>
                          </>
                        )}
                      </button>
                      <button
                        onClick={() => {
                          track({
                            name: 'filing_viewed',
                            props: { ticker: f.ticker, form: f.form, period: f.period_label, source: hideTickerColumn ? 'drilldown' : 'latest' },
                          })
                          navigate(`/filings/${f.path}`)
                        }}
                        className="inline-flex items-center gap-1 px-2 py-1.5 rounded-md text-[11px] font-medium bg-white text-slate-600 border border-slate-200 hover:bg-slate-100 hover:text-slate-900 transition-colors"
                        title="View filing"
                      >
                        <Eye className="w-3.5 h-3.5" />
                        <span className="hidden sm:inline">View</span>
                      </button>
                      {edgarUrl && (
                        <a
                          href={edgarUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          onClick={() => track({
                            name: 'filing_edgar_opened',
                            props: { ticker: f.ticker, form: f.form, period: f.period_label },
                          })}
                          className="inline-flex items-center gap-1 px-2 py-1.5 rounded-md text-[11px] font-medium bg-white text-slate-500 border border-slate-200 hover:bg-slate-100 hover:text-slate-900 transition-colors"
                          title="Open original on SEC EDGAR"
                        >
                          <ExternalLink className="w-3.5 h-3.5" />
                          <span className="hidden sm:inline">EDGAR</span>
                        </a>
                      )}
                    </div>
                  </td>
                </tr>
                {isExpanded && (
                  <tr key={`${f.path}-meta`} className="bg-slate-50/60">
                    <td colSpan={hideTickerColumn ? 5 : 6} className="px-12 py-3">
                      <FilingMetaPanel filing={f} edgarUrl={edgarUrl} />
                    </td>
                  </tr>
                )}
              </>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function FilingMetaPanel({
  filing,
  edgarUrl,
}: {
  filing: Filing
  edgarUrl: string | null
}) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-x-6 gap-y-2 text-xs">
      <MetaItem icon={Calendar} label="Filed with SEC">
        <span className="font-mono">{formatFilingDate(filing.filing_date)}</span>
      </MetaItem>
      <MetaItem icon={Hash} label="Accession">
        {edgarUrl ? (
          <a
            href={edgarUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="font-mono text-[#0a1628] hover:underline"
          >
            {formatAccession(filing.accession)}
          </a>
        ) : (
          <span className="font-mono text-slate-700">{formatAccession(filing.accession)}</span>
        )}
      </MetaItem>
      <MetaItem icon={Layers} label="Sections parsed">
        <span className="text-slate-700">
          {filing.section_count ?? 0}
        </span>
      </MetaItem>
      <MetaItem icon={FileText} label="Exhibits / size">
        <span className="text-slate-700">
          {filing.exhibit_count ?? 0} exhibits
          {filing.filing_chars ? <span className="text-slate-400"> · {formatBytes(filing.filing_chars)}</span> : null}
        </span>
      </MetaItem>
    </div>
  )
}

function MetaItem({
  icon: Icon,
  label,
  children,
}: {
  icon: React.ComponentType<{ className?: string }>
  label: string
  children: React.ReactNode
}) {
  return (
    <div className="flex items-start gap-2 min-w-0">
      <Icon className="w-3.5 h-3.5 text-slate-400 mt-0.5 shrink-0" />
      <div className="min-w-0">
        <div className="text-[10px] uppercase tracking-wider text-slate-400 font-medium">
          {label}
        </div>
        <div className="mt-0.5 truncate">{children}</div>
      </div>
    </div>
  )
}

function FormBadge({ form }: { form: string }) {
  const styles =
    form === '10-K'
      ? 'bg-blue-50 text-blue-700 ring-blue-100'
      : form === '10-Q'
      ? 'bg-purple-50 text-purple-700 ring-purple-100'
      : form === '8-K'
      ? 'bg-amber-50 text-amber-700 ring-amber-100'
      : 'bg-slate-100 text-slate-600 ring-slate-200'
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-semibold tracking-tight ring-1 ring-inset ${styles}`}
    >
      {form}
    </span>
  )
}
