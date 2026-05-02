import { Pin, X } from 'lucide-react'
import { useScope } from '../lib/scopeStore'

/**
 * ScopeChips — horizontal strip showing the user's pinned filings.
 *
 * Mounts above the chat input on /chat. Each chip is removable; the whole
 * strip can be cleared. Auto-hides when the scope is empty.
 */
export default function ScopeChips() {
  const { scope, remove, clear, count, max } = useScope()

  if (count === 0) return null

  return (
    <div className="mb-2 flex items-start gap-2">
      <div className="flex items-center gap-1 text-[11px] font-medium text-slate-500 pt-1.5 shrink-0">
        <Pin className="w-3 h-3" />
        Scope:
      </div>
      <div className="flex-1 flex flex-wrap gap-1.5">
        {scope.map((f) => (
          <span
            key={f.path}
            className="inline-flex items-center gap-1 pl-2 pr-1 py-1 rounded-full bg-emerald-50 border border-emerald-200 text-[11px] text-emerald-800"
            title={`${f.ticker} ${f.form} ${f.period_label} (filed ${f.filing_date})`}
          >
            <span className="font-semibold">{f.ticker}</span>
            <span className="text-emerald-600">·</span>
            <span>{f.form}</span>
            <span className="text-emerald-600">·</span>
            <span className="font-mono">{f.period_label}</span>
            <button
              onClick={() => remove(f.path)}
              className="ml-0.5 p-0.5 rounded-full hover:bg-emerald-100"
              title="Remove from scope"
            >
              <X className="w-3 h-3" />
            </button>
          </span>
        ))}
        <button
          onClick={clear}
          className="inline-flex items-center px-2 py-1 rounded-full text-[11px] text-slate-500 hover:text-slate-900 hover:bg-slate-100 transition-colors"
        >
          Clear all
        </button>
        {count >= max && (
          <span className="inline-flex items-center px-2 py-1 text-[11px] text-amber-700">
            (max {max} reached)
          </span>
        )}
      </div>
    </div>
  )
}
