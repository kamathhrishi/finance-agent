import { useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown, Check, Sparkles, Lock } from 'lucide-react'
import {
  MODEL_GROUPS,
  findModel,
  type ModelId,
} from '../lib/models'

interface ModelSelectorProps {
  value: ModelId
  onChange: (id: ModelId) => void
  /** Tightens vertical padding for inline use next to a chat input. */
  compact?: boolean
  /** Aligns the dropdown panel — defaults to 'left' (under the trigger). */
  align?: 'left' | 'right'
}

/**
 * Dropdown styled like Cursor / ChatGPT model picker.
 *
 * Trigger shows the current model name. Panel groups models under a heading
 * and short blurb; each row has label + description + check (selected) /
 * lock (disabled). Disabled rows are non-clickable and visibly greyed.
 */
export default function ModelSelector({
  value,
  onChange,
  compact = false,
  align = 'left',
}: ModelSelectorProps) {
  const [open, setOpen] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  const current = findModel(value) || findModel('gpt-5.4-mini')!

  useEffect(() => {
    if (!open) return
    const onClick = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onClick)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onClick)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  return (
    <div ref={containerRef} className="relative inline-block">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className={`inline-flex items-center justify-between gap-2 rounded-md border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 hover:border-slate-300 transition-colors w-[160px] ${
          compact ? 'px-2.5 py-1 text-[11px] h-7' : 'px-3 py-1.5 text-xs h-8'
        }`}
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <span className="inline-flex items-center gap-1.5 min-w-0">
          <Sparkles className="w-3 h-3 text-slate-400 shrink-0" />
          <span className="font-medium truncate">{current.label}</span>
        </span>
        <ChevronDown
          className={`w-3 h-3 text-slate-400 shrink-0 transition-transform ${open ? 'rotate-180' : ''}`}
        />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.12 }}
            className={`absolute z-50 mt-1.5 w-[300px] bg-white border border-slate-200 rounded-lg shadow-xl overflow-hidden ${
              align === 'right' ? 'right-0' : 'left-0'
            }`}
            role="listbox"
          >
            <div className="max-h-[480px] overflow-y-auto">
              {MODEL_GROUPS.map((group) => (
                <div key={group.heading} className="py-2">
                  <div className="px-3 pb-1.5">
                    <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                      {group.heading}
                    </div>
                    <div className="text-[11px] text-slate-400 mt-0.5">{group.blurb}</div>
                  </div>
                  <div className="px-1">
                    {group.models.map((m) => {
                      const selected = m.id === value
                      const disabled = !!m.disabled
                      return (
                        <button
                          key={m.id}
                          type="button"
                          disabled={disabled}
                          onClick={() => {
                            if (disabled) return
                            onChange(m.id)
                            setOpen(false)
                          }}
                          className={`w-full text-left flex items-start gap-2.5 px-2.5 py-2 rounded-md transition-colors ${
                            selected
                              ? 'bg-slate-50'
                              : disabled
                              ? 'opacity-50 cursor-not-allowed'
                              : 'hover:bg-slate-50 cursor-pointer'
                          }`}
                          aria-selected={selected}
                          role="option"
                        >
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-1.5">
                              <span className="text-[13px] font-semibold text-slate-900">
                                {m.label}
                              </span>
                              {disabled && (
                                <span className="inline-flex items-center gap-0.5 text-[9px] font-medium uppercase tracking-wider text-slate-400 bg-slate-100 px-1.5 py-0.5 rounded">
                                  <Lock className="w-2.5 h-2.5" />
                                  Soon
                                </span>
                              )}
                            </div>
                            <div
                              className="text-[11px] text-slate-500 mt-0.5 truncate"
                              title={m.description}
                            >
                              {m.description}
                            </div>
                          </div>
                          <div className="w-4 h-4 mt-0.5 flex items-center justify-center shrink-0">
                            {selected ? (
                              <Check className="w-3.5 h-3.5 text-emerald-600" />
                            ) : null}
                          </div>
                        </button>
                      )
                    })}
                  </div>
                </div>
              ))}
            </div>
            <div className="border-t border-slate-100 px-3 py-2 text-[10px] text-slate-400">
              More models coming soon.
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
