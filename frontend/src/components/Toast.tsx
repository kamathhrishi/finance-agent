import { useEffect, useRef, useState, useSyncExternalStore } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, CheckCircle2, AlertCircle, Info } from 'lucide-react'

/**
 * Tiny toast system — global pubsub + <Toaster /> renderer.
 *
 * Usage:
 *   import { showToast } from './Toast'
 *   showToast({ title: 'Pinned to chat', body: 'NVDA 10-K added.', action: { label: 'Open chat', href: '/chat' } })
 *
 * Mount the Toaster once at app root.
 */

export type ToastVariant = 'success' | 'info' | 'error'

export interface ToastInput {
  title: string
  body?: string
  variant?: ToastVariant
  durationMs?: number
  action?: {
    label: string
    href?: string
    onClick?: () => void
  }
}

interface ToastEntry extends ToastInput {
  id: string
}

type Listener = () => void

const MAX_VISIBLE_TOASTS = 2

class ToastStore {
  private items: ToastEntry[] = []
  private listeners: Set<Listener> = new Set()

  subscribe = (l: Listener) => {
    this.listeners.add(l)
    return () => {
      this.listeners.delete(l)
    }
  }

  getSnapshot = (): ToastEntry[] => this.items

  add(t: ToastInput) {
    const id = `t_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
    // Cap concurrent toasts: drop the oldest when a new one arrives so the
    // stack stays small and never covers the page.
    const next = [...this.items, { ...t, id }]
    if (next.length > MAX_VISIBLE_TOASTS) {
      next.splice(0, next.length - MAX_VISIBLE_TOASTS)
    }
    this.items = next
    this.emit()
  }

  remove(id: string) {
    const next = this.items.filter((t) => t.id !== id)
    if (next.length === this.items.length) return
    this.items = next
    this.emit()
  }

  private emit() {
    this.listeners.forEach((l) => l())
  }
}

const _store = new ToastStore()

export function showToast(t: ToastInput) {
  _store.add(t)
}

export function Toaster() {
  const items = useSyncExternalStore(_store.subscribe, _store.getSnapshot, _store.getSnapshot)

  return (
    <div className="fixed bottom-4 right-4 z-[100] flex flex-col gap-2 w-[320px] pointer-events-none">
      <AnimatePresence>
        {items.map((t) => (
          <ToastCard key={t.id} toast={t} onClose={() => _store.remove(t.id)} />
        ))}
      </AnimatePresence>
    </div>
  )
}

function ToastCard({ toast, onClose }: { toast: ToastEntry; onClose: () => void }) {
  const [hovered, setHovered] = useState(false)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const duration = toast.durationMs ?? 5000

  useEffect(() => {
    if (hovered) {
      if (timer.current) clearTimeout(timer.current)
      return
    }
    timer.current = setTimeout(onClose, duration)
    return () => {
      if (timer.current) clearTimeout(timer.current)
    }
  }, [hovered, duration, onClose])

  const variant = toast.variant || 'success'
  const accent = {
    success: 'text-emerald-600',
    info: 'text-blue-600',
    error: 'text-red-600',
  }[variant]
  const Icon = {
    success: CheckCircle2,
    info: Info,
    error: AlertCircle,
  }[variant]

  return (
    <motion.div
      layout
      initial={{ y: 20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      exit={{ y: 20, opacity: 0, transition: { duration: 0.15 } }}
      transition={{ type: 'spring', damping: 24, stiffness: 320 }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className="pointer-events-auto bg-white border border-slate-200 rounded-lg shadow-lg px-3.5 py-3 flex items-start gap-2.5 w-full"
    >
      <Icon className={`w-4 h-4 mt-0.5 shrink-0 ${accent}`} />
      <div className="flex-1 min-w-0">
        <div className="text-sm font-semibold text-slate-900">{toast.title}</div>
        {toast.body && <div className="text-xs text-slate-600 mt-0.5">{toast.body}</div>}
        {toast.action && (
          <div className="mt-2">
            {toast.action.href ? (
              <a
                href={toast.action.href}
                onClick={() => {
                  toast.action?.onClick?.()
                  onClose()
                }}
                className="inline-flex items-center text-xs font-semibold text-[#0a1628] hover:underline"
              >
                {toast.action.label} →
              </a>
            ) : (
              <button
                onClick={() => {
                  toast.action?.onClick?.()
                  onClose()
                }}
                className="text-xs font-semibold text-[#0a1628] hover:underline"
              >
                {toast.action.label} →
              </button>
            )}
          </div>
        )}
      </div>
      <button
        onClick={onClose}
        className="text-slate-400 hover:text-slate-700 transition-colors"
        aria-label="Dismiss"
      >
        <X className="w-3.5 h-3.5" />
      </button>
    </motion.div>
  )
}
