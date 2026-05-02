/**
 * Per-user "scope" store — set of filings the user has pinned to their chat.
 *
 * Survives tab switches AND page refreshes (localStorage backed). Strict
 * isolation: keyed by Clerk userId so two users on the same browser cannot
 * see each other's pinned filings. An anonymous user has their own bucket
 * that doesn't leak to a signed-in user (or vice versa).
 *
 * The store is an external (non-React) singleton subscribed to by React via
 * useSyncExternalStore. That means all mounted components — Companies tab,
 * Latest tab, Chat page, the floating "scope pill" header — stay in sync
 * the instant a filing is added or removed, no prop drilling.
 *
 * Cleared on sign-out so the next person on this device starts clean.
 */
import { useEffect, useSyncExternalStore } from 'react'
import { useAuth } from '@clerk/clerk-react'
import type { Filing } from './coverageApi'

export interface ScopedFiling {
  ticker: string
  company_name: string
  form: string
  period_label: string
  filing_date: string
  path: string
}

const KEY_PREFIX = 'scope:v1:'
const MAX_SCOPED_FILINGS = 10 // generous; UI warns when adding the 11th

// ─── Internal store ────────────────────────────────────────────────────────

type Listener = () => void

class ScopeStore {
  private userKey: string = `${KEY_PREFIX}anon`
  private items: ScopedFiling[] = []
  private listeners: Set<Listener> = new Set()
  private initialized: boolean = false

  setUser(userId: string | null | undefined) {
    const next = `${KEY_PREFIX}${userId || 'anon'}`
    // First call MUST always read localStorage. Otherwise an anon user whose
    // userKey happens to match the default ('scope:v1:anon') would never see
    // their persisted scope after a navigation/reload — the short-circuit
    // would skip the localStorage read.
    if (next === this.userKey && this.initialized) return
    this.userKey = next
    this.items = this.readFromLocalStorage()
    this.initialized = true
    this.emit()
  }

  /** Snapshot reference — must be stable when contents haven't changed for
   *  useSyncExternalStore to avoid re-render loops. */
  getSnapshot = (): ScopedFiling[] => this.items

  subscribe = (listener: Listener): (() => void) => {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }

  isInScope = (path: string): boolean => {
    return this.items.some((f) => f.path === path)
  }

  add(f: Filing | ScopedFiling) {
    if (this.isInScope(f.path)) return
    if (this.items.length >= MAX_SCOPED_FILINGS) return
    const next: ScopedFiling = {
      ticker: f.ticker,
      company_name: f.company_name,
      form: f.form,
      period_label: f.period_label,
      filing_date: f.filing_date,
      path: f.path,
    }
    this.items = [...this.items, next]
    this.persist()
    this.emit()
  }

  remove(path: string) {
    const next = this.items.filter((f) => f.path !== path)
    if (next.length === this.items.length) return
    this.items = next
    this.persist()
    this.emit()
  }

  toggle(f: Filing | ScopedFiling) {
    if (this.isInScope(f.path)) this.remove(f.path)
    else this.add(f)
  }

  clear() {
    if (this.items.length === 0) return
    this.items = []
    this.persist()
    this.emit()
  }

  /** Wipe a specific user's bucket — call on sign-out before re-anonymising. */
  wipeFor(userId: string | null | undefined) {
    if (typeof window === 'undefined') return
    try {
      window.localStorage.removeItem(`${KEY_PREFIX}${userId || 'anon'}`)
    } catch {
      /* ignore */
    }
  }

  // ─── Internals ────────────────────────────────────────────────────────

  private readFromLocalStorage(): ScopedFiling[] {
    if (typeof window === 'undefined') return []
    try {
      const raw = window.localStorage.getItem(this.userKey)
      if (!raw) return []
      const parsed = JSON.parse(raw)
      if (!Array.isArray(parsed)) return []
      return parsed.filter(
        (x) => x && typeof x.path === 'string' && typeof x.ticker === 'string',
      )
    } catch {
      return []
    }
  }

  private persist() {
    if (typeof window === 'undefined') return
    try {
      window.localStorage.setItem(this.userKey, JSON.stringify(this.items))
    } catch {
      /* quota / private browsing — silently degrade */
    }
  }

  private emit() {
    this.listeners.forEach((l) => l())
  }
}

const scopeStore = new ScopeStore()

// Re-sync to localStorage when another tab updates the scope for the same user.
if (typeof window !== 'undefined') {
  window.addEventListener('storage', (e) => {
    if (e.key && e.key.startsWith(KEY_PREFIX)) {
      // Force re-read by calling setUser with the same id (no-op if same key,
      // so we use a tiny dance: clear key then set it)
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const current = (scopeStore as any).userKey as string
      const userId = current.slice(KEY_PREFIX.length)
      // Trigger re-read by toggling
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      ;(scopeStore as any).userKey = ''
      scopeStore.setUser(userId)
    }
  })
}

// ─── React hook ────────────────────────────────────────────────────────────

export interface UseScope {
  scope: ScopedFiling[]
  isInScope: (path: string) => boolean
  add: (f: Filing | ScopedFiling) => void
  remove: (path: string) => void
  toggle: (f: Filing | ScopedFiling) => void
  clear: () => void
  count: number
  max: number
}

/**
 * useScope — React hook that returns the current user's scope plus mutators.
 *
 * Self-binds to Clerk so sign-in / sign-out / user-switch events
 * automatically re-key the store. Components don't need to do anything.
 */
export function useScope(): UseScope {
  const { userId, isSignedIn } = useAuth()

  // Bind store to current user. Runs whenever Clerk transitions.
  useEffect(() => {
    scopeStore.setUser(userId)
  }, [userId, isSignedIn])

  const scope = useSyncExternalStore(
    scopeStore.subscribe,
    scopeStore.getSnapshot,
    scopeStore.getSnapshot,
  )

  return {
    scope,
    isInScope: (path: string) => scopeStore.isInScope(path),
    add: (f) => scopeStore.add(f),
    remove: (path) => scopeStore.remove(path),
    toggle: (f) => scopeStore.toggle(f),
    clear: () => scopeStore.clear(),
    count: scope.length,
    max: MAX_SCOPED_FILINGS,
  }
}

// Exposed for sign-out cleanup hooks
export const _scopeStoreInternal = scopeStore
