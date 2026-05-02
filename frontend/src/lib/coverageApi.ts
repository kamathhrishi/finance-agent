/**
 * Typed wrapper for the /coverage/* endpoints (Companies + Latest Filings tabs).
 * All endpoints are public; no auth token required.
 */
import { config } from './config'

export interface Filing {
  ticker: string
  company_name: string
  form: '10-K' | '10-Q' | '8-K' | string
  fiscal_year_label: string | null
  quarter_label: string | null
  period_label: string
  filing_date: string // YYYY-MM-DD — when SEC received the document
  accession: string
  path: string // e.g. "filings/AAPL/10-K/FY2024"
  section_count?: number
  exhibit_count?: number
  filing_chars?: number
}

/** Format the 18-char accession with dashes: 0001045810-25-000023 */
export function formatAccession(accession: string): string {
  if (!accession) return ''
  const raw = accession.replace(/-/g, '')
  if (raw.length !== 18) return accession
  return `${raw.slice(0, 10)}-${raw.slice(10, 12)}-${raw.slice(12)}`
}

/** "412345" → "412 KB" — rough size hint for the main filing markdown. */
export function formatBytes(n: number | undefined): string {
  if (!n) return ''
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${Math.round(n / 1024)} KB`
  return `${(n / (1024 * 1024)).toFixed(1)} MB`
}

export interface CompanySummary {
  ticker: string
  cik: string
  company_name: string
  counts: Record<string, number> // per-form counts (kept in API but not displayed)
  total: number
  latest_filing_date: string | null
}

export interface CompanyDetail extends CompanySummary {
  filings: Filing[]
}

export interface CoverageStatus {
  generated_at: string
  company_count: number
  filing_count: number // received but intentionally not surfaced
  by_form: Record<string, number> // received but intentionally not surfaced
}

const base = () => config.apiBaseUrl || ''

export async function fetchCoverageStatus(): Promise<CoverageStatus> {
  const r = await fetch(`${base()}/coverage/status`)
  if (!r.ok) throw new Error(`coverage/status failed: ${r.status}`)
  return r.json()
}

export async function fetchCoverageCompanies(q?: string): Promise<CompanySummary[]> {
  const params = new URLSearchParams()
  if (q) params.set('q', q)
  const r = await fetch(`${base()}/coverage/companies${params.toString() ? `?${params}` : ''}`)
  if (!r.ok) throw new Error(`coverage/companies failed: ${r.status}`)
  return r.json()
}

export async function fetchCoverageCompany(ticker: string): Promise<CompanyDetail> {
  const r = await fetch(`${base()}/coverage/companies/${encodeURIComponent(ticker)}`)
  if (!r.ok) throw new Error(`coverage/companies/${ticker} failed: ${r.status}`)
  return r.json()
}

export interface LatestFilingsParams {
  limit?: number
  offset?: number
  form?: '10-K' | '10-Q' | '8-K'
  ticker?: string
}

export interface LatestFilingsResponse {
  items: Filing[]
  total: number
  offset: number
  limit: number
}

export async function fetchLatestFilings(
  p: LatestFilingsParams = {},
): Promise<LatestFilingsResponse> {
  const params = new URLSearchParams()
  if (p.limit != null) params.set('limit', String(p.limit))
  if (p.offset != null) params.set('offset', String(p.offset))
  if (p.form) params.set('form', p.form)
  if (p.ticker) params.set('ticker', p.ticker)
  const r = await fetch(`${base()}/coverage/latest${params.toString() ? `?${params}` : ''}`)
  if (!r.ok) throw new Error(`coverage/latest failed: ${r.status}`)
  return r.json()
}

/**
 * Build the SEC EDGAR filing-index URL.
 *
 * The corpus stores accessions as 18-digit no-dash strings (e.g.
 * "000104581025000023"). EDGAR's directory layout requires the accession with
 * dashes ("0001045810-25-000023") AND the leading-zeros stripped from the CIK
 * in the path. The `-index.htm` suffix lands on the human-friendly index page
 * with the list of attached documents (10-K + exhibits).
 *
 *   raw "000104581025000023"
 *      → cik_int  = 1045810
 *      → acc_dash = "0001045810-25-000023"
 *      → https://www.sec.gov/Archives/edgar/data/1045810/0001045810-25-000023-index.htm
 */
export function buildEdgarUrl(accession: string): string | null {
  if (!accession || accession.length < 18) return null
  const raw = accession.replace(/-/g, '')
  if (raw.length !== 18) return null
  const cikInt = parseInt(raw.slice(0, 10), 10)
  if (Number.isNaN(cikInt)) return null
  const accDashed = `${raw.slice(0, 10)}-${raw.slice(10, 12)}-${raw.slice(12)}`
  return `https://www.sec.gov/Archives/edgar/data/${cikInt}/${accDashed}-index.htm`
}

/** "2026-04-27" → "Apr 27, 2026" */
export function formatFilingDate(d: string): string {
  if (!d) return ''
  // Avoid timezone shifts by parsing as plain UTC date
  const [y, m, day] = d.split('-').map((s) => parseInt(s, 10))
  if (!y || !m || !day) return d
  const dt = new Date(Date.UTC(y, m - 1, day))
  return dt.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    timeZone: 'UTC',
  })
}

/** "2026-05-02T18:08:53Z" → "5 min ago" / "yesterday" / etc */
export function relativeTime(iso: string): string {
  if (!iso) return ''
  const then = new Date(iso).getTime()
  const now = Date.now()
  const sec = Math.max(1, Math.floor((now - then) / 1000))
  if (sec < 60) return `${sec}s ago`
  const min = Math.floor(sec / 60)
  if (min < 60) return `${min} min ago`
  const hr = Math.floor(min / 60)
  if (hr < 24) return `${hr}h ago`
  const day = Math.floor(hr / 24)
  if (day === 1) return 'yesterday'
  if (day < 7) return `${day} days ago`
  const wk = Math.floor(day / 7)
  if (wk < 5) return `${wk}w ago`
  const mo = Math.floor(day / 30)
  return `${mo}mo ago`
}
