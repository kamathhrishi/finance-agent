import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams, useLocation } from 'react-router-dom'
import SECFilingViewer from '../components/SECFilingViewer'
import { fetchCoverageCompany, type Filing } from '../lib/coverageApi'

/**
 * /filings/:path  — opens the SECFilingViewer in modal mode for a corpus filing.
 *
 * Path is the markdown file path under data_root, e.g.
 *   filings/NVDA/10-K/FY2025/filing.md
 *   filings/NVDA/10-K/FY2025/sections/item-7-mda.md
 *
 * If the path is a filing dir (no trailing .md), we default to filing.md.
 *
 * Closing the viewer takes you back to wherever you came from.
 */
export default function FilingViewerPage() {
  const navigate = useNavigate()
  const location = useLocation()
  const params = useParams<{ '*': string }>()
  const rawPath = params['*'] || ''

  // Default to filing.md if the path looks like a directory
  const fullPath = rawPath.endsWith('.md') ? rawPath : `${rawPath.replace(/\/$/, '')}/filing.md`

  // Parse ticker / form / fiscal_year out of the path so the viewer header
  // can show meaningful metadata even before the document loads.
  const meta = useMemo(() => parseFilingPath(fullPath), [fullPath])

  // For 8-K filings (where filing_date is the period), we need to look up the
  // actual filing_date. Try to fetch it from the company endpoint.
  const [filing, setFiling] = useState<Filing | null>(null)
  useEffect(() => {
    if (!meta.ticker) return
    let cancelled = false
    fetchCoverageCompany(meta.ticker)
      .then((d) => {
        if (cancelled) return
        // Match by path prefix (rawPath might be the dir; metadata path is the dir too)
        const dirPath = rawPath.endsWith('.md')
          ? rawPath.replace(/\/[^/]*\.md$/, '')
          : rawPath.replace(/\/$/, '')
        const match = d.filings.find((f) => f.path === dirPath)
        if (match) setFiling(match)
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [meta.ticker, rawPath])

  const goBack = () => {
    // Prefer history if there's somewhere to go back to in this app
    if (location.key !== 'default') navigate(-1)
    else navigate('/companies')
  }

  if (!meta.ticker || !meta.form) {
    return (
      <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div className="bg-white rounded-xl shadow-2xl max-w-md p-6 text-center">
          <h2 className="text-lg font-semibold text-slate-900 mb-2">Couldn't open filing</h2>
          <p className="text-sm text-slate-600 mb-4">This filing link appears to be invalid.</p>
          <button
            onClick={goBack}
            className="px-4 py-2 rounded-lg bg-slate-900 text-white text-sm hover:bg-slate-800"
          >
            Go back
          </button>
        </div>
      </div>
    )
  }

  return (
    <SECFilingViewer
      isOpen={true}
      onClose={goBack}
      sourceBackend="fs_research"
      path={fullPath}
      ticker={meta.ticker}
      filingType={meta.form}
      fiscalYear={meta.fiscalYear || 0}
      quarter={meta.quarter}
      filingDate={filing?.filing_date}
      panelMode={false}
    />
  )
}

interface ParsedPath {
  ticker?: string
  form?: '10-K' | '10-Q' | '8-K'
  fiscalYear?: number
  quarter?: number
  filingDate?: string // for 8-K
}

function parseFilingPath(path: string): ParsedPath {
  // filings/<TICKER>/<FORM>/<PERIOD>/...
  const m = path.match(
    /^filings\/([A-Z][A-Z0-9._-]{0,9})\/(10-K|10-Q|8-K)\/(FY\d{4}(?:\/Q[1-4])?|\d{4}-\d{2}-\d{2})/,
  )
  if (!m) return {}
  const [, ticker, form, period] = m
  if (form === '10-K' || form === '10-Q' || form === '8-K') {
    if (form === '8-K') {
      return { ticker, form, filingDate: period }
    }
    if (period.includes('/Q')) {
      const [fy, qPart] = period.split('/')
      return {
        ticker,
        form,
        fiscalYear: parseInt(fy.replace('FY', ''), 10),
        quarter: parseInt(qPart.replace('Q', ''), 10),
      }
    }
    return { ticker, form, fiscalYear: parseInt(period.replace('FY', ''), 10) }
  }
  return {}
}
