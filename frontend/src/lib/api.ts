import { config } from './config'

export interface ReasoningStep {
  message: string
  step?: string
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  data?: any
}

export interface Source {
  type?: string
  company?: string
  ticker?: string
  quarter?: number
  year?: number
  chunk_text?: string
  chunk_id?: string
  chunk_length?: number
  relevance_score?: number
  transcript_available?: boolean
  fiscal_year?: number
  section?: string
  chunk_type?: string
  path?: string
  filing_date?: string
  filing_type?: string
  char_offset?: number
  url?: string
  published_date?: string
  page?: number
  title?: string
  marker?: string
  similarity?: number
  // agent: line-range citations into local markdown corpus
  source_backend?: 'sec' | 'fs_research'
  line_start?: number
  line_end?: number
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  reasoning?: ReasoningStep[]
  timestamp?: Date
  isStreaming?: boolean
}

export interface SSEEvent {
  type: string
  content?: string
  token?: string
  message?: string
  step?: string
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  data?: any
  answer?: string
  citations?: Source[]
  conversation_id?: string
}

export interface Conversation {
  id: string
  title?: string
  messages?: {
    id: string
    role: string
    content?: string
    citations?: Source[]
    reasoning?: ReasoningStep[]
    created_at: string
  }[]
  created_at?: string
  updated_at: string
}

// Portfolio types
export interface Holding {
  id: string
  symbol: string
  company_name?: string
  quantity?: number
  purchase_price?: number
  shares?: number
  average_cost?: number
  current_price?: number
  market_value?: number
  gain_loss?: number
  gain_loss_pct?: number
  created_at: string
}

export interface WatchlistItem {
  id: string
  symbol: string
  company_name?: string
  current_price?: number
  change_pct?: number
  created_at: string
  notes?: string
}

export interface PortfolioSummary {
  holdings: Holding[]
  watchlist: WatchlistItem[]
  holdings_count?: number
  watchlist_count?: number
  total_value?: number
  total_gain_loss?: number
  total_gain_loss_pct?: number
}

// Generic authenticated GET helper
async function authenticatedGet(path: string, token?: string | null) {
  const headers: Record<string, string> = {}
  if (token) headers.Authorization = `Bearer ${token}`
  const response = await fetch(`${config.apiBaseUrl}${path}`, { headers })
  if (!response.ok) {
    const text = await response.text()
    throw new Error(`${response.status}: ${text}`)
  }
  return response.json()
}

export interface ScopedFilingPayload {
  ticker: string
  form: string
  period_label: string
  filing_date: string
  path: string
}

export async function* streamChat(
  message: string,
  options?: {
    conversationId?: string
    authToken?: string | null
    scopedFilings?: ScopedFilingPayload[]
    model?: string
  }
): AsyncGenerator<SSEEvent> {
  const { conversationId, authToken, scopedFilings, model } = options || {}
  const endpoint = authToken ? '/chat/message/stream-v2' : '/chat/landing/demo/stream-v2'
  const url = `${config.apiBaseUrl}${endpoint}`
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (authToken) {
    headers.Authorization = `Bearer ${authToken}`
  }

  const response = await fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify({
      message,
      conversation_id: conversationId,
      scoped_filings: scopedFilings && scopedFilings.length > 0 ? scopedFilings : undefined,
      model: model || undefined,
    }),
  })

  if (!response.ok) {
    const text = await response.text()
    throw new Error(`HTTP error! status: ${response.status}, message: ${text}`)
  }

  const reader = response.body?.getReader()
  if (!reader) throw new Error('No response body')

  const decoder = new TextDecoder()
  let buffer = ''

  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6).trim()
          if (data && data !== '[DONE]') {
            try {
              yield JSON.parse(data)
            } catch (e) {
              console.warn('Failed to parse SSE event:', data, e)
            }
          }
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}

export async function fetchConversations(token: string): Promise<Conversation[]> {
  const response = await fetch(`${config.apiBaseUrl}/chat/conversations`, {
    headers: { Authorization: `Bearer ${token}` },
  })
  if (!response.ok) throw new Error(`Failed to fetch conversations: ${response.status}`)
  return (await response.json()).conversations || []
}

export async function fetchConversation(id: string, token: string): Promise<Conversation> {
  const response = await fetch(`${config.apiBaseUrl}/chat/conversations/${id}`, {
    headers: { Authorization: `Bearer ${token}` },
  })
  if (!response.ok) throw new Error(`Failed to fetch conversation: ${response.status}`)
  const data = (await response.json()).conversation
  if (!data) throw new Error('No conversation data returned from server')
  return { ...data, messages: data.messages || [] }
}

export function generateMessageId(): string {
  return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
}

// Company functions
export async function fetchCompanyProfile(ticker: string, token?: string | null) {
  return authenticatedGet(`/companies/${encodeURIComponent(ticker)}`, token)
}

export async function fetchIncomeStatement(ticker: string, token?: string | null, years = 5) {
  return authenticatedGet(`/companies/${encodeURIComponent(ticker)}/income-statement?years=${years}`, token)
}

export async function fetchBalanceSheet(ticker: string, token?: string | null, years = 5) {
  return authenticatedGet(`/companies/${encodeURIComponent(ticker)}/balance-sheet?years=${years}`, token)
}

export async function fetchCashFlow(ticker: string, token?: string | null, years = 5) {
  return authenticatedGet(`/companies/${encodeURIComponent(ticker)}/cash-flow?years=${years}`, token)
}

export async function fetchProductSegments(ticker: string, token?: string | null) {
  return authenticatedGet(`/companies/${encodeURIComponent(ticker)}/product-segments`, token)
}

export async function fetchGeographicSegments(ticker: string, token?: string | null) {
  return authenticatedGet(`/companies/${encodeURIComponent(ticker)}/geographic-segments`, token)
}

export async function fetchAvailableTranscripts(ticker: string, token?: string | null) {
  return authenticatedGet(`/transcript/${encodeURIComponent(ticker)}/available`, token)
}

export async function fetchTranscript(ticker: string, year: number, quarter: number | string, token?: string | null) {
  return authenticatedGet(`/transcript/${encodeURIComponent(ticker)}/${year}/${quarter}`, token)
}

export async function fetchAvailableSECFilings(ticker: string, token?: string | null) {
  return authenticatedGet(`/sec-filings/${encodeURIComponent(ticker)}/available`, token)
}

export async function searchCompanies(query: string, token?: string | null) {
  return authenticatedGet(`/companies/search?query=${encodeURIComponent(query)}`, token)
}

export async function searchCompaniesPublic(query: string, limit = 8) {
  const response = await fetch(
    `${config.apiBaseUrl}/companies/public/search?query=${encodeURIComponent(query)}&limit=${limit}`
  )
  if (!response.ok) {
    const text = await response.text()
    throw new Error(`${response.status}: ${text}`)
  }
  return response.json()
}

// Portfolio functions
export async function fetchPortfolioSummary(token: string | null): Promise<PortfolioSummary> {
  return authenticatedGet('/portfolio/summary', token)
}

export async function addHolding(symbol: string, companyName: string, token: string | null) {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (token) headers.Authorization = `Bearer ${token}`
  const response = await fetch(`${config.apiBaseUrl}/portfolio/holdings`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ symbol, company_name: companyName }),
  })
  if (!response.ok) {
    const text = await response.text()
    throw new Error(`${response.status}: ${text}`)
  }
  return response.json()
}

export async function deleteHolding(symbol: string, token: string | null) {
  const headers: Record<string, string> = {}
  if (token) headers.Authorization = `Bearer ${token}`
  const response = await fetch(`${config.apiBaseUrl}/portfolio/holdings/${encodeURIComponent(symbol)}`, {
    method: 'DELETE',
    headers,
  })
  if (!response.ok) {
    const text = await response.text()
    throw new Error(`${response.status}: ${text}`)
  }
}

export async function addToWatchlist(symbol: string, companyName: string, token: string | null) {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (token) headers.Authorization = `Bearer ${token}`
  const response = await fetch(`${config.apiBaseUrl}/portfolio/watchlist`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ symbol, company_name: companyName }),
  })
  if (!response.ok) {
    const text = await response.text()
    throw new Error(`${response.status}: ${text}`)
  }
  return response.json()
}

export async function removeFromWatchlist(symbol: string, token: string | null) {
  const headers: Record<string, string> = {}
  if (token) headers.Authorization = `Bearer ${token}`
  const response = await fetch(`${config.apiBaseUrl}/portfolio/watchlist/${encodeURIComponent(symbol)}`, {
    method: 'DELETE',
    headers,
  })
  if (!response.ok) {
    const text = await response.text()
    throw new Error(`${response.status}: ${text}`)
  }
}
