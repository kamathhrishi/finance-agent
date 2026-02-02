import { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, Download, Building2, Calendar, FileText, Highlighter } from 'lucide-react'
import { config } from '../lib/config'

interface TranscriptChunk {
  chunk_text: string
  chunk_id?: string
  relevance_score?: number
}

interface TranscriptModalProps {
  isOpen: boolean
  onClose: () => void
  company: string
  ticker: string
  quarter: string  // Format: "2025_Q2" or "Q2 2025"
  relevantChunks: TranscriptChunk[]
}

interface TranscriptData {
  success: boolean
  transcript_text: string
  metadata?: {
    date?: string
    title?: string
  }
}

// Parse quarter string to year and quarter number
function parseQuarter(quarter: string | number | null | undefined): { year: number; quarterNum: number } | null {
  // Ensure quarter is a string
  if (quarter == null) return null
  const quarterStr = String(quarter)

  // Handle formats: "2025_Q2", "2025_q2", "Q2 2025", "Q2_2025", "2025 Q2"
  const patterns = [
    /(\d{4})[-_\s]?[Qq](\d)/,  // 2025_Q2, 2025-Q2, 2025 Q2
    /[Qq](\d)[-_\s]?(\d{4})/,  // Q2_2025, Q2-2025, Q2 2025
  ]

  for (const pattern of patterns) {
    const match = quarterStr.match(pattern)
    if (match) {
      if (pattern === patterns[0]) {
        return { year: parseInt(match[1]), quarterNum: parseInt(match[2]) }
      } else {
        return { year: parseInt(match[2]), quarterNum: parseInt(match[1]) }
      }
    }
  }
  return null
}

// Format transcript text with speaker sections
function formatTranscriptWithSpeakers(text: string): string {
  if (!text) return ''

  const lines = text.split('\n')
  let formatted = ''
  let currentSpeaker = ''
  let currentContent: string[] = []

  const flushSpeaker = () => {
    if (currentSpeaker && currentContent.length > 0) {
      formatted += `<div class="speaker-section mb-4">
        <div class="speaker-name font-semibold text-[#0083f1] mb-1">${currentSpeaker}</div>
        <div class="speaker-content text-slate-700 leading-relaxed">${currentContent.join('\n')}</div>
      </div>`
    }
    currentContent = []
  }

  for (const line of lines) {
    const match = line.match(/^([A-Z][a-zA-Z\s\-'.]+(?:\s*[-–]\s*[A-Za-z\s,]+)?)\s*[:–-]\s*(.*)$/)
    if (match) {
      flushSpeaker()
      currentSpeaker = match[1].trim()
      if (match[2]) {
        currentContent.push(match[2])
      }
    } else if (line.trim()) {
      currentContent.push(line)
    }
  }
  flushSpeaker()

  return formatted || `<div class="text-slate-700 leading-relaxed whitespace-pre-wrap">${text}</div>`
}

// Highlight relevant chunks in transcript
function highlightRelevantChunks(html: string, chunks: TranscriptChunk[]): string {
  if (!chunks || chunks.length === 0) return html

  let result = html

  for (const chunk of chunks) {
    if (!chunk.chunk_text || chunk.chunk_text.length < 20) continue

    // Escape special regex characters
    const escapedText = chunk.chunk_text
      .replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
      .replace(/\s+/g, '\\s+')  // Handle flexible whitespace

    try {
      const regex = new RegExp(`(${escapedText.substring(0, 200)})`, 'gi')
      const score = chunk.relevance_score || 0.5
      const opacity = Math.max(0.2, Math.min(1.0, score))

      result = result.replace(regex, (match) =>
        `<mark class="chunk-highlight" style="background-color: rgba(59, 130, 246, ${opacity}); padding: 2px 4px; border-radius: 3px; cursor: pointer;" title="Relevance: ${Math.round(score * 100)}%">${match}</mark>`
      )
    } catch (e) {
      // Regex failed, skip this chunk
      console.warn('Failed to highlight chunk:', e)
    }
  }

  return result
}

export default function TranscriptModal({
  isOpen,
  onClose,
  company,
  ticker,
  quarter,
  relevantChunks
}: TranscriptModalProps) {
  const [transcript, setTranscript] = useState<TranscriptData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Fetch transcript when modal opens
  useEffect(() => {
    if (!isOpen) return

    const fetchTranscript = async () => {
      setLoading(true)
      setError(null)

      const parsed = parseQuarter(quarter)
      if (!parsed) {
        setError(`Invalid quarter format: ${quarter}`)
        setLoading(false)
        return
      }

      try {
        // Try demo endpoint first (no auth required)
        const response = await fetch(
          `${config.apiBaseUrl}/demo/transcript/${ticker}/${parsed.year}/${parsed.quarterNum}`
        )

        if (!response.ok) {
          throw new Error(`Transcript not found for ${ticker} Q${parsed.quarterNum} ${parsed.year}`)
        }

        const data = await response.json()
        setTranscript(data)
      } catch (err) {
        console.error('Failed to fetch transcript:', err)
        setError(err instanceof Error ? err.message : 'Failed to load transcript')
      } finally {
        setLoading(false)
      }
    }

    fetchTranscript()
  }, [isOpen, ticker, quarter])

  // Handle escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }

    if (isOpen) {
      document.addEventListener('keydown', handleEscape)
      document.body.style.overflow = 'hidden'
    }

    return () => {
      document.removeEventListener('keydown', handleEscape)
      document.body.style.overflow = ''
    }
  }, [isOpen, onClose])

  // Download transcript
  const handleDownload = useCallback(() => {
    if (!transcript?.transcript_text) return

    const blob = new Blob([transcript.transcript_text], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${ticker}_${quarter}_transcript.txt`
    a.click()
    URL.revokeObjectURL(url)
  }, [transcript, ticker, quarter])

  // Process and highlight transcript
  const processedTranscript = transcript?.transcript_text
    ? highlightRelevantChunks(
        formatTranscriptWithSpeakers(transcript.transcript_text),
        relevantChunks
      )
    : ''

  const parsed = parseQuarter(quarter)
  const displayQuarter = parsed ? `Q${parsed.quarterNum} ${parsed.year}` : quarter

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
            className="relative w-full max-w-4xl max-h-[90vh] bg-white rounded-2xl shadow-2xl flex flex-col overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 bg-gradient-to-r from-slate-50 to-white">
              <div>
                <h2 className="text-xl font-bold text-slate-900">
                  {company || ticker} {displayQuarter} Earnings Transcript
                </h2>
                {relevantChunks.length > 0 && (
                  <p className="text-sm text-slate-500 mt-0.5">
                    {relevantChunks.length} relevant section{relevantChunks.length > 1 ? 's' : ''} highlighted
                  </p>
                )}
              </div>
              <button
                onClick={onClose}
                className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Metadata bar */}
            <div className="flex items-center gap-4 px-6 py-3 bg-slate-50 border-b border-slate-100 text-sm text-slate-600">
              <div className="flex items-center gap-1.5">
                <Building2 className="w-4 h-4 text-[#0083f1]" />
                <span className="font-medium">{ticker}</span>
              </div>
              <div className="flex items-center gap-1.5">
                <Calendar className="w-4 h-4 text-[#0083f1]" />
                <span>{displayQuarter}</span>
              </div>
              {transcript?.metadata?.date && (
                <div className="flex items-center gap-1.5">
                  <FileText className="w-4 h-4 text-[#0083f1]" />
                  <span>{transcript.metadata.date}</span>
                </div>
              )}
              {relevantChunks.length > 0 && (
                <div className="flex items-center gap-1.5 ml-auto">
                  <Highlighter className="w-4 h-4 text-blue-500" />
                  <span className="text-blue-600 font-medium">
                    {relevantChunks.length} highlighted
                  </span>
                </div>
              )}
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto px-6 py-4">
              {loading ? (
                <div className="flex flex-col items-center justify-center py-12">
                  <div className="w-8 h-8 border-3 border-[#0083f1] border-t-transparent rounded-full animate-spin" />
                  <p className="mt-4 text-slate-500">Loading transcript...</p>
                </div>
              ) : error ? (
                <div className="flex flex-col items-center justify-center py-12 text-center">
                  <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mb-4">
                    <X className="w-8 h-8 text-red-500" />
                  </div>
                  <p className="text-red-600 font-medium">{error}</p>
                  <p className="text-slate-500 text-sm mt-2">
                    The transcript may not be available in our database.
                  </p>
                </div>
              ) : (
                <div
                  className="prose prose-slate max-w-none"
                  dangerouslySetInnerHTML={{ __html: processedTranscript }}
                />
              )}
            </div>

            {/* Footer */}
            <div className="flex items-center justify-between px-6 py-4 border-t border-slate-200 bg-slate-50">
              <p className="text-sm text-slate-500">
                Click highlighted sections for more details
              </p>
              <button
                onClick={handleDownload}
                disabled={!transcript?.transcript_text}
                className="flex items-center gap-2 px-4 py-2 bg-[#0083f1] text-white rounded-lg hover:bg-[#0070d8] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <Download className="w-4 h-4" />
                Download
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
