import { motion, AnimatePresence } from 'framer-motion'
import { User, ChevronDown, ChevronUp, FileText, Newspaper, Link as LinkIcon, ExternalLink, Table, Expand, Shrink, Eye } from 'lucide-react'
import StrataLensLogo from './StrataLensLogo'
import TranscriptModal from './TranscriptModal'
import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { ChatMessage as ChatMessageType, Source } from '../lib/api'
import ReasoningTrace from './ReasoningTrace'

interface ChatMessageProps {
  message: ChatMessageType
}

// Citation type detection
function getCitationType(source: Source): 'transcript' | '10k' | 'news' {
  if (source.type === 'news' || source.url?.includes('http')) return 'news'
  if (source.type === '10-K' || source.section || source.fiscal_year) return '10k'
  return 'transcript'
}

// Get display title for citation
function getCitationTitle(source: Source, type: 'transcript' | '10k' | 'news'): string {
  if (type === 'transcript') {
    const company = source.company || source.ticker || 'Unknown Company'
    // Ensure quarter is a string before calling replace
    const quarterStr = source.quarter != null ? String(source.quarter) : ''
    const quarter = quarterStr.replace('_', ' ')
    return quarter ? `${company} - ${quarter}` : company
  }
  if (type === '10k') {
    const ticker = source.ticker || 'Unknown'
    const year = source.fiscal_year ? `FY${source.fiscal_year}` : ''
    return year ? `${ticker} ${year} 10-K` : `${ticker} 10-K`
  }
  // News
  return source.title || 'News Article'
}

// Citation badge component
function CitationBadge({ type }: { type: 'transcript' | '10k' | 'news' }) {
  const badges = {
    transcript: { icon: <FileText className="w-3 h-3" />, label: 'Transcript', color: 'bg-blue-100 text-blue-700' },
    '10k': { icon: <Table className="w-3 h-3" />, label: '10-K', color: 'bg-purple-100 text-purple-700' },
    news: { icon: <Newspaper className="w-3 h-3" />, label: 'News', color: 'bg-emerald-100 text-emerald-700' },
  }
  const badge = badges[type]

  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded ${badge.color}`}>
      {badge.icon}
      {badge.label}
    </span>
  )
}

// Individual citation card - clickable to expand
interface CitationCardProps {
  source: Source
  onViewTranscript?: (source: Source) => void
}

function CitationCard({ source, onViewTranscript }: CitationCardProps) {
  const [expanded, setExpanded] = useState(false)
  const type = getCitationType(source)
  const title = getCitationTitle(source, type)

  // Subtitle based on type
  const subtitle = type === 'transcript'
    ? source.transcript_available ? 'Full transcript available' : ''
    : type === '10k'
      ? source.section || ''
      : source.published_date || ''

  const hasContent = source.chunk_text && source.chunk_text.length > 0
  const canViewTranscript = type === 'transcript' && source.ticker && source.quarter

  return (
    <div
      className={`bg-white border rounded-lg overflow-hidden transition-all ${
        expanded ? 'border-[#0083f1]/30 shadow-sm' : 'border-slate-200'
      }`}
    >
      <div className="p-3">
        <div className="flex items-start justify-between gap-2">
          <div
            className="flex-1 min-w-0 cursor-pointer"
            onClick={() => hasContent && setExpanded(!expanded)}
          >
            {/* Badge and marker */}
            <div className="flex items-center gap-2 mb-1.5">
              <CitationBadge type={type} />
              {source.marker && (
                <span className="text-xs font-mono text-slate-400">{source.marker}</span>
              )}
            </div>

            {/* Title - prominent */}
            <h4 className="font-semibold text-slate-900 text-sm">{title}</h4>

            {/* Subtitle */}
            {subtitle && (
              <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>
            )}
          </div>

          {/* Actions */}
          <div className="flex items-center gap-1 flex-shrink-0">
            {/* View Full Transcript button - only for transcripts */}
            {canViewTranscript && onViewTranscript && (
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  onViewTranscript(source)
                }}
                className="flex items-center gap-1.5 px-2 py-1 text-xs font-medium text-[#0083f1] bg-[#0083f1]/5 hover:bg-[#0083f1]/10 rounded transition-colors"
                title="View full transcript with highlighted sections"
              >
                <Eye className="w-3.5 h-3.5" />
                View Transcript
              </button>
            )}
            {type === 'news' && source.url && (
              <a
                href={source.url}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
                className="p-1.5 text-slate-400 hover:text-[#0083f1] hover:bg-[#0083f1]/5 rounded transition-colors"
                title="Open article"
              >
                <ExternalLink className="w-4 h-4" />
              </a>
            )}
            {hasContent && (
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  setExpanded(!expanded)
                }}
                className="p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded transition-colors"
                title={expanded ? 'Show less' : 'Show more'}
              >
                {expanded ? <Shrink className="w-4 h-4" /> : <Expand className="w-4 h-4" />}
              </button>
            )}
          </div>
        </div>

        {/* Preview text - only when collapsed */}
        {hasContent && !expanded && (
          <p
            className="text-xs text-slate-500 mt-2 line-clamp-2 leading-relaxed cursor-pointer"
            onClick={() => setExpanded(true)}
          >
            {source.chunk_text!.substring(0, 150)}{source.chunk_text!.length > 150 ? '...' : ''}
          </p>
        )}
      </div>

      {/* Expanded content */}
      <AnimatePresence>
        {expanded && hasContent && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="border-t border-slate-100"
          >
            <div className="p-3 bg-slate-50 text-sm text-slate-700 whitespace-pre-wrap max-h-64 overflow-y-auto leading-relaxed">
              {source.chunk_text}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// Citations section - COLLAPSED by default
interface CitationsSectionProps {
  sources: Source[]
  onViewTranscript: (source: Source) => void
}

function CitationsSection({ sources, onViewTranscript }: CitationsSectionProps) {
  const [expanded, setExpanded] = useState(false) // Collapsed by default

  // Group by type
  const transcripts = sources.filter(s => getCitationType(s) === 'transcript')
  const tenKs = sources.filter(s => getCitationType(s) === '10k')
  const news = sources.filter(s => getCitationType(s) === 'news')

  const parts = []
  if (transcripts.length > 0) parts.push(`${transcripts.length} transcript${transcripts.length > 1 ? 's' : ''}`)
  if (tenKs.length > 0) parts.push(`${tenKs.length} 10-K`)
  if (news.length > 0) parts.push(`${news.length} news`)

  return (
    <div className="mt-4 rounded-xl border border-slate-200 overflow-hidden">
      {/* Header - clickable */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-3 bg-slate-50 hover:bg-slate-100 transition-colors text-left"
      >
        <div className="flex items-center gap-2">
          <LinkIcon className="w-4 h-4 text-[#0083f1]" />
          <span className="font-medium text-slate-800">
            {sources.length} source{sources.length > 1 ? 's' : ''}
          </span>
          <span className="text-sm text-slate-500">
            ({parts.join(', ')})
          </span>
        </div>
        {expanded ? (
          <ChevronUp className="w-4 h-4 text-slate-400" />
        ) : (
          <ChevronDown className="w-4 h-4 text-slate-400" />
        )}
      </button>

      {/* Content */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0 }}
            animate={{ height: 'auto' }}
            exit={{ height: 0 }}
            className="overflow-hidden"
          >
            <div className="p-3 space-y-4 bg-white">
              {/* Transcript sources */}
              {transcripts.length > 0 && (
                <div>
                  <h5 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                    <FileText className="w-3.5 h-3.5" />
                    Document Sources ({transcripts.length})
                  </h5>
                  <div className="space-y-2">
                    {transcripts.map((source, idx) => (
                      <CitationCard
                        key={`transcript-${idx}`}
                        source={source}
                        onViewTranscript={onViewTranscript}
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* 10-K sources */}
              {tenKs.length > 0 && (
                <div>
                  <h5 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                    <Table className="w-3.5 h-3.5" />
                    10-K SEC Filings ({tenKs.length})
                  </h5>
                  <div className="space-y-2">
                    {tenKs.map((source, idx) => (
                      <CitationCard key={`10k-${idx}`} source={source} />
                    ))}
                  </div>
                </div>
              )}

              {/* News sources */}
              {news.length > 0 && (
                <div>
                  <h5 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                    <Newspaper className="w-3.5 h-3.5" />
                    Web Sources ({news.length})
                  </h5>
                  <div className="space-y-2">
                    {news.map((source, idx) => (
                      <CitationCard key={`news-${idx}`} source={source} />
                    ))}
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const [showReasoning, setShowReasoning] = useState(true)
  const [transcriptModal, setTranscriptModal] = useState<{
    isOpen: boolean
    source: Source | null
  }>({ isOpen: false, source: null })

  const isUser = message.role === 'user'
  const hasReasoning = message.reasoning && message.reasoning.length > 0
  const hasSources = message.sources && message.sources.length > 0

  // Get all relevant chunks for highlighting in transcript
  const getRelevantChunks = (ticker: string, quarter: string) => {
    if (!message.sources) return []
    return message.sources
      .filter(s => {
        const sourceType = getCitationType(s)
        if (sourceType !== 'transcript') return false
        const matchesTicker = s.ticker === ticker || s.company === ticker
        // Ensure both quarters are strings before comparing
        const sQuarter = s.quarter != null ? String(s.quarter) : ''
        const qQuarter = quarter != null ? String(quarter) : ''
        const matchesQuarter = sQuarter === qQuarter ||
          sQuarter.replace('_', ' ') === qQuarter.replace('_', ' ')
        return matchesTicker && matchesQuarter
      })
      .map(s => ({
        chunk_text: s.chunk_text || '',
        chunk_id: s.chunk_id,
        relevance_score: s.relevance_score || 0.5
      }))
  }

  const handleViewTranscript = (source: Source) => {
    setTranscriptModal({ isOpen: true, source })
  }

  const handleCloseTranscript = () => {
    setTranscriptModal({ isOpen: false, source: null })
  }

  return (
    <>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className={`flex gap-4 ${isUser ? 'flex-row-reverse' : ''}`}
      >
        {/* Avatar */}
        <div className={`flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center ${
          isUser
            ? 'bg-slate-200'
            : 'bg-gradient-to-br from-[#0083f1] to-[#0070d8] shadow-sm'
        }`}>
          {isUser ? (
            <User className="w-5 h-5 text-slate-600" />
          ) : (
            <StrataLensLogo size={18} className="text-white" />
          )}
        </div>

        {/* Message content */}
        <div className={`flex-1 min-w-0 ${isUser ? 'flex flex-col items-end' : ''}`}>
          {/* Reasoning trace (for assistant) - lighter colors */}
          {!isUser && hasReasoning && (
            <div className="w-full mb-3">
              <button
                onClick={() => setShowReasoning(!showReasoning)}
                className="flex items-center gap-2 text-sm text-slate-400 hover:text-[#0083f1] mb-2 transition-colors"
              >
                {showReasoning ? (
                  <ChevronUp className="w-4 h-4" />
                ) : (
                  <ChevronDown className="w-4 h-4" />
                )}
                {showReasoning ? 'Hide' : 'Show'} reasoning
              </button>
              {showReasoning && (
                <ReasoningTrace steps={message.reasoning!} isStreaming={message.isStreaming} />
              )}
            </div>
          )}

          {/* Message bubble - DARKER text for actual chat content */}
          <div
            className={`
              ${isUser
                ? 'bg-gradient-to-br from-[#0083f1] to-[#0070d8] text-white rounded-2xl rounded-tr-md px-4 py-3 shadow-sm max-w-[85%]'
                : 'w-full'
              }
              ${message.isStreaming && !message.content ? 'min-w-[100px]' : ''}
            `}
          >
            {message.content ? (
              isUser ? (
                <div className="whitespace-pre-wrap">{message.content}</div>
              ) : (
                <div className="prose prose-slate max-w-none
                  prose-headings:text-slate-900 prose-headings:font-bold
                  prose-h1:text-xl prose-h1:mb-4 prose-h1:mt-6
                  prose-h2:text-lg prose-h2:mb-3 prose-h2:mt-5
                  prose-h3:text-base prose-h3:mb-2 prose-h3:mt-4
                  prose-p:text-slate-800 prose-p:leading-relaxed prose-p:mb-3
                  prose-strong:text-slate-900 prose-strong:font-bold
                  prose-ul:my-2 prose-ol:my-2
                  prose-li:text-slate-800 prose-li:my-1
                  prose-code:text-[#0083f1] prose-code:bg-blue-50 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-sm prose-code:font-medium prose-code:before:content-[''] prose-code:after:content-['']
                  prose-pre:bg-slate-900 prose-pre:text-slate-100 prose-pre:rounded-xl prose-pre:p-4
                  prose-blockquote:border-l-[#0083f1] prose-blockquote:bg-blue-50/50 prose-blockquote:py-1 prose-blockquote:px-4 prose-blockquote:rounded-r-lg prose-blockquote:text-slate-700
                  prose-table:border-collapse prose-table:w-full
                  prose-th:bg-slate-100 prose-th:text-slate-900 prose-th:font-semibold prose-th:text-left prose-th:p-2 prose-th:border prose-th:border-slate-200
                  prose-td:p-2 prose-td:border prose-td:border-slate-200 prose-td:text-slate-800
                  prose-a:text-[#0083f1] prose-a:font-medium prose-a:no-underline hover:prose-a:underline
                ">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {message.content}
                  </ReactMarkdown>
                  {message.isStreaming && (
                    <span className="inline-block w-2 h-4 ml-1 bg-[#0083f1] animate-pulse rounded-sm" />
                  )}
                </div>
              )
            ) : null}
          </div>

          {/* Sources - only for assistant messages when done streaming */}
          {!isUser && hasSources && !message.isStreaming && (
            <CitationsSection
              sources={message.sources!}
              onViewTranscript={handleViewTranscript}
            />
          )}
        </div>
      </motion.div>

      {/* Transcript Modal */}
      {transcriptModal.source && (
        <TranscriptModal
          isOpen={transcriptModal.isOpen}
          onClose={handleCloseTranscript}
          company={transcriptModal.source.company || transcriptModal.source.ticker || ''}
          ticker={transcriptModal.source.ticker || ''}
          quarter={transcriptModal.source.quarter || ''}
          relevantChunks={getRelevantChunks(
            transcriptModal.source.ticker || '',
            transcriptModal.source.quarter || ''
          )}
        />
      )}
    </>
  )
}
