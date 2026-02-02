import { useEffect, useRef, useState, useCallback } from 'react'
import { useSearchParams } from 'react-router-dom'
import { motion } from 'framer-motion'
import { MessageSquare, Info } from 'lucide-react'
import ChatInput from '../components/ChatInput'
import ChatMessage from '../components/ChatMessage'
import Sidebar from '../components/Sidebar'
import AboutModal from '../components/AboutModal'
import { useChat } from '../hooks/useChat'

export default function ChatPage() {
  const [searchParams] = useSearchParams()
  const { messages, isLoading, sendMessage, clearMessages } = useChat()
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const messagesContainerRef = useRef<HTMLDivElement>(null)
  const hasExecutedInitialQuery = useRef(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [aboutOpen, setAboutOpen] = useState(false)
  const lastScrollTime = useRef(0)
  const isUserScrolling = useRef(false)

  // Auto-execute query from URL parameter
  useEffect(() => {
    const query = searchParams.get('q')
    if (query && !hasExecutedInitialQuery.current && messages.length === 0) {
      hasExecutedInitialQuery.current = true
      sendMessage(decodeURIComponent(query))
    }
  }, [searchParams, sendMessage, messages.length])

  // Throttled scroll function to avoid animation collisions
  const scrollToBottom = useCallback((force = false) => {
    const now = Date.now()
    // Throttle to max once every 100ms, unless forced
    if (!force && now - lastScrollTime.current < 100) return
    // Don't auto-scroll if user is scrolling
    if (isUserScrolling.current) return

    lastScrollTime.current = now
    messagesEndRef.current?.scrollIntoView({ behavior: 'auto' })
  }, [])

  // Track user scrolling to avoid fighting
  useEffect(() => {
    const container = messagesContainerRef.current
    if (!container) return

    let scrollTimeout: ReturnType<typeof setTimeout>
    const handleScroll = () => {
      isUserScrolling.current = true
      clearTimeout(scrollTimeout)
      // Reset after 1 second of no scrolling
      scrollTimeout = setTimeout(() => {
        isUserScrolling.current = false
      }, 1000)
    }

    container.addEventListener('scroll', handleScroll)
    return () => {
      container.removeEventListener('scroll', handleScroll)
      clearTimeout(scrollTimeout)
    }
  }, [])

  // Auto-scroll on new messages - throttled and non-smooth during streaming
  useEffect(() => {
    const lastMessage = messages[messages.length - 1]
    const isStreaming = lastMessage?.isStreaming

    if (isStreaming) {
      // During streaming, use instant scroll (no animation fighting)
      scrollToBottom()
    } else {
      // After streaming completes, do a smooth final scroll
      setTimeout(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
      }, 100)
    }
  }, [messages, scrollToBottom])

  const isEmpty = messages.length === 0

  return (
    <div className="min-h-screen bg-gradient-to-br from-white via-blue-50/20 to-slate-50/30">
      {/* Sidebar */}
      <Sidebar
        isCollapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      {/* Main content area - shifts based on sidebar */}
      <div
        className={`min-h-screen flex flex-col transition-all duration-200 ${
          sidebarCollapsed ? 'lg:pl-[72px]' : 'lg:pl-[240px]'
        }`}
      >
        {/* Header bar */}
        <header className="sticky top-0 z-20 bg-white/95 backdrop-blur-md border-b border-slate-200/60">
          <div className="flex items-center justify-between h-14 px-4 lg:px-6">
            <div className="flex items-center gap-3">
              <div className="lg:hidden w-10" /> {/* Spacer for mobile menu button */}
              <h1 className="text-lg font-semibold text-slate-800">Chat</h1>
              {messages.length > 0 && (
                <span className="text-sm text-slate-400">
                  {messages.filter(m => m.role === 'user').length} messages
                </span>
              )}
            </div>

            <div className="flex items-center gap-2">
              {messages.length > 0 && (
                <button
                  onClick={clearMessages}
                  className="text-sm text-slate-500 hover:text-[#0083f1] font-medium transition-colors px-3 py-1.5 hover:bg-[#0083f1]/5 rounded-lg"
                >
                  New chat
                </button>
              )}
              <button
                onClick={() => setAboutOpen(true)}
                className="flex items-center gap-1.5 text-sm text-slate-500 hover:text-[#0083f1] font-medium transition-colors px-3 py-1.5 hover:bg-[#0083f1]/5 rounded-lg"
              >
                <Info className="w-4 h-4" />
                About
              </button>
            </div>
          </div>
        </header>

        {/* Chat area */}
        <main ref={messagesContainerRef} className="flex-1 pb-32 overflow-y-auto">
          <div className="max-w-4xl mx-auto px-4 lg:px-6">
            {isEmpty ? (
              // Empty state
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex flex-col items-center justify-center min-h-[calc(100vh-200px)] text-center py-12"
              >
                <div className="w-16 h-16 bg-gradient-to-br from-[#0083f1] to-[#0070d8] rounded-2xl flex items-center justify-center shadow-lg mb-6">
                  <MessageSquare className="w-8 h-8 text-white" />
                </div>
                <h1 className="text-2xl font-bold text-slate-900 mb-2">
                  What would you like to know?
                </h1>
                <p className="text-slate-500 max-w-md mb-8">
                  Ask any question about US public companies. I can analyze 10-K filings, earnings
                  transcripts, and financial data.
                </p>

                {/* Data Coverage Badges */}
                <div className="flex flex-wrap justify-center gap-3 mb-8">
                  <span className="flex items-center gap-1.5 px-3 py-1.5 bg-white border border-slate-200 rounded-full text-xs text-slate-600">
                    <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full"></span>
                    9,000+ US companies
                  </span>
                  <span className="flex items-center gap-1.5 px-3 py-1.5 bg-white border border-slate-200 rounded-full text-xs text-slate-600">
                    <span className="w-1.5 h-1.5 bg-[#0083f1] rounded-full"></span>
                    Earnings calls 2022-2025
                  </span>
                  <span className="flex items-center gap-1.5 px-3 py-1.5 bg-white border border-slate-200 rounded-full text-xs text-slate-600">
                    <span className="w-1.5 h-1.5 bg-purple-500 rounded-full"></span>
                    10-K filings
                  </span>
                </div>

                {/* Example queries */}
                <div className="grid sm:grid-cols-2 gap-3 w-full max-w-2xl">
                  {[
                    "What is $AAPL's revenue breakdown by segment?",
                    "How has $NVDA's gross margin changed over time?",
                    "What are the main risks mentioned in $TSLA's 10-K?",
                    "What did $MSFT's CEO say about AI in the last earnings call?",
                  ].map((query, index) => (
                    <button
                      key={index}
                      onClick={() => sendMessage(query)}
                      className="p-4 text-left bg-white border border-slate-200 rounded-xl hover:border-[#0083f1]/50 hover:shadow-md hover:shadow-[#0083f1]/5 transition-all text-sm text-slate-700"
                    >
                      {query}
                    </button>
                  ))}
                </div>
              </motion.div>
            ) : (
              // Messages
              <div className="py-6 space-y-6">
                {messages.map((message) => (
                  <ChatMessage key={message.id} message={message} />
                ))}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>
        </main>

        {/* Fixed input at bottom */}
        <div className="fixed bottom-0 right-0 left-0 lg:left-[var(--sidebar-width)] bg-gradient-to-t from-white via-white to-transparent pt-6 pb-4 transition-all duration-200"
          style={{ '--sidebar-width': sidebarCollapsed ? '72px' : '240px' } as React.CSSProperties}
        >
          <div className="max-w-4xl mx-auto px-4 lg:px-6">
            <ChatInput
              onSubmit={sendMessage}
              isLoading={isLoading}
              placeholder="Ask a question... (mention tickers with $)"
              autoFocus={!searchParams.get('q')}
            />
            <p className="text-center text-xs text-slate-400 mt-3">
              StrataLens uses AI to analyze SEC filings and earnings transcripts.
              Always verify important financial decisions.
            </p>
          </div>
        </div>
      </div>

      {/* About Modal */}
      <AboutModal isOpen={aboutOpen} onClose={() => setAboutOpen(false)} />
    </div>
  )
}
