import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { useState, useEffect, useRef } from 'react'
import StrataLensLogo from '../components/StrataLensLogo'
import AboutModal from '../components/AboutModal'
import { Check, X, Shield, Globe, Send, ArrowRight, ChevronRight } from 'lucide-react'

// Mock data for animated UIs - aligned with example queries
const filingData = [
  {
    ticker: "INTC",
    metrics: [
      { label: "Foundry Revenue", value: "$4.7B", change: "+12%" },
      { label: "Client Computing", value: "$29.3B", change: "-8%" },
      { label: "Operating Margin", value: "1.2%", change: "-5.1%" },
    ]
  },
  {
    ticker: "MSFT",
    metrics: [
      { label: "Azure Revenue", value: "$118.5B", change: "+29%" },
      { label: "AI Services", value: "$5.2B", change: "+60%" },
      { label: "Cloud Margin", value: "44.6%", change: "+2.1%" },
    ]
  },
  {
    ticker: "META",
    metrics: [
      { label: "AI Capex", value: "$37.5B", change: "+42%" },
      { label: "Revenue", value: "$134.9B", change: "+16%" },
      { label: "Operating Margin", value: "41%", change: "+7%" },
    ]
  },
]

const chatData = [
  {
    question: "$INTC foundry business commentary?",
    answer: "Intel Foundry Services reported $4.7B revenue. Management cited 'significant progress' with external customers and expects breakeven by 2027..."
  },
  {
    question: "Compare $MSFT and $GOOGL cloud",
    answer: "Azure grew 29% vs Google Cloud's 26%. Microsoft leads in enterprise AI integration while Google focuses on AI infrastructure..."
  },
  {
    question: "$META AI capex guidance?",
    answer: "Meta expects $37-40B in 2024 capex, primarily for AI infrastructure. Zuckerberg: 'AI is our biggest investment area'..."
  },
]

const transcriptData = [
  {
    company: "Intel Corp.",
    quarter: "Q4 2024",
    speakers: [
      { role: "CEO", text: "Intel Foundry is on track. We're making significant progress with external customers..." },
      { role: "CFO", text: "We expect foundry to reach breakeven by 2027 as we scale..." },
    ]
  },
  {
    company: "Microsoft",
    quarter: "Q2 2025",
    speakers: [
      { role: "CEO", text: "Azure continues to take share. AI is now a $5B+ annual run rate business..." },
      { role: "CFO", text: "Cloud gross margin expanded to 72%, driven by AI services..." },
    ]
  },
  {
    company: "Meta Platforms",
    quarter: "Q4 2024",
    speakers: [
      { role: "CEO", text: "We're investing aggressively in AI infrastructure. Llama is seeing strong adoption..." },
      { role: "CFO", text: "Capex guidance for next year is $37-40B, primarily for AI..." },
    ]
  },
]

const exampleQueries = [
  "$INTC management commentary on foundry business in last 3 quarters",
  "Compare $MSFT and $GOOGL cloud segment",
  "$META AI capex commentary in last 3 quarters",
]

export default function LandingPage() {
  const navigate = useNavigate()
  const [filingIndex, setFilingIndex] = useState(0)
  const [chatIndex, setChatIndex] = useState(0)
  const [transcriptIndex, setTranscriptIndex] = useState(0)
  const [inputValue, setInputValue] = useState('')
  const [aboutOpen, setAboutOpen] = useState(false)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    const interval = setInterval(() => setFilingIndex(i => (i + 1) % filingData.length), 4000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    const interval = setInterval(() => setChatIndex(i => (i + 1) % chatData.length), 5000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    const interval = setInterval(() => setTranscriptIndex(i => (i + 1) % transcriptData.length), 4500)
    return () => clearInterval(interval)
  }, [])

  const handleSubmit = (query: string) => {
    if (!query.trim()) return
    navigate(`/chat?q=${encodeURIComponent(query)}`)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(inputValue)
    }
  }

  return (
    <div className="min-h-screen bg-white font-sans antialiased">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 h-16 bg-white/90 backdrop-blur-xl border-b border-slate-200/80">
        <div className="max-w-6xl mx-auto px-6 h-full flex items-center justify-between">
          <a href="/" className="flex items-center gap-2.5 group">
            <div className="w-9 h-9 bg-gradient-to-br from-[#0066cc] to-[#0052a3] rounded-xl flex items-center justify-center shadow-md shadow-blue-500/20 group-hover:shadow-lg group-hover:shadow-blue-500/30 transition-shadow">
              <StrataLensLogo size={17} className="text-white" />
            </div>
            <span className="text-lg font-bold text-slate-900 tracking-tight">StrataLens</span>
          </a>
          <div className="hidden md:flex items-center gap-6">
            <a href="#features" className="text-slate-600 text-sm font-medium hover:text-slate-900 transition-colors px-2 py-1">Features</a>
            <a href="#why" className="text-slate-600 text-sm font-medium hover:text-slate-900 transition-colors px-2 py-1">Why StrataLens</a>
            <button
              onClick={() => setAboutOpen(true)}
              className="text-slate-600 text-sm font-medium hover:text-slate-900 transition-colors px-2 py-1"
            >
              About
            </button>
            <button
              onClick={() => navigate('/chat')}
              className="px-5 py-2.5 bg-gradient-to-br from-[#0066cc] to-[#0052a3] text-white text-sm font-semibold rounded-xl hover:from-[#0052a3] hover:to-[#003d7a] transition-all shadow-md shadow-blue-500/25 hover:shadow-lg hover:shadow-blue-500/35 hover:-translate-y-0.5"
            >
              Get Started
            </button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-16 min-h-screen flex items-center relative overflow-hidden">
        {/* Background */}
        <div className="absolute inset-0 bg-gradient-to-b from-white via-slate-50/80 to-slate-100" />
        <div className="absolute top-0 right-0 w-[800px] h-[800px] bg-gradient-radial from-blue-100/40 via-transparent to-transparent" />
        <div className="absolute bottom-0 left-0 w-[600px] h-[600px] bg-gradient-radial from-slate-200/50 via-transparent to-transparent" />

        {/* Dot pattern */}
        <div className="absolute inset-0 opacity-[0.03]" style={{
          backgroundImage: `radial-gradient(circle, #0066cc 1px, transparent 1px)`,
          backgroundSize: '32px 32px'
        }} />

        <div className="max-w-6xl mx-auto px-6 py-20 relative z-10 w-full">
          <div className="max-w-3xl mx-auto text-center">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
            >
              {/* Badge */}
              <div className="inline-flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-50 to-slate-50 border border-blue-200/60 rounded-full mb-8 shadow-sm">
                <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
                <span className="text-sm font-medium text-slate-700">AI-Powered Equity Research</span>
              </div>

              {/* Headline */}
              <h1 className="text-4xl md:text-5xl font-bold leading-[1.1] tracking-tight text-slate-800 mb-5">
                Equity Research Copilot
              </h1>

              {/* Subheadline */}
              <p className="text-base md:text-lg text-slate-500 leading-relaxed mb-8 max-w-lg mx-auto">
                Quick insights from U.S. public markets data
              </p>

              {/* Data pills */}
              <div className="flex flex-wrap justify-center gap-3 mb-10">
                {[
                  { color: 'bg-emerald-500', text: '9,000+ US companies' },
                  { color: 'bg-blue-500', text: 'Earnings calls 2022-2025' },
                  { color: 'bg-purple-500', text: '10-K SEC filings' },
                ].map((pill, i) => (
                  <span key={i} className="flex items-center gap-2 px-4 py-2 bg-white border border-slate-200 rounded-full text-sm text-slate-600 shadow-sm">
                    <span className={`w-2 h-2 ${pill.color} rounded-full`} />
                    {pill.text}
                  </span>
                ))}
              </div>
            </motion.div>

            {/* Search Input - Enterprise Style */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
              className="max-w-2xl mx-auto mb-8"
            >
              <div className="relative bg-white rounded-2xl border-2 border-slate-200 shadow-xl shadow-slate-200/50 hover:border-[#0066cc]/40 hover:shadow-blue-100/50 transition-all focus-within:border-[#0066cc] focus-within:shadow-blue-100/60">
                <textarea
                  ref={inputRef}
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="What do you want to know? (mention tickers with $)"
                  className="w-full px-6 py-5 pr-14 text-base text-slate-900 placeholder:text-slate-400 bg-transparent resize-none focus:outline-none min-h-[60px] max-h-[120px]"
                  rows={1}
                  autoFocus
                />
                <button
                  onClick={() => handleSubmit(inputValue)}
                  disabled={!inputValue.trim()}
                  className="absolute right-4 bottom-4 w-10 h-10 bg-gradient-to-br from-[#0066cc] to-[#0052a3] text-white rounded-xl flex items-center justify-center disabled:opacity-40 disabled:cursor-not-allowed hover:from-[#0052a3] hover:to-[#003d7a] transition-all shadow-md shadow-blue-500/25 hover:shadow-lg disabled:shadow-none"
                >
                  <Send className="w-4 h-4" />
                </button>
              </div>
            </motion.div>

            {/* Example Queries - Vertical List */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.4 }}
              className="max-w-2xl mx-auto"
            >
              <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Try an example</p>
              <div className="space-y-2">
                {exampleQueries.map((query, i) => (
                  <button
                    key={i}
                    onClick={() => handleSubmit(query)}
                    className="w-full flex items-center justify-between px-5 py-3.5 bg-white border border-slate-200 rounded-xl text-left text-slate-700 hover:border-[#0066cc]/50 hover:bg-blue-50/30 hover:text-slate-900 transition-all group shadow-sm hover:shadow-md"
                  >
                    <span className="font-medium">{query}</span>
                    <ChevronRight className="w-4 h-4 text-slate-400 group-hover:text-[#0066cc] group-hover:translate-x-1 transition-all" />
                  </button>
                ))}
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-28 bg-white relative">
        <div className="max-w-6xl mx-auto px-6">
          {/* Section Header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mb-20 text-center"
          >
            <span className="inline-block px-4 py-1.5 bg-blue-50 text-[#0066cc] text-xs font-bold uppercase tracking-wider rounded-full mb-4">
              Features
            </span>
            <h2 className="text-4xl md:text-5xl font-extrabold text-slate-900 tracking-tight mb-4">
              Everything you need
            </h2>
            <p className="text-lg text-slate-500 max-w-md mx-auto">
              10-K filings, earnings transcripts, financial metrics—delivered instantly.
            </p>
          </motion.div>

          {/* Feature 01 - 10-K Analysis */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center mb-32"
          >
            <div>
              <span className="inline-block px-3 py-1 bg-purple-100 text-purple-700 text-xs font-bold uppercase tracking-wider rounded-full mb-4">
                SEC Filings
              </span>
              <h3 className="text-3xl font-extrabold text-slate-900 mb-4 tracking-tight">
                Extract insights from 10-K filings
              </h3>
              <p className="text-slate-500 leading-relaxed mb-6">
                Get instant answers from annual reports. Revenue breakdowns, risk factors, segment analysis—all sourced directly from official SEC filings.
              </p>
              <ul className="space-y-3">
                {[
                  "Financial metrics extracted automatically",
                  "Year-over-year comparisons",
                  "Risk factors and MD&A analysis"
                ].map((item, i) => (
                  <li key={i} className="flex items-center gap-3">
                    <div className="w-5 h-5 bg-emerald-100 rounded-full flex items-center justify-center">
                      <Check className="w-3 h-3 text-emerald-600" />
                    </div>
                    <span className="text-slate-700">{item}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div className="relative">
              <div className="bg-slate-900 rounded-3xl p-10 shadow-2xl">
                <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
                  <div className="flex items-center justify-between px-5 py-4 border-b border-slate-100">
                    <span className="font-bold text-slate-900">10-K Analysis</span>
                    <AnimatePresence mode="wait">
                      <motion.span
                        key={filingIndex}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="text-xs font-medium text-slate-400 bg-slate-100 px-2 py-1 rounded"
                      >
                        {filingData[filingIndex].ticker} FY2024
                      </motion.span>
                    </AnimatePresence>
                  </div>
                  <div className="p-5">
                    <AnimatePresence mode="wait">
                      <motion.div
                        key={filingIndex}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        className="space-y-3"
                      >
                        {filingData[filingIndex].metrics.map((metric, i) => (
                          <div key={i} className="flex items-center justify-between p-4 bg-slate-50 rounded-xl">
                            <span className="text-slate-600 font-medium">{metric.label}</span>
                            <div className="flex items-center gap-3">
                              <span className="font-bold text-slate-900 text-lg">{metric.value}</span>
                              <span className={`text-sm font-semibold px-2 py-0.5 rounded ${metric.change.startsWith('+') ? 'text-emerald-700 bg-emerald-100' : 'text-red-700 bg-red-100'}`}>
                                {metric.change}
                              </span>
                            </div>
                          </div>
                        ))}
                      </motion.div>
                    </AnimatePresence>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Feature 02 - AI Chat */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center mb-32"
          >
            <div className="order-2 lg:order-1 relative">
              <div className="bg-slate-900 rounded-3xl p-10 shadow-2xl">
                <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
                  <div className="px-5 py-4 border-b border-slate-100">
                    <span className="font-bold text-slate-900">Research Assistant</span>
                  </div>
                  <div className="p-5 min-h-[220px]">
                    <AnimatePresence mode="wait">
                      <motion.div
                        key={chatIndex}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="space-y-4"
                      >
                        <div className="flex justify-end">
                          <div className="bg-gradient-to-br from-[#0066cc] to-[#0052a3] text-white px-4 py-3 rounded-2xl rounded-br-md text-sm max-w-[85%] shadow-md">
                            {chatData[chatIndex].question}
                          </div>
                        </div>
                        <div className="flex justify-start">
                          <div className="bg-slate-100 text-slate-700 px-4 py-3 rounded-2xl rounded-bl-md text-sm max-w-[85%] leading-relaxed">
                            {chatData[chatIndex].answer}
                          </div>
                        </div>
                      </motion.div>
                    </AnimatePresence>
                  </div>
                </div>
              </div>
            </div>
            <div className="order-1 lg:order-2">
              <span className="inline-block px-3 py-1 bg-blue-100 text-blue-700 text-xs font-bold uppercase tracking-wider rounded-full mb-4">
                AI Research
              </span>
              <h3 className="text-3xl font-extrabold text-slate-900 mb-4 tracking-tight">
                Ask questions in plain English
              </h3>
              <p className="text-slate-500 leading-relaxed mb-6">
                No more digging through documents. Ask complex questions and get comprehensive answers backed by real data.
              </p>
              <ul className="space-y-3">
                {[
                  "Answers sourced from official filings",
                  "Citations for every claim",
                  "Follow-up questions supported"
                ].map((item, i) => (
                  <li key={i} className="flex items-center gap-3">
                    <div className="w-5 h-5 bg-emerald-100 rounded-full flex items-center justify-center">
                      <Check className="w-3 h-3 text-emerald-600" />
                    </div>
                    <span className="text-slate-700">{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          </motion.div>

          {/* Feature 03 - Earnings Transcripts */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center"
          >
            <div>
              <span className="inline-block px-3 py-1 bg-emerald-100 text-emerald-700 text-xs font-bold uppercase tracking-wider rounded-full mb-4">
                Earnings Calls
              </span>
              <h3 className="text-3xl font-extrabold text-slate-900 mb-4 tracking-tight">
                Search earnings transcripts
              </h3>
              <p className="text-slate-500 leading-relaxed mb-6">
                Find what management said about any topic. Guidance, competitive positioning, strategic priorities—searchable across all calls.
              </p>
              <ul className="space-y-3">
                {[
                  "Management commentary and guidance",
                  "Analyst Q&A insights",
                  "Historical transcript search"
                ].map((item, i) => (
                  <li key={i} className="flex items-center gap-3">
                    <div className="w-5 h-5 bg-emerald-100 rounded-full flex items-center justify-center">
                      <Check className="w-3 h-3 text-emerald-600" />
                    </div>
                    <span className="text-slate-700">{item}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div className="relative">
              <div className="bg-slate-900 rounded-3xl p-10 shadow-2xl">
                <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
                  <div className="flex items-center justify-between px-5 py-4 border-b border-slate-100">
                    <span className="font-bold text-slate-900">Earnings Call</span>
                    <AnimatePresence mode="wait">
                      <motion.span
                        key={transcriptIndex}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="text-xs font-medium text-slate-400 bg-slate-100 px-2 py-1 rounded"
                      >
                        {transcriptData[transcriptIndex].quarter}
                      </motion.span>
                    </AnimatePresence>
                  </div>
                  <div className="p-5">
                    <AnimatePresence mode="wait">
                      <motion.div
                        key={transcriptIndex}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                      >
                        <p className="font-bold text-slate-900 mb-4">{transcriptData[transcriptIndex].company}</p>
                        <div className="space-y-3">
                          {transcriptData[transcriptIndex].speakers.map((speaker, i) => (
                            <div key={i} className="p-4 bg-slate-50 rounded-xl">
                              <span className="inline-block px-2 py-0.5 bg-slate-200 text-slate-700 text-xs font-bold rounded mb-2">
                                {speaker.role}
                              </span>
                              <p className="text-sm text-slate-600 leading-relaxed">"{speaker.text}"</p>
                            </div>
                          ))}
                        </div>
                      </motion.div>
                    </AnimatePresence>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Why StrataLens Section */}
      <section id="why" className="py-28 bg-gradient-to-b from-slate-100 to-slate-50 relative">
        <div className="absolute inset-0 opacity-30" style={{
          backgroundImage: `radial-gradient(circle, #94a3b8 1px, transparent 1px)`,
          backgroundSize: '24px 24px'
        }} />
        <div className="max-w-5xl mx-auto px-6 relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <span className="inline-block px-4 py-1.5 bg-white text-slate-600 text-xs font-bold uppercase tracking-wider rounded-full mb-4 shadow-sm border border-slate-200">
              Comparison
            </span>
            <h2 className="text-4xl md:text-5xl font-extrabold text-slate-900 tracking-tight mb-4">
              Why StrataLens?
            </h2>
            <p className="text-lg text-slate-500">
              Built for serious equity research
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-8">
            {/* StrataLens Card */}
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="bg-white rounded-3xl p-8 border-2 border-slate-200 shadow-xl hover:border-[#0066cc]/50 hover:shadow-2xl transition-all"
            >
              <div className="flex items-center mb-8">
                <div className="w-14 h-14 bg-gradient-to-br from-[#0066cc] to-[#0052a3] rounded-2xl flex items-center justify-center mr-4 shadow-lg shadow-blue-500/25">
                  <Shield className="w-7 h-7 text-white" />
                </div>
                <div>
                  <h3 className="text-2xl font-extrabold text-slate-900">StrataLens</h3>
                  <p className="text-sm text-emerald-600 font-semibold">Primary Sources</p>
                </div>
              </div>
              <div className="space-y-5">
                {[
                  { title: "Official SEC Filings", desc: "10-K, 10-Q reports directly from the SEC" },
                  { title: "Earnings Transcripts", desc: "Word-for-word executive commentary" },
                  { title: "Primary Sources", desc: "Same documents institutional investors use" },
                  { title: "Verifiable & Accurate", desc: "Citation-backed, traceable insights" },
                ].map((item, i) => (
                  <div key={i} className="flex items-start gap-4">
                    <div className="w-6 h-6 bg-emerald-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                      <Check className="w-4 h-4 text-emerald-600" />
                    </div>
                    <div>
                      <p className="font-bold text-slate-900">{item.title}</p>
                      <p className="text-sm text-slate-500">{item.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>

            {/* LLM + Web Search Card */}
            <motion.div
              initial={{ opacity: 0, x: 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="bg-slate-100 rounded-3xl p-8 border-2 border-slate-300"
            >
              <div className="flex items-center mb-8">
                <div className="w-14 h-14 bg-slate-400 rounded-2xl flex items-center justify-center mr-4">
                  <Globe className="w-7 h-7 text-white" />
                </div>
                <div>
                  <h3 className="text-2xl font-extrabold text-slate-500">LLM + Web Search</h3>
                  <p className="text-sm text-slate-400 font-semibold">Secondary Sources</p>
                </div>
              </div>
              <div className="space-y-5">
                {[
                  { title: "News Articles", desc: "Second-hand interpretations and opinions" },
                  { title: "Web Content", desc: "Unverified, potentially outdated information" },
                  { title: "Secondary Sources", desc: "Filtered through journalists and bloggers" },
                  { title: "No Verification", desc: "Cannot trace back to original sources" },
                ].map((item, i) => (
                  <div key={i} className="flex items-start gap-4">
                    <div className="w-6 h-6 bg-slate-300 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                      <X className="w-4 h-4 text-slate-500" />
                    </div>
                    <div>
                      <p className="font-bold text-slate-500">{item.title}</p>
                      <p className="text-sm text-slate-400">{item.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-28 bg-gradient-to-br from-[#0052a3] via-[#0066cc] to-[#0083f1] relative overflow-hidden">
        <div className="absolute inset-0 opacity-10" style={{
          backgroundImage: `radial-gradient(circle, white 1px, transparent 1px)`,
          backgroundSize: '40px 40px'
        }} />
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="max-w-2xl mx-auto px-6 text-center relative z-10"
        >
          <h2 className="text-4xl md:text-5xl font-extrabold text-white mb-6 tracking-tight">
            Start researching in seconds
          </h2>
          <p className="text-lg text-blue-100 mb-10">
            No signup required. Just ask a question and get instant insights from SEC filings and earnings calls.
          </p>
          <button
            onClick={() => navigate('/chat')}
            className="inline-flex items-center gap-3 px-8 py-4 bg-white text-[#0066cc] text-base font-bold rounded-xl hover:bg-blue-50 transition-all shadow-xl hover:shadow-2xl hover:-translate-y-0.5"
          >
            Get Started Free
            <ArrowRight className="w-5 h-5" />
          </button>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-6 bg-slate-900">
        <div className="max-w-6xl mx-auto">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 bg-gradient-to-br from-[#0066cc] to-[#0052a3] rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/20">
                <StrataLensLogo size={14} className="text-white" />
              </div>
              <span className="text-base font-bold text-white">StrataLens</span>
            </div>
            <p className="text-sm text-slate-400">
              AI-powered equity research. Built for investors.
            </p>
            <a
              href="https://github.com/kamathhrishi/stratalens-ai"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-sm text-slate-400 hover:text-white transition-colors"
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
              </svg>
              <span>Open Source</span>
            </a>
          </div>
        </div>
      </footer>

      {/* About Modal */}
      <AboutModal isOpen={aboutOpen} onClose={() => setAboutOpen(false)} />
    </div>
  )
}
