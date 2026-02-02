import { motion } from 'framer-motion'
import { useEffect, useState } from 'react'
import { Bot, User } from 'lucide-react'

const conversation = [
  { role: 'user', text: "What is Apple's revenue for 2024?" },
  {
    role: 'assistant',
    text: "Apple's total revenue for FY2024 was $383.3 billion, representing a 2% year-over-year increase...",
  },
]

export default function ChatMock() {
  const [visibleMessages, setVisibleMessages] = useState(0)
  const [typingText, setTypingText] = useState('')
  const [isTyping, setIsTyping] = useState(false)

  useEffect(() => {
    const showNextMessage = () => {
      if (visibleMessages < conversation.length) {
        setVisibleMessages((v) => v + 1)
      }
    }

    const timer = setTimeout(showNextMessage, visibleMessages === 0 ? 500 : 2000)
    return () => clearTimeout(timer)
  }, [visibleMessages])

  useEffect(() => {
    if (visibleMessages === 2) {
      setIsTyping(true)
      const text = conversation[1].text
      let index = 0

      const typeInterval = setInterval(() => {
        if (index <= text.length) {
          setTypingText(text.slice(0, index))
          index++
        } else {
          clearInterval(typeInterval)
          setIsTyping(false)
        }
      }, 30)

      return () => clearInterval(typeInterval)
    }
  }, [visibleMessages])

  // Reset animation loop
  useEffect(() => {
    if (visibleMessages >= conversation.length && !isTyping) {
      const resetTimer = setTimeout(() => {
        setVisibleMessages(0)
        setTypingText('')
      }, 4000)
      return () => clearTimeout(resetTimer)
    }
  }, [visibleMessages, isTyping])

  return (
    <div className="w-full h-full bg-white rounded-xl p-4 flex flex-col">
      <div className="flex-1 space-y-3 overflow-hidden">
        {/* User message */}
        {visibleMessages >= 1 && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex justify-end gap-2"
          >
            <div className="bg-slate-100 rounded-2xl rounded-tr-md px-3 py-2 text-sm text-slate-900 max-w-[85%]">
              {conversation[0].text}
            </div>
            <div className="w-7 h-7 rounded-lg bg-slate-200 flex items-center justify-center flex-shrink-0">
              <User className="w-4 h-4 text-slate-600" />
            </div>
          </motion.div>
        )}

        {/* Assistant message */}
        {visibleMessages >= 2 && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex gap-2"
          >
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-blue-600 to-blue-700 flex items-center justify-center flex-shrink-0">
              <Bot className="w-4 h-4 text-white" />
            </div>
            <div className="bg-white border border-slate-200 rounded-2xl rounded-tl-md px-3 py-2 text-sm text-slate-700 max-w-[85%]">
              {typingText}
              {isTyping && <span className="animate-pulse">|</span>}
            </div>
          </motion.div>
        )}
      </div>

      {/* Input mock */}
      <div className="mt-3 flex items-center gap-2 p-2 bg-slate-50 rounded-xl border border-slate-200">
        <div className="flex-1 text-sm text-slate-400">Ask a follow-up...</div>
        <div className="w-7 h-7 rounded-lg bg-slate-200 flex items-center justify-center">
          <svg className="w-4 h-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
          </svg>
        </div>
      </div>
    </div>
  )
}
