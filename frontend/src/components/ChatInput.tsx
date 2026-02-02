import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Send, Loader2 } from 'lucide-react'

interface ChatInputProps {
  onSubmit: (message: string) => void
  isLoading?: boolean
  placeholder?: string
  autoFocus?: boolean
  initialValue?: string
  size?: 'default' | 'large'
}

export default function ChatInput({
  onSubmit,
  isLoading = false,
  placeholder = 'Ask about any company...',
  autoFocus = false,
  initialValue = '',
  size = 'default',
}: ChatInputProps) {
  const [value, setValue] = useState(initialValue)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (autoFocus && inputRef.current) {
      inputRef.current.focus()
    }
  }, [autoFocus])

  useEffect(() => {
    if (initialValue) {
      setValue(initialValue)
    }
  }, [initialValue])

  const handleSubmit = () => {
    if (value.trim() && !isLoading) {
      onSubmit(value.trim())
      setValue('')
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  // Auto-resize textarea
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto'
      inputRef.current.style.height = `${Math.min(inputRef.current.scrollHeight, 150)}px`
    }
  }, [value])

  const isLarge = size === 'large'

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`relative flex items-end gap-3 bg-white border border-slate-200 rounded-2xl shadow-lg shadow-slate-900/5 hover:border-[#0083f1]/30 hover:shadow-[#0083f1]/5 transition-all ${
        isLarge ? 'p-4' : 'p-3'
      }`}
    >
      <textarea
        ref={inputRef}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        disabled={isLoading}
        rows={1}
        className={`flex-1 resize-none bg-transparent border-none outline-none placeholder:text-slate-400 text-slate-900 ${
          isLarge ? 'text-lg' : 'text-base'
        } scrollbar-thin`}
      />
      <button
        onClick={handleSubmit}
        disabled={!value.trim() || isLoading}
        className={`flex-shrink-0 flex items-center justify-center rounded-xl transition-all duration-200 ${
          isLarge ? 'w-12 h-12' : 'w-10 h-10'
        } ${
          value.trim() && !isLoading
            ? 'bg-gradient-to-br from-[#0083f1] to-[#0070d8] hover:from-[#0070d8] hover:to-[#005cb6] text-white shadow-lg shadow-[#0083f1]/25'
            : 'bg-slate-100 text-slate-400 cursor-not-allowed'
        }`}
      >
        {isLoading ? (
          <Loader2 className={`${isLarge ? 'w-5 h-5' : 'w-4 h-4'} animate-spin`} />
        ) : (
          <Send className={`${isLarge ? 'w-5 h-5' : 'w-4 h-4'}`} />
        )}
      </button>
    </motion.div>
  )
}
