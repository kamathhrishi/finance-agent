import { motion } from 'framer-motion'
import { useEffect, useState } from 'react'
import { Mic, Play, Users } from 'lucide-react'

const transcriptLines = [
  { speaker: 'CEO', text: 'We achieved record Services revenue this quarter...' },
  { speaker: 'CFO', text: 'Our gross margin expanded by 80 basis points...' },
  { speaker: 'Analyst', text: 'Can you discuss the iPhone demand trends?' },
  { speaker: 'CEO', text: 'We saw strong demand across all geographies...' },
]

export default function TranscriptMock() {
  const [visibleLines, setVisibleLines] = useState(1)
  const [currentHighlight, setCurrentHighlight] = useState(0)

  useEffect(() => {
    const lineInterval = setInterval(() => {
      setVisibleLines((v) => {
        if (v >= transcriptLines.length) {
          return 1 // Reset
        }
        return v + 1
      })
    }, 2500)

    return () => clearInterval(lineInterval)
  }, [])

  useEffect(() => {
    const highlightInterval = setInterval(() => {
      setCurrentHighlight((h) => (h + 1) % visibleLines)
    }, 1200)

    return () => clearInterval(highlightInterval)
  }, [visibleLines])

  const getSpeakerColor = (speaker: string) => {
    switch (speaker) {
      case 'CEO':
        return 'bg-blue-100 text-blue-700'
      case 'CFO':
        return 'bg-emerald-100 text-emerald-700'
      case 'Analyst':
        return 'bg-amber-100 text-amber-700'
      default:
        return 'bg-slate-100 text-slate-700'
    }
  }

  return (
    <div className="w-full h-full bg-white rounded-xl p-4 flex flex-col">
      {/* Header */}
      <div className="flex items-center gap-2 mb-3 pb-2 border-b border-slate-100">
        <Mic className="w-4 h-4 text-blue-600" />
        <span className="text-sm font-semibold text-slate-900">Q4 2024 Earnings Call</span>
        <div className="ml-auto flex items-center gap-1 text-xs text-slate-400">
          <Users className="w-3 h-3" />
          <span>AAPL</span>
        </div>
      </div>

      {/* Transcript lines */}
      <div className="flex-1 space-y-2 overflow-hidden">
        {transcriptLines.slice(0, visibleLines).map((line, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, x: -10 }}
            animate={{
              opacity: 1,
              x: 0,
              backgroundColor: currentHighlight === index ? '#eff6ff' : '#ffffff',
            }}
            transition={{ duration: 0.3 }}
            className="flex gap-2 p-2 rounded-lg"
          >
            <span
              className={`flex-shrink-0 px-1.5 py-0.5 rounded text-xs font-medium ${getSpeakerColor(
                line.speaker
              )}`}
            >
              {line.speaker}
            </span>
            <span className="text-xs text-slate-600 line-clamp-2">{line.text}</span>
          </motion.div>
        ))}
      </div>

      {/* Play indicator */}
      <div className="mt-2 flex items-center gap-2">
        <div className="w-6 h-6 rounded-full bg-blue-600 flex items-center justify-center">
          <Play className="w-3 h-3 text-white ml-0.5" />
        </div>
        <div className="flex-1 h-1 bg-slate-100 rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-blue-600 rounded-full"
            animate={{ width: `${(visibleLines / transcriptLines.length) * 100}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
        <span className="text-xs text-slate-400">1:24:30</span>
      </div>
    </div>
  )
}
