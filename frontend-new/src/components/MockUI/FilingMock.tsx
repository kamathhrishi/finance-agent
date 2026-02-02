import { motion } from 'framer-motion'
import { useEffect, useState } from 'react'
import { FileText, TrendingUp } from 'lucide-react'

const filingData = [
  { label: 'Total Revenue', value: '$383.3B', change: '+2.1%', positive: true },
  { label: 'Net Income', value: '$93.7B', change: '-3.4%', positive: false },
  { label: 'Gross Margin', value: '46.2%', change: '+0.8%', positive: true },
  { label: 'R&D Expenses', value: '$29.9B', change: '+5.2%', positive: false },
]

export default function FilingMock() {
  const [highlightIndex, setHighlightIndex] = useState(-1)

  useEffect(() => {
    const interval = setInterval(() => {
      setHighlightIndex((i) => (i + 1) % (filingData.length + 1) - 1)
    }, 1500)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="w-full h-full bg-white rounded-xl p-4 flex flex-col">
      {/* Header */}
      <div className="flex items-center gap-2 mb-3 pb-2 border-b border-slate-100">
        <FileText className="w-4 h-4 text-blue-600" />
        <span className="text-sm font-semibold text-slate-900">AAPL 10-K 2024</span>
        <span className="ml-auto text-xs text-slate-400">Annual Report</span>
      </div>

      {/* Metrics grid */}
      <div className="flex-1 grid grid-cols-2 gap-2">
        {filingData.map((item, index) => (
          <motion.div
            key={index}
            animate={{
              scale: highlightIndex === index ? 1.02 : 1,
              borderColor: highlightIndex === index ? '#2563eb' : '#e2e8f0',
            }}
            transition={{ duration: 0.2 }}
            className="p-2.5 rounded-lg border bg-white"
          >
            <div className="text-xs text-slate-500 mb-1">{item.label}</div>
            <div className="flex items-baseline gap-2">
              <span className="text-base font-semibold text-slate-900">{item.value}</span>
              <span
                className={`text-xs font-medium ${
                  item.positive ? 'text-emerald-600' : 'text-red-500'
                }`}
              >
                {item.change}
              </span>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Bottom highlight bar */}
      <motion.div
        animate={{
          opacity: highlightIndex >= 0 ? 1 : 0.5,
        }}
        className="mt-3 flex items-center gap-2 p-2 bg-blue-50 rounded-lg border border-blue-100"
      >
        <TrendingUp className="w-4 h-4 text-blue-600" />
        <span className="text-xs text-blue-700">
          {highlightIndex >= 0
            ? `Analyzing ${filingData[highlightIndex]?.label}...`
            : 'AI extracting key metrics'}
        </span>
      </motion.div>
    </div>
  )
}
