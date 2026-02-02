import { motion } from 'framer-motion'
import type { ReactNode } from 'react'

interface FeatureCardProps {
  title: string
  description: string
  icon: ReactNode
  children: ReactNode
  delay?: number
}

export default function FeatureCard({
  title,
  description,
  icon,
  children,
  delay = 0,
}: FeatureCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-50px' }}
      transition={{ duration: 0.6, delay }}
      className="group"
    >
      <div className="relative bg-white rounded-2xl border border-slate-200 shadow-xl shadow-slate-900/5 overflow-hidden hover:shadow-2xl hover:shadow-slate-900/10 transition-all duration-500">
        {/* Mock UI container */}
        <div className="aspect-[4/3] bg-gradient-to-br from-slate-50 to-slate-100 p-4 border-b border-slate-200">
          <div className="w-full h-full rounded-xl overflow-hidden shadow-inner">
            {children}
          </div>
        </div>

        {/* Content */}
        <div className="p-6">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-xl bg-blue-100 flex items-center justify-center text-blue-600 group-hover:bg-blue-600 group-hover:text-white transition-colors duration-300">
              {icon}
            </div>
            <h3 className="text-lg font-semibold text-slate-900">{title}</h3>
          </div>
          <p className="text-slate-600 leading-relaxed">{description}</p>
        </div>
      </div>
    </motion.div>
  )
}
