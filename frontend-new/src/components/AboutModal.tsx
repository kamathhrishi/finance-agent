import { motion, AnimatePresence } from 'framer-motion'
import { X, ExternalLink } from 'lucide-react'
import { useEffect } from 'react'
import StrataLensLogo from './StrataLensLogo'

interface AboutModalProps {
  isOpen: boolean
  onClose: () => void
}

export default function AboutModal({ isOpen, onClose }: AboutModalProps) {
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
            className="relative w-full max-w-md bg-white rounded-2xl shadow-2xl overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 bg-gradient-to-r from-slate-50 to-white">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-gradient-to-br from-[#0083f1] to-[#0070d8] rounded-xl flex items-center justify-center shadow-sm">
                  <StrataLensLogo size={20} className="text-white" />
                </div>
                <h2 className="text-xl font-bold text-slate-900">About StrataLens</h2>
              </div>
              <button
                onClick={onClose}
                className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Content */}
            <div className="px-6 py-5">
              <p className="text-slate-600 leading-relaxed mb-6">
                AI equity research copilot providing quick insights from U.S. public markets data.
              </p>

              <div className="mb-6">
                <p className="text-sm text-slate-500 mb-1">Created by</p>
                <p className="text-lg font-semibold text-slate-900">Hrishikesh Kamath</p>
              </div>

              <a
                href="https://kamathhrishi.github.io"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 text-[#0083f1] hover:text-[#0070d8] font-medium transition-colors mb-6"
              >
                Visit kamathhrishi.github.io
                <ExternalLink className="w-4 h-4" />
              </a>

              <div className="p-4 bg-slate-50 rounded-xl">
                <p className="text-sm text-slate-600 leading-relaxed">
                  I spent several months building StrataLens as a product, learning extensively about the financial industry and working with analysts. While I've decided to open source it rather than continue commercial development, it was an incredibly valuable journey. Feel free to use and contribute!
                </p>
              </div>
            </div>

            {/* Footer */}
            <div className="px-6 py-4 border-t border-slate-200 bg-slate-50">
              <button
                onClick={onClose}
                className="w-full py-2.5 bg-slate-900 text-white rounded-lg hover:bg-slate-800 transition-colors font-medium"
              >
                Close
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
