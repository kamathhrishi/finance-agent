import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Link, useLocation } from 'react-router-dom'
import {
  MessageSquare,
  Building2,
  Filter,
  Layers,
  LineChart,
  Lock,
  ChevronLeft,
  ChevronRight,
  Menu,
  X,
  Info,
} from 'lucide-react'
import StrataLensLogo from './StrataLensLogo'
import AboutModal from './AboutModal'

interface SidebarItem {
  id: string
  label: string
  icon: React.ReactNode
  path: string
  authRequired: boolean
}

const sidebarItems: SidebarItem[] = [
  {
    id: 'chat',
    label: 'Chat',
    icon: <MessageSquare className="w-5 h-5" />,
    path: '/chat',
    authRequired: false,
  },
  {
    id: 'companies',
    label: 'Companies',
    icon: <Building2 className="w-5 h-5" />,
    path: '/companies',
    authRequired: true,
  },
  {
    id: 'screener',
    label: 'Screener',
    icon: <Filter className="w-5 h-5" />,
    path: '/screener',
    authRequired: true,
  },
  {
    id: 'collections',
    label: 'Collections',
    icon: <Layers className="w-5 h-5" />,
    path: '/collections',
    authRequired: true,
  },
  {
    id: 'charting',
    label: 'Charting',
    icon: <LineChart className="w-5 h-5" />,
    path: '/charting',
    authRequired: true,
  },
]

interface SidebarProps {
  isCollapsed?: boolean
  onToggle?: () => void
}

export default function Sidebar({ isCollapsed = false, onToggle }: SidebarProps) {
  const location = useLocation()
  const [mobileOpen, setMobileOpen] = useState(false)
  const [aboutOpen, setAboutOpen] = useState(false)

  const isActive = (path: string) => location.pathname === path

  const SidebarContent = () => (
    <div className="flex flex-col h-full">
      {/* Logo */}
      <div className={`flex items-center ${isCollapsed ? 'justify-center' : 'gap-2.5'} p-4 border-b border-slate-200/60`}>
        <Link to="/" className="flex items-center gap-2.5 group">
          <div className="w-9 h-9 bg-gradient-to-br from-[#0083f1] to-[#0070d8] rounded-xl flex items-center justify-center shadow-sm group-hover:shadow-md transition-shadow">
            <StrataLensLogo size={18} className="text-white" />
          </div>
          {!isCollapsed && (
            <span className="text-lg font-semibold text-slate-800 tracking-tight">StrataLens</span>
          )}
        </Link>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-3 space-y-1">
        {sidebarItems.map((item) => (
          <Link
            key={item.id}
            to={item.authRequired ? '#' : item.path}
            onClick={(e) => {
              if (item.authRequired) {
                e.preventDefault()
                // TODO: Show auth modal
              }
            }}
            className={`
              flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200 group relative
              ${isActive(item.path)
                ? 'bg-gradient-to-r from-[#0083f1]/10 to-[#0083f1]/5 text-[#0083f1] font-medium'
                : item.authRequired
                  ? 'text-slate-400 hover:bg-slate-100 cursor-not-allowed'
                  : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900'
              }
              ${isCollapsed ? 'justify-center' : ''}
            `}
          >
            <span className={`flex-shrink-0 ${isActive(item.path) ? 'text-[#0083f1]' : ''}`}>
              {item.icon}
            </span>
            {!isCollapsed && (
              <>
                <span className="flex-1">{item.label}</span>
                {item.authRequired && (
                  <Lock className="w-3.5 h-3.5 opacity-50" />
                )}
              </>
            )}

            {/* Tooltip for collapsed state */}
            {isCollapsed && (
              <div className="absolute left-full ml-2 px-2 py-1 bg-slate-900 text-white text-sm rounded-md opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all whitespace-nowrap z-50">
                {item.label}
                {item.authRequired && ' (Sign in required)'}
              </div>
            )}
          </Link>
        ))}
      </nav>

      {/* About button */}
      <div className="p-3 border-t border-slate-200/60">
        <button
          onClick={() => setAboutOpen(true)}
          className={`
            w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200 group relative
            text-slate-600 hover:bg-slate-100 hover:text-slate-900
            ${isCollapsed ? 'justify-center' : ''}
          `}
        >
          <Info className="w-5 h-5 flex-shrink-0" />
          {!isCollapsed && <span className="flex-1 text-left">About</span>}

          {/* Tooltip for collapsed state */}
          {isCollapsed && (
            <div className="absolute left-full ml-2 px-2 py-1 bg-slate-900 text-white text-sm rounded-md opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all whitespace-nowrap z-50">
              About
            </div>
          )}
        </button>
      </div>

      {/* Collapse toggle - desktop only */}
      {onToggle && (
        <div className="p-3 border-t border-slate-200/60 hidden lg:block">
          <button
            onClick={onToggle}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 text-slate-500 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-colors"
          >
            {isCollapsed ? (
              <ChevronRight className="w-4 h-4" />
            ) : (
              <>
                <ChevronLeft className="w-4 h-4" />
                <span className="text-sm">Collapse</span>
              </>
            )}
          </button>
        </div>
      )}
    </div>
  )

  return (
    <>
      {/* Mobile toggle button */}
      <button
        onClick={() => setMobileOpen(true)}
        className="lg:hidden fixed top-4 left-4 z-50 p-2 bg-white border border-slate-200 rounded-lg shadow-sm hover:bg-slate-50 transition-colors"
      >
        <Menu className="w-5 h-5 text-slate-600" />
      </button>

      {/* Mobile sidebar overlay */}
      <AnimatePresence>
        {mobileOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setMobileOpen(false)}
              className="lg:hidden fixed inset-0 bg-black/50 z-40"
            />
            <motion.aside
              initial={{ x: -280 }}
              animate={{ x: 0 }}
              exit={{ x: -280 }}
              transition={{ type: 'spring', damping: 25, stiffness: 300 }}
              className="lg:hidden fixed left-0 top-0 bottom-0 w-[280px] bg-white border-r border-slate-200 z-50 shadow-xl"
            >
              <button
                onClick={() => setMobileOpen(false)}
                className="absolute top-4 right-4 p-1.5 hover:bg-slate-100 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-slate-500" />
              </button>
              <SidebarContent />
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      {/* Desktop sidebar */}
      <motion.aside
        initial={false}
        animate={{ width: isCollapsed ? 72 : 240 }}
        transition={{ duration: 0.2 }}
        className="hidden lg:block fixed left-0 top-0 bottom-0 bg-white border-r border-slate-200/60 z-30"
      >
        <SidebarContent />
      </motion.aside>

      {/* About Modal */}
      <AboutModal isOpen={aboutOpen} onClose={() => setAboutOpen(false)} />
    </>
  )
}
