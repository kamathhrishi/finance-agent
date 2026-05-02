import { motion, AnimatePresence } from 'framer-motion'
import { X, Search, FileText } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'

interface CoverageModalProps {
  isOpen: boolean
  onClose: () => void
}

// Tech universe — every ticker we currently have full SEC filing coverage for
// (10-K, 10-Q, 8-K + substantive exhibits, last 5 years). Generated from
// fs_research_agent/tech_universe.json. Names are SEC's display titles,
// lightly cleaned.
const COVERAGE: { t: string; n: string }[] = [
  { t: "AAPL", n: "Apple Inc." }, { t: "ABNB", n: "Airbnb, Inc." }, { t: "ADBE", n: "Adobe Inc." },
  { t: "ADI", n: "Analog Devices Inc." }, { t: "ADSK", n: "Autodesk, Inc." }, { t: "AFRM", n: "Affirm Holdings, Inc." },
  { t: "AKAM", n: "Akamai Technologies Inc." }, { t: "ALAB", n: "Astera Labs, Inc." }, { t: "AMAT", n: "Applied Materials Inc." },
  { t: "AMD", n: "Advanced Micro Devices Inc." }, { t: "AMKR", n: "Amkor Technology, Inc." }, { t: "APH", n: "Amphenol Corp" },
  { t: "APP", n: "AppLovin Corp" }, { t: "AVGO", n: "Broadcom Inc." }, { t: "BR", n: "Broadridge Financial Solutions, Inc." },
  { t: "BSY", n: "Bentley Systems Inc." }, { t: "CACI", n: "CACI International Inc." }, { t: "CDNS", n: "Cadence Design Systems Inc." },
  { t: "CDW", n: "CDW Corp" }, { t: "CHTR", n: "Charter Communications, Inc." }, { t: "CIEN", n: "Ciena Corp" },
  { t: "CMCSA", n: "Comcast Corp" }, { t: "COHR", n: "Coherent Corp." }, { t: "COIN", n: "Coinbase Global, Inc." },
  { t: "CPAY", n: "Corpay, Inc." }, { t: "CRDO", n: "Credo Technology Group" }, { t: "CRM", n: "Salesforce, Inc." },
  { t: "CRWD", n: "CrowdStrike Holdings, Inc." }, { t: "CRWV", n: "CoreWeave, Inc." }, { t: "CSCO", n: "Cisco Systems, Inc." },
  { t: "CTSH", n: "Cognizant Technology Solutions" }, { t: "DASH", n: "DoorDash, Inc." }, { t: "DDOG", n: "Datadog, Inc." },
  { t: "DELL", n: "Dell Technologies Inc." }, { t: "DIS", n: "Walt Disney Co" }, { t: "DT", n: "Dynatrace, Inc." },
  { t: "EA", n: "Electronic Arts Inc." }, { t: "ENTG", n: "Entegris Inc." }, { t: "FICO", n: "Fair Isaac Corp" },
  { t: "FIS", n: "Fidelity National Information Services" }, { t: "FOX", n: "Fox Corp" }, { t: "FOXA", n: "Fox Corp (Class A)" },
  { t: "FSLR", n: "First Solar, Inc." }, { t: "FTNT", n: "Fortinet, Inc." }, { t: "FTV", n: "Fortive Corp" },
  { t: "GDDY", n: "GoDaddy Inc." }, { t: "GEHC", n: "GE Healthcare Technologies Inc." }, { t: "GEN", n: "Gen Digital Inc." },
  { t: "GFS", n: "GlobalFoundries Inc." }, { t: "GLW", n: "Corning Inc." }, { t: "GOOG", n: "Alphabet Inc. (Class C)" },
  { t: "GWRE", n: "Guidewire Software, Inc." }, { t: "HPE", n: "Hewlett Packard Enterprise Co" }, { t: "HPQ", n: "HP Inc." },
  { t: "HUBS", n: "HubSpot Inc." }, { t: "IBM", n: "IBM Corp" }, { t: "INTC", n: "Intel Corp" },
  { t: "INTU", n: "Intuit Inc." }, { t: "IT", n: "Gartner Inc." }, { t: "JBL", n: "Jabil Inc." },
  { t: "JKHY", n: "Jack Henry & Associates Inc." }, { t: "KEYS", n: "Keysight Technologies, Inc." }, { t: "KLAC", n: "KLA Corp" },
  { t: "LBRDB", n: "Liberty Broadband Corp" }, { t: "LDOS", n: "Leidos Holdings, Inc." }, { t: "LITE", n: "Lumentum Holdings Inc." },
  { t: "LRCX", n: "Lam Research Corp" }, { t: "LSCC", n: "Lattice Semiconductor Corp" }, { t: "LYV", n: "Live Nation Entertainment, Inc." },
  { t: "MCHP", n: "Microchip Technology Inc." }, { t: "MDB", n: "MongoDB, Inc." }, { t: "META", n: "Meta Platforms, Inc." },
  { t: "MKSI", n: "MKS Inc." }, { t: "MPWR", n: "Monolithic Power Systems Inc." }, { t: "MRVL", n: "Marvell Technology, Inc." },
  { t: "MSFT", n: "Microsoft Corp" }, { t: "MSI", n: "Motorola Solutions, Inc." }, { t: "MSTR", n: "Strategy (formerly MicroStrategy) Inc." },
  { t: "MTSI", n: "MACOM Technology Solutions" }, { t: "MU", n: "Micron Technology Inc." }, { t: "NET", n: "Cloudflare, Inc." },
  { t: "NOW", n: "ServiceNow, Inc." }, { t: "NTAP", n: "NetApp, Inc." }, { t: "NTNX", n: "Nutanix, Inc." },
  { t: "NVDA", n: "NVIDIA Corp" }, { t: "NWS", n: "News Corp" }, { t: "NWSA", n: "News Corp (Class A)" },
  { t: "NXT", n: "Nextracker Inc." }, { t: "NYT", n: "New York Times Co" }, { t: "OKTA", n: "Okta, Inc." },
  { t: "OMC", n: "Omnicom Group Inc." }, { t: "ON", n: "ON Semiconductor Corp" }, { t: "ONTO", n: "Onto Innovation Inc." },
  { t: "ORCL", n: "Oracle Corp" }, { t: "PINS", n: "Pinterest, Inc." }, { t: "PLTR", n: "Palantir Technologies Inc." },
  { t: "PSKY", n: "Paramount Skydance Corp" }, { t: "PSTG", n: "Pure Storage, Inc." }, { t: "PTC", n: "PTC Inc." },
  { t: "QCOM", n: "Qualcomm Inc." }, { t: "QXO", n: "QXO, Inc." }, { t: "RBLX", n: "Roblox Corp" },
  { t: "RMBS", n: "Rambus Inc." }, { t: "ROKU", n: "Roku, Inc." }, { t: "SATS", n: "EchoStar Corp" },
  { t: "SITM", n: "SiTime Corp" }, { t: "SMCI", n: "Super Micro Computer, Inc." }, { t: "SNDK", n: "SanDisk Corp" },
  { t: "SNOW", n: "Snowflake Inc." }, { t: "SNPS", n: "Synopsys Inc." }, { t: "SNX", n: "TD SYNNEX Corp" },
  { t: "T", n: "AT&T Inc." }, { t: "TDY", n: "Teledyne Technologies Inc." }, { t: "TER", n: "Teradyne, Inc." },
  { t: "TMUS", n: "T-Mobile US, Inc." }, { t: "TRMB", n: "Trimble Inc." }, { t: "TTD", n: "The Trade Desk, Inc." },
  { t: "TWLO", n: "Twilio Inc." }, { t: "TXN", n: "Texas Instruments Inc." }, { t: "TYL", n: "Tyler Technologies Inc." },
  { t: "UBER", n: "Uber Technologies, Inc." }, { t: "UI", n: "Ubiquiti Inc." }, { t: "VG", n: "Venture Global, Inc." },
  { t: "VRSN", n: "VeriSign Inc." }, { t: "VZ", n: "Verizon Communications Inc." }, { t: "WBD", n: "Warner Bros. Discovery, Inc." },
  { t: "WDAY", n: "Workday, Inc." }, { t: "WDC", n: "Western Digital Corp" }, { t: "WMG", n: "Warner Music Group Corp." },
  { t: "XYZ", n: "Block, Inc. (formerly Square)" }, { t: "Z", n: "Zillow Group, Inc." }, { t: "ZBRA", n: "Zebra Technologies Corp" },
  { t: "ZG", n: "Zillow Group, Inc. (Class A)" }, { t: "ZM", n: "Zoom Communications, Inc." }, { t: "ZS", n: "Zscaler, Inc." },
]

export default function CoverageModal({ isOpen, onClose }: CoverageModalProps) {
  const [query, setQuery] = useState('')

  useEffect(() => {
    const onEscape = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    if (isOpen) {
      document.addEventListener('keydown', onEscape)
      document.body.style.overflow = 'hidden'
      setQuery('')
    }
    return () => {
      document.removeEventListener('keydown', onEscape)
      document.body.style.overflow = ''
    }
  }, [isOpen, onClose])

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return COVERAGE
    return COVERAGE.filter(c => c.t.toLowerCase().includes(q) || c.n.toLowerCase().includes(q))
  }, [query])

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
            className="relative w-full max-w-3xl max-h-[85vh] bg-white rounded-2xl shadow-2xl flex flex-col overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 bg-gradient-to-r from-slate-50 to-white shrink-0">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-[#0a1628] rounded-xl flex items-center justify-center shadow-sm">
                  <FileText className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h2 className="text-lg font-semibold text-slate-900">Coverage</h2>
                  <p className="text-xs text-slate-500 mt-0.5">
                    {COVERAGE.length} tech companies — full 10-K, 10-Q, 8-K SEC filings
                  </p>
                </div>
              </div>
              <button
                onClick={onClose}
                className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-slate-100 text-slate-500 hover:text-slate-900 transition-colors"
                aria-label="Close coverage"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Coverage stats strip */}
            <div className="grid grid-cols-3 gap-px bg-slate-200 border-b border-slate-200 shrink-0">
              <div className="bg-white px-5 py-3">
                <div className="text-2xl font-semibold text-[#0a1628]">{COVERAGE.length}</div>
                <div className="text-xs text-slate-500 mt-0.5">Tech Companies</div>
              </div>
              <div className="bg-white px-5 py-3">
                <div className="text-2xl font-semibold text-[#0a1628]">3+</div>
                <div className="text-xs text-slate-500 mt-0.5">Years Coverage</div>
              </div>
              <div className="bg-white px-5 py-3">
                <div className="text-2xl font-semibold text-[#0a1628]">10-K · 10-Q · 8-K</div>
                <div className="text-xs text-slate-500 mt-0.5">Filing Types</div>
              </div>
            </div>

            {/* Search */}
            <div className="px-6 py-3 border-b border-slate-200 shrink-0">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
                <input
                  type="text"
                  autoFocus
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Filter by ticker or company name..."
                  className="w-full pl-9 pr-3 py-2 text-sm bg-slate-50 border border-slate-200 rounded-lg placeholder:text-slate-400 focus:outline-none focus:border-[#0a1628] focus:ring-1 focus:ring-[#0a1628]"
                />
              </div>
              {query && (
                <p className="text-xs text-slate-400 mt-2">
                  {filtered.length} of {COVERAGE.length} match
                </p>
              )}
            </div>

            {/* Ticker grid */}
            <div className="flex-1 overflow-y-auto px-6 py-4">
              {filtered.length === 0 ? (
                <p className="text-sm text-slate-500 text-center py-12">
                  No matches. We currently cover {COVERAGE.length} tech companies — request additions via the chat.
                </p>
              ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
                  {filtered.map((c) => (
                    <div
                      key={c.t}
                      className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-slate-50 transition-colors"
                    >
                      <span className="text-xs font-mono font-semibold text-[#0a1628] bg-slate-100 px-2 py-0.5 rounded min-w-[3.25rem] text-center">
                        {c.t}
                      </span>
                      <span className="text-sm text-slate-700 truncate">{c.n}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Footer */}
            <div className="px-6 py-3 border-t border-slate-200 bg-slate-50 text-xs text-slate-500 flex items-center justify-between shrink-0">
              <span>Updated continuously from SEC EDGAR.</span>
              <span className="text-slate-400">Don't see your company? Ask in chat.</span>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
