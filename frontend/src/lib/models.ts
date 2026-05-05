/**
 * Model catalogue for the chat selector UI.
 *
 * `id` is the value sent to the backend (the Pydantic schema accepts the bare
 * display id and the adapter maps it to a dated OpenAI model below).
 *
 * `disabled: true` greys the row in the dropdown so users see the future
 * lineup but can't pick anything beyond what's wired up server-side. To
 * enable a model, drop the `disabled` flag here AND make sure the backend
 * mapping in `agent/orchestrator_adapter.py::MODEL_ALIASES`
 * resolves it to a real OpenAI model id.
 */
// Display ids the UI knows about. Add disabled-future models back here AND
// in the catalogue below if you want them visible in the picker as
// "coming soon"; otherwise keep this lean.
export type ModelId = 'gpt-5.4-mini' | 'gpt-5.4-nano'

export interface ModelEntry {
  id: ModelId
  label: string
  description: string
  disabled?: boolean
}

export interface ModelGroup {
  heading: string
  blurb: string
  models: ModelEntry[]
}

/** Default selection when nothing is stored in localStorage. */
export const DEFAULT_MODEL_ID: ModelId = 'gpt-5.4-mini'

export const MODEL_GROUPS: ModelGroup[] = [
  {
    heading: 'Frontier models',
    blurb: "OpenAI's most advanced models, recommended for most tasks.",
    models: [
      {
        id: 'gpt-5.4-mini',
        label: 'GPT-5.4 mini',
        description: 'Strong mini model for research and subagents.',
      },
      {
        id: 'gpt-5.4-nano',
        label: 'GPT-5.4 nano',
        description: 'Cheapest GPT-5.4-class model for high-volume tasks.',
      },
    ],
  },
]

export function findModel(id: string | null | undefined): ModelEntry | undefined {
  if (!id) return undefined
  for (const g of MODEL_GROUPS) {
    const m = g.models.find((m) => m.id === id)
    if (m) return m
  }
  return undefined
}

const STORAGE_KEY = 'fs_research:selected_model:v1'

export function getStoredModel(): ModelId {
  if (typeof window === 'undefined') return DEFAULT_MODEL_ID
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return DEFAULT_MODEL_ID
    const m = findModel(raw)
    if (m && !m.disabled) return m.id
  } catch {
    /* ignore */
  }
  return DEFAULT_MODEL_ID
}

export function setStoredModel(id: ModelId): void {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(STORAGE_KEY, id)
  } catch {
    /* ignore */
  }
}
