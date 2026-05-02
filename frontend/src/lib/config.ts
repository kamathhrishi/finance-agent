const AUTH_DISABLED = import.meta.env.VITE_AUTH_DISABLED === 'true'

export const config = {
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL || '',
  authEnabled: !AUTH_DISABLED,
  appName: 'StrataLens',
  appDescription: 'AI-Powered Equity Research',
  githubUrl: 'https://github.com/KamathHrishi/stratalens',
}
