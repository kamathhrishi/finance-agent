import { useAuth } from '@clerk/clerk-react'

const AUTH_DISABLED = import.meta.env.VITE_AUTH_DISABLED === 'true'

export function useAuthStatus() {
  const { isSignedIn, getToken } = useAuth()
  const authEnabled = !AUTH_DISABLED
  const canAccess = AUTH_DISABLED || isSignedIn

  const getOptionalToken = async () => {
    if (!authEnabled || !isSignedIn) return null
    return getToken()
  }

  return {
    authEnabled,
    canAccess,
    isSignedIn,
    getOptionalToken,
  }
}
