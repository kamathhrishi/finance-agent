// =============================================================================
// ENVIRONMENT CONFIGURATION
// =============================================================================
// 
// To switch between environments, simply change the ENVIRONMENT variable below:
// - 'local' for development (localhost:8000)
// - 'production' for live server (apiurmarketcast.com)
//
// =============================================================================

const ENVIRONMENT = 'local'; // Change this to 'local' or 'production'

const ENVIRONMENTS = {
    local: {
        apiBaseUrl: 'http://localhost:8000',
        websocketUrl: 'ws://localhost:8000',
        description: 'Local Development Server'
    },
    production: {
        apiBaseUrl: 'https://web-production-835f4.up.railway.app',
        websocketUrl: 'wss://web-production-835f4.up.railway.app',
        description: 'Production Server (Railway)'
    }
};

// Clerk configuration - publishable key will be fetched from backend
// or can be set directly here for faster initialization
const CLERK_CONFIG = {
    // Clerk publishable key for authentication
    publishableKey: 'pk_test_aW1wcm92ZWQtYmFib29uLTk2LmNsZXJrLmFjY291bnRzLmRldiQ',
    signInUrl: '/sign-in',
    signUpUrl: '/sign-up',
    afterSignInUrl: '/',
    afterSignUpUrl: '/'
};

// Auth configuration - set to true to hide all auth UI
const AUTH_DISABLED = true;

// Export the configuration
window.STRATALENS_CONFIG = {
    environment: ENVIRONMENT,
    apiBaseUrl: ENVIRONMENTS[ENVIRONMENT].apiBaseUrl,
    websocketUrl: ENVIRONMENTS[ENVIRONMENT].websocketUrl,
    description: ENVIRONMENTS[ENVIRONMENT].description,
    pageSize: 20,
    toastDuration: 4000,
    chartDataLimit: 50,
    clerk: CLERK_CONFIG,
    authDisabled: AUTH_DISABLED
};

