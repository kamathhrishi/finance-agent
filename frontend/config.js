// =============================================================================
// ENVIRONMENT CONFIGURATION
// =============================================================================
// 
// To switch between environments, simply change the ENVIRONMENT variable below:
// - 'local' for development (localhost:8000)
// - 'production' for live server (apiurmarketcast.com)
//
// =============================================================================

const ENVIRONMENT = 'production'; // Change this to 'local' or 'production'

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

// Export the configuration
window.STRATALENS_CONFIG = {
    environment: ENVIRONMENT,
    apiBaseUrl: ENVIRONMENTS[ENVIRONMENT].apiBaseUrl,
    websocketUrl: ENVIRONMENTS[ENVIRONMENT].websocketUrl,
    description: ENVIRONMENTS[ENVIRONMENT].description,
    pageSize: 20,
    toastDuration: 4000,
    chartDataLimit: 50
};

