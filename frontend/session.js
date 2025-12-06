// =============================================================================
// SESSION MANAGEMENT FOR ANONYMOUS USERS
// =============================================================================
//
// This module handles session ID generation and storage for anonymous users.
// Session IDs enable:
// - Conversation memory across messages (follow-up questions work)
// - Rate limiting per session (not per IP - better for office networks)
// - Multiple tabs can have separate conversations (if using sessionStorage)
//
// =============================================================================

/**
 * Generate a unique session ID
 * Format: session_<timestamp>_<random>
 * Example: session_1696723456789_a8f3c2
 */
function generateSessionId() {
    const timestamp = Date.now();
    const random = Math.random().toString(36).substring(2, 8);
    return `session_${timestamp}_${random}`;
}

/**
 * Get or create session ID for anonymous users
 * 
 * Storage Strategy:
 * - Uses sessionStorage by default (separate session per tab/window)
 * - Each browser tab/window gets its own conversation
 * - Session persists across page reloads within same tab
 * - Session cleared when tab/window is closed
 * - Perfect for: "Ask question, close browser, open again = new conversation"
 * 
 * @returns {string} Session ID
 */
function getOrCreateSessionId() {
    const STORAGE_KEY = 'stratalens_session_id';
    
    // Try to get existing session ID from sessionStorage (tab-specific)
    let sessionId = sessionStorage.getItem(STORAGE_KEY);
    
    if (!sessionId) {
        // Generate new session ID
        sessionId = generateSessionId();
        sessionStorage.setItem(STORAGE_KEY, sessionId);
    } else {
    }
    
    return sessionId;
}

/**
 * Get or create persistent session ID
 * 
 * Alternative storage strategy using localStorage:
 * - Session persists across all tabs and browser sessions
 * - Same session ID even if you close and reopen browser
 * - Good for: Continuing same conversation across browser restarts
 * - Use this if you want continuity across browser sessions
 * 
 * @returns {string} Persistent session ID
 */
function getOrCreatePersistentSessionId() {
    const STORAGE_KEY = 'stratalens_persistent_session_id';
    
    // Try to get existing persistent session ID
    let sessionId = localStorage.getItem(STORAGE_KEY);
    
    if (!sessionId) {
        // Generate new session ID
        sessionId = generateSessionId();
        localStorage.setItem(STORAGE_KEY, sessionId);
    } else {
    }
    
    return sessionId;
}

/**
 * Clear session ID (useful for testing or "reset conversation")
 */
function clearSessionId() {
    sessionStorage.removeItem('stratalens_session_id');
    localStorage.removeItem('stratalens_persistent_session_id');
}

/**
 * Get session info for debugging
 */
function getSessionInfo() {
    const currentId = sessionStorage.getItem('stratalens_session_id');
    const persistentId = localStorage.getItem('stratalens_persistent_session_id');
    
    return {
        current: currentId || 'none',
        persistent: persistentId || 'none',
        active: getOrCreateSessionId()
    };
}

// Export functions
window.SessionManager = {
    getOrCreateSessionId,           // Default: tab/window-specific (sessionStorage)
    getOrCreatePersistentSessionId, // Alternative: persistent across browser sessions (localStorage)
    clearSessionId,
    getSessionInfo,
    generateSessionId
};

// Auto-initialize on load
document.addEventListener('DOMContentLoaded', () => {
    const sessionId = getOrCreateSessionId();
});

