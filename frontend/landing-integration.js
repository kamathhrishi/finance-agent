/**
 * Landing Page Integration Script
 * Handles the transition between the landing page and the full platform
 */

// Global state for landing page integration
window.LANDING_STATE = {
    hasUsedFreeMessage: false,
    isTransitioned: false,
    initialQuery: null
};

/**
 * Initialize landing page integration
 */
function initLandingIntegration() {
    
    // First priority: Check localStorage for landing state
    const storedState = localStorage.getItem('LANDING_STATE');
    if (storedState) {
        try {
            const landingState = JSON.parse(storedState);
            // Check if it's recent (within last 5 seconds)
            if (landingState.timestamp && (Date.now() - landingState.timestamp) < 5000) {
                window.LANDING_STATE = {
                    fromLanding: true,
                    isTransitioned: true,
                    initialQuery: landingState.initialQuery,
                    selectedMode: landingState.selectedMode
                };
                
                // Restore the selected mode FIRST (before query execution)
                if (landingState.selectedMode) {
                    restoreLandingMode(landingState.selectedMode);
                }
                
                // Then handle the initial query
                if (landingState.initialQuery) {
                    insertInitialQuery();
                }
                
                // Clear the stored state after use
                localStorage.removeItem('LANDING_STATE');
                
                // Apply landing restrictions if not authenticated
                if (!isUserAuthenticated()) {
                    applyLandingRestrictions();
                }
                
                return; // Exit early since we found the state
            }
        } catch (e) {
            localStorage.removeItem('LANDING_STATE');
        }
    }
    
    // Second priority: Check if we have preserved landing state from app.js
    if (window.LANDING_STATE && window.LANDING_STATE.fromLanding) {
        window.LANDING_STATE.isTransitioned = true;
        
        // Restore the selected mode if available
        if (window.LANDING_STATE.selectedMode) {
            restoreLandingMode(window.LANDING_STATE.selectedMode);
        }
        
        if (window.LANDING_STATE.initialQuery) {
            
            // Insert and auto-send the query
            insertInitialQuery();
        }
        
        // Apply landing restrictions if not authenticated
        if (!isUserAuthenticated()) {
            applyLandingRestrictions();
        }
    } else {
        // Fallback: Check URL parameters (in case app.js hasn't run yet)
        const urlParams = new URLSearchParams(window.location.search);
        const fromLanding = urlParams.get('from') === 'landing';
        const query = urlParams.get('query');
        const mode = urlParams.get('mode');
        
        if (fromLanding) {
            window.LANDING_STATE = window.LANDING_STATE || {};
            window.LANDING_STATE.fromLanding = true;
            window.LANDING_STATE.isTransitioned = true;
            
            // Restore the selected mode
            if (mode) {
                window.LANDING_STATE.selectedMode = mode;
                restoreLandingMode(mode);
            }
            
            if (query) {
                window.LANDING_STATE.initialQuery = decodeURIComponent(query);
                
                // Insert and auto-send the query
                insertInitialQuery();
            }
            
            // Apply landing restrictions if not authenticated
            if (!isUserAuthenticated()) {
                applyLandingRestrictions();
            }
        } else {
        }
    }
}

/**
 * Restore the mode selection from landing page
 */
function restoreLandingMode(mode) {
    
    // Function to attempt mode restoration
    const attemptRestore = () => {
        const modeItems = document.querySelectorAll('#modeDropdown .mode-item');
        
        if (modeItems.length === 0) {
            return false;
        }
        
        let restored = false;
        modeItems.forEach(item => {
            const itemMode = item.getAttribute('data-mode');
            if (itemMode === mode) {
                // Activate this mode
                modeItems.forEach(i => i.classList.remove('active'));
                item.classList.add('active');
                
                // Update the button display
                const toggleBtn = document.getElementById('modeToggleBtn');
                if (toggleBtn) {
                    const icon = mode === 'agent' ? 'fa-magnifying-glass-chart' : 'fa-bolt';
                    const text = mode === 'agent' ? 'Agent' : 'Ask';
                    
                    const iconEl = toggleBtn.querySelector('i:first-child');
                    if (iconEl) iconEl.className = `fas ${icon}`;
                    
                    const currentMode = document.getElementById('currentMode');
                    if (currentMode) currentMode.textContent = text;
                }
                
                restored = true;
            }
        });
        
        return restored;
    };
    
    // Try immediately first
    if (attemptRestore()) {
        return;
    }
    
    // Retry with increasing delays if needed
    let retries = 0;
    const maxRetries = 10;
    const retryInterval = setInterval(() => {
        retries++;
        
        if (attemptRestore()) {
            clearInterval(retryInterval);
        } else if (retries >= maxRetries) {
            clearInterval(retryInterval);
        }
    }, 200); // Check every 200ms
}

/**
 * Check if user is authenticated
 */
function isUserAuthenticated() {
    const token = localStorage.getItem('authToken');
    const user = localStorage.getItem('currentUser');
    return token && user; // Real authentication check
}

/**
 * Apply restrictions for users coming from landing page
 */
function applyLandingRestrictions() {
    
    // Don't lock features aggressively, just add subtle hints to sign up
    // Users can explore the platform freely
    
    // Set chat as default section using the existing app's function
    setTimeout(() => {
        if (typeof switchToSection === 'function') {
            switchToSection('chat');
        } else if (typeof window.switchToSection === 'function') {
            window.switchToSection('chat');
        }
        
        if (window.LANDING_STATE.initialQuery) {
            insertInitialQuery();
        }
    }, 500);
    
    // Add a subtle banner encouraging sign-up instead of locking features
    addLandingBanner();
}

/**
 * Banner functionality removed - no more promotional banners
 */
function addLandingBanner() {
    // Banner functionality removed
    return;
}

/**
 * Show upgrade modal for restricted features
 */
function showUpgradeModal(featureName) {
    // Use existing auth modal if available
    if (typeof showAuthModal === 'function') {
        showAuthModal('register');
        return;
    }
    
    // Fallback: create simple modal
    const modal = document.createElement('div');
    modal.className = 'fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4';
    modal.innerHTML = `
        <div class="bg-white rounded-2xl p-8 max-w-md w-full shadow-2xl text-center">
            <div class="w-16 h-16 bg-orange-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <i class="fas fa-lock text-orange-600 text-2xl"></i>
            </div>
            <h3 class="text-xl font-bold text-gray-800 mb-2">${featureName} is Locked</h3>
            <p class="text-gray-600 mb-6">Sign up to unlock ${featureName} and all platform features. It's free!</p>
            <div class="space-y-3">
                <button onclick="window.location.href='/'" class="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 rounded-lg transition-colors">
                    Sign Up Free
                </button>
                <button onclick="this.closest('div[class*=\\'fixed\\']').remove()" class="w-full text-gray-500 hover:text-gray-700 font-medium py-2">
                    Maybe Later
                </button>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
}

/**
 * Insert the initial query from landing page into chat
 */
function insertInitialQuery() {
    const chatInput = document.getElementById('chatInput');
    if (chatInput && window.LANDING_STATE.initialQuery) {
        
        chatInput.value = window.LANDING_STATE.initialQuery;
        chatInput.focus();
        
        // Function to check if mode is properly set
        const isModeReady = () => {
            // If we're expecting a specific mode, verify it's set
            if (window.LANDING_STATE.selectedMode) {
                const activeItem = document.querySelector('#modeDropdown .mode-item.active');
                const currentMode = activeItem ? activeItem.getAttribute('data-mode') : 'ask';
                return currentMode === window.LANDING_STATE.selectedMode;
            }
            // If no specific mode, we're ready
            return true;
        };
        
        // Check if auth is ready (either auth disabled, Clerk initialized, or token in localStorage)
        const isAuthReady = () => {
            // If auth is disabled, we're always ready
            if (window.STRATALENS_CONFIG?.authDisabled) {
                return true;
            }
            // Check if Clerk is loaded and has a session
            if (window.strataAuth?.initialized && window.strataAuth?.clerk?.loaded) {
                return true;
            }
            // Fallback: check localStorage for token
            const token = localStorage.getItem('authToken');
            return !!token;
        };

        // Auto-send the query with retry mechanism
        const attemptAutoSend = (retryCount = 0) => {
            // Wait for auth to be ready before sending
            if (!isAuthReady()) {
                console.log('Auto-send waiting for auth...', retryCount);
                return false;
            }

            // Verify mode is set correctly before sending
            if (!isModeReady()) {
                return false;
            }

            // Method 1: Try using the global chat interface (most reliable)
            if (window.chatInterface && typeof window.chatInterface.sendMessage === 'function') {
                window.chatInterface.sendMessage();
                return true;
            }

            // Method 2: Try triggering the send button click
            const sendButton = document.getElementById('sendChatButton');
            if (sendButton && typeof sendButton.click === 'function') {
                sendButton.click();
                return true;
            }

            // Method 3: Simulate Enter key press on the input
            const enterEvent = new KeyboardEvent('keydown', {
                key: 'Enter',
                code: 'Enter',
                keyCode: 13,
                which: 13,
                bubbles: true,
                cancelable: true
            });
            chatInput.dispatchEvent(enterEvent);
            return true;
        };
        
        // Wait for mode to be restored, then send
        let attempts = 0;
        const maxAttempts = 25; // Try for up to 5 seconds
        const sendInterval = setInterval(() => {
            attempts++;

            if (attemptAutoSend(attempts - 1)) {
                clearInterval(sendInterval);
                console.log('Auto-send successful on attempt', attempts);
            } else if (attempts >= maxAttempts) {
                clearInterval(sendInterval);
                console.warn('Auto-send failed after', maxAttempts, 'attempts');
            }
        }, 200); // Check every 200ms

        // Clear the initial query after attempting to send to prevent re-sends
        window.LANDING_STATE.initialQuery = null;
    }
}

/**
 * Handle successful authentication from landing page flow
 */
function handleLandingAuthSuccess() {
    
    // Remove restrictions
    removeLandingRestrictions();
    
    // Show welcome message using various possible chat interfaces
    const chatInterface = window.chatInterface || window.ChatInterface;
    if (chatInterface && typeof chatInterface.addMessage === 'function') {
        chatInterface.addMessage('assistant', 
            '🎉 Welcome to StrataLens! You now have full access to all platform features. Feel free to explore charts, screener, companies, and more!');
    }
    
    // Update UI to reflect authenticated state
    updateUIForAuthenticatedUser();
    
    // Reset landing state
    window.LANDING_STATE.hasUsedFreeMessage = false;
}

/**
 * Update UI elements after successful authentication
 */
function updateUIForAuthenticatedUser() {
    // Remove landing restrictions
    removeLandingRestrictions();
    
    // Update any auth-related UI elements
    const authButtons = document.querySelectorAll('[onclick*="showAuthModal"]');
    authButtons.forEach(btn => {
        btn.style.display = 'none';
    });
    
    // Show user profile or logout options if available
    const userProfile = document.getElementById('userProfile');
    if (userProfile) {
        userProfile.style.display = 'block';
    }
}

/**
 * Remove landing page restrictions after authentication
 */
function removeLandingRestrictions() {
    // Remove the landing banner
    const banner = document.getElementById('landing-banner');
    if (banner) {
        banner.remove();
    }
    
    // Re-enable any restricted navigation items (if any were restricted)
    const restrictedItems = document.querySelectorAll('.landing-restricted');
    restrictedItems.forEach(item => {
        item.classList.remove('landing-restricted', 'opacity-50', 'pointer-events-none');
        item.removeAttribute('title');
    });
    
    // Remove any upgrade prompts
    const overlays = document.querySelectorAll('.landing-upgrade-prompt');
    overlays.forEach(overlay => {
        overlay.remove();
    });
}

/**
 * Create a smooth transition effect between landing and platform
 */
function createTransitionEffect() {
    // Add a subtle fade-in effect when transitioning from landing
    if (window.LANDING_STATE.isTransitioned) {
        document.body.style.opacity = '0';
        document.body.style.transition = 'opacity 0.5s ease-in-out';
        
        setTimeout(() => {
            document.body.style.opacity = '1';
        }, 100);
        
        // Clean up
        setTimeout(() => {
            document.body.style.transition = '';
        }, 600);
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize immediately but mark that we're initializing
    if (!window.LANDING_INTEGRATION_INITIALIZED) {
        window.LANDING_INTEGRATION_INITIALIZED = true;
        
        // Wait a bit for other scripts to load
        setTimeout(() => {
            initLandingIntegration();
            createTransitionEffect();
        }, 300); // Reduced delay for faster initialization
    }
});

// Also initialize when called from app.js (fallback)
window.addEventListener('load', function() {
    setTimeout(() => {
        if (!window.LANDING_INTEGRATION_COMPLETED) {
            initLandingIntegration();
            createTransitionEffect();
            window.LANDING_INTEGRATION_COMPLETED = true;
        }
    }, 200);
});

// Export functions for global access
window.initLandingIntegration = initLandingIntegration;
window.handleLandingAuthSuccess = handleLandingAuthSuccess;
window.removeLandingRestrictions = removeLandingRestrictions;
