/**
 * StrataLens Authentication Module
 *
 * Handles authentication using Clerk.
 * Provides methods for:
 * - Initializing Clerk
 * - Getting auth tokens for API calls
 * - Managing auth state
 * - Sign in/out flows
 */

class StrataLensAuth {
    constructor() {
        this.clerk = null;
        this.initialized = false;
        this.initPromise = null;
        this.currentUser = null;
        this.onAuthChangeCallbacks = [];
    }

    /**
     * Initialize Clerk authentication
     * @returns {Promise<void>}
     */
    async init() {
        // Return existing promise if already initializing
        if (this.initPromise) {
            return this.initPromise;
        }

        this.initPromise = this._doInit();
        return this.initPromise;
    }

    async _doInit() {
        try {
            // Wait for Clerk script to load
            await this._waitForClerk();

            this.clerk = window.Clerk;

            // Check if Clerk is already loaded (auto-init from script tag)
            if (this.clerk.loaded) {
                console.log('Clerk already loaded, using existing instance');
            } else {
                // Get publishable key from config or backend
                let publishableKey = window.STRATALENS_CONFIG?.clerk?.publishableKey;

                if (!publishableKey) {
                    // Fetch from backend
                    try {
                        const response = await fetch(`${window.STRATALENS_CONFIG?.apiBaseUrl || ''}/auth/clerk/config`);
                        if (response.ok) {
                            const config = await response.json();
                            publishableKey = config.publishableKey;
                        }
                    } catch (e) {
                        console.warn('Could not fetch Clerk config from backend:', e);
                    }
                }

                if (!publishableKey) {
                    console.warn('Clerk publishable key not configured. Auth features will be limited.');
                    this.initialized = true;
                    return;
                }

                // Initialize Clerk
                await this.clerk.load({
                    publishableKey: publishableKey
                });
            }

            // Initial auth state - check if user is already signed in
            // This handles the case where user signed in on another page and was redirected
            if (this.clerk.user && this.clerk.session) {
                console.log('User already signed in:', this.clerk.user.primaryEmailAddress?.emailAddress);

                // Immediately store token and user in localStorage
                try {
                    const token = await this.clerk.session.getToken();
                    if (token) {
                        localStorage.setItem('authToken', token);
                        const userData = {
                            id: this.clerk.user.id,
                            email: this.clerk.user.primaryEmailAddress?.emailAddress,
                            fullName: this.clerk.user.fullName || this.clerk.user.firstName,
                            firstName: this.clerk.user.firstName,
                            lastName: this.clerk.user.lastName,
                            imageUrl: this.clerk.user.imageUrl
                        };
                        localStorage.setItem('currentUser', JSON.stringify(userData));
                        this.currentUser = userData;
                        console.log('Stored token and user from existing Clerk session');
                    }
                } catch (e) {
                    console.warn('Could not get token from existing session:', e);
                }
            }

            // Set up auth state listener for future changes
            this.clerk.addListener(({ user, session }) => {
                this._handleAuthChange(user, session);
            });

            this.initialized = true;
            console.log('Clerk authentication initialized, user:', this.clerk.user ? 'signed in' : 'not signed in');

        } catch (error) {
            console.error('Failed to initialize Clerk:', error);
            this.initialized = true; // Mark as initialized to prevent retry loops
            throw error;
        }
    }

    /**
     * Wait for Clerk to be fully ready
     */
    async _waitForClerk(timeout = 10000) {
        const start = Date.now();

        // First wait for Clerk object to exist
        while (!window.Clerk && Date.now() - start < timeout) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        if (!window.Clerk) {
            throw new Error('Clerk script failed to load');
        }

        // If Clerk has a load method and isn't loaded, wait for it
        // The script tag with data-clerk-publishable-key auto-calls load()
        // We need to wait until Clerk is fully ready
        while (!window.Clerk.loaded && Date.now() - start < timeout) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }

        console.log('Clerk ready, loaded:', window.Clerk.loaded, 'user:', !!window.Clerk.user);
    }

    /**
     * Handle auth state changes
     */
    async _handleAuthChange(user, session) {
        const wasAuthenticated = !!this.currentUser;
        const isAuthenticated = !!user;

        if (user && session) {
            this.currentUser = {
                id: user.id,
                email: user.primaryEmailAddress?.emailAddress,
                fullName: user.fullName || user.firstName || user.username,
                firstName: user.firstName,
                lastName: user.lastName,
                imageUrl: user.imageUrl,
                username: user.username
            };

            // Get token from Clerk and store in localStorage
            try {
                const token = await session.getToken();
                if (token) {
                    localStorage.setItem('authToken', token);
                    console.log('Stored auth token in localStorage');
                }
            } catch (e) {
                console.warn('Could not get token from Clerk session:', e);
            }

            // Store user info in localStorage
            localStorage.setItem('currentUser', JSON.stringify(this.currentUser));
            console.log('User signed in:', this.currentUser.email);

            // Update sidebar auth state - hide lock icons
            if (typeof window.updateSidebarAuthState === 'function') {
                window.updateSidebarAuthState();
            }
        } else {
            this.currentUser = null;
            localStorage.removeItem('currentUser');
            localStorage.removeItem('authToken');
            console.log('User signed out, cleared localStorage');

            // Update sidebar auth state - show lock icons
            if (typeof window.updateSidebarAuthState === 'function') {
                window.updateSidebarAuthState();
            }
        }

        // Notify listeners
        this.onAuthChangeCallbacks.forEach(callback => {
            try {
                callback(this.currentUser, wasAuthenticated !== isAuthenticated);
            } catch (e) {
                console.error('Auth change callback error:', e);
            }
        });
    }

    /**
     * Register a callback for auth state changes
     * @param {Function} callback - Called with (user, didChange)
     */
    onAuthChange(callback) {
        this.onAuthChangeCallbacks.push(callback);

        // Call immediately with current state
        if (this.initialized) {
            callback(this.currentUser, false);
        }
    }

    /**
     * Check if user is authenticated
     * @returns {boolean}
     */
    isAuthenticated() {
        return !!this.clerk?.session;
    }

    /**
     * Get current user info
     * @returns {Object|null}
     */
    getUser() {
        return this.currentUser;
    }

    /**
     * Get auth token for API calls
     * This is the primary method to get tokens - use this instead of localStorage
     * @returns {Promise<string|null>}
     */
    async getToken() {
        if (!this.clerk?.session) {
            return null;
        }

        try {
            const token = await this.clerk.session.getToken();
            return token;
        } catch (error) {
            console.error('Failed to get auth token:', error);
            return null;
        }
    }

    /**
     * Get auth headers for fetch requests
     * @returns {Promise<Object>}
     */
    async getAuthHeaders() {
        const token = await this.getToken();
        if (token) {
            return {
                'Authorization': `Bearer ${token}`
            };
        }
        return {};
    }

    /**
     * Open Clerk sign-in modal
     */
    openSignIn() {
        if (!this.clerk) {
            console.error('Clerk not initialized');
            // Fallback: show custom auth modal
            if (typeof showAuthModal === 'function') {
                showAuthModal();
            }
            return;
        }

        this.clerk.openSignIn({
            afterSignInUrl: window.location.href,
            afterSignUpUrl: window.location.href
        });
    }

    /**
     * Open Clerk sign-up modal
     */
    openSignUp() {
        if (!this.clerk) {
            console.error('Clerk not initialized');
            return;
        }

        this.clerk.openSignUp({
            afterSignInUrl: window.location.href,
            afterSignUpUrl: window.location.href
        });
    }

    /**
     * Open Clerk user profile
     */
    openUserProfile() {
        if (!this.clerk) {
            console.error('Clerk not initialized');
            return;
        }

        this.clerk.openUserProfile();
    }

    /**
     * Sign out
     */
    async signOut() {
        if (!this.clerk) {
            // Legacy fallback
            localStorage.removeItem('authToken');
            localStorage.removeItem('currentUser');
            window.location.reload();
            return;
        }

        await this.clerk.signOut();
        window.location.reload();
    }

    /**
     * Mount Clerk user button component
     * @param {string|HTMLElement} element - Element or selector to mount to
     */
    mountUserButton(element) {
        if (!this.clerk) {
            console.warn('Clerk not initialized, cannot mount user button');
            return;
        }

        const el = typeof element === 'string' ? document.querySelector(element) : element;
        if (!el) {
            console.warn('User button mount element not found');
            return;
        }

        this.clerk.mountUserButton(el, {
            afterSignOutUrl: window.location.origin
        });
    }

    /**
     * Mount Clerk sign-in component
     * @param {string|HTMLElement} element - Element or selector to mount to
     */
    mountSignIn(element) {
        if (!this.clerk) {
            console.warn('Clerk not initialized, cannot mount sign-in');
            return;
        }

        const el = typeof element === 'string' ? document.querySelector(element) : element;
        if (!el) {
            console.warn('Sign-in mount element not found');
            return;
        }

        this.clerk.mountSignIn(el, {
            afterSignInUrl: window.location.href,
            afterSignUpUrl: window.location.href
        });
    }
}

// Create global auth instance
window.strataAuth = new StrataLensAuth();

// Helper function for getting auth token (for backward compatibility)
async function getAuthToken() {
    // First try Clerk
    if (window.strataAuth?.initialized) {
        const token = await window.strataAuth.getToken();
        if (token) return token;
    }

    // Fallback to legacy localStorage token
    return localStorage.getItem('authToken');
}

// Helper function to check if authenticated
function isAuthenticated() {
    if (window.strataAuth?.isAuthenticated()) {
        return true;
    }

    // Fallback to legacy check
    const token = localStorage.getItem('authToken');
    const user = localStorage.getItem('currentUser');
    return !!(token && user);
}

// Helper function for authenticated fetch
async function authenticatedFetch(url, options = {}) {
    const authHeaders = await window.strataAuth?.getAuthHeaders() || {};

    // Fallback to legacy token
    if (!authHeaders.Authorization) {
        const token = localStorage.getItem('authToken');
        if (token) {
            authHeaders.Authorization = `Bearer ${token}`;
        }
    }

    return fetch(url, {
        ...options,
        headers: {
            ...options.headers,
            ...authHeaders
        }
    });
}

// Global helper functions for onclick handlers

/**
 * Open Clerk sign-in modal (called from navbar buttons)
 */
async function openClerkSignIn() {
    if (!window.strataAuth?.initialized) {
        await window.strataAuth?.init();
    }

    if (window.strataAuth?.clerk) {
        window.strataAuth.openSignIn();
    } else {
        // Fallback to custom modal if it exists
        if (typeof showAuthModal === 'function') {
            showAuthModal();
        } else {
            alert('Sign-in is loading. Please try again in a moment.');
        }
    }
}

/**
 * Open Clerk user profile (called from user menu)
 */
function openUserProfile() {
    if (window.strataAuth?.clerk) {
        window.strataAuth.openUserProfile();
    }
}

/**
 * Sign out (called from user menu)
 */
async function signOut() {
    if (window.strataAuth) {
        await window.strataAuth.signOut();
    } else {
        // Legacy fallback
        localStorage.removeItem('authToken');
        localStorage.removeItem('currentUser');
        window.location.reload();
    }
}

/**
 * Update navbar UI based on auth state - uses localStorage for reliability
 */
function updateNavbarAuthUI() {
    const navSignInBtn = document.getElementById('navSignInBtn');
    const userMenuApp = document.getElementById('userMenuApp');
    const userAvatarApp = document.getElementById('userAvatarApp');
    const userNameApp = document.getElementById('userNameApp');

    // If auth is disabled, hide all auth UI
    if (window.STRATALENS_CONFIG?.authDisabled) {
        if (navSignInBtn) navSignInBtn.classList.add('hidden');
        if (userMenuApp) userMenuApp.classList.add('hidden');
        return;
    }

    // Simply check localStorage - this is the source of truth
    const token = localStorage.getItem('authToken');
    const userStr = localStorage.getItem('currentUser');
    let user = null;

    if (token && userStr) {
        try {
            user = JSON.parse(userStr);
        } catch (e) {
            console.warn('Failed to parse currentUser from localStorage');
        }
    }

    console.log('updateNavbarAuthUI, user:', user ? user.email : 'none', 'token:', token ? 'yes' : 'no');

    if (user && token) {
        // User is signed in - hide sign in, show user menu
        if (navSignInBtn) navSignInBtn.classList.add('hidden');
        if (userMenuApp) {
            userMenuApp.classList.remove('hidden');
            userMenuApp.classList.add('flex');
        }

        // Update user info
        if (userAvatarApp && user.imageUrl) {
            userAvatarApp.src = user.imageUrl;
            userAvatarApp.alt = user.fullName || 'User';
        }
        if (userNameApp) {
            userNameApp.textContent = user.firstName || user.fullName || 'User';
        }
    } else {
        // User is signed out - show sign in, hide user menu
        if (navSignInBtn) navSignInBtn.classList.remove('hidden');
        if (userMenuApp) {
            userMenuApp.classList.add('hidden');
            userMenuApp.classList.remove('flex');
        }
    }
}

// Register auth change listener to update UI
if (window.strataAuth) {
    window.strataAuth.onAuthChange((user, didChange) => {
        console.log('Auth state changed:', user ? 'signed in' : 'signed out');
        updateNavbarAuthUI();
    });
}

// Auto-initialize Clerk on page load
document.addEventListener('DOMContentLoaded', async () => {
    // First, immediately check localStorage and update UI
    // This shows the correct state right away if user is already signed in
    updateNavbarAuthUI();

    // Then initialize Clerk to sync with their session
    console.log('Initializing Clerk authentication...');
    try {
        await window.strataAuth.init();
        console.log('Clerk initialized, updating navbar UI');
        // Update again after Clerk is ready (may have fresh token)
        updateNavbarAuthUI();
    } catch (e) {
        console.warn('Clerk init failed:', e);
        // UI already updated from localStorage above
    }
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { StrataLensAuth, getAuthToken, isAuthenticated, authenticatedFetch, openClerkSignIn, openUserProfile, signOut };
}
