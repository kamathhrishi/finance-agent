// --- POSTHOG HELPER FUNCTIONS ---
// Safe PostHog tracking function that won't throw errors if PostHog isn't loaded
function trackEvent(eventName, properties = {}) {
    try {
        if (typeof posthog !== 'undefined' && posthog && posthog.capture) {
            posthog.capture(eventName, properties);
        } else {
        }
    } catch (error) {
    }
}

// Safe PostHog identify function
function identifyUser(userId, properties = {}) {
    try {
        if (typeof posthog !== 'undefined' && posthog && posthog.identify) {
            posthog.identify(userId, properties);
        } else {
        }
    } catch (error) {
    }
}

// Safe PostHog reset function
function resetUser() {
    try {
        if (typeof posthog !== 'undefined' && posthog && posthog.reset) {
            posthog.reset();
        } else {
        }
    } catch (error) {
    }
}

// --- CONFIGURATION ---
// Use external configuration from config.js
const CONFIG = window.STRATALENS_CONFIG || {
    // Fallback configuration if config.js fails to load
    apiBaseUrl: 'https://apiurmarketcast.com',
    websocketUrl: 'wss://apiurmarketcast.com',
    pageSize: 20,
    toastDuration: 4000,
    chartDataLimit: 50,
    environment: 'local'
};

// Global variables for toast functionality and streaming
let toastTimeout;
let eventSource = null;
let chartInstance = null;
let isSearchInProgress = false;

// WebSocket state
let wsClient = null;
let isUsingWebSocket = false;


$(document).ready(function() {


            // Check for token in URL parameters (from OAuth/magic link redirects)
            const urlParams = new URLSearchParams(window.location.search);
            const tokenFromUrl = urlParams.get('token');
            
            if (tokenFromUrl) {
                localStorage.setItem('authToken', tokenFromUrl);
                
                // Clear URL parameters
                window.history.replaceState({}, document.title, window.location.pathname);
                
                // Validate token and get user info
                validateTokenAndSetUser(tokenFromUrl);
                return; // Skip normal auth check
            }

            // --- STATE ---
            let currentUser = null;
            let currentData = [];
            let currentQuery = '';
            let currentPage = 1;
            let totalPages = 1;
            let totalRecords = 0;
            let lastApiResponse = null;

            // WebSocket state
            let persistentSessionId = null;

            // Filing data state
            let currentFilingData = null;
            let hasFilingData = false;

            // Sorting state
            let currentSortColumn = null;
            let currentSortDirection = null;

            // =============================================================================
            // WEBSOCKET CLIENT FOR REAL-TIME ANALYSIS
            // =============================================================================

            class StrataLensWebSocketClient {
                constructor(url, userId = null) {
                    this.url = url;
                    this.userId = userId;
                    this.ws = null;
                    this.isConnected = false;
                    this.reconnectAttempts = 0;
                    this.maxReconnectAttempts = 5;
                    this.reconnectDelay = 1000;
                    this.activeSessions = new Map();
                    this.eventHandlers = new Map();
                    this.shouldReconnect = true;
                    this.pingInterval = null;
                }

                // Event handler management
                on(event, handler) {
                    if (!this.eventHandlers.has(event)) {
                        this.eventHandlers.set(event, []);
                    }
                    this.eventHandlers.get(event).push(handler);
                }

                off(event, handler) {
                    if (this.eventHandlers.has(event)) {
                        const handlers = this.eventHandlers.get(event);
                        const index = handlers.indexOf(handler);
                        if (index > -1) {
                            handlers.splice(index, 1);
                        }
                    }
                }

                emit(event, data) {
                    if (this.eventHandlers.has(event)) {
                        this.eventHandlers.get(event).forEach(handler => handler(data));
                    }
                }

                // Connection management
                async connect() {
                    try {
                        if (this.ws?.readyState === WebSocket.OPEN) {
                            return;
                        }

                        this.ws = new WebSocket(this.url);

                        this.ws.onopen = () => {
                            this.isConnected = true;
                            this.reconnectAttempts = 0;
                            this.startPingPong();
                            this.emit('connected', { timestamp: new Date() });

                            // Try to reconnect to any stored sessions
                            this.reconnectStoredSessions();
                        };

                        this.ws.onmessage = (event) => {
                            try {
                                const message = JSON.parse(event.data);
                                this.handleMessage(message);
                            } catch (error) {
                            }
                        };

                        this.ws.onclose = (event) => {
                            this.isConnected = false;
                            this.stopPingPong();
                            this.emit('disconnected', { code: event.code, reason: event.reason });

                            if (this.shouldReconnect) {
                                this.scheduleReconnect();
                            }
                        };

                        this.ws.onerror = (error) => {
                            this.emit('error', { error });
                        };

                    } catch (error) {
                        this.emit('error', { error });
                        this.scheduleReconnect();
                    }
                }

                disconnect() {
                    this.shouldReconnect = false;
                    this.stopPingPong();
                    if (this.ws) {
                        this.ws.close();
                        this.ws = null;
                    }
                    this.isConnected = false;
                }

                scheduleReconnect() {
                    if (!this.shouldReconnect || this.reconnectAttempts >= this.maxReconnectAttempts) {
                        return;
                    }

                    this.reconnectAttempts++;
                    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

                    setTimeout(() => {
                        this.connect();
                    }, delay);
                }

                startPingPong() {
                    this.pingInterval = setInterval(() => {
                        if (this.isConnected) {
                            this.send({ type: 'ping' });
                        }
                    }, 30000); // Ping every 30 seconds
                }

                stopPingPong() {
                    if (this.pingInterval) {
                        clearInterval(this.pingInterval);
                        this.pingInterval = null;
                    }
                }

                // Message handling
                handleMessage(message) {
                    const { type } = message;

                    switch (type) {
                        case 'connection_established':
                        case 'connected':  // ✅ Backend uses 'connected'
                            this.emit('ready', message);
                            break;

                        case 'query_received':  // ✅ Backend sends this after receiving query
                            this.emit('query_received', message);
                            break;

                        case 'session_created':
                            this.handleSessionCreated(message);
                            break;

                        case 'session_reconnected':
                            this.handleSessionReconnected(message);
                            break;

                        case 'status_update':
                        case 'reasoning_update':
                        case 'reasoning':  // ✅ NEW: Handle reasoning events from backend
                            this.handleAnalysisUpdate(message);
                            break;

                        case 'result':  // ✅ Backend sends 'result' not 'analysis_complete'
                            this.handleAnalysisComplete(message);
                            break;

                        case 'analysis_complete':
                            this.handleAnalysisComplete(message);
                            break;

                        case 'analysis_cancelled':
                        case 'cancelled':  // ✅ Backend also sends 'cancelled'
                            this.handleAnalysisCancelled(message);
                            break;

                        case 'error':
                            // Check if this is a rate limit error
                            if (message.error === 'RATE_LIMIT_EXCEEDED') {
                                showRateLimitError(message.message || 'Rate limit exceeded');
                            } else {
                                this.handleError(message);
                            }
                            break;

                        case 'pong':
                            // Heartbeat response
                            break;

                        case 'chat_received':
                            this.emit('chat_received', message);
                            break;

                        case 'chat_result':
                            this.emit('chat_result', message);
                            break;

                        case 'chat_error':
                            this.emit('chat_error', message);
                            break;

                        default:
                    }
                }

                handleSessionCreated(message) {
                    const { session_id, query, user_id } = message;

                    // Store session info
                    this.activeSessions.set(session_id, {
                        query,
                        user_id,
                        status: 'processing',
                        created_at: new Date()
                    });

                    // Store persistent session ID
                    persistentSessionId = session_id;
                    this.storeSessionInBrowser(session_id, query);

                    this.emit('session_created', message);
                }

                handleSessionReconnected(message) {
                    const { session_id, status, progress } = message;
                    this.emit('session_reconnected', message);
                }

                handleAnalysisUpdate(message) {
                    this.emit('analysis_update', message);
                }

                handleAnalysisComplete(message) {
                    const { result, session_id } = message;

                    // Update session status
                    if (this.activeSessions.has(session_id)) {
                        this.activeSessions.get(session_id).status = 'completed';
                    }

                    // Handle both old and new result formats
                    const finalResult = result || message;
                    this.emit('analysis_complete', { result: finalResult, session_id });
                }

                handleAnalysisCancelled(message) {
                    const { session_id } = message;

                    if (this.activeSessions.has(session_id)) {
                        this.activeSessions.get(session_id).status = 'cancelled';
                    }

                    this.emit('analysis_cancelled', message);
                }

                handleError(message) {
                    this.emit('websocket_error', message);
                }

                // Session management
                storeSessionInBrowser(sessionId, query) {
                    try {
                        const sessionData = {
                            session_id: sessionId,
                            query: query,
                            timestamp: new Date().toISOString()
                        };
                        localStorage.setItem('stratalens_active_session', JSON.stringify(sessionData));
                    } catch (error) {
                    }
                }

                getStoredSession() {
                    try {
                        const stored = localStorage.getItem('stratalens_active_session');
                        return stored ? JSON.parse(stored) : null;
                    } catch (error) {
                        return null;
                    }
                }

                reconnectStoredSessions() {
                    const stored = this.getStoredSession();
                    if (stored && stored.session_id) {
                        this.reconnectToSession(stored.session_id);
                    }
                }

                // Public API methods
                send(message) {
                    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                        this.ws.send(JSON.stringify(message));
                        return true;
                    } else {
                        return false;
                    }
                }

                startAnalysis(query, options = {}) {
                    const message = {
                        type: 'query',           // ✅ Fixed: Use correct message type
                        question: query,         // ✅ Fixed: Use correct field name
                        user_id: this.userId || 'anonymous',
                        page: options.page || 1,
                        page_size: options.page_size || CONFIG.pageSize
                    };

                    return this.send(message);
                }

                reconnectToSession(sessionId) {
                    const message = {
                        type: 'reconnect_session',
                        session_id: sessionId
                    };

                    return this.send(message);
                }

                cancelAnalysis(sessionId) {
                    const message = {
                        type: 'cancel_analysis',
                        session_id: sessionId
                    };

                    return this.send(message);
                }

                getSessionStatus(sessionId) {
                    const message = {
                        type: 'get_session_status',
                        session_id: sessionId
                    };

                    return this.send(message);
                }

                getUserSessions() {
                    const message = {
                        type: 'get_user_sessions',
                        user_id: this.userId
                    };

                    return this.send(message);
                }

                sendChatMessage(message, comprehensive = true) {
                    if (!this.isConnected) {
                        return false;
                    }

                    const chatMessage = {
                        type: 'chat',
                        message: message,
                        comprehensive: comprehensive
                    };

                    return this.send(chatMessage);
                }
            }

            // WebSocket integration functions
            function initializeWebSocket() {
                if (!currentUser || wsClient) {
                    return;
                }

                try {
                    // For single-server: ws://localhost:8000/ws/{user_id}
                    const websocketUrl = `${CONFIG.websocketUrl}/ws/${currentUser.id}`;
                    wsClient = new StrataLensWebSocketClient(websocketUrl, currentUser.id);

                    // Set up event handlers
                    wsClient.on('connected', handleWebSocketConnected);
                    wsClient.on('disconnected', handleWebSocketDisconnected);
                    wsClient.on('session_created', handleWebSocketSessionCreated);
                    wsClient.on('analysis_update', handleWebSocketAnalysisUpdate);
                    wsClient.on('analysis_complete', handleWebSocketAnalysisComplete);
                    wsClient.on('websocket_error', handleWebSocketError);
                    wsClient.on('chat_received', handleWebSocketChatReceived);
                    wsClient.on('chat_result', handleWebSocketChatResult);
                    wsClient.on('chat_error', handleWebSocketChatError);
                    wsClient.on('chat_stream', handleWebSocketChatStream); // Re-enabled for streaming
                    wsClient.on('chat_progress', handleWebSocketChatProgress);

                    // Attempt to connect
                    wsClient.connect();

                } catch (error) {
                    wsClient = null;
                }
            }

            function handleWebSocketConnected() {
                isUsingWebSocket = true;
            }

            function handleWebSocketDisconnected() {
                isUsingWebSocket = false;
            }

            function handleWebSocketSessionCreated(data) {
                persistentSessionId = data.session_id;

                // Session info toast removed - silent mode
            }

            function handleWebSocketAnalysisUpdate(data) {
                // Handle real-time analysis updates
                if (data.type === 'reasoning_update') {
                    const reasoning = {
                        event_type: data.step,
                        message: data.message,
                        details: data.details || {}
                    };
                    handleReasoningEvent(reasoning);
                } else if (data.type === 'reasoning') {
                    // ✅ NEW: Handle reasoning events from backend WebSocket
                    const reasoning = data.event || {};
                    handleReasoningEvent(reasoning);
                } else if (data.type === 'status_update') {
                    updatePanelStatus('processing', data.message);
                }

                // Update progress if available
                if (data.progress !== undefined) {
                    updateProgressBar(data.progress);
                }
            }

            function handleWebSocketAnalysisComplete(data) {
    const result = data.result;

    if (result) {
        // Use existing result handling (this already calls setLoadingState(false))
        handleFinalResult(result);
    } else {
        showToast('❌ Analysis completed but no results received', 'error');
        setLoadingState(false);
        updateSearchButtonState(false);
        updatePanelStatus('error', 'No results received');

        // Don't auto-clear sidebar indicator - let user see completed state until they click search tab
    }
}

            function handleWebSocketError(data) {
                // WebSocket error toast removed - silent mode
                handleStreamingError(data.message);
                updateSearchButtonState(false);
            }

            function handleWebSocketChatReceived(data) {
                // Show that the message was received and is being processed
                showTypingIndicator();
            }

            function handleWebSocketChatResult(data) {
                
                hideTypingIndicator();
                
                // Set loading state for chat
                // isChatLoading = false; // OLD CHAT REMOVED
                const sendButton = document.getElementById('sendChatButton');
                if (sendButton) {
                    sendButton.disabled = false;
                    sendButton.innerHTML = '<i class="fas fa-paper-plane"></i>';
                }
                
                // Clean up old streaming message element if it exists
                const streamingMessage = document.getElementById('streaming-chat-message');
                if (streamingMessage) {
                    streamingMessage.remove();
                }
                
                // ✅ FIXED: Check data.success (not data.result.success)
                if (data.success && data.result) {
                    const result = data.result;
                    
                    // Extract the answer from the result
                    const answer = result.answer || result.message || 'No response generated';
                    const citations = result.citations || [];
                    
                    
                    // Finalize streaming message if it exists, otherwise add complete message
                    if (window.chatInterface) {
                        if (window.chatInterface.streamingMessageId) {
                            // Finalize the streaming message with citations
                            window.chatInterface.finalizeStreamingMessage(
                                window.chatInterface.streamingMessageId, 
                                citations, 
                                null  // reasoningHTML if available
                            );
                            window.chatInterface.streamingMessageId = null;
                        } else {
                            // No streaming occurred, add complete message
                            window.chatInterface.addMessage('assistant', answer, citations, 'normal');
                        }
                    }
                    
                    // ✅ BACKGROUND PROCESSING: Track notifications when tab not visible
                    if (!chatTabVisible || document.hidden) {
                        pendingChatNotifications++;
                        updateChatTabIndicator();
                    }
                    
                } else {
                    // ✅ BETTER ERROR HANDLING: More specific error messages
                    let errorMessage = 'Unknown error occurred';
                    
                    if (data.result) {
                        if (data.result.error) {
                            errorMessage = data.result.error;
                        } else if (data.result.answer) {
                            // Sometimes the "error" is actually in the answer field
                            errorMessage = data.result.answer;
                        }
                    } else if (data.message) {
                        errorMessage = data.message;
                    }
                    
                    
                    // Add error message
                    if (window.chatInterface) {
                        window.chatInterface.addMessage('assistant', `I apologize, but an error occurred: ${errorMessage}`, [], 'error');
                    }
                    
                    // ✅ BACKGROUND PROCESSING: Track error notifications when tab not visible
                    if (!chatTabVisible || document.hidden) {
                        pendingChatNotifications++;
                        updateChatTabIndicator();
                    }
                }
            }

            function handleWebSocketChatError(data) {
                
                hideTypingIndicator();
                
                // Set loading state for chat
                // isChatLoading = false; // OLD CHAT REMOVED
                const sendButton = document.getElementById('sendChatButton');
                if (sendButton) {
                    sendButton.disabled = false;
                    sendButton.innerHTML = '<i class="fas fa-paper-plane"></i>';
                }
                
                // Clean up streaming message if it exists
                if (window.chatInterface && window.chatInterface.streamingMessageId) {
                    const streamingMsg = document.getElementById(window.chatInterface.streamingMessageId);
                    if (streamingMsg && streamingMsg.parentNode) {
                        streamingMsg.parentNode.removeChild(streamingMsg);
                    }
                    window.chatInterface.streamingMessageId = null;
                }
                
                // ✅ IMPROVED ERROR MESSAGES: More specific error handling
                let errorMessage = 'Failed to process chat message';
                
                if (data.error === 'RATE_LIMIT_EXCEEDED') {
                    errorMessage = `Rate limit exceeded: ${data.message}`;
                } else if (data.error === 'RAG_SYSTEM_UNAVAILABLE') {
                    errorMessage = 'Chat system is temporarily unavailable. Please try again later.';
                } else if (data.error === 'AUTHENTICATION_FAILED') {
                    errorMessage = 'Authentication failed. Please log in again.';
                } else if (data.message) {
                    errorMessage = data.message;
                } else if (data.error) {
                    errorMessage = `Error: ${data.error}`;
                }
                
                
                // Show error to user via chat interface
                if (window.chatInterface) {
                    window.chatInterface.addMessage('assistant', errorMessage, [], 'error');
                }
            }

            function handleWebSocketChatStream(data) {
                
                // Forward streaming to chat interface if available
                if (window.chatInterface) {
                    // Check if streaming message exists, create if needed
                    if (!window.chatInterface.streamingMessageId) {
                        window.chatInterface.streamingMessageId = 'streaming-' + Date.now();
                        window.chatInterface.createStreamingMessage(window.chatInterface.streamingMessageId);
                    }
                    
                    // Append token to streaming message
                    if (data.content) {
                        window.chatInterface.appendTokenToMessage(window.chatInterface.streamingMessageId, data.content);
                    }
                }
            }

            function handleWebSocketChatProgress(data) {
                
                // Update or create progress indicator
                let progressIndicator = document.getElementById('chat-progress-indicator');
                if (!progressIndicator) {
                    // Create progress indicator
                    progressIndicator = document.createElement('div');
                    progressIndicator.id = 'chat-progress-indicator';
                    progressIndicator.className = 'chat-progress-indicator';
                    
                    const progressBar = document.createElement('div');
                    progressBar.className = 'progress-bar';
                    
                    const progressFill = document.createElement('div');
                    progressFill.className = 'progress-fill';
                    
                    const progressText = document.createElement('div');
                    progressText.className = 'progress-text';
                    
                    progressBar.appendChild(progressFill);
                    progressIndicator.appendChild(progressBar);
                    progressIndicator.appendChild(progressText);
                    
                    // Add to chat messages container
                    const messagesContainer = document.getElementById('chatMessages');
                    if (messagesContainer) {
                        messagesContainer.appendChild(progressIndicator);
                    }
                }
                
                // Update progress
                const progressFill = progressIndicator.querySelector('.progress-fill');
                const progressText = progressIndicator.querySelector('.progress-text');
                
                if (progressFill) {
                    const percentage = Math.round(data.progress * 100);
                    progressFill.style.width = `${percentage}%`;
                }
                
                if (progressText) {
                    progressText.textContent = data.message || 'Processing...';
                }
                
                // Hide progress indicator when completed
                if (data.stage === 'completed') {
                    setTimeout(() => {
                        if (progressIndicator && progressIndicator.parentNode) {
                            progressIndicator.parentNode.removeChild(progressIndicator);
                        }
                    }, 2000); // Hide after 2 seconds
                }
            }

            function updateProgressBar(progress) {
                const progressBar = $('#progressBar');
                if (progressBar.length) {
                    const percentage = Math.round(progress * 100);
                    progressBar.css('width', `${percentage}%`);
                }
            }

            // --- PROMPT LIBRARY DATA ---
            const promptLibrary = {
                available: [
                    { query: "Show me all data center stocks that have increased their R&D spend in last quarter beyond their revenue growth and revenue growth is more than 20%", description: "Data Center R&D Leaders with High Growth" },
                    { query: "Consumer companies that have grown their revenue by more than 20% in last 10 years", description: "High Growth Consumer Companies" },
                    { query: "Companies with gross margins improvement excluding financials and mining in latest quarter", description: "Margin Improvement Leaders" },
                    { query: "show me all semiconductor companies above billon dollar market cap with their total r&d spend to revenue between 2015-2020 and their revenue cagr from 2020-2024", description: "Semiconductor R&D Analysis" },
                    { query: "What is the Revenue, cash, margins and net income data for last 10 years for $AAPL, $MSFT, $NFLX, $TSLA and $GOOG", description: "10-Year Financial History: Tech Giants" },
                    { query: "which 50 billion plus market cap companies have grown their profits by more than 50% in latest quarter", description: "Latest Quarter: High Growth Large Caps" },
                    { query: "all consumer companies above billion dollar market cap with gross margins above 80% and with their marketing to revenue spend", description: "High Margin Consumer Companies" },
                    { query: "compare crypto mining companies revenue growth from 2015-2020 to 2020-2024", description: "Crypto Mining Growth Analysis" }
                ],
                comingSoon: [
                    { query: "Show me companies with highest revenue exposure to Europe", description: "European Market Leaders", preview: "Analyze companies with significant European revenue exposure, including geographic segment breakdowns and regional growth trends." },
                    { query: "Show companies with more than 40% exposure to China", description: "Geopolitical Exposure", preview: "Identify companies with significant exposure to China and analyze geopolitical risk factors." },
                    { query: "Companies with more than 20% exposure to data centers", description: "AI Proxy Plays", preview: "Find companies heavily invested in data center infrastructure as AI proxy investments." },
                    { query: "companies who build enterprise software with above 20% roce, that are above their 200DMA and less than 10% of their all time high", description: "Enterprise Software Value Plays", preview: "Real-time market data analysis combining financial metrics with technical indicators and price performance." },
                    { query: "companies who which had lowest volaity in revenue growth and stock price from 2006-2011", description: "Crisis-Resilient Companies", preview: "Historical analysis of companies that maintained stability during the 2008 financial crisis and subsequent recovery period." }
                ]
            };

            // --- INITIALIZATION ---
            function init() {
                
                // Ensure authenticated controls are hidden by default
                $('#authenticatedControls').addClass('hidden');

                // Set up event listeners FIRST
                setupEventListeners();
                
                // Initialize suggestions section to be visible and expanded by default
                $('.screener-categories').removeClass('hidden');
                $('#suggestionsContent').removeClass('hidden');
                $('#suggestionsChevron').removeClass('fa-chevron-down').addClass('fa-chevron-up');

                // Mobile devices are now supported

                showAuthLoading();
                Chart.register(ChartDataLabels);

                
                // Check if user came from landing page
                const urlParams = new URLSearchParams(window.location.search);
                const fromLanding = urlParams.get('from') === 'landing';
                
                checkAuthenticationStatus().then(isValid => {
                    hideAuthLoading();
                    
                    if (isValid) {
                        showMainApp();
                        // Force chat section as default with explicit activation
                        forceActivateSearchSection();
                        validateTokenWithBackend();

                        // Initialize WebSocket for real-time features
                        initializeWebSocket();

                        // Load usage data to show indicator
                        loadUsageData();
                        
                        // Start with fresh conversation
                        setTimeout(() => {
                            if (typeof startNewChat === 'function') {
                                startNewChat();
                            }
                        }, 1000);
                    } else if (fromLanding) {
                        
                        // Store landing state BEFORE cleaning URL
                        const initialQuery = urlParams.get('query');
                        if (initialQuery) {
                            window.LANDING_STATE = {
                                fromLanding: true,
                                initialQuery: decodeURIComponent(initialQuery),
                                hasUsedFreeMessage: false
                            };
                        }
                        
                        showMainApp();
                        forceActivateSearchSection();
                        // Landing integration will handle the banner and query transfer
                        
                        // Clean up URL to remove the landing params
                        window.history.replaceState({}, document.title, window.location.pathname);
                    } else {
                        showMainApp();
                        forceActivateSearchSection();
                    }
                }).catch(error => {
                    hideAuthLoading();
                    
                    if (fromLanding) {
                        showMainApp();
                        forceActivateSearchSection();
                        // Clean up URL to remove the landing params
                        window.history.replaceState({}, document.title, window.location.pathname);
                    } else {
                        showMainApp();
                        forceActivateSearchSection();
                    }
                });
            }

            // Mobile devices are now fully supported - no detection needed


            
            // Initialize modals and global event listeners
            setupUsageModal();
            setupOnboardingModal();
            setupProfileModal();
            setupChangePasswordModal();
            setupChartingEventListeners();
            // initializeChat(); // DISABLED - using new chat.js
            setupChatHistoryAndStats(); // Setup chat history and stats modals
            
            // Initialize the application
            init();
        });


            // --- AUTHENTICATION ---
            function showAuthLoading() {
                $('#authLoadingScreen').show();
            }
            function hideAuthLoading() {
                $('#authLoadingScreen').hide();
            }

            async function checkAuthenticationStatus() {
                const token = localStorage.getItem('authToken');
                const user = JSON.parse(localStorage.getItem('currentUser') || 'null');


                if (!token || !user) {
                    clearAuthData();
                    return false;
                }

                try {
                    const payload = JSON.parse(atob(token.split('.')[1]));
                    if (payload.exp && Date.now() >= payload.exp * 1000) {
                        clearAuthData();
                        return false;
                    }
                } catch (e) {
                    clearAuthData();
                    return false;
                }

                // Quick backend validation with timeout - if it fails, go to login
                try {
                    const controller = new AbortController();
                    const timeoutId = setTimeout(() => controller.abort(), 3000); // 3 second timeout

                    const response = await fetch(`${CONFIG.apiBaseUrl}/auth/validate`, {
                        headers: { 'Authorization': `Bearer ${token}` },
                        signal: controller.signal
                    });

                    clearTimeout(timeoutId);

                    if (!response.ok) {
                        clearAuthData();
                        return false;
                    }

                } catch (error) {
                    clearAuthData();
                    return false;
                }

                // Set current user and continue
                currentUser = user;

                // Fetch latest onboarding status from backend (non-blocking with timeout)
                fetchOnboardingStatusWithTimeout(token, user);

                return true;
            }

            async function fetchOnboardingStatusWithTimeout(token, user) {
                try {

                    // Add timeout to prevent hanging
                    const controller = new AbortController();
                    const timeoutId = setTimeout(() => {
                        controller.abort();
                    }, 3000); // 3 second timeout

                    const response = await fetch(`${CONFIG.apiBaseUrl}/user/onboarding-status`, {
                        headers: { 'Authorization': `Bearer ${token}` },
                        signal: controller.signal
                    });

                    clearTimeout(timeoutId);


                    if (response.ok) {
                        const onboardingData = await response.json();
                        user.has_completed_onboarding = onboardingData.has_completed_onboarding;
                        user.should_show_onboarding = onboardingData.should_show_onboarding;
                        user.query_count = onboardingData.query_count;
                        localStorage.setItem('currentUser', JSON.stringify(user));
                        currentUser = user; // Update current user with fresh data
                    } else {
                        const errorText = await response.text();
                        // Don't redirect to login for onboarding status failure - just use cached data
                    }
                } catch (error) {
                    if (error.name === 'AbortError') {
                    } else {
                    }
                    // Continue with cached user data - don't block authentication
                }
            }

            async function validateTokenAndSetUser(token) {
                try {
                    showAuthLoading();
                    
                    const response = await fetch(`${CONFIG.apiBaseUrl}/auth/validate`, {
                        headers: { 'Authorization': `Bearer ${token}` }
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        
                        // Set user data
                        currentUser = {
                            id: data.user_id,
                            email: data.user?.email || '',
                            full_name: data.user?.full_name || '',
                            is_admin: data.user?.is_admin || false
                        };
                        localStorage.setItem('currentUser', JSON.stringify(currentUser));
                        
                        hideAuthLoading();
                        showMainApp();
                        showToast('Authentication successful!', 'success');
                        
                        // Track successful OAuth/magic link login
                        trackEvent('oauth_magic_link_success');
                        
                    } else {
                        throw new Error('Token validation failed');
                    }
                    
                } catch (error) {
                    clearAuthData();
                    hideAuthLoading();
                    showAuthModal('login');
                    showToast('Authentication failed. Please try again.', 'error');
                }
            }

            async function validateTokenWithBackend() {
                try {
                    const token = localStorage.getItem('authToken');

                    // Add timeout to prevent hanging
                    const controller = new AbortController();
                    const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout

                    const response = await fetch(`${CONFIG.apiBaseUrl}/auth/validate`, {
                        headers: { 'Authorization': `Bearer ${token}` },
                        signal: controller.signal
                    });

                    clearTimeout(timeoutId);

                    if (!response.ok) {
                        throw new Error('Token invalid on backend');
                    }
                } catch (error) {
                    if (error.name === 'AbortError') {
                    } else {
                        // If backend validation fails, redirect to login for security
                        clearAuthData();
                        showAuthModal('login');
                    }
                }
            }

            function clearAuthData() {
                localStorage.removeItem('authToken');
                localStorage.removeItem('currentUser');
                currentUser = null;
            }

            function showAuthModal(defaultTab = 'register') {
                $('#authModal').removeClass('hidden').addClass('flex');
                $('#mainApp').addClass('hidden');
            }

            function hideAuthModal() {
                $('#authModal').addClass('hidden').removeClass('flex');
                $('#mainApp').removeClass('hidden');
            }

            function showAboutModal() {
                $('#aboutModal').removeClass('hidden').addClass('flex');
            }

            function hideAboutModal() {
                $('#aboutModal').addClass('hidden').removeClass('flex');
            }

            function showMagicLinkModal() {
                $('#authModal').addClass('hidden').removeClass('flex');
                $('#magicLinkModal').removeClass('hidden').addClass('flex');
            }

            function hideMagicLinkModal() {
                $('#magicLinkModal').addClass('hidden').removeClass('flex');
                $('#authModal').removeClass('hidden').addClass('flex');
            }

            function showMainApp() {
                $('#authModal').addClass('hidden').removeClass('flex');
                $('#magicLinkModal').addClass('hidden').removeClass('flex');
                $('#mainApp').removeClass('hidden');
                
                // Only show authenticated controls if user is actually authenticated
                const token = localStorage.getItem('authToken');
                const user = JSON.parse(localStorage.getItem('currentUser') || 'null');
                
                if (token && user && currentUser) {
                    // User is authenticated - show all controls
                    $('#authenticatedControls').removeClass('hidden');
                    $('#userName').text(currentUser?.full_name || currentUser?.email || 'User');
                } else {
                    // User is anonymous - hide controls
                    $('#authenticatedControls').addClass('hidden');
                }

                // Hide chat section for non-admin users
                // Chat is now available to all users - make sure it's visible
                const chatMenuItem = $('.sidebar-menu-item[data-section="chat"]');
                if (chatMenuItem.length > 0) {
                    chatMenuItem.show();
                } else {
                }

                // Onboarding modal is now only shown when user clicks the onboarding button

            }

            function logout() {
                // Track logout
                trackEvent('user_logout');
                
                // Hide authenticated controls
                $('#authenticatedControls').addClass('hidden');
                
                resetUser();

                clearAuthData();
                showAuthModal('register');
                showToast('Logged out successfully', 'info');
            }

            // =============================================================================
            // ONBOARDING SYSTEM
            // =============================================================================

            function showOnboardingModal() {
    trackEvent('show_onboarding');
    $('#onboardingModal').removeClass('hidden').addClass('flex');
            }

            function hideOnboardingModal() {
                $('#onboardingModal').removeClass('flex').addClass('hidden');
            }

            async function completeOnboarding() {
                try {
                    const token = localStorage.getItem('authToken');
                    if (!token) {
                        logout();
                        return;
                    }

                    const response = await fetch(`${CONFIG.apiBaseUrl}/user/complete-onboarding`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${token}`
                        },
                        body: JSON.stringify({
                            user_id: currentUser.id
                        })
                    });

                    if (!response.ok) {
                        throw new Error('Failed to complete onboarding');
                    }

                            // Update local user data - mark as completed
        currentUser.has_completed_onboarding = true;
        localStorage.setItem('currentUser', JSON.stringify(currentUser));

        // Track onboarding completion
        trackEvent('complete_onboarding', {
            user_id: currentUser.id
        });

        hideOnboardingModal();
        showToast('Onboarding completed! Welcome to StrataLens!', 'success');

                    } catch (error) {
        // Track onboarding error
        trackEvent('onboarding_error', {
            error: error.message
        });

        showToast('Failed to complete onboarding. Please try again.', 'error');
                }
            }

            // --- SETUP LISTENERS ---


            // Note: hasMultipleSeries is available as ChartUtils.hasMultipleSeries

            // =============================================================================
            // NEW: SERVER-SIDE SORTING FUNCTIONALITY
            // =============================================================================

            function setupServerSideSorting() {
                // Remove existing click handlers
                $(document).off('click', '.data-grid thead th');

                // Add new server-side sorting handlers
                $(document).on('click', '.data-grid thead th', function() {
                    handleServerSideSorting(this);
                });
            }

            async function handleServerSideSorting(thElement) {
                const $th = $(thElement);
                const $table = $th.closest('table');
                const columnIndex = $th.index();

                // Sorting works on the entire dataset, not just the current page
                // No page restriction needed

                // Get column name from the original columns array
                let columnName;

                if ($table.attr('id') === 'singleSheetTable') {
                    // Single sheet sorting
                    if (!lastApiResponse || !lastApiResponse.columns) {
                        showToast('No data available to sort', 'warning');
                        return;
                    }
                    columnName = lastApiResponse.columns[columnIndex];
                } else {
                    // Multi-sheet sorting
                    const activeSheetData = getCurrentActiveSheetData();
                    if (!activeSheetData || !activeSheetData.columns) {
                        showToast('No sheet data available to sort', 'warning');
                        return;
                    }
                    columnName = activeSheetData.columns[columnIndex];
                }

                if (!columnName) {
                    showToast('Column not found for sorting', 'warning');
                    return;
                }

                // Determine sort direction
                const currentSort = $th.data('sort') || 'none';
                let newDirection = 'asc';
                if (currentSort === 'asc') newDirection = 'desc';
                else if (currentSort === 'desc') newDirection = 'asc';

                // Clear all sort indicators
                $table.find('thead th').removeData('sort').find('.sort-icon')
                    .removeClass('fa-sort-up fa-sort-down').addClass('fa-sort');

                // Set new sort indicator
                $th.data('sort', newDirection);
                const $icon = $th.find('.sort-icon');
                $icon.removeClass('fa-sort');
                if (newDirection === 'asc') $icon.addClass('fa-sort-up');
                else $icon.addClass('fa-sort-down');

                // Update global sort state
                currentSortColumn = columnName;
                currentSortDirection = newDirection;

                // Show loading state
                const $tbody = $table.find('tbody');
                const originalContent = $tbody.html();
                $tbody.html(`
                    <tr>
                        <td colspan="99" class="text-center p-8">
                            <div class="flex items-center justify-center space-x-3">
                                <i class="fas fa-spinner fa-spin text-xl text-accent-primary"></i>
                                <span>Sorting data on server...</span>
                            </div>
                        </td>
                    </tr>
                `);

                try {
                    // Call server-side sorting
                    if ($table.attr('id') === 'singleSheetTable') {
                        await sortSingleSheetData(columnName, newDirection);
                    } else {
                        await sortMultiSheetData(columnName, newDirection);
                    }

                    showToast(`Sorted by ${getFriendlyColumnName(columnName)} (${newDirection === 'asc' ? 'ascending' : 'descending'})`, 'success', 2000);

                } catch (error) {

                    // Restore original content on error
                    $tbody.html(originalContent);

                    // Reset sort indicator and global state
                    $th.removeData('sort').find('.sort-icon')
                        .removeClass('fa-sort-up fa-sort-down').addClass('fa-sort');
                    currentSortColumn = null;
                    currentSortDirection = null;

                    const errorMessage = error.message || 'Sorting failed. Please try again.';
                    showToast(errorMessage, 'error');
                }
            }

            async function sortSingleSheetData(columnName, direction) {
                try {
                    const token = localStorage.getItem('authToken');
                    if (!token) {
                        logout();
                        return;
                    }

                    const response = await fetch(`${CONFIG.apiBaseUrl}/screener/query/sort`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${token}`
                        },
                        body: JSON.stringify({
                            column: columnName,
                            direction: direction
                        })
                    });

                    if (response.status === 401) {
                        logout();
                        return;
                    }

                    const result = await response.json();

                    if (!response.ok) {
                        throw new Error(result.detail || `HTTP error! status: ${response.status}`);
                    }

                    if (!result.success) {
                        throw new Error(result.message || 'Sorting failed');
                    }

                    // Update the current data with sorted results
                    currentData = result.data_rows;

                    // Update lastApiResponse to maintain consistency
                    if (lastApiResponse) {
                        lastApiResponse.data_rows = result.data_rows;
                        lastApiResponse.columns = result.columns;
                        lastApiResponse.friendly_columns = result.friendly_columns;
                        lastApiResponse.pagination_info = result.pagination_info;
                        lastApiResponse.total_rows = result.total_rows;
                    }

                    // Update pagination info after sorting (maintain current page)
                    if (result.pagination_info) {
                        totalPages = result.pagination_info.total_pages;
                        totalRecords = result.pagination_info.total_records;
                    }

                    // Re-render the table with sorted data and preserve sort state
                    renderTableWithSortState(result.columns, result.data_rows, result.friendly_columns, columnName, direction);

                    // Update any other UI elements that depend on the data
                    updateDataDependentUI(result);


                } catch (error) {
                    throw error;
                }
            }



            function updateDataDependentUI(result) {
                // Update any UI elements that depend on the current data
                // For example, update result count displays
                if (result.total_rows !== undefined) {
                    $('#singleSheetCount').text(`${result.total_rows.toLocaleString()} results`);
                }

                // Update pagination if needed
                if (result.pagination_info) {
                    renderPagination(result.pagination_info);
                }
            }

            function getFriendlyColumnName(columnName) {
                // Get friendly column name from current data
                if (lastApiResponse && lastApiResponse.friendly_columns && lastApiResponse.friendly_columns[columnName]) {
                    return lastApiResponse.friendly_columns[columnName];
                }

                // Fallback to formatted column name
                return ChartUtils.formatColumnName(columnName);
            }

                // Note: formatColumnName is available as ChartUtils.formatColumnName



    // Helper function to handle company name/ticker clicks - make it globally accessible
    window.handleCompanyClick = function(rowIndex, columnKey) {
        try {
            // Get the row data from current single-sheet data
            let row;
            if (currentData && currentData[rowIndex]) {
                row = currentData[rowIndex];
            } else {
                showToast('Unable to find row data', 'warning');
                return;
            }


            // Try to find symbol from the row data
            let symbol = null;
            let companyName = null;

            // Look for symbol in common column names
            const symbolColumns = ['symbol', 'ticker', 'stock_symbol', 'company_symbol'];
            for (const col of symbolColumns) {
                if (row[col]) {
                    symbol = row[col];
                    break;
                }
            }

            // Look for company name in common column names
            const nameColumns = ['companyName', 'company_name', 'name', 'company'];
            for (const col of nameColumns) {
                if (row[col]) {
                    companyName = row[col];
                    break;
                }
            }

            // If we clicked on a symbol column, use that as symbol
            if (columnKey.toLowerCase().includes('symbol') || columnKey.toLowerCase().includes('ticker')) {
                symbol = row[columnKey];
            }

            // If we clicked on a name column, use that as company name and try to find symbol
            if (columnKey.toLowerCase().includes('name') && !columnKey.toLowerCase().includes('symbol')) {
                companyName = row[columnKey];
                if (!symbol) {
                    // If no symbol found, use the first few words of company name as fallback
                    symbol = companyName ? companyName.split(' ')[0].toUpperCase() : null;
                }
            }


            if (symbol) {
                // Switch to companies tab and load the company profile
                switchToSection('companies');
                loadCompanyProfile(symbol, companyName);

                // Show loading toast
                showToast(`Loading profile for ${symbol}...`, 'info', 2000);
            } else {
                showToast('Unable to identify company symbol for profile loading', 'warning');
            }
        } catch (error) {
            showToast('Error loading company profile', 'error');
        }
    }

            // Enhanced table rendering function that preserves sort state
            function renderTableWithSortState(columns, dataRows, friendlyColumns, currentSortColumn = null, currentSortDirection = null) {
                if (!columns || columns.length === 0) return;

                // Check if we have actual filing links for this dataset
                const hasActualLinks = hasActualFilingLinks();

                // Generate table header with friendly column names and sort indicators
                const headerHtml = columns.map((colKey, index) => {
                    const displayName = friendlyColumns[colKey] || colKey.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

                    // Determine sort icon
                    let sortIcon = 'fa-sort';
                    let sortData = '';

                    if (currentSortColumn === colKey) {
                        if (currentSortDirection === 'asc') {
                            sortIcon = 'fa-sort-up';
                            sortData = 'data-sort="asc"';
                        } else if (currentSortDirection === 'desc') {
                            sortIcon = 'fa-sort-down';
                            sortData = 'data-sort="desc"';
                        }
                    }

                    return `<th title="Sort by ${displayName}" ${sortData}>${displayName} <i class="fas ${sortIcon} sort-icon ml-1"></i></th>`;
                }).join('');

                // Add Sources column if we have actual filing links
                const sourcesHeader = hasActualLinks ? '<th class="text-center">Sources</th>' : '';
                $('#singleSheetTableHead').html(`<tr>${headerHtml}${sourcesHeader}</tr>`);

                // Handle empty data case
                if (!dataRows || dataRows.length === 0) {
                    const colspan = columns.length + (hasActualLinks ? 1 : 0);
                    $('#singleSheetTableBody').html(`<tr><td colspan="${colspan}" class="text-center py-8">No data.</td></tr>`);
                    return;
                }

                // Generate table rows with proper formatting and styling
                const rowsHtml = dataRows.map((row, rowIndex) => {
                    const cellsHtml = columns.map(colKey => {
                        const value = row[colKey];
                        let cellClass = '';
                        let displayValue;
                        const lowerColKey = colKey.toLowerCase();

                        // Handle null/undefined values
                        if (value === null || value === undefined) {
                            displayValue = '<span class="text-text-tertiary">—</span>';
                        }
                        // Handle company symbols and names with links
                        else if (lowerColKey.includes('symbol') || lowerColKey.includes('ticker') || lowerColKey.includes('name')) {
                            cellClass = 'symbol-cell';
                            const truncatedValue = lowerColKey.includes('name') ? truncateCompanyName(value) : value;
                            displayValue = `<a href="#" class="company-link" onclick="event.preventDefault(); event.stopPropagation(); handleCompanyClick(${rowIndex}, '${colKey}', false, 0);" title="Click to view company profile: ${value}">${truncatedValue}</a>`;
                        }
                        // Handle sector and industry with badges
                        else if (lowerColKey.includes('sector') || lowerColKey.includes('industry')) {
                            displayValue = `<span class="sector-badge ${ChartUtils.getSectorClass(value)}">${value}</span>`;
                        }
                        // Use backend-formatted values directly (no re-formatting)
                        else {
                            const isTruncated = ChartUtils.isValueTruncated(value);
                            displayValue = truncateDisplayValue(value);

                            // Add expand button for truncated values
                            if (isTruncated) {
                                displayValue = `<span class="truncated-value">${displayValue}</span><button class="expand-value-btn ml-1 text-accent-primary hover:text-accent-secondary transition-colors" onclick="expandTruncatedValue('${colKey}', ${rowIndex}, 0)" title="Click to view full value"><i class="fas fa-expand-alt text-xs"></i></button>`;
                            }

                            // Apply appropriate CSS classes for styling based on column type
                            if (lowerColKey.includes('year') || lowerColKey.includes('calendaryear')) {
                                cellClass = 'number-cell';
                            }
                            else if (lowerColKey.includes('cagr') || lowerColKey.includes('percent') || lowerColKey.includes('growth') || lowerColKey.includes('margin')) {
                                cellClass = 'number-cell';
                                const numericValue = parseFloat(value);
                                if (!isNaN(numericValue)) {
                                    if (numericValue > 0) cellClass += ' data-positive';
                                    else if (numericValue < 0) cellClass += ' data-negative';
                                }
                            }
                            else if (lowerColKey.includes('price') || lowerColKey.includes('mktcap') || lowerColKey.includes('revenue') || lowerColKey.includes('income')) {
                                cellClass = 'number-cell';
                            }
                            else if (lowerColKey.includes('ratio') && (lowerColKey.includes('pe') || lowerColKey.includes('pb'))) {
                                cellClass = 'number-cell';
                            }
                            else {
                                const numericCheck = parseFloat(value);
                                if (!isNaN(numericCheck)) {
                                    cellClass = 'number-cell';
                                }
                            }
                        }

                        return `<td class="${cellClass}">${displayValue}</td>`;
                    }).join('');

                    // Add Sources column cell if we have actual filing links
                    let sourcesCell = '';
                    if (hasActualLinks) {
                        const rowFilingData = getRowFilingData(row, rowIndex);
                        if (rowFilingData && rowFilingData.length > 0) {
                            // Check if this specific row has actual clickable links
                            let hasRowLinks = false;
                            for (const filing of rowFilingData) {
                                // Only check for actual clickable links, not metadata like CIK or period
                                if ((filing.link && filing.link !== '-' && filing.link !== 'No filing link available' &&
                                     (filing.link.includes('http') || filing.link.includes('www') || filing.link.includes('.com') || filing.link.includes('.gov'))) ||
                                    (filing.finalLink && filing.finalLink !== '-' && filing.finalLink !== 'No filing link available' &&
                                     (filing.finalLink.includes('http') || filing.finalLink.includes('www') || filing.finalLink.includes('.com') || filing.finalLink.includes('.gov')))) {
                                    hasRowLinks = true;
                                    break;
                                }

                                // Also check for any other URL-like fields
                                Object.keys(filing).forEach(key => {
                                    const lowerKey = key.toLowerCase();
                                    const value = filing[key];
                                    if (value && value !== '-' && value !== 'No filing link available' &&
                                        (lowerKey.includes('url') || lowerKey.includes('href') ||
                                         (lowerKey.includes('link') && key !== 'link' && key !== 'finalLink')) &&
                                        (value.includes('http') || value.includes('www') || value.includes('.com') || value.includes('.gov'))) {
                                        hasRowLinks = true;
                                    }
                                });

                                if (hasRowLinks) break;
                            }

                            if (hasRowLinks) {
                                sourcesCell = `<td class="text-center">
                                    <button class="view-sources-btn" onclick="openSecSourcesModal(${rowIndex})" title="View SEC filing sources for this row">
                                        <i class="fas fa-file-contract"></i>
                                        View Sources
                                    </button>
                                </td>`;
                            } else {
                                sourcesCell = `<td class="text-center">
                                    <span class="text-text-tertiary text-xs">No sources</span>
                                </td>`;
                            }
                        } else {
                            sourcesCell = `<td class="text-center">
                                <span class="text-text-tertiary text-xs">No sources</span>
                            </td>`;
                        }
                    }

                    return `<tr>${cellsHtml}${sourcesCell}</tr>`;
                }).join('');

                $('#singleSheetTableBody').html(rowsHtml);
            }

            // Note: filing functions are available as ChartUtils.isFilingColumn and ChartUtils.separateDataAndFilingColumns

            // Update the existing renderTable function to filter out filing columns
            function renderTable(columns, dataRows, friendlyColumns) {
                // Separate data and filing columns
                const { dataColumns, dataFriendlyColumns } = ChartUtils.separateDataAndFilingColumns(columns, friendlyColumns);

                // Render only data columns in the main table with current sorting state
                renderTableWithSortState(dataColumns, dataRows, dataFriendlyColumns, currentSortColumn, currentSortDirection);
                // Setup server-side sorting after table is rendered
                setupServerSideSorting();
            }

            // =============================================================================
            // SEC SOURCES MODAL FUNCTIONALITY
            // =============================================================================

            // Check if there are any actual filing links available in the data
            function hasActualFilingLinks() {
                if (!hasFilingData || !currentFilingData || currentFilingData.length === 0) {
                    return false;
                }

                // Check if any row has actual filing links
                if (currentData && currentData.length > 0) {
                    for (let i = 0; i < currentData.length; i++) {
                        const rowFilingData = getRowFilingData(currentData[i], i);
                        if (rowFilingData && rowFilingData.length > 0) {
                            // Check if any filing has actual clickable links
                            for (const filing of rowFilingData) {
                                // Only check for actual clickable links, not metadata like CIK or period
                                if ((filing.link && filing.link !== '-' && filing.link !== 'No filing link available' &&
                                     (filing.link.includes('http') || filing.link.includes('www') || filing.link.includes('.com') || filing.link.includes('.gov'))) ||
                                    (filing.finalLink && filing.finalLink !== '-' && filing.finalLink !== 'No filing link available' &&
                                     (filing.finalLink.includes('http') || filing.finalLink.includes('www') || filing.finalLink.includes('.com') || filing.finalLink.includes('.gov')))) {
                                    return true;
                                }

                                // Also check for any other URL-like fields
                                Object.keys(filing).forEach(key => {
                                    const lowerKey = key.toLowerCase();
                                    const value = filing[key];
                                    if (value && value !== '-' && value !== 'No filing link available' &&
                                        (lowerKey.includes('url') || lowerKey.includes('href') ||
                                         (lowerKey.includes('link') && key !== 'link' && key !== 'finalLink')) &&
                                        (value.includes('http') || value.includes('www') || value.includes('.com') || value.includes('.gov'))) {
                                        return true;
                                    }
                                });
                            }
                        }
                    }
                }


                return false;
            }

            // Get filing data for a specific row
            window.getRowFilingData = function(row, rowIndex) {

                // First, try to extract filing data directly from the row itself
                // This handles cases where filing data was included in the table but filtered out
                const rowFilingData = {};
                let hasRowFilingData = false;

                Object.keys(row).forEach(key => {
                    const lowerKey = key.toLowerCase();
                    if (lowerKey.includes('link') || lowerKey.includes('filing') ||
                        lowerKey.includes('accepted') || lowerKey.includes('cik') ||
                        lowerKey.includes('filling') || lowerKey === 'period') {
                        rowFilingData[key] = row[key];
                        if (row[key] && row[key] !== '-' && row[key] !== 'No filing link available') {
                            hasRowFilingData = true;
                        }
                    }
                });


                // If we found filing data in the row itself, use it
                if (hasRowFilingData) {
                    // Add company context
                    const filing = {...rowFilingData};
                    if (row.symbol) filing.symbol = row.symbol;
                    if (row.companyName) filing.companyName = row.companyName;
                    if (row.name && !filing.companyName) filing.companyName = row.name;

                    return [filing];
                }

                if (!currentFilingData || currentFilingData.length === 0) {
                    return [];
                }

                // For single-sheet data, look for filing data that matches this row
                if (currentFilingData.length === 1 && currentFilingData[0].filings) {
                    const filings = currentFilingData[0].filings;

                    // Try to match by symbol, company name, or row index
                    const matchingFilings = filings.filter(filing => {
                        if (row.symbol && filing.symbol && row.symbol === filing.symbol) {
                            return true;
                        }
                        if (row.companyName && filing.companyName && row.companyName === filing.companyName) {
                            return true;
                        }
                        if (row.name && filing.companyName && row.name === filing.companyName) {
                            return true;
                        }
                        return false;
                    });


                    // If no matches found by data, return all filings (for simple cases where all rows share the same filing data)
                    if (matchingFilings.length === 0) {
                        return filings.length > 0 ? filings : [];
                    }

                    return matchingFilings;
                }


                return [];
            }

            // Open SEC Sources Modal for a specific row
            window.openSecSourcesModal = function(rowIndex) {
                const row = currentData[rowIndex];
                if (!row) return;

                const rowFilingData = getRowFilingData(row, rowIndex);
                if (!rowFilingData || rowFilingData.length === 0) {
                    showToast('No SEC filing sources available for this row', 'warning');
                    return;
                }

                // Generate modal content
                const modalContent = renderSecSourcesModalContent(row, rowFilingData);
                $('#secSourcesContent').html(modalContent);

                // Show modal
                $('#secSourcesModal').removeClass('hidden');
            }

            // Render content for SEC Sources Modal
            window.renderSecSourcesModalContent = function(row, filingData) {

                let content = `<div class="space-y-4">`;

                // Render filing data
                filingData.forEach((filing, index) => {
                    content += `
                        <div class="filing-source-card">
                            <div class="flex justify-between items-start mb-3">
                                <div class="flex flex-col">
                                    <h5 class="font-semibold text-text-primary">Filing ${index + 1}</h5>
                    `;

                    // Show key row data inline with filing info
                    if (row.symbol || row.companyName || row.name) {
                        content += `<div class="text-sm text-text-secondary mt-1">`;
                        if (row.symbol) content += `${row.symbol}`;
                        if (row.companyName) content += `${row.symbol ? ' • ' : ''}${row.companyName}`;
                        if (row.name && !row.companyName) content += `${row.symbol ? ' • ' : ''}${row.name}`;
                        content += `</div>`;
                    }

                    content += `
                                </div>
                                <div class="flex gap-2">
                    `;

                    // Add filing links with enhanced URL validation and formatting
                    if (filing.link && filing.link !== '-' && filing.link !== 'No filing link available') {
                        const cleanLink = filing.link.replace(/<[^>]*>/g, '').trim();
                        // Ensure the link has a proper protocol
                        const formattedLink = cleanLink.startsWith('http') ? cleanLink : `https://${cleanLink}`;
                        content += `
                            <a href="${formattedLink}" target="_blank" rel="noopener noreferrer" class="filing-link" onclick="event.stopPropagation();">
                                <i class="fas fa-external-link-alt"></i>
                                View Filing
                            </a>
                        `;
                    }

                    if (filing.finalLink && filing.finalLink !== '-' && filing.finalLink !== 'No filing link available') {
                        const cleanFinalLink = filing.finalLink.replace(/<[^>]*>/g, '').trim();
                        // Ensure the link has a proper protocol
                        const formattedFinalLink = cleanFinalLink.startsWith('http') ? cleanFinalLink : `https://${cleanFinalLink}`;
                        content += `
                            <a href="${formattedFinalLink}" target="_blank" rel="noopener noreferrer" class="filing-link" onclick="event.stopPropagation();">
                                <i class="fas fa-file-alt"></i>
                                Alt Link
                            </a>
                        `;
                    }

                    // Check for any other potential link fields in the filing data
                    Object.keys(filing).forEach(key => {
                        const lowerKey = key.toLowerCase();
                        const value = filing[key];

                        // Look for any field that might contain a URL but isn't already handled
                        if (value && value !== '-' && value !== 'No filing link available' &&
                            (lowerKey.includes('url') || lowerKey.includes('href') ||
                             (lowerKey.includes('link') && key !== 'link' && key !== 'finalLink')) &&
                            (value.includes('http') || value.includes('www') || value.includes('.com') || value.includes('.gov'))) {

                            const cleanValue = value.replace(/<[^>]*>/g, '').trim();
                            const formattedValue = cleanValue.startsWith('http') ? cleanValue : `https://${cleanValue}`;

                            // Convert field name to friendly label
                            let linkLabel = key.replace(/([A-Z])/g, ' $1').replace(/_/g, ' ')
                                              .replace(/\b\w/g, l => l.toUpperCase()).trim();

                            content += `
                                <a href="${formattedValue}" target="_blank" rel="noopener noreferrer" class="filing-link" onclick="event.stopPropagation();">
                                    <i class="fas fa-external-link-alt"></i>
                                    ${linkLabel}
                                </a>
                            `;
                        }
                    });

                    content += `
                                </div>
                            </div>

                            <div class="filing-metadata">
                    `;

                    // Add filing metadata with comprehensive field checking

                    // Add ALL non-empty fields as metadata
                    Object.keys(filing).forEach(key => {
                        const value = filing[key];
                        if (value && value !== '-' && value !== 'No filing link available' &&
                            key !== 'symbol' && key !== 'companyName' && key !== 'link' && key !== 'finalLink') {

                            let label = key;
                            const lowerKey = key.toLowerCase();

                            // Convert common field names to friendly labels
                            if (lowerKey.includes('filling') || lowerKey.includes('filing')) {
                                label = 'Filing Date';
                            } else if (lowerKey.includes('accepted')) {
                                label = 'Accepted Date';
                            } else if (lowerKey.includes('cik')) {
                                label = 'SEC CIK';
                            } else if (lowerKey.includes('period')) {
                                label = 'Period';
                            } else {
                                // Convert camelCase or snake_case to Title Case
                                label = key.replace(/([A-Z])/g, ' $1').replace(/_/g, ' ')
                                          .replace(/\b\w/g, l => l.toUpperCase()).trim();
                            }

                            // Check if the value looks like a URL and make it clickable
                            let displayValue = value;
                            if (typeof value === 'string' &&
                                (value.includes('http') || value.includes('www') ||
                                 value.includes('.com') || value.includes('.gov') ||
                                 value.includes('.org') || value.includes('.edu'))) {

                                const cleanValue = value.replace(/<[^>]*>/g, '').trim();
                                const formattedValue = cleanValue.startsWith('http') ? cleanValue : `https://${cleanValue}`;
                                displayValue = `<a href="${formattedValue}" target="_blank" rel="noopener noreferrer" class="text-blue-600 hover:text-blue-800 underline" onclick="event.stopPropagation();">${cleanValue}</a>`;
                            }

                            content += `
                                <div class="filing-metadata-item">
                                    <div class="filing-metadata-label">${label}</div>
                                    <div class="filing-metadata-value">${displayValue}</div>
                                </div>
                            `;
                        }
                    });

                    content += `
                            </div>
                        </div>
                    `;
                });

                content += `</div>`;

                return content;
            }

            // Close SEC Sources Modal
            window.closeSecSourcesModal = function() {
                $('#secSourcesModal').addClass('hidden');
            }


            // =============================================================================
            // END: SEC SOURCES MODAL FUNCTIONALITY
            // =============================================================================

            // =============================================================================
            // END: SERVER-SIDE SORTING FUNCTIONALITY
            // =============================================================================

function setupEventListeners() {
    // Mobile devices are now fully supported

    // Login form handler removed - using request access modal only
    $('#logoutBtn').on('click', logout);
    $('#onboardingBtn').on('click', showOnboardingModal);
    $('#searchBtn').on('click', performSearchWithStreaming);
    $('#queryInput').on('keypress', e => { if (e.which === 13) performSearchWithStreaming(); });
    
    // Expand suggestions when user focuses on search input (if collapsed)
    $('#queryInput').on('focus', function() {
        const categoriesSection = $('.screener-categories');
        if (categoriesSection.hasClass('collapsed')) {
            expandScreenerSuggestions();
        }
    });

    // Character count updates
    $('#queryInput').on('input', function() {
        const count = $(this).val().length;
        $('#queryCharCount').text(count);

        // Change color when approaching limit
        if (count > 900) {
            $('#queryCharCount').addClass('text-red-500').removeClass('text-text-tertiary');
        } else if (count > 800) {
            $('#queryCharCount').addClass('text-yellow-500').removeClass('text-text-tertiary text-red-500');
        } else {
            $('#queryCharCount').removeClass('text-yellow-500 text-red-500').addClass('text-text-tertiary');
        }
    });

    $('#companySearchInput').on('input', function() {
        const count = $(this).val().length;
        $('#companyCharCount').text(count);

        // Change color when approaching limit
        if (count > 90) {
            $('#companyCharCount').addClass('text-red-500').removeClass('text-text-tertiary');
        } else if (count > 80) {
            $('#companyCharCount').addClass('text-yellow-500').removeClass('text-text-tertiary text-red-500');
        } else {
            $('#companyCharCount').removeClass('text-yellow-500 text-red-500').addClass('text-text-tertiary');
        }
    });

    $(document).on('click', '.example-query', function() {
        $('#queryInput').val($(this).data('query'));
    });

    // New screener query suggestion buttons
    $(document).on('click', '.query-suggestion-btn', function() {
        const query = $(this).data('query');
        $('#queryInput').val(query);
        // Auto-scroll to search input for better UX
        $('html, body').animate({
            scrollTop: $('#queryInput').offset().top - 100
        }, 500);
        // Focus on input but let user decide when to search
        $('#queryInput').focus();
        // Highlight the search button to indicate next action
        $('#searchBtn').addClass('pulse-highlight');
        setTimeout(() => {
            $('#searchBtn').removeClass('pulse-highlight');
        }, 2000);
    });

    // Quick action buttons
    $(document).on('click', '.quick-action-btn', function() {
        const query = $(this).data('query');
        $('#queryInput').val(query);
        // Auto-scroll to search input for better UX
        $('html, body').animate({
            scrollTop: $('#queryInput').offset().top - 100
        }, 300);
        $('#queryInput').focus();
    });

    $(document).on('click', '.prompt-library-btn, #promptLibraryBtn', openPromptLibrary);
    $('#closeModalBtn').on('click', closePromptLibrary);
    $('#promptLibraryModal').on('click', function(e) { if (e.target === this) closePromptLibrary(); });
    $(document).on('click', '.prompt-item-btn', function() {
        $('#queryInput').val($(this).data('query'));
        closePromptLibrary();
    });

    // Mobile navigation functionality
    $('#mobileAboutBtn').on('click', function() {
        showAboutModal();
    });
    
    $('#mobileSignInBtn').on('click', function() {
        showAuthModal();
    });
    
    $('#mobileGetStartedBtn').on('click', function() {
        showAuthModal();
    });
    
    // Desktop About button
    $('#desktopAboutBtn').on('click', function() {
        showAboutModal();
    });
    
    // Sidebar toggle functionality
    $(document).on('click', '#sidebarToggle', function(e) {
        e.preventDefault();
        toggleSidebar();
    });

    // Close sidebar when clicking backdrop
    $(document).on('click', '.sidebar-backdrop', function(e) {
        e.preventDefault();
        closeSidebar();
    });

    // Desktop sidebar functionality
    $(document).on('click', '.sidebar-menu-item', function(e) {
        e.preventDefault();
    const $item = $(this);
    const section = $item.data('section');

    // Intercept clicks on disabled (non-chat) items and show Coming Soon modal
    if ($item.hasClass('disabled') || $item.attr('aria-disabled') === 'true') {
        if (section !== 'chat') {
            const modal = document.getElementById('comingSoonModal');
            if (modal) {
                modal.classList.add('open');
            }
            return;
        }
    }
        handleSidebarNavigation(section);
    });

    // Export buttons
    $('#singleSheetExportBtn').on('click', exportSingleSheetData);

    // Pagination controls
    $('#pageSizeSelect').on('change', function() {
        CONFIG.pageSize = parseInt($(this).val());
        currentPage = 1;
        if (currentQuery) fetchPage(1);
    });

    $('#firstPageBtn').on('click', () => goToPage(1));
    $('#prevPageBtn').on('click', () => goToPage(currentPage - 1));
    $('#nextPageBtn').on('click', () => goToPage(currentPage + 1));
    $('#lastPageBtn').on('click', () => goToPage(totalPages));
    $('#pageJumpInput').on('change', function() {
        const page = parseInt($(this).val());
        goToPage(page >= 1 && page <= totalPages ? page : currentPage);
    });

    $('#aiPanelToggle').on('click', () => $('#aiReasoningPanel').toggleClass('expanded'));

    // Charting Listeners - removed view toggles
    $('#closeChartModalBtn').on('click', closeChartModal);
    $('#chartModal').on('click', function(e) { if (e.target === this) closeChartModal(); });
    $('#updateChartBtn').on('click', updateChart);
    $('#exportModalChartBtn').on('click', exportModalChart);
    // Use event delegation to ensure dropdown changes are always captured
    $(document).on('change', '#chartDataPointsSelect', function() {
        const selectedValue = $(this).val();
        const chartType = $('#chartModal .chart-type-toggle button.active').data('type');
        updateChart();
    });

    // Chart type toggle - line charts for time series, bar charts for comparisons
    $('#chartModal .chart-type-toggle button').on('click', function() {
        const chartType = $(this).data('type');
        const selectedSheet = $('#chartSheetSelector').val();


        $(this).addClass('active').siblings().removeClass('active');

        // Update UI based on chart type
        if (chartType === 'line') {
            // Line charts are for time series analysis
            $('#chartDataPointsLabel').text('Time Period');
            $('#chartDataPointsSelect').empty().append(`
                <option value="all">All Time Periods</option>
                <option value="3">Last 3 Years</option>
                <option value="5">Last 5 Years</option>
                <option value="8">Last 8 Years</option>
                <option value="10">Last 10 Years</option>
                <option value="15">Last 15 Years</option>
                <option value="20">Last 20 Years</option>
            `).val('all');
        } else {
            // Bar charts are for comparing top performers
            $('#chartDataPointsLabel').text('Show Companies');
            $('#chartDataPointsSelect').empty().append(`
                <option value="all">All Companies</option>
                <option value="5">Top 5</option>
                <option value="8">Top 8</option>
                <option value="10">Top 10</option>
                <option value="15">Top 15</option>
                <option value="20">Top 20</option>
                <option value="50">Top 50</option>
            `).val('10');
        }

        ChartFunctions.populateChartSelectors({
            lastApiResponse: lastApiResponse,
            currentData: currentData
        });
        updateChart();
    });

    // Update axis selectors
    $('#chartXAxis, #chartYAxis').on('change', updateChart);

    // Close dropdowns when clicking outside
    $(document).on('click', function(e) {
        if (!$(e.target).closest('.export-dropdown').length) {
            $('.export-dropdown').removeClass('active');
        }
    });

    // SEC Sources Modal event listeners
    $('#closeSecSourcesModalBtn, #closeSecSourcesFooterBtn').on('click', window.closeSecSourcesModal);
    $('#secSourcesModal').on('click', function(e) {
        if (e.target === this) window.closeSecSourcesModal();
    });

    // Screens functionality event listeners
    $('#saveScreenBtn, #saveMultiScreenBtn').on('click', openSaveScreenModal);
    $('#saveScreenForm').on('submit', handleSaveScreen);
    $('#closeSaveScreenModalBtn, #cancelSaveScreenBtn').on('click', closeSaveScreenModal);
    $('#saveScreenModal').on('click', function(e) {
        if (e.target === this) closeSaveScreenModal();
    });

    // Plot Screen Stocks Modal event listeners
    $('#closePlotScreenStocksModalBtn, #cancelPlotScreenStocksBtn').on('click', closePlotScreenStocksModal);
    $('#plotScreenStocksModal').on('click', function(e) {
        if (e.target === this) closePlotScreenStocksModal();
    });
    $('#plotScreenStocksBtn').on('click', handlePlotScreenStocks);

    // Screens page event listeners
    $('#refreshScreensBtn').on('click', loadUserScreens);
    $('#screensFilterType').on('change', loadUserScreens);

    // Collections navigation event listeners
    $(document).on('click', '.collection-nav-btn', function(e) {
        e.preventDefault();
        const collectionType = $(this).data('collection-type');
        switchCollectionType(collectionType);
    });

    // Screens pagination event listeners
    $('#screensFirstPageBtn').on('click', () => goToScreensPage(1));
    
    // Suggestions toggle functionality
    $('#suggestionsToggle').on('click', function() {
        const content = $('#suggestionsContent');
        const chevron = $('#suggestionsChevron');
        
        if (content.hasClass('hidden')) {
            content.removeClass('hidden');
            chevron.removeClass('fa-chevron-down').addClass('fa-chevron-up');
        } else {
            content.addClass('hidden');
            chevron.removeClass('fa-chevron-up').addClass('fa-chevron-down');
        }
    });
    $('#screensPrevPageBtn').on('click', () => goToScreensPage(screensPage - 1));
    $('#screensNextPageBtn').on('click', () => goToScreensPage(screensPage + 1));
    $('#screensLastPageBtn').on('click', () => goToScreensPage(screensTotalPages));
    $('#screensPageJumpInput').on('change', function() {
        const page = parseInt($(this).val());
        goToScreensPage(page >= 1 && page <= screensTotalPages ? page : screensPage);
    });

    // Company search event listeners
    $('#companySearchInput').on('input', handleCompanySearchInput); // Real-time search
    $('#companyClearBtn').on('click', clearCompanySearch);

    // Company profile tab switching
    $(document).on('click', '.tab-btn', function() {
        const tabName = $(this).data('tab');
        switchCompanyTab(tabName);
    });

    // Financial tab switching removed - using unified view

    $(document).on('click', '.company-quick-search', function() {
        const searchTerm = $(this).data('search');
        $('#companySearchInput').val(searchTerm);
        // Trigger input event to show suggestions
        handleCompanySearchInput();
    });

    // Handle keyboard navigation in suggestions
    $('#companySearchInput').on('keydown', function(e) {
        const $suggestions = $('#companySuggestions');
        if ($suggestions.hasClass('hidden')) return;

    // Window resize handler
    $(window).on('resize', function() {
        // Handle responsive behavior
    });

        const $items = $('.company-suggestion');
        let $selected = $('.company-suggestion.selected');

        if (e.which === 38) { // Up arrow
            e.preventDefault();
            if ($selected.length === 0) {
                $items.last().addClass('selected');
            } else {
                $selected.removeClass('selected');
                const $prev = $selected.prev('.company-suggestion');
                if ($prev.length) {
                    $prev.addClass('selected');
                } else {
                    $items.last().addClass('selected');
                }
            }
        } else if (e.which === 40) { // Down arrow
            e.preventDefault();
            if ($selected.length === 0) {
                $items.first().addClass('selected');
            } else {
                $selected.removeClass('selected');
                const $next = $selected.next('.company-suggestion');
                if ($next.length) {
                    $next.addClass('selected');
                } else {
                    $items.first().addClass('selected');
                }
            }
        } else if (e.which === 13) { // Enter
            if ($selected.length) {
                e.preventDefault();
                const symbol = $selected.data('symbol');
                const name = $selected.data('name');
                selectCompanySuggestion(symbol, name);
            }
        } else if (e.which === 27) { // Escape
            hideSuggestions();
        }
    });

    // Hide suggestions when clicking outside
    $(document).on('click', function(e) {
        if (!$(e.target).closest('#companySearchInput, #companySuggestions').length) {
            hideSuggestions();
        }
    });

    // Clear completed indicator when user interacts with results
    setupResultInteractionListeners();

    // Error reporting modal event listeners
    $('#closeErrorReportingModalBtn, #closeErrorReportingFooterBtn').on('click', closeErrorReportingModal);
    $('#errorReportingModal').on('click', function(e) {
        if (e.target === this) {
            closeErrorReportingModal();
        }
    });

}

// Note: Previously had listeners to auto-clear the sidebar indicator on user interaction
// Now we only clear when user manually clicks on the search tab
function setupResultInteractionListeners() {
}

// --- SIDEBAR FUNCTIONALITY ---
function toggleSidebar() {
    const sidebar = $('.app-sidebar');
    const backdrop = $('.sidebar-backdrop');

    if (sidebar.hasClass('open')) {
        closeSidebar();
    } else {
        openSidebar();
    }
}

function openSidebar() {
    $('.app-sidebar').addClass('open');
    $('.sidebar-backdrop').addClass('open');
    $('body').addClass('overflow-hidden');
}

function closeSidebar() {
    $('.app-sidebar').removeClass('open');
    $('.sidebar-backdrop').removeClass('open');
    $('body').removeClass('overflow-hidden');
}

// Coming Soon modal close handlers
$(document).on('click', '[data-close-coming-soon]', function() {
    $('#comingSoonModal').removeClass('open');
});


// Manual banner controls for testing
window.showMobileBanner = function() {
    const banner = document.getElementById('mobileWarningBanner');
    if (banner) {
        banner.classList.remove('hidden');
        banner.style.display = 'block'; // Force display
        try { localStorage.removeItem('mobileBannerDismissed'); } catch (e) {}
    } else {
    }
};

window.hideMobileBanner = function() {
    const banner = document.getElementById('mobileWarningBanner');
    if (banner) {
        banner.classList.add('hidden');
        try { localStorage.setItem('mobileBannerDismissed', 'true'); } catch (e) {}
    }
};

window.checkMobileBanner = function() {
    const banner = document.getElementById('mobileWarningBanner');
    return banner;
};



// Mobile warning banner functionality
function showMobileBannerIfNeeded() {
    
    try {
        if (localStorage.getItem('mobileBannerDismissed') === 'true') {
            return;
        }
    } catch (e) {}
    
    // Check for mobile devices
    const isSmallViewport = window.innerWidth <= 768 || Math.min(window.innerWidth, window.innerHeight) <= 820;
    const isTouch = ('ontouchstart' in window) || navigator.maxTouchPoints > 0;
    const isMobileUserAgent = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    
    
    if ((isSmallViewport && isTouch) || isMobileUserAgent) {
        const banner = document.getElementById('mobileWarningBanner');
        if (banner) {
            banner.classList.remove('hidden');
            banner.style.display = 'block';
        } else {
        }
    } else {
    }
}

// Handle banner dismissal
$(document).on('click', '#dismissMobileBanner', function() {
    const banner = document.getElementById('mobileWarningBanner');
    if (banner) {
        banner.classList.add('hidden');
        try { localStorage.setItem('mobileBannerDismissed', 'true'); } catch (e) {}
    }
});


// Trigger after DOM ready
$(function() {
    // Show mobile banner only
    setTimeout(showMobileBannerIfNeeded, 100);
});


function handleSidebarNavigation(section) {
    // Remove active class from all menu items
    $('.sidebar-menu-item').removeClass('active');

    // Add active class to clicked item
    $(`.sidebar-menu-item[data-section="${section}"]`).addClass('active');

    // Handle different sections
    switch(section) {
        case 'search':
            switchToSection('search');
            // Clear the sidebar search indicator when user manually clicks on search tab
            updateSidebarSearchIndicator('cleared');
            // Expand suggestions when switching to screener section
            setTimeout(() => {
                const categoriesSection = $('.screener-categories');
                if (categoriesSection.hasClass('collapsed')) {
                    expandScreenerSuggestions();
                }
            }, 100);
            if (window.innerWidth <= 1024) {
                closeSidebar();
            }
            break;

        case 'screens':
            switchToSection('screens');
            if (window.innerWidth <= 1024) closeSidebar();
            break;

        case 'companies':
            switchToSection('companies');
            if (window.innerWidth <= 1024) closeSidebar();
            break;

        case 'charting':
            switchToSection('charting');
            initializeChartingSection();
            if (window.innerWidth <= 1024) closeSidebar();
            break;


        case 'chat':
            switchToSection('chat');
            // Show chat controls in navbar
            $('#chatNavControls').removeClass('hidden');
            // initializeChat(); // DISABLED - using new chat.js
            break;

        default:
    }
}

// Close sidebar on window resize if screen becomes large
$(window).on('resize', function() {
    if (window.innerWidth > 1024) {
        closeSidebar();
    }
});

// =============================================================================
// COMPANY SEARCH FUNCTIONALITY
// =============================================================================

let searchTimeout = null;

// Real-time search suggestions with debouncing
function handleCompanySearchInput() {
    const query = $('#companySearchInput').val().trim();

    // Clear previous timeout
    if (searchTimeout) {
        clearTimeout(searchTimeout);
    }

    // Hide suggestions if query is too short
    if (query.length < 2) {
        hideSuggestions();
        return;
    }

    // Check character limit
    if (query.length > 100) {
        showToast('Company search query cannot exceed 100 characters', 'warning');
        return;
    }

    // Debounce the search to avoid too many API calls
    searchTimeout = setTimeout(() => {
        fetchCompanySuggestions(query);
    }, 300); // 300ms delay
}

// Fetch company suggestions for autocomplete
async function fetchCompanySuggestions(query) {
    try {
        const token = localStorage.getItem('authToken');
        if (!token) {
            logout();
            return;
        }

        const params = new URLSearchParams({
            query: query,
            limit: 8 // Fewer results for suggestions
        });

        const response = await fetch(`${CONFIG.apiBaseUrl}/companies/search?${params.toString()}`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (response.status === 401) {
            logout();
            return;
        }

        const result = await response.json();

        if (response.ok && result.companies) {
            currentSuggestions = result.companies;
            showSuggestions(result.companies);
        } else {
            hideSuggestions();
        }

    } catch (error) {
        hideSuggestions();
    }
}

// Show suggestions dropdown
function showSuggestions(companies) {
    const $suggestions = $('#companySuggestions');

    if (!companies || companies.length === 0) {
        hideSuggestions();
        return;
    }

    let suggestionsHtml = '';

    companies.forEach(company => {
        const marketCapText = company.marketCap ? ChartUtils.formatMarketCap(company.marketCap) : '';
        const sectorText = company.sector ? company.sector : '';

        suggestionsHtml += `
            <div class="company-suggestion p-3 hover:bg-bg-secondary cursor-pointer border-b border-border-primary last:border-b-0" data-symbol="${company.symbol}" data-name="${company.companyName}">
                <div class="flex items-center justify-between">
                    <div class="flex items-center gap-3">
                        <div class="flex flex-col">
                            <div class="flex items-center gap-2">
                                <span class="font-semibold text-text-primary">${company.symbol}</span>
                                <span class="text-sm text-text-secondary">${company.exchangeShortName || ''}</span>
                            </div>
                            <span class="text-sm text-text-secondary line-clamp-1">${company.companyName}</span>
                        </div>
                    </div>
                    <div class="flex flex-col items-end text-xs text-text-tertiary">
                        ${marketCapText ? `<span>${marketCapText}</span>` : ''}
                        ${sectorText ? `<span>${sectorText}</span>` : ''}
                    </div>
                </div>
            </div>
        `;
    });

    $suggestions.html(suggestionsHtml).removeClass('hidden');

    // Force bring to front with extreme z-index
    $suggestions.css({
        'z-index': '999999999',
        'position': 'absolute',
        'background': 'var(--bg-primary)',
        'border': '1px solid var(--border-primary)',
        'box-shadow': '0 4px 12px rgba(0, 0, 0, 0.15)'
    });

    // Add click handlers for suggestions
    $('.company-suggestion').on('click', function() {
        const symbol = $(this).data('symbol');
        const name = $(this).data('name');
        selectCompanySuggestion(symbol, name);
    });
}

// Hide suggestions dropdown
function hideSuggestions() {
    $('#companySuggestions').addClass('hidden').empty();
}

// Select a company suggestion
function selectCompanySuggestion(symbol, name) {
    $('#companySearchInput').val(name);
    hideSuggestions();
    viewCompanyDetails(symbol);
}

// Use formatMarketCap from ChartUtils instead of local implementation
// Note: formatMarketCap is available as ChartUtils.formatMarketCap

// View company details - load full profile
async function viewCompanyDetails(symbol) {
    loadCompanyProfile(symbol);
}



function initializeCompaniesSection() {
    // Reset to empty state
    clearCompanySearch();

    // Hide company profile when switching to companies section
    $('#companyProfileSection').addClass('hidden');

    // Show empty state by default
    hideAllCompanyContainers();
    $('#companiesEmptyContainer').removeClass('hidden');

}

// =============================================================================
// COMPANY PROFILE FUNCTIONALITY
// =============================================================================

let currentCompanyData = null;

// NEW: Uniform tab system setup and event handlers
function setupUniformTabEvents() {

    // Remove any existing event handlers to avoid conflicts
    $(document).off('click', '.uniform-tab-btn');
    $(document).off('click', '.financial-btn'); // Note: .financial-btn no longer used

    // Main tab switching
    $(document).on('click', '.uniform-tab-btn', function() {
        const $btn = $(this);
        if ($btn.hasClass('disabled') || $btn.attr('aria-disabled') === 'true') {
            return;
        }
        const tabName = $btn.data('uniform-tab');
        switchUniformTab(tabName);
    });

    // Financial sub-tab switching removed - using unified view

}

// NEW: Uniform tab switching logic
function switchUniformTab(tabName) {

    // Update tab button states
    $('.uniform-tab-btn').removeClass('active');
    $(`.uniform-tab-btn[data-uniform-tab="${tabName}"]`).addClass('active');

    // Disable other tabs visually and functionally
    $('.uniform-tab-btn').each(function() {
        const isActive = $(this).data('uniform-tab') === tabName;
        if (isActive) {
            $(this).removeClass('disabled').attr('aria-disabled', 'false');
        } else {
            $(this).addClass('disabled').attr('aria-disabled', 'true');
        }
    });

    // Update tab content visibility
    $('.uniform-tab-content').removeClass('active');
    $(`.uniform-tab-content[data-uniform-content="${tabName}"]`).addClass('active');

    // Load data based on tab
    if (currentCompanyData && currentCompanyData.symbol) {
        const symbol = currentCompanyData.symbol;

        switch(tabName) {
            case 'financials':
                // Load unified financial data (all three statements)
                loadUniformFinancialData(symbol);
                break;
            case 'ttm':
                // Load both current TTM data and historical data
                loadUniformTTMData(symbol);
                loadUniformTTMHistory(symbol, 5); // Load last 5 quarters
                break;
            case 'segments':
                loadUniformProductSegments(symbol);
                break;
            case 'geography':
                loadUniformGeographicSegments(symbol);
                break;
            default:
        }
    }

}

// Financial sub-tab switching removed - using unified financial view

// NEW: Uniform data loading functions
async function loadUniformFinancialData(symbol) {

    try {
        const token = localStorage.getItem('authToken');
        if (!token) {
            logout();
            return;
        }

        // Load all financial statements in parallel
        const [incomeResponse, balanceResponse, cashflowResponse] = await Promise.all([
            fetch(`${CONFIG.apiBaseUrl}/companies/${symbol}/income-statement?years=5`, {
                headers: { 'Authorization': `Bearer ${token}` }
            }),
            fetch(`${CONFIG.apiBaseUrl}/companies/${symbol}/balance-sheet?years=5`, {
                headers: { 'Authorization': `Bearer ${token}` }
            }),
            fetch(`${CONFIG.apiBaseUrl}/companies/${symbol}/cash-flow?years=5`, {
                headers: { 'Authorization': `Bearer ${token}` }
            })
        ]);

        if (incomeResponse.ok) {
            const incomeData = await incomeResponse.json();
            renderUniformFinancialTable('income', incomeData);
        } else {
            showUniformError('financials', 'Failed to load income statement');
        }

        if (balanceResponse.ok) {
            const balanceData = await balanceResponse.json();
            renderUniformFinancialTable('balance', balanceData);
        } else {
            showUniformError('financials', 'Failed to load balance sheet');
        }

        if (cashflowResponse.ok) {
            const cashflowData = await cashflowResponse.json();
            renderUniformFinancialTable('cashflow', cashflowData);
        } else {
            showUniformError('financials', 'Failed to load cash flow');
        }

    } catch (error) {
        showUniformError('financials', 'Failed to load financial data');
    }
}

async function loadUniformTTMData(symbol) {

    try {
        const token = localStorage.getItem('authToken');
        if (!token) {
            logout();
            return;
        }

        const response = await fetch(`${CONFIG.apiBaseUrl}/companies/${symbol}/ttm-metrics`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });

        if (response.ok) {
            const data = await response.json();
            renderUniformTTMData(data.metrics || {});
        } else {
            showUniformError('ttm', 'Failed to load TTM metrics');
        }

    } catch (error) {
        showUniformError('ttm', 'Failed to load TTM metrics');
    }
}

async function loadUniformTTMHistory(symbol, quarters = 5) {

    try {
        const token = localStorage.getItem('authToken');
        if (!token) {
            logout();
            return;
        }

        const response = await fetch(`${CONFIG.apiBaseUrl}/companies/${symbol}/ttm-history?quarters=${quarters}`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });

        if (response.ok) {
            const data = await response.json();
            renderUniformTTMHistory(data.quarters || [], symbol);
        } else {
            showUniformError('ttm', 'Failed to load TTM history');
        }

    } catch (error) {
        showUniformError('ttm', 'Failed to load TTM history');
    }
}

async function loadUniformProductSegments(symbol) {

    try {
        const token = localStorage.getItem('authToken');
        if (!token) {
            logout();
            return;
        }

        const response = await fetch(`${CONFIG.apiBaseUrl}/companies/${symbol}/product-segments`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });

        if (response.status === 401) {
            logout();
            return;
        }

        if (response.ok) {
            const data = await response.json();
            renderUniformSegmentsTable('product', data.segments || []);
        } else {
            showUniformError('segments', 'No product segment data available');
        }

    } catch (error) {
        showUniformError('segments', 'Failed to load product segments');
    }
}

async function loadUniformGeographicSegments(symbol) {

    try {
        const token = localStorage.getItem('authToken');
        if (!token) {
            logout();
            return;
        }

        const response = await fetch(`${CONFIG.apiBaseUrl}/companies/${symbol}/geographic-segments`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });

        if (response.ok) {
            const data = await response.json();
            renderUniformSegmentsTable('geographic', data.segments || []);
        } else {
            showUniformError('geography', 'No geographic segment data available');
        }

    } catch (error) {
        showUniformError('geography', 'Failed to load geographic segments');
    }
}

// NEW: Uniform rendering functions
function renderUniformTTMData(metrics) {

    // Income Statement TTM data - ENHANCED with YoY growth metrics
    const incomeHtml = `
        <div class="ttm-item">
            <span class="ttm-label">Revenue (TTM)</span>
            <span class="ttm-value">${formatCurrency(metrics.revenueTTM, 'revenue') || '--'}</span>
            ${metrics.revenue_yoy_growth_pct !== null && metrics.revenue_yoy_growth_pct !== undefined ? `<span class="ttm-growth ${metrics.revenue_yoy_growth_pct >= 0 ? 'positive' : 'negative'}">(${metrics.revenue_yoy_growth_pct >= 0 ? '+' : ''}${metrics.revenue_yoy_growth_pct}%)</span>` : ''}
        </div>
        <div class="ttm-item">
            <span class="ttm-label">Cost of Revenue (TTM)</span>
            <span class="ttm-value">${formatCurrency(metrics.costOfRevenueTTM, 'costOfRevenue') || '--'}</span>
        </div>
        <div class="ttm-item">
            <span class="ttm-label">Gross Profit (TTM)</span>
            <span class="ttm-value ${getFinancialColorClass(metrics.grossProfitTTM, 'grossProfit')}">${formatCurrency(metrics.grossProfitTTM, 'grossProfit') || '--'}</span>
            ${metrics.grossProfit_yoy_growth_pct !== null && metrics.grossProfit_yoy_growth_pct !== undefined ? `<span class="ttm-growth ${metrics.grossProfit_yoy_growth_pct >= 0 ? 'positive' : 'negative'}">(${metrics.grossProfit_yoy_growth_pct >= 0 ? '+' : ''}${metrics.grossProfit_yoy_growth_pct}%)</span>` : ''}
        </div>
        <div class="ttm-item">
            <span class="ttm-label">Operating Expenses (TTM)</span>
            <span class="ttm-value">${formatCurrency(metrics.operatingExpensesTTM, 'operatingExpenses') || '--'}</span>
        </div>
        <div class="ttm-item">
            <span class="ttm-label">Operating Income (TTM)</span>
            <span class="ttm-value ${getFinancialColorClass(metrics.operatingIncomeTTM, 'operatingIncome')}">${formatCurrency(metrics.operatingIncomeTTM, 'operatingIncome') || '--'}</span>
            ${metrics.operatingIncome_yoy_growth_pct !== null && metrics.operatingIncome_yoy_growth_pct !== undefined ? `<span class="ttm-growth ${metrics.operatingIncome_yoy_growth_pct >= 0 ? 'positive' : 'negative'}">(${metrics.operatingIncome_yoy_growth_pct >= 0 ? '+' : ''}${metrics.operatingIncome_yoy_growth_pct}%)</span>` : ''}
        </div>
        <div class="ttm-item">
            <span class="ttm-label">Income Before Tax (TTM)</span>
            <span class="ttm-value ${getFinancialColorClass(metrics.incomeBeforeTaxTTM, 'incomeBeforeTax')}">${formatCurrency(metrics.incomeBeforeTaxTTM, 'incomeBeforeTax') || '--'}</span>
        </div>
        <div class="ttm-item">
            <span class="ttm-label">Income Tax Expense (TTM)</span>
            <span class="ttm-value">${formatCurrency(metrics.incomeTaxExpenseTTM, 'incomeTaxExpense') || '--'}</span>
        </div>
        <div class="ttm-item">
            <span class="ttm-label">Net Income (TTM)</span>
            <span class="ttm-value ${getFinancialColorClass(metrics.netIncomeTTM, 'netIncome')}">${formatCurrency(metrics.netIncomeTTM, 'netIncome') || '--'}</span>
            ${metrics.netIncome_yoy_growth_pct !== null && metrics.netIncome_yoy_growth_pct !== undefined ? `<span class="ttm-growth ${metrics.netIncome_yoy_growth_pct >= 0 ? 'positive' : 'negative'}">(${metrics.netIncome_yoy_growth_pct >= 0 ? '+' : ''}${metrics.netIncome_yoy_growth_pct}%)</span>` : ''}
        </div>
        <div class="ttm-item">
            <span class="ttm-label">Earnings Per Share (TTM)</span>
            <span class="ttm-value">${formatCurrency(metrics.epsTTM, 'eps') || '--'}</span>
        </div>
        <div class="ttm-item">
            <span class="ttm-label">EPS Diluted (TTM)</span>
            <span class="ttm-value">${formatCurrency(metrics.epsDilutedTTM, 'epsDiluted') || '--'}</span>
        </div>
    `;

    // Balance Sheet TTM data - ENHANCED with YoY growth metrics
    const balanceHtml = `
        <div class="ttm-item">
            <span class="ttm-label">Cash & Cash Equivalents (TTM)</span>
            <span class="ttm-value">${formatCurrency(metrics.cashTTM, 'cash') || '--'}</span>
            ${metrics.cash_yoy_growth_pct !== null && metrics.cash_yoy_growth_pct !== undefined ? `<span class="ttm-growth ${metrics.cash_yoy_growth_pct >= 0 ? 'positive' : 'negative'}">(${metrics.cash_yoy_growth_pct >= 0 ? '+' : ''}${metrics.cash_yoy_growth_pct}%)</span>` : ''}
        </div>
        <div class="ttm-item">
            <span class="ttm-label">Current Assets (TTM)</span>
            <span class="ttm-value">${formatCurrency(metrics.currentAssetsTTM, 'currentAssets') || '--'}</span>
        </div>
        <div class="ttm-item">
            <span class="ttm-label">Total Assets (TTM)</span>
            <span class="ttm-value">${formatCurrency(metrics.totalAssetsTTM, 'totalAssets') || '--'}</span>
            ${metrics.totalAssets_yoy_growth_pct !== null && metrics.totalAssets_yoy_growth_pct !== undefined ? `<span class="ttm-growth ${metrics.totalAssets_yoy_growth_pct >= 0 ? 'positive' : 'negative'}">(${metrics.totalAssets_yoy_growth_pct >= 0 ? '+' : ''}${metrics.totalAssets_yoy_growth_pct}%)</span>` : ''}
        </div>
        <div class="ttm-item">
            <span class="ttm-label">Current Liabilities (TTM)</span>
            <span class="ttm-value">${formatCurrency(metrics.currentLiabilitiesTTM, 'currentLiabilities') || '--'}</span>
        </div>
        <div class="ttm-item">
            <span class="ttm-label">Long-term Debt (TTM)</span>
            <span class="ttm-value">${formatCurrency(metrics.longTermDebtTTM, 'longTermDebt') || '--'}</span>
        </div>
        <div class="ttm-item">
            <span class="ttm-label">Total Liabilities (TTM)</span>
            <span class="ttm-value">${formatCurrency(metrics.totalLiabilitiesTTM, 'totalLiabilities') || '--'}</span>
        </div>
        <div class="ttm-item">
            <span class="ttm-label">Total Debt (TTM)</span>
            <span class="ttm-value">${formatCurrency(metrics.totalDebtTTM, 'totalDebt') || '--'}</span>
        </div>
        <div class="ttm-item">
            <span class="ttm-label">Stockholders Equity (TTM)</span>
            <span class="ttm-value">${formatCurrency(metrics.stockholdersEquityTTM, 'stockholdersEquity') || '--'}</span>
        </div>
        <div class="ttm-item">
            <span class="ttm-label">Working Capital (TTM)</span>
            <span class="ttm-value ${getFinancialColorClass(metrics.currentAssetsTTM - metrics.currentLiabilitiesTTM, 'workingCapital')}">${formatCurrency((metrics.currentAssetsTTM || 0) - (metrics.currentLiabilitiesTTM || 0), 'workingCapital') || '--'}</span>
        </div>
    `;

    // Cash Flow TTM data - ENHANCED with YoY growth metrics
    const cashflowHtml = `
        <div class="ttm-item">
            <span class="ttm-label">Operating Cash Flow (TTM)</span>
            <span class="ttm-value ${getFinancialColorClass(metrics.operatingCashFlowTTM, 'operatingCashFlow')}">${formatCurrency(metrics.operatingCashFlowTTM, 'operatingCashFlow') || '--'}</span>
            ${metrics.operatingCashFlow_yoy_growth_pct !== null && metrics.operatingCashFlow_yoy_growth_pct !== undefined ? `<span class="ttm-growth ${metrics.operatingCashFlow_yoy_growth_pct >= 0 ? 'positive' : 'negative'}">(${metrics.operatingCashFlow_yoy_growth_pct >= 0 ? '+' : ''}${metrics.operatingCashFlow_yoy_growth_pct}%)</span>` : ''}
        </div>
        <div class="ttm-item">
            <span class="ttm-label">Investing Cash Flow (TTM)</span>
            <span class="ttm-value ${getFinancialColorClass(metrics.investingCashFlowTTM, 'investingCashFlow')}">${formatCurrency(metrics.investingCashFlowTTM, 'investingCashFlow') || '--'}</span>
        </div>
        <div class="ttm-item">
            <span class="ttm-label">Financing Cash Flow (TTM)</span>
            <span class="ttm-value ${getFinancialColorClass(metrics.financingCashFlowTTM, 'financingCashFlow')}">${formatCurrency(metrics.financingCashFlowTTM, 'financingCashFlow') || '--'}</span>
        </div>
        <div class="ttm-item">
            <span class="ttm-label">Capital Expenditures (TTM)</span>
            <span class="ttm-value">${formatCurrency(metrics.capexTTM, 'capex') || '--'}</span>
        </div>
        <div class="ttm-item">
            <span class="ttm-label">Free Cash Flow (TTM)</span>
            <span class="ttm-value ${getFinancialColorClass(metrics.freeCashFlowTTM, 'freeCashFlow')}">${formatCurrency(metrics.freeCashFlowTTM, 'freeCashFlow') || '--'}</span>
            ${metrics.freeCashFlow_yoy_growth_pct !== null && metrics.freeCashFlow_yoy_growth_pct !== undefined ? `<span class="ttm-growth ${metrics.freeCashFlow_yoy_growth_pct >= 0 ? 'positive' : 'negative'}">(${metrics.freeCashFlow_yoy_growth_pct >= 0 ? '+' : ''}${metrics.freeCashFlow_yoy_growth_pct}%)</span>` : ''}
        </div>
        <div class="ttm-item">
            <span class="ttm-label">Net Change in Cash (TTM)</span>
            <span class="ttm-value ${getFinancialColorClass(metrics.netChangeInCashTTM, 'netChangeInCash')}">${formatCurrency(metrics.netChangeInCashTTM, 'netChangeInCash') || '--'}</span>
        </div>
    `;

    // Insert the data
    $('#ttmIncomeData').html(incomeHtml);
    $('#ttmBalanceData').html(balanceHtml);
    $('#ttmCashflowData').html(cashflowHtml);

}

function renderUniformTTMHistory(quarters, symbol) {
    // TTM Historical Data rendering is disabled
    // This function is now a no-op
    return;
}

function renderUniformSegmentsTable(type, segments) {

    const containerId = type === 'product' ? '#productSegmentsContainer' : '#geographicSegmentsContainer';
    const container = $(containerId);

    if (!segments || segments.length === 0) {
        container.html(`
            <div class="no-data">
                <i class="fas fa-info-circle"></i>
                <span>No ${type} segment data available</span>
            </div>
        `);
        return;
    }

    const segmentLabel = type === 'product' ? 'Product/Service' : 'Geographic Region';

    const tableHtml = `
        <div class="segments-table-container">
            <div class="segments-header-info">
                <h4 class="segments-title">${type.charAt(0).toUpperCase() + type.slice(1)} Segments (Historical Annual Data)</h4>
                <p class="segments-subtitle">${segments.length} segments found - Latest completed year data</p>
            </div>
            <div class="segments-table-wrapper">
                <table class="segments-data-table">
                    <thead>
                        <tr>
                            <th class="segment-name-col">${segmentLabel}</th>
                            <th class="segment-revenue-col">Revenue</th>
                            <th class="segment-percentage-col">% of Total</th>
                            <th class="segment-year-col">Year</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${segments.map((segment, index) => {
                            const segmentName = segment.segment || segment.name || segment.product || segment.description || segment.segmentName || '--';
                            const revenue = segment.revenue || segment.amount || segment.value || '--';
                            const percentage = segment.percentage || segment.percent || segment.share || '--';
                            const year = segment.year || segment.period || segment.calendarYear || '--';

                            return `
                                <tr class="segment-row">
                                    <td class="segment-name">${segmentName}</td>
                                    <td class="segment-revenue">${formatCurrency(revenue, 'revenue') || revenue || '--'}</td>
                                    <td class="segment-percentage">${formatPercentage(percentage) || percentage || '--'}</td>
                                    <td class="segment-year">${year}</td>
                                </tr>
                            `;
                        }).join('')}
                    </tbody>
                </table>
            </div>
        </div>
    `;

    container.html(tableHtml);
}

function renderUniformFinancialTable(financialType, data) {

    // Map to correct container IDs
    const containerMap = {
        'income': '#incomeStatementData',
        'balance': '#balanceSheetData',
        'cashflow': '#cashFlowData'
    };

    const containerId = containerMap[financialType];
    const container = $(containerId);

    if (!container.length) {
        return;
    }

    if (!data || !data.statements || data.statements.length === 0) {
        container.html(`
            <div class="no-data">
                <i class="fas fa-info-circle"></i>
                <span>No ${financialType} statement data available</span>
            </div>
        `);
        return;
    }

    const statements = data.statements.slice(0, 5); // Last 5 years
    const years = statements.map(s => s.calendarYear);

    // Define key metrics for each statement type
    const metricsMap = {
        'income': [
            { key: 'revenue', label: 'Revenue' },
            { key: 'costOfRevenue', label: 'Cost of Revenue' },
            { key: 'grossProfit', label: 'Gross Profit' },
            { key: 'operatingExpenses', label: 'Operating Expenses' },
            { key: 'operatingIncome', label: 'Operating Income' },
            { key: 'netIncome', label: 'Net Income' },
            { key: 'eps', label: 'Earnings Per Share' }
        ],
        'balance': [
            { key: 'totalAssets', label: 'Total Assets' },
            { key: 'totalCurrentAssets', label: 'Current Assets' },
            { key: 'totalLiabilities', label: 'Total Liabilities' },
            { key: 'totalCurrentLiabilities', label: 'Current Liabilities' },
            { key: 'totalDebt', label: 'Total Debt' },
            { key: 'totalStockholdersEquity', label: 'Shareholders Equity' },
            { key: 'retainedEarnings', label: 'Retained Earnings' }
        ],
        'cashflow': [
            { key: 'operatingCashFlow', label: 'Operating Cash Flow' },
            { key: 'netCashUsedForInvestingActivites', label: 'Investing Cash Flow' },
            { key: 'netCashUsedProvidedByFinancingActivities', label: 'Financing Cash Flow' },
            { key: 'netChangeInCash', label: 'Net Change in Cash' },
            { key: 'freeCashFlow', label: 'Free Cash Flow' },
            { key: 'capitalExpenditure', label: 'Capital Expenditure' }
        ]
    };

    const metrics = metricsMap[financialType] || [];

    // Create responsive financial table
    const tableHtml = `
        <div class="financial-table-container">
            <div class="table-header-info">
                <h4 class="table-title">${financialType.charAt(0).toUpperCase() + financialType.slice(1)} Statement</h4>
                <p class="table-subtitle">${statements.length} years of data (in USD)</p>
            </div>
            <div class="financial-table-wrapper">
                <table class="financial-data-table">
                    <thead>
                        <tr>
                            <th class="metric-header">Metric</th>
                            ${years.map(year => `<th class="year-header">${year}</th>`).join('')}
                        </tr>
                    </thead>
                    <tbody>
                        ${metrics.map(metric => `
                            <tr class="metric-row">
                                <td class="metric-label">${metric.label}</td>
                                ${years.map(year => {
                                    const statement = statements.find(s => s.calendarYear === year);
                                    const value = statement ? statement[metric.key] : null;
                                    const formattedValue = formatFinancialStatementValue(value, metric.key);
                                    const colorClass = getFinancialColorClass(value, metric.key);
                                    return `<td class="metric-value ${colorClass}">${formattedValue}</td>`;
                                }).join('')}
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        </div>
    `;

    container.html(tableHtml);
}

function showUniformError(tabType, message) {

    const errorHtml = `
        <div class="error-message">
            <div class="text-center p-6">
                <div class="w-12 h-12 bg-red-100 dark:bg-red-900/50 rounded-full flex items-center justify-center mx-auto mb-4">
                    <i class="fas fa-exclamation-triangle text-red-500 text-xl"></i>
                </div>
                <h4 class="font-semibold text-text-primary mb-2">Failed to Load Data</h4>
                <p class="text-sm text-text-secondary mb-4">${message}</p>
                <div class="flex flex-col sm:flex-row gap-3 justify-center items-center">
                    <button onclick="retryUniformDataLoad('${tabType}')" class="btn btn-primary text-sm">
                        <i class="fas fa-redo mr-2"></i>
                        <span>Try Again</span>
                    </button>
                    <button onclick="openErrorReportingModal()" class="btn btn-secondary text-sm">
                        <i class="fas fa-life-ring mr-2"></i>
                        <span>Need Help?</span>
                    </button>
                </div>
            </div>
        </div>
    `;

    switch(tabType) {
        case 'ttm':
            $('#ttmIncomeData, #ttmBalanceData, #ttmCashflowData').html(errorHtml);
            break;
        case 'segments':
            $('#productSegmentsContainer').html(errorHtml);
            break;
        case 'geography':
            $('#geographicSegmentsContainer').html(errorHtml);
            break;
        case 'financials':
            $('#incomeStatementData, #balanceSheetData, #cashFlowData').html(errorHtml);
            break;
        default:
    }
}


// Load and display company profile
async function loadCompanyProfile(symbol, companyName = null) {
    try {
        // Track company profile view
        trackEvent('view_company_profile', {
            symbol: symbol,
            company_name: companyName
        });


        // Hide all other containers first (including empty state)
        hideAllCompanyContainers();

        // Show profile section and loading state
        $('#companyProfileSection').removeClass('hidden');
        showCompanyProfileLoading();

        const token = localStorage.getItem('authToken');
        if (!token) {
            logout();
            return;
        }

        // Fetch detailed company data
        const response = await fetch(`${CONFIG.apiBaseUrl}/companies/${symbol}`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (response.status === 401) {
            logout();
            return;
        }

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || `HTTP error! status: ${response.status}`);
        }

        currentCompanyData = result.company;
        renderCompanyProfile(currentCompanyData);

        showToast(`Loaded profile for ${symbol}`, 'success', 2000);

    } catch (error) {
        showCompanyProfileError(error.message);
    }
}

// Show loading state for company profile
function showCompanyProfileLoading() {
    $('#companyProfileSection').html(`
        <div class="profile-loading">
            <div class="flex flex-col items-center justify-center space-y-4">
                <div class="relative w-12 h-12">
                    <div class="w-12 h-12 border-2 border-slate-200 dark:border-slate-700 rounded-full"></div>
                    <div class="w-12 h-12 border-2 border-accent-primary border-t-transparent rounded-full animate-spin absolute top-0 left-0"></div>
                </div>
                <div>
                    <p class="font-medium text-text-primary">Loading company profile...</p>
                    <p class="text-sm text-text-secondary">Fetching detailed information</p>
                </div>
            </div>
        </div>
    `);
}

// Show error state for company profile
function showCompanyProfileError(message) {
    $('#companyProfileSection').html(`
        <div class="p-8 text-center">
            <div class="w-12 h-12 bg-red-100 dark:bg-red-900/50 rounded-full flex items-center justify-center mx-auto mb-4">
                <i class="fas fa-exclamation-triangle text-red-500 text-xl"></i>
            </div>
            <h3 class="text-lg font-semibold text-text-primary mb-2">Failed to Load Profile</h3>
            <p class="text-sm text-text-secondary mb-4">${message}</p>
            <div class="flex flex-col sm:flex-row gap-3 justify-center items-center">
                <button onclick="loadCompanyProfile(currentCompanySymbol)" class="btn btn-primary">
                    <i class="fas fa-redo mr-2"></i>
                    <span>Try Again</span>
                </button>
                <button onclick="openErrorReportingModal()" class="btn btn-secondary text-sm">
                    <i class="fas fa-life-ring mr-2"></i>
                    <span>Need Help?</span>
                </button>
            </div>
        </div>
    `);
}

// NEW: Clean, uniform company profile renderer
function renderCompanyProfile(company) {

    // DEBUG: Check the actual values being received for the problematic fields

    // Test formatNumber with these values

    // Test formatNumber with these values

    // Create the new uniform structure
    $('#companyProfileSection').html(`
        <div class="company-profile-container">
            <!-- Company Header -->
            <div class="company-header">
                <div class="company-info">
                    <div class="company-logo">
                        <i class="fas fa-building"></i>
                    </div>
                    <div class="company-details">
                        <h2 class="company-name">${company.companyName || 'N/A'}</h2>
                        <div class="company-meta">
                            <span class="company-symbol">${company.symbol || 'N/A'}</span>
                            <span class="company-exchange">${company.exchangeShortName || 'N/A'}</span>
                        </div>
                    </div>
                </div>
                <div class="company-price">
                    <div class="price-value">${formatCurrency(company.price) || '--'}</div>
                    <div class="price-change ${getChangeClass(company.changes)}">${formatChange(company.changes) || '--'}</div>
                </div>
            </div>

            <!-- Company Stats -->
            <div class="company-stats">
                <div class="stat-item">
                    <span class="stat-label">Market Cap</span>
                    <span class="stat-value">${ChartUtils.formatMarketCap(company.mktCap || company.marketCap) || '--'}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">P/E Ratio</span>
                    <span class="stat-value">${formatNumber(company.pe || company.priceToEarningsRatio) || '--'}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">P/S Ratio</span>
                    <span class="stat-value">${formatNumber(company.priceToSalesRatio || company.psRatio) || '--'}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">P/FCF Ratio</span>
                    <span class="stat-value">${formatNumber(company.priceToFreeCashFlowRatio || company.pfcfRatio) || '--'}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Revenue (TTM)</span>
                    <span class="stat-value">${formatCurrency(company.revenue || company.revenueTTM) || '--'}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">P/B Ratio</span>
                    <span class="stat-value">${formatNumber(company.priceToBookRatio || company.pbRatio) || '--'}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">EV/EBITDA</span>
                    <span class="stat-value">${formatNumber(company.evToEbitda || company.enterpriseValueOverEBITDA) || '--'}</span>
                </div>
            </div>

            <!-- Uniform Tab System -->
            <div class="uniform-tabs">
                <!-- Tab Navigation -->
                <div class="tab-nav">
                    <button class="uniform-tab-btn active" data-uniform-tab="profile">
                        <i class="fas fa-user"></i>
                        <span>Profile</span>
                    </button>
                    <button class="uniform-tab-btn" data-uniform-tab="metrics">
                        <i class="fas fa-chart-line"></i>
                        <span>Key Metrics</span>
                    </button>
                    <button class="uniform-tab-btn" data-uniform-tab="financials">
                        <i class="fas fa-file-invoice-dollar"></i>
                        <span>Financials</span>
                    </button>
                    <button class="uniform-tab-btn" data-uniform-tab="ttm">
                        <i class="fas fa-calendar-alt"></i>
                        <span>TTM Metrics</span>
                    </button>
                    <button class="uniform-tab-btn" data-uniform-tab="segments">
                        <i class="fas fa-chart-pie"></i>
                        <span>Product Segments</span>
                    </button>
                    <button class="uniform-tab-btn" data-uniform-tab="geography">
                        <i class="fas fa-globe"></i>
                        <span>Geographic Segments</span>
                    </button>
                </div>

                <!-- Tab Content Area -->
                <div class="tab-content-area">
                    <!-- Profile Tab -->
                    <div class="uniform-tab-content active" data-uniform-content="profile">
                        <div class="tab-section">
                            <h3 class="section-title">Company Profile</h3>
                            <div class="profile-grid">
                                <div class="profile-description">
                                    <h4>Business Description</h4>
                                    <p>${company.description || 'No description available.'}</p>
                                </div>
                                <div class="profile-details">
                                    <h4>Company Details</h4>
                                    <div class="detail-list">
                                        <div class="detail-item">
                                            <span class="detail-key">Sector</span>
                                            <span class="detail-value">${company.sector || '--'}</span>
                                        </div>
                                        <div class="detail-item">
                                            <span class="detail-key">Industry</span>
                                            <span class="detail-value">${company.industry || '--'}</span>
                                        </div>
                                        <div class="detail-item">
                                            <span class="detail-key">Country</span>
                                            <span class="detail-value">${company.country || '--'}</span>
                                        </div>
                                        <div class="detail-item">
                                            <span class="detail-key">Website</span>
                                            <span class="detail-value">
                                                ${company.website ? `<a href="${company.website}" target="_blank">${company.website}</a>` : '--'}
                                            </span>
                                        </div>
                                        <div class="detail-item">
                                            <span class="detail-key">CEO</span>
                                            <span class="detail-value">${company.ceo || '--'}</span>
                                        </div>
                                        <div class="detail-item">
                                            <span class="detail-key">IPO Date</span>
                                            <span class="detail-value">${formatDate(company.ipoDate) || '--'}</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Key Metrics Tab -->
                    <div class="uniform-tab-content" data-uniform-content="metrics">
                        <div class="tab-section">
                            <h3 class="section-title">Key Metrics</h3>
                            <div class="metrics-grid">
                                <div class="metrics-group">
                                    <h4>Price & Trading</h4>
                                    <div class="metric-list">
                                        <div class="metric-item">
                                            <span class="metric-key">Previous Close</span>
                                            <span class="metric-value">${formatCurrency(company.previousClose) || '--'}</span>
                                        </div>
                                        <div class="metric-item">
                                            <span class="metric-key">52-Week Range</span>
                                            <span class="metric-value">${formatRangeFromString(company.range) || '--'}</span>
                                        </div>
                                        <div class="metric-item">
                                            <span class="metric-key">Volume</span>
                                            <span class="metric-value">${formatNumber(company.volume) || '--'}</span>
                                        </div>
                                        <div class="metric-item">
                                            <span class="metric-key">Avg Volume (10D)</span>
                                            <span class="metric-value">${formatNumber(company.avgVolume) || '--'}</span>
                                        </div>
                                        <div class="metric-item">
                                            <span class="metric-key">Beta</span>
                                            <span class="metric-value">${formatNumber(company.beta, 2) || '--'}</span>
                                        </div>
                                    </div>
                                </div>
                                <div class="metrics-group">
                                    <h4>Valuation Ratios</h4>
                                    <div class="metric-list">
                                        <div class="metric-item">
                                            <span class="metric-key">P/E Ratio</span>
                                            <span class="metric-value">${formatNumber(company.pe || company.priceToEarningsRatio || company.peRatioKM) || '--'}</span>
                                        </div>
                                        <div class="metric-item">
                                            <span class="metric-key">P/B Ratio</span>
                                            <span class="metric-value">${formatNumber(company.priceToBookRatio || company.pbRatio || company.pbRatioKM) || '--'}</span>
                                        </div>
                                        <div class="metric-item">
                                            <span class="metric-key">P/S Ratio</span>
                                            <span class="metric-value">${formatNumber(company.priceToSalesRatio || company.psRatio) || '--'}</span>
                                        </div>
                                        <div class="metric-item">
                                            <span class="metric-key">P/FCF Ratio</span>
                                            <span class="metric-value">${formatNumber(company.priceToFreeCashFlowRatio || company.pfcfRatio) || '--'}</span>
                                        </div>
                                        <div class="metric-item">
                                            <span class="metric-key">EV/Revenue</span>
                                            <span class="metric-value">${formatNumber(company.evToRevenue || company.evToSales) || '--'}</span>
                                        </div>
                                        <div class="metric-item">
                                            <span class="metric-key">EV/EBITDA</span>
                                            <span class="metric-value">${formatNumber(company.evToEbitda || company.enterpriseValueOverEBITDA) || '--'}</span>
                                        </div>
                                        <div class="metric-item">
                                            <span class="metric-key">PEG Ratio</span>
                                            <span class="metric-value">${formatNumber(company.pegRatio) || '--'}</span>
                                        </div>
                                    </div>
                                </div>
                                <div class="metrics-group">
                                    <h4>Financial Health</h4>
                                    <div class="metric-list">
                                        <div class="metric-item">
                                            <span class="metric-key">Debt-to-Equity</span>
                                            <span class="metric-value">${formatNumber(company.debtEquityRatio || company.debtToEquityRatio) || '--'}</span>
                                        </div>
                                        <div class="metric-item">
                                            <span class="metric-key">Current Ratio</span>
                                            <span class="metric-value">${formatNumber(company.currentRatio) || '--'}</span>
                                        </div>
                                        <div class="metric-item">
                                            <span class="metric-key">Quick Ratio</span>
                                            <span class="metric-value">${formatNumber(company.quickRatio) || '--'}</span>
                                        </div>
                                        <div class="metric-item">
                                            <span class="metric-key">ROE</span>
                                            <span class="metric-value">${formatPercentage(company.roe || company.returnOnEquity || company.roeKM) || '--'}</span>
                                        </div>
                                        <div class="metric-item">
                                            <span class="metric-key">ROA</span>
                                            <span class="metric-value">${formatPercentage(company.roa || company.returnOnAssets) || '--'}</span>
                                        </div>
                                        <div class="metric-item">
                                            <span class="metric-key">ROIC</span>
                                            <span class="metric-value">${formatPercentage(company.roic || company.returnOnCapitalEmployed) || '--'}</span>
                                        </div>
                                    </div>
                                </div>
                                <div class="metrics-group">
                                    <h4>Profitability & Margins</h4>
                                    <div class="metric-list">
                                        <div class="metric-item">
                                            <span class="metric-key">Gross Margin</span>
                                            <span class="metric-value">${formatPercentage(company.grossProfitMargin) || '--'}</span>
                                        </div>
                                        <div class="metric-item">
                                            <span class="metric-key">Operating Margin</span>
                                            <span class="metric-value">${formatPercentage(company.operatingProfitMargin) || '--'}</span>
                                        </div>
                                        <div class="metric-item">
                                            <span class="metric-key">Net Margin</span>
                                            <span class="metric-value">${formatPercentage(company.netProfitMargin) || '--'}</span>
                                        </div>
                                        <div class="metric-item">
                                            <span class="metric-key">Dividend Yield</span>
                                            <span class="metric-value">${formatPercentage(company.dividendYield || company.dividendYieldRatio || company.dividendYieldKM) || '--'}</span>
                                        </div>
                                        <div class="metric-item">
                                            <span class="metric-key">Payout Ratio</span>
                                            <span class="metric-value">${formatPercentage(company.payoutRatio || company.payoutRatioKM) || '--'}</span>
                                        </div>
                                        <div class="metric-item">
                                            <span class="metric-key">Earnings Yield</span>
                                            <span class="metric-value">${formatPercentage(company.earningsYield) || '--'}</span>
                                        </div>
                                        <div class="metric-item">
                                            <span class="metric-key">Free Cash Flow Yield</span>
                                            <span class="metric-value">${formatPercentage(company.freeCashFlowYield) || '--'}</span>
                                        </div>
                                    </div>
                                </div>
                                <div class="metrics-group">
                                    <h4>Size & Scale</h4>
                                    <div class="metric-list">
                                        <div class="metric-item">
                                            <span class="metric-key">Enterprise Value</span>
                                            <span class="metric-value">${formatCurrency(company.enterpriseValue) || '--'}</span>
                                        </div>
                                        <div class="metric-item">
                                            <span class="metric-key">Book Value per Share</span>
                                            <span class="metric-value">${formatCurrency(company.bookValuePerShare) || '--'}</span>
                                        </div>
                                        <div class="metric-item">
                                            <span class="metric-key">Shares Outstanding</span>
                                            <span class="metric-value">${formatLargeNumberFixed2(company.sharesOutstanding) || '--'}</span>
                                        </div>
                                        <div class="metric-item">
                                            <span class="metric-key">Float</span>
                                            <span class="metric-value">${formatLargeNumberFixed2(company.sharesFloat || company.sharesOutstandingDiluted) || '--'}</span>
                                        </div>
                                        <div class="metric-item">
                                            <span class="metric-key">Working Capital</span>
                                            <span class="metric-value">${formatCurrency(company.workingCapital || company.workingCapitalCalculated) || '--'}</span>
                                        </div>
                                        <div class="metric-item">
                                            <span class="metric-key">Cash per Share</span>
                                            <span class="metric-value">${formatCurrency(company.cashPerShare) || '--'}</span>
                                        </div>
                                        <div class="metric-item">
                                            <span class="metric-key">Free Cash Flow per Share</span>
                                            <span class="metric-value">${formatCurrency(company.freeCashFlowPerShare) || '--'}</span>
                                        </div>
                                        <div class="metric-item">
                                            <span class="metric-key">Employees</span>
                                            <span class="metric-value">${formatLargeNumberFixed2(company.fullTimeEmployees) || '--'}</span>
                                        </div>
                                        <div class="metric-item">
                                            <span class="metric-key">Founded</span>
                                            <span class="metric-value">${formatDate(company.ipoDate) || '--'}</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Financials Tab -->
                    <div class="uniform-tab-content" data-uniform-content="financials">
                        <div class="tab-section">
                            <h3 class="section-title">Financial Statements</h3>
                            <div class="financials-unified-container">
                                <!-- Income Statement -->
                                <div id="incomeStatementData">
                                    <div class="data-loading">
                                        <i class="fas fa-spinner fa-spin"></i>
                                        <span>Loading income statement...</span>
                                    </div>
                                </div>

                                <!-- Balance Sheet -->
                                <div id="balanceSheetData">
                                    <div class="data-loading">
                                        <i class="fas fa-spinner fa-spin"></i>
                                        <span>Loading balance sheet...</span>
                                    </div>
                                </div>

                                <!-- Cash Flow -->
                                <div id="cashFlowData">
                                    <div class="data-loading">
                                        <i class="fas fa-spinner fa-spin"></i>
                                        <span>Loading cash flow...</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- TTM Metrics Tab -->
                    <div class="uniform-tab-content" data-uniform-content="ttm">
                        <div class="tab-section">
                            <h3 class="section-title">TTM (Trailing Twelve Months) Metrics</h3>
                            <div class="ttm-container">
                                <div class="ttm-grid">
                                    <div class="ttm-section">
                                        <h4>Income Statement (TTM)</h4>
                                        <div class="ttm-data" id="ttmIncomeData">
                                            <div class="data-loading">
                                                <i class="fas fa-spinner fa-spin"></i>
                                                <span>Loading TTM income data...</span>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="ttm-section">
                                        <h4>Balance Sheet (TTM)</h4>
                                        <div class="ttm-data" id="ttmBalanceData">
                                            <div class="data-loading">
                                                <i class="fas fa-spinner fa-spin"></i>
                                                <span>Loading TTM balance data...</span>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="ttm-section">
                                        <h4>Cash Flow (TTM)</h4>
                                        <div class="ttm-data" id="ttmCashflowData">
                                            <div class="data-loading">
                                                <i class="fas fa-spinner fa-spin"></i>
                                                <span>Loading TTM cashflow data...</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Product Segments Tab -->
                    <div class="uniform-tab-content" data-uniform-content="segments">
                        <div class="tab-section">
                            <h3 class="section-title">Revenue by Product Segment</h3>
                            <div class="segments-container">
                                <div class="segments-data" id="productSegmentsContainer">
                                    <div class="data-loading">
                                        <i class="fas fa-spinner fa-spin"></i>
                                        <span>Loading product segments...</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Geographic Segments Tab -->
                    <div class="uniform-tab-content" data-uniform-content="geography">
                        <div class="tab-section">
                            <h3 class="section-title">Revenue by Geographic Region</h3>
                            <div class="segments-container">
                                <div class="segments-data" id="geographicSegmentsContainer">
                                    <div class="data-loading">
                                        <i class="fas fa-spinner fa-spin"></i>
                                        <span>Loading geographic segments...</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Close Button -->
            <div class="company-actions">
                <button onclick="hideCompanyProfile()" class="btn btn-secondary">
                    <i class="fas fa-times"></i>
                    <span>Close Profile</span>
                </button>
            </div>
        </div>
    `);

    // Set up the new uniform event handlers
    setupUniformTabEvents();

}

// Hide all company containers
function hideAllCompanyContainers() {
    $('#companiesEmptyContainer, #companyProfileSection').addClass('hidden');
}

// Clear company search
function clearCompanySearch() {
    $('#companySearchInput').val('');
    currentCompanies = [];
    lastCompanySearchQuery = '';
    hideSuggestions(); // Hide suggestions when clearing
    if (searchTimeout) {
        clearTimeout(searchTimeout); // Clear any pending search
    }
    $('#companiesCount').text('Type to search companies');
    hideAllCompanyContainers();
    $('#companiesEmptyContainer').removeClass('hidden');

    // Hide company profile when clearing search
    $('#companyProfileSection').addClass('hidden');
}


// Helper Functions
function formatFinancialStatementValue(value, key) {
    if (!value || isNaN(value)) return '--';

    if (key.toLowerCase().includes('eps')) {
        return `$${Number(value).toFixed(2)}`;
    }

    return formatCurrency(value, key);
}

function getFinancialColorClass(value, key) {
    if (!value || isNaN(value)) return '';

    // For most financial metrics, positive is good
    const positiveMetrics = ['revenue', 'grossProfit', 'operatingIncome', 'netIncome', 'freeCashFlow', 'operatingCashFlow', 'roe', 'roa', 'roic'];
    const negativeMetrics = ['costOfRevenue', 'operatingExpenses', 'totalDebt'];

    if (positiveMetrics.some(metric => key.toLowerCase().includes(metric))) {
        return value > 0 ? 'financial-positive' : 'financial-negative';
    } else if (negativeMetrics.some(metric => key.toLowerCase().includes(metric))) {
        return value < 0 ? 'financial-positive' : 'financial-negative';
    }

    return '';
}

function showFinancialError(message) {
    $('#incomeStatementData, #balanceSheetData, #cashFlowData').html(`
        <div class="text-center p-6">
            <div class="w-12 h-12 bg-red-100 dark:bg-red-900/50 rounded-full flex items-center justify-center mx-auto mb-4">
                <i class="fas fa-exclamation-triangle text-red-500 text-xl"></i>
            </div>
            <h4 class="font-semibold text-text-primary mb-2">Failed to Load Financial Data</h4>
            <p class="text-sm text-text-secondary mb-4">${message}</p>
            <div class="flex flex-col sm:flex-row gap-3 justify-center items-center">
                <button onclick="retryFinancialDataLoad()" class="btn btn-primary text-sm">
                    <i class="fas fa-redo mr-2"></i>
                    <span>Try Again</span>
                </button>
                <button onclick="openErrorReportingModal()" class="btn btn-secondary text-sm">
                    <i class="fas fa-life-ring mr-2"></i>
                    <span>Need Help?</span>
                </button>
            </div>
        </div>
    `);
}

function showTTMError(message) {
    $('#ttmProfitabilityData, #ttmEfficiencyData, #ttmLiquidityData').html(`
        <div class="text-center p-6">
            <div class="w-12 h-12 bg-red-100 dark:bg-red-900/50 rounded-full flex items-center justify-center mx-auto mb-4">
                <i class="fas fa-exclamation-triangle text-red-500 text-xl"></i>
            </div>
            <h4 class="font-semibold text-text-primary mb-2">Failed to Load TTM Data</h4>
            <p class="text-sm text-text-secondary mb-4">${message}</p>
            <div class="flex flex-col sm:flex-row gap-3 justify-center items-center">
                <button onclick="retryTTMDataLoad()" class="btn btn-primary text-sm">
                    <i class="fas fa-redo mr-2"></i>
                    <span>Try Again</span>
                </button>
                <button onclick="openErrorReportingModal()" class="btn btn-secondary text-sm">
                    <i class="fas fa-life-ring mr-2"></i>
                    <span>Need Help?</span>
                </button>
            </div>
        </div>
    `);
}

function showSegmentError(type, message) {
    const containerId = type === 'product' ? '#productSegmentsData' : '#geographicSegmentsData';
    $(containerId).html(`
        <div class="text-center p-6">
            <div class="w-12 h-12 bg-red-100 dark:bg-red-900/50 rounded-full flex items-center justify-center mx-auto mb-4">
                <i class="fas fa-exclamation-triangle text-red-500 text-xl"></i>
            </div>
            <h4 class="font-semibold text-text-primary mb-2">Failed to Load ${type === 'product' ? 'Product' : 'Geographic'} Segments</h4>
            <p class="text-sm text-text-secondary mb-4">${message}</p>
            <div class="flex flex-col sm:flex-row gap-3 justify-center items-center">
                <button onclick="retrySegmentDataLoad('${type}')" class="btn btn-primary text-sm">
                    <i class="fas fa-redo mr-2"></i>
                    <span>Try Again</span>
                </button>
                <button onclick="openErrorReportingModal()" class="btn btn-secondary text-sm">
                    <i class="fas fa-life-ring mr-2"></i>
                    <span>Need Help?</span>
                </button>
            </div>
        </div>
    `);
}

// Hide company profile
function hideCompanyProfile() {
    $('#companyProfileSection').addClass('hidden');
    currentCompanyData = null;

    // Show the empty state
    hideAllCompanyContainers();
    $('#companiesEmptyContainer').removeClass('hidden');

    // Update companies count
    $('#companiesCount').text('Type to search companies');
}

// Make function globally available
window.hideCompanyProfile = hideCompanyProfile;

// --- LOGIN ---
async function handleLogin(e) {
    e.preventDefault();
    setLoginLoading(true);
    const username = $('#username').val().trim();
    const password = $('#password').val();

    // Track login attempt
    trackEvent('login_attempt', {
        username_length: username.length
    });

    if (!username || !password) {
        showToast('Please enter both username and password', 'warning');
        setLoginLoading(false);
        return;
    }

    try {
        const response = await fetch(`${CONFIG.apiBaseUrl}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });
        
        // Check if response is ok before trying to parse JSON
        if (!response.ok) {
            let errorMessage = 'Login failed';
            try {
                const errorData = await response.json();
                errorMessage = errorData.detail || errorMessage;
            } catch (parseError) {
                // If JSON parsing fails, use status text
                errorMessage = response.statusText || errorMessage;
            }
            throw new Error(errorMessage);
        }
        
        const data = await response.json();
        localStorage.setItem('authToken', data.access_token);
        localStorage.setItem('currentUser', JSON.stringify(data.user));
        currentUser = data.user;

        // Track successful login
        identifyUser(data.user.id, {
            username: data.user.username,
            email: data.user.email,
            full_name: data.user.full_name,
            company: data.user.company
        });

        trackEvent('user_login_success');

        // Fetch onboarding status immediately after login
        try {
            const response = await fetch(`${CONFIG.apiBaseUrl}/user/onboarding-status`, {
                headers: { 'Authorization': `Bearer ${data.access_token}` }
            });

            if (response.ok) {
                const onboardingData = await response.json();
                currentUser.query_count = onboardingData.query_count;
                localStorage.setItem('currentUser', JSON.stringify(currentUser));
            }
        } catch (error) {
        }

        showMainApp();
        showToast('Logged in successfully!', 'success');
    } catch (error) {
        // Track login error
        trackEvent('user_login_error', {
            error: error.message
        });

        let errorMessage;
        if (error.message.includes('Failed to fetch') || error.message.includes('ERR_NETWORK_CHANGED')) {
            errorMessage = "Network connection error. Please check your internet connection and try again.";
        } else if (error.message.includes('ERR_INTERNET_DISCONNECTED')) {
            errorMessage = "No internet connection. Please check your network and try again.";
        } else if (error.message.includes('ERR_CONNECTION_REFUSED')) {
            errorMessage = "Could not connect to server. Please try again later.";
        } else {
            errorMessage = `Login Failed: ${error.message}`;
        }
        
        showToast(errorMessage, 'error');
    } finally {
        setLoginLoading(false);
    }
}

function setLoginLoading(loading) {
    $('#loginBtnText').text(loading ? 'Signing in...' : 'Sign In');
    $('#loginSpinner').toggleClass('hidden', !loading);
    $('#loginForm button').prop('disabled', loading);
}

// --- PROMPT LIBRARY ---
const promptLibrary = {
    available: [
        {
            query: "Consumer companies that have grown their revenue by more than 20% in last 10 years",
            description: "High growth consumer companies with 20%+ revenue CAGR"
        },
        {
            query: "which 50 billion plus market cap companies have grown their profits by more than 50% in latest quarter",
            description: "Large cap companies with exceptional profit growth"
        },
        {
            query: "Companies with gross margins improvement excluding financials and mining in latest quarter",
            description: "Companies showing margin expansion trends"
        },
        {
            query: "Show me all data center stocks that have increased their R&D spend in last quarter beyond their revenue growth and revenue growth is more than 20%",
            description: "Data center companies with accelerating R&D investment"
        },
        {
            query: "show me all semiconductor companies above billon dollar market cap with their total r&d spend to revenue between 2015-2020 and their revenue cagr from 2020-2024",
            description: "Semiconductor R&D trends and revenue growth analysis"
        },
        {
            query: "What is the Revenue, cash, margins and net income data for last 10 years for $AAPL, $MSFT, $NFLX, $TSLA and $GOOG",
            description: "Comprehensive 10-year financial analysis of tech giants"
        },
        {
            query: "all consumer companies above billion dollar market cap with gross margins above 80% and with their marketing to revenue spend",
            description: "High-margin consumer companies with marketing spend analysis"
        },
        {
            query: "compare crypto mining companies revenue growth from 2015-2020 to 2020-2024",
            description: "Crypto mining revenue growth comparison across periods"
        }
    ],
    comingSoon: [
        {
            query: "Geographic revenue breakdown for multinational tech companies",
            description: "Regional analysis of tech company revenues",
            preview: "Will show revenue by region for major tech companies"
        },
        {
            query: "ESG scores correlation with financial performance",
            description: "Environmental, Social, Governance impact analysis",
            preview: "Analyze ESG ratings vs financial metrics"
        }
    ]
};

function populatePromptLibrary() {
    const container = $('#promptCategoriesContainer').empty();

                // Example queries section
            const availableHtml = `
                <div class="space-y-4">
                    <div class="flex items-center space-x-3">
                        <div class="w-8 h-8 bg-green-100 dark:bg-green-900/50 rounded-full flex items-center justify-center">
                            <i class="fas fa-check text-green-600 dark:text-green-400 text-sm"></i>
                        </div>
                        <div>
                            <h4 class="text-lg font-semibold text-text-primary">Example Queries</h4>
                            <p class="text-sm text-text-secondary">Click any query to run it immediately</p>
                        </div>
                    </div>
                    <div class="grid gap-3">
                        ${promptLibrary.available.map(p => `
                            <button class="prompt-item-btn w-full text-left p-3 bg-bg-tertiary dark:bg-slate-700/60 rounded-lg hover:bg-slate-200/50 dark:hover:bg-slate-700 transition-all border border-transparent hover:border-accent-primary/30" data-query="${p.query}">
                                <div class="font-medium text-sm text-text-primary font-mono">${p.query}</div>
                                <div class="text-xs text-text-secondary mt-0.5">${p.description}</div>
                            </button>
                        `).join('')}
                    </div>
                </div>
            `;

    // Coming soon section
    const comingSoonHtml = `
        <div class="space-y-4 mt-8 pt-8 border-t border-border-primary">
            <div class="flex items-center space-x-3">
                <div class="w-8 h-8 bg-blue-100 dark:bg-blue-900/50 rounded-full flex items-center justify-center">
                    <i class="fas fa-clock text-blue-600 dark:text-blue-400 text-sm"></i>
                </div>
                <div>
                    <h4 class="text-lg font-semibold text-text-primary">Coming Soon</h4>
                    <p class="text-sm text-text-secondary">Geographic and advanced data center queries</p>
                </div>
            </div>
            <div class="grid gap-3">
                ${promptLibrary.comingSoon.map(p => `
                    <div class="w-full text-left p-3 bg-bg-tertiary/50 dark:bg-slate-700/30 rounded-lg border border-dashed border-border-primary cursor-not-allowed opacity-75">
                        <div class="font-medium text-sm text-text-primary/70 font-mono">${p.query}</div>
                        <div class="text-xs text-text-secondary mt-0.5">${p.description}</div>
                        <div class="text-xs text-blue-600 dark:text-blue-400 mt-2 bg-blue-50 dark:bg-blue-900/20 p-2 rounded">
                            <i class="fas fa-info-circle mr-1"></i>
                            <strong>Preview:</strong> ${p.preview}
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;

    container.append(availableHtml + comingSoonHtml);
}

function openPromptLibrary() {
    trackEvent('open_prompt_library');
    populatePromptLibrary();
    $('#promptLibraryModal').removeClass('hidden').addClass('flex');
    $('body').addClass('overflow-hidden');
}

function closePromptLibrary() {
    $('#promptLibraryModal').addClass('hidden').removeClass('flex');
    $('body').removeClass('overflow-hidden');
}

// Expose functions to global scope
window.openPromptLibrary = openPromptLibrary;
window.closePromptLibrary = closePromptLibrary;
window.stopSearch = stopSearch;

// Clear previous query data function
function clearPreviousQueryData() {
    // Clear chart instance
    chartInstance = ChartFunctions.destroyExistingChart(chartInstance);

    // Close chart modal if open
    if (!$('#chartModal').hasClass('hidden')) {
        closeChartModal();
    }

    // Reset data variables
    currentData = [];
    lastApiResponse = null;

    // Reset sorting state
    currentSortColumn = null;
    currentSortDirection = null;

    // Clear chart selectors
    $('#chartXAxis, #chartYAxis, #chartSheetSelector').empty();

    // Don't auto-clear sidebar indicators - let them persist until user clicks search tab

}

// --- STOP SEARCH FUNCTIONALITY ---
async function stopSearch() {
    
    try {
        // Call backend cancellation endpoint
        const token = localStorage.getItem('authToken');
        const response = await fetch(`${CONFIG.apiBaseUrl}/screener/cancel`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        if (response.ok) {
            const result = await response.json();
            if (result.success) {
            } else {
            }
        } else {
        }
    } catch (error) {
    }
    
    // Close EventSource if active
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }
    
    // Close WebSocket if active
    if (wsClient && wsClient.isConnected) {
        wsClient.disconnect();
    }
    
    // Reset search state
    isSearchInProgress = false;
    updateSearchButtonState(false);
    
    // Clear loading containers
    $('#loadingContainer').addClass('hidden');
    $('#aiReasoningPanel').addClass('hidden');
    
    // Reset sidebar indicator
    updateSidebarSearchIndicator('completed');
    
    showToast('Search stopped', 'info');
}

// --- SEARCH BUTTON STATE MANAGEMENT ---
function updateSearchButtonState(isLoading) {
    const searchBtn = $('#searchBtn');
    
    if (isLoading) {
        // Convert to stop button
        searchBtn.removeClass('btn-primary').addClass('btn-secondary');
        searchBtn.find('i').removeClass('fa-search').addClass('fa-stop');
        searchBtn.find('span').text('Stop');
        searchBtn.attr('onclick', 'stopSearch()');
        isSearchInProgress = true;
    } else {
        // Convert back to search button
        searchBtn.removeClass('btn-secondary').addClass('btn-primary');
        searchBtn.find('i').removeClass('fa-stop').addClass('fa-search');
        searchBtn.find('span').text('Execute');
        searchBtn.attr('onclick', 'if($(this).prop(\'disabled\')) return false;');
        isSearchInProgress = false;
    }
}

// --- SEARCH ORCHESTRATION (STREAMING) ---
async function performSearchWithStreaming() {
    const query = $('#queryInput').val().trim();
    if (!query) {
        showToast('Please enter a query.', 'warning');
        return;
    }

    // Track search query
    trackEvent('search_query', {
        query: query,
        query_length: query.length,
        source: isViewingSavedScreen ? 'saved_screen' : 'direct_search'
    });

    // Check character limit
    if (query.length > 1000) {
        showToast('Search query cannot exceed 1000 characters', 'warning');
        return;
    }

    // Check if user has reached their daily limit
    const searchBtn = $('#searchBtn');
    if (searchBtn.prop('disabled')) {
        showToast('You have reached your daily query limit. Please try again tomorrow.', 'warning');
        return;
    }

    // Update button state to show stop button
    updateSearchButtonState(true);

    if (eventSource) {
        eventSource.close();
    }

    currentQuery = query;
    currentQuestion = query; // Store for expand functionality
    currentScreenId = null; // Reset screen tracking
    isViewingSavedScreen = false; // Reset screen viewing flag
    currentPage = 1;

    // Clear previous data when starting new query
    clearPreviousQueryData();

    // Collapse query suggestions when search starts
    collapseScreenerSuggestions();

    setLoadingState(true);
    resetAiReasoningPanel();

    // Prefer WebSocket if available, fallback to EventSource
    if (isUsingWebSocket && wsClient && wsClient.isConnected) {
        try {
            const success = wsClient.startAnalysis(query, {
                page: currentPage,
                page_size: CONFIG.pageSize
            });

            if (!success) {
                throw new Error('Failed to start WebSocket analysis');
            }

            // WebSocket connection toast removed - silent mode
            return; // WebSocket will handle the rest
        } catch (error) {
            isUsingWebSocket = false;
            // WebSocket fallback toast removed - silent mode
        }
    }

    // Fallback to EventSource (existing implementation)
    try {
        const token = localStorage.getItem('authToken');
        const params = new URLSearchParams({
            question: query,
            page: currentPage,
            page_size: CONFIG.pageSize,
        });
        params.append('token', token);

        eventSource = new EventSource(`${CONFIG.apiBaseUrl}/screener/query/stream?${params.toString()}`);

        eventSource.onmessage = function(event) {
            const data = JSON.parse(event.data);

            if (data.type === 'reasoning') {
                handleReasoningEvent(data.event);
            } else if (data.type === 'thinking_steps') {
                handleThinkingStepsSequentially(data.steps);
            } else if (data.type === 'result') {
                // FIXED: Check if this is multi-sheet result
                handleFinalResult(data.data);
                eventSource.close();
                eventSource = null;
                updateSearchButtonState(false);
            } else if (data.type === 'error') {

                // Check if this is a rate limit error
                if (data.error === 'RATE_LIMIT_EXCEEDED') {
                    showRateLimitError(data.message || 'Rate limit exceeded');
                } else {
                    handleStreamingError(data.message || 'An error occurred during processing.');
                }

                eventSource.close();
                eventSource = null;
                updateSearchButtonState(false);
            }
        };

        eventSource.onerror = function(error) {
            handleStreamingError('Failed to connect to the streaming service.');
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }
            updateSearchButtonState(false);
        };
    } catch (error) {
        handleStreamingError('An unexpected error occurred while starting the search.');
    }
}

function handleStreamingError(message) {
    showError(message);
    setLoadingState(false);
    updatePanelStatus('error', 'Analysis failed');

    // Don't auto-clear on errors - let user see the completed state until they click search tab
}

// FIXED: Enhanced result handling with proper multi-sheet detection
function handleFinalResult(result) {

    if (!result) {
        showError("Received empty result from server.", false);
        return;
    }

    // Only show the warning banner for semantic queries
    if (result.is_semantic_query) {
        const warningHtml = `
            <div class="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-700 rounded-lg p-4">
                <div class="flex items-center">
                    <div class="flex-shrink-0">
                        <i class="fas fa-exclamation-triangle text-yellow-400 dark:text-yellow-300"></i>
                    </div>
                    <div class="ml-3">
                        <div class="text-sm text-yellow-700 dark:text-yellow-300">
                            <p>
                                ⚠️ Note: Results for this query may not be exhaustive as of now. Will be improved in coming iterations
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        `;
        $('#warningBanner').html(warningHtml).removeClass('hidden');
    } else {
        $('#warningBanner').addClass('hidden');
    }

    lastApiResponse = result;


    // Handle single-sheet result
    handleSingleSheetResult(result);

    // Update panel status to completed with proper messaging
    const hasError = result.error || (result.data && result.data.error);

    if (hasError) {
        const errorMessage = result.message || (result.data ? result.data.message : 'An unexpected error occurred.');
        const showRetry = result.show_retry === true || (result.data ? result.data.show_retry === true : false);
        updatePanelStatus('error', `Analysis failed: ${result.error || result.data.error}`);
        showError(errorMessage, showRetry);
    } else {
        // Calculate result summary
        const dataCount = result.data_rows ? result.data_rows.length : 0;
        const resultSummary = `${dataCount.toLocaleString()} results found`;
        updatePanelStatus('completed', resultSummary);
    }

    // Show completed state when results are ready
    setLoadingState(false);

    updateSidebarSearchIndicator('completed');

    // Keep reasoning panel visible after completion - don't auto-collapse

    // Update usage indicator after successful query
    if (!hasError) {
        // Refresh usage data from backend to ensure accuracy
        loadUsageData();

        // SIMPLE: Hide onboarding after successful query
        setTimeout(() => {
            hideOnboardingModal();
        }, 1000);
    }
}


// Handle single-sheet results
function handleSingleSheetResult(result) {

    // Extract filing data for transparency
    currentFilingData = extractFilingData(result);
    hasFilingData = currentFilingData && currentFilingData.length > 0;


    currentData = result.data_rows || [];
    const pagination = result.pagination_info;

    if (pagination) {
        totalRecords = pagination.total_records;
        totalPages = pagination.total_pages;
        currentPage = pagination.current_page;
    } else {
        const dataRowCount = result.data_rows ? result.data_rows.length : 0;
        totalRecords = dataRowCount;
        totalPages = 1;
        currentPage = 1;
    }

    if (totalRecords === 0) {
        showNoResults();
    } else {
        renderSingleSheetResults(result);
        hideAllResultContainers();
        $('#singleSheetContainer').removeClass('hidden');

        // Ensure loading state is cleared
        setLoadingState(false);
    }
}


// Existing single-sheet result rendering
function renderSingleSheetResults(apiResult) {
    $('#singleSheetTitle').text(`${totalRecords.toLocaleString()} Companies`);
    $('#singleSheetTime').text(`As of ${moment().format('HH:mm:ss')}`);

    // Update data tab badge
    $('#singleSheetDataCount').text(totalRecords.toLocaleString());

    // Add filing tab if we have filing data
    renderSingleSheetTabs();

    // Render the main data table with current sorting state
    renderTable(apiResult.columns || [], apiResult.data_rows, apiResult.friendly_columns || {});
    renderPagination(apiResult.pagination_info);
}

// Render single-sheet tabs (data only)
function renderSingleSheetTabs() {
    const tabsContainer = $('#singleSheetTabs');

    // Remove any existing filing tabs
    tabsContainer.find('.filing-tab').remove();

    // Add click handlers for single-sheet tabs
    tabsContainer.off('click', '[data-single-tab]'); // Remove old handlers
    tabsContainer.on('click', '[data-single-tab]', function() {
        const tabType = $(this).data('single-tab');
        activateSingleSheetTab(tabType);
    });
}

// Activate single-sheet tab
function activateSingleSheetTab(tabType) {
    // Update tab states
    $('#singleSheetTabs .sheet-tab').removeClass('active');
    $(`#singleSheetTabs [data-single-tab="${tabType}"]`).addClass('active');

    // Update content states
    $('#singleSheetTabContents .sheet-content').removeClass('active').hide();

    if (tabType === 'data') {
        $('#single-data-content').addClass('active').show();
    }
}

function hideAllResultContainers() {
    $('#singleSheetContainer, #noResultsContainer, #errorContainer, #loadingContainer').addClass('hidden');
    // Hide entire suggestions section when showing results
    $('.screener-categories').addClass('hidden');
    // Keep reasoning panel visible - don't hide it
    // Ensure loading state is always cleared when hiding containers
    updateSearchButtonState(false);
}

// --- REAL-TIME AI REASONING ---
function resetAiReasoningPanel() {
    $('#aiReasoningPanel').removeClass('hidden').css('opacity', 1);
    $('#reasoningStepsContainer').empty();

    // Reset the active step tracking
    currentActiveStepId = null;

    // Show initial progress bar
    $('#initialLoadingProgress').removeClass('hidden');
    $('#progressBar').css('width', '10%');

    updatePanelStatus('processing', 'Initializing analysis...');
    $('#aiReasoningPanel').addClass('expanded');

    clearTimeout(window.reasoningPanelTimeout);
}

function handleReasoningEvent(event) {
    const { event_type, message, details } = event;
    const stepId = details?.step || 'general-' + Date.now();

    // Skip individual step messages since we handle them sequentially now
    if (message.startsWith('💭 Step ') || message.startsWith('✓ Step ')) {
        return;
    }

    // Update progress bar based on step
    updateProgressBar(stepId, event_type);

    switch (event_type) {
        case 'step_start':
            addReasoningStep(stepId, message, 'active');
            if (stepId === 'intent_analysis' || stepId === 'strategy_analysis' || stepId === 'multi_sheet_execution' || stepId === 'table_selection' || stepId === 'sql_generation' || stepId === 'sql_execution') {
                updatePanelStatus('processing', message);
            }
            // Add warning for semantic querying
            if (message.toLowerCase().includes('semantic') || (details && details.semantic_processing)) {
                addReasoningInfo(stepId, "⚠️ Note: Results may not be exhaustive as of now. Will be improved in coming iterations");
            }
            break;
        case 'step_complete':
            updateReasoningStep(stepId, message, 'completed');

            if (stepId === 'final_result') {
                if (message.includes('✅') && (message.includes('results found') || message.includes('completed'))) {
                    updatePanelStatus('completed', message.replace('✅ ', ''));
                }
            }
            break;
        case 'step_error':
            // Don't create new reasoning steps for retries - just update existing ones
            const attempt = details?.attempt || 1;
            if (attempt < 3) {
                // For retries, just update the panel status without creating new steps
                updatePanelStatus('processing', 'Analyzing...');
            } else {
                // Final attempt - show actual error
                updateReasoningStep(stepId, message, 'failed');

                if (stepId === 'final_result' || stepId === 'error_handling') {
                    updatePanelStatus('error', message.replace('❌ ', ''));
                }
            }
            break;
        case 'info':
            // Hide progress bar when first real content starts appearing
            if (message.includes('💭') && $('#initialLoadingProgress').is(':visible')) {
                $('#initialLoadingProgress').fadeOut(300);
            }

            // Enhanced handling for AI streaming responses
            if (message.includes('🤖 Starting AI analysis') || message.includes('🔍 Analyzing patterns')) {
                addReasoningStreamingUpdate(stepId, message);
            } else if (message.includes('💭')) {
                addReasoningThinkingStep(stepId, message);
            } else {
                addReasoningInfo(stepId, message);
            }
            break;
        case 'step_warning':
            // Handle warnings (like zero results retry) with a different style
            // Don't create new steps for warnings, just update existing ones
            updatePanelStatus('processing', message);
            break;
    }
}

// New function for handling streaming updates from AI
function addReasoningStreamingUpdate(stepId, message) {
    let stepEl = $(`#step-${stepId}`);
    if (stepEl.length === 0) {
        addReasoningStep(stepId, 'AI Analysis', 'active');
        stepEl = $(`#step-${stepId}`);
    }

    // Clean message to remove extra newlines and whitespace
    const cleanMessage = message.replace(/\n+/g, ' ').replace(/\s+/g, ' ').trim();

    // Update or add streaming indicator
    let streamingIndicator = stepEl.find('.streaming-indicator');
    if (streamingIndicator.length === 0) {
        const detailsContainer = stepEl.find('.step-details');
        detailsContainer.append(`<div class="streaming-indicator text-blue-600 dark:text-blue-300 font-mono text-xs mt-2"></div>`);
        streamingIndicator = stepEl.find('.streaming-indicator');
    }

    // Animate the streaming text with typing effect
    typewriterEffect(streamingIndicator, cleanMessage, 30);

    // Scroll to keep in view
    const container = $('#reasoningStepsContainer')[0];
    setTimeout(() => {
        container.scrollTo({
            top: container.scrollHeight,
            behavior: 'smooth'
        });
    }, 100);
}

// New function for handling AI thinking steps with typing effect
function addReasoningThinkingStep(stepId, message) {
    let stepEl = $(`#step-${stepId}`);
    if (stepEl.length === 0) {
        return;
    }

    const detailsContainer = stepEl.find('.step-details');
    const thinkingId = `thinking-${Date.now()}`;

    // Check if this is a step completion message
    if (message.includes('✓ Step') && message.includes('complete')) {
        detailsContainer.append(`
            <div class="step-completion text-green-600 dark:text-green-400 text-xs mt-1 pl-2 border-l-2 border-green-200 dark:border-green-700 font-medium">
                ${message}
            </div>
        `);
    } else {
        // Clean the message to remove extra newlines and whitespace
        const cleanMessage = message.replace(/\n+/g, ' ').replace(/\s+/g, ' ').trim();

        // Add thinking container with typing animation
        detailsContainer.append(`
            <div id="${thinkingId}" class="thinking-step text-blue-600 dark:text-blue-300 italic text-xs mt-1 pl-2 border-l-2 border-blue-200 dark:border-blue-700">
                <span class="thinking-text"></span>
                <span class="thinking-cursor">|</span>
            </div>
        `);

        const thinkingElement = $(`#${thinkingId} .thinking-text`);
        typewriterEffect(thinkingElement, cleanMessage, 25); // Faster typing for better flow

        // Remove cursor after typing is complete
        setTimeout(() => {
            $(`#${thinkingId} .thinking-cursor`).fadeOut(300);
        }, cleanMessage.length * 25 + 300);
    }

    // Scroll to keep in view
    const container = $('#reasoningStepsContainer')[0];
    setTimeout(() => {
        container.scrollTo({
            top: container.scrollHeight,
            behavior: 'smooth'
        });
    }, 100);
}

// Handle thinking steps sequentially with delays
async function handleThinkingStepsSequentially(steps) {
    for (let i = 0; i < steps.length; i++) {
        const step = steps[i];
        const stepNumber = i + 1;

        // Clean the step text
        const cleanStep = step.replace('..', '.').replace(/\n+/g, ' ').replace(/\s+/g, ' ').trim();
        if (!cleanStep) continue;

        // Add final period if missing
        const finalStep = cleanStep.endsWith('.') ? cleanStep : cleanStep + '.';

        // Show the step with typewriter effect
        const stepId = 'intent_analysis';
        let stepEl = $(`#step-${stepId}`);
        if (stepEl.length === 0) {
            addReasoningStep(stepId, 'Analyzing your request...', 'active');
            stepEl = $(`#step-${stepId}`);
        }

        const detailsContainer = stepEl.find('.step-details');
        const thinkingId = `thinking-${Date.now()}-${i}`;

        // Add thinking container
        detailsContainer.append(`
            <div id="${thinkingId}" class="thinking-step text-blue-600 dark:text-blue-300 italic text-xs mt-1 pl-2 border-l-2 border-blue-200 dark:border-blue-700">
                <span class="thinking-text"></span>
                <span class="thinking-cursor">|</span>
            </div>
        `);

        const thinkingElement = $(`#${thinkingId} .thinking-text`);

        // Start typewriter effect
        await typewriterEffectAsync(thinkingElement, `💭 ${finalStep}`, 20);

        // Remove cursor
        $(`#${thinkingId} .thinking-cursor`).fadeOut(300);

        // Show completion for numbered steps (except the last one)
        if (finalStep.startsWith('Step ') && i < steps.length - 1) {
            await new Promise(resolve => setTimeout(resolve, 300)); // Small delay before completion

            detailsContainer.append(`
                <div class="step-completion text-green-600 dark:text-green-400 text-xs mt-1 pl-2 border-l-2 border-green-200 dark:border-green-700 font-medium">
                    ✓ Step ${stepNumber} complete
                </div>
            `);

            // Delay before next step
            await new Promise(resolve => setTimeout(resolve, 800));
        }

        // Scroll to keep in view
        const container = $('#reasoningStepsContainer')[0];
        container.scrollTo({
            top: container.scrollHeight,
            behavior: 'smooth'
        });
    }
}

// Async version of typewriter effect
function typewriterEffectAsync(element, text, speed = 50) {
    return new Promise((resolve) => {
        element.empty();
        let i = 0;

        function typeCharacter() {
            if (i < text.length) {
                element.append(text.charAt(i));
                i++;
                setTimeout(typeCharacter, speed);
            } else {
                resolve();
            }
        }

        typeCharacter();
    });
}

// Enhanced typewriter effect for streaming text
function typewriterEffect(element, text, speed = 50) {
    element.empty();
    let i = 0;

    function typeCharacter() {
        if (i < text.length) {
            element.append(text.charAt(i));
            i++;
            setTimeout(typeCharacter, speed);
        }
    }

    typeCharacter();
}

// Update progress bar based on analysis step
function updateProgressBar(stepId, eventType) {
    const progressBar = $('#progressBar');
    let progress = 10; // Default starting progress

    // Define progress milestones
    const stepProgress = {
        'intent_analysis': eventType === 'step_start' ? 25 : 40,
        'table_selection': eventType === 'step_start' ? 50 : 60,
        'sql_generation': eventType === 'step_start' ? 70 : 80,
        'sql_execution': eventType === 'step_start' ? 85 : 95,
        'final_result': 100
    };

    progress = stepProgress[stepId] || progress;

    if (eventType === 'step_complete' && stepId === 'final_result') {
        // Hide progress bar when analysis is complete
        setTimeout(() => {
            $('#initialLoadingProgress').fadeOut(300);
        }, 500);
    }

    progressBar.css('width', progress + '%');
}

// Add a global variable to track the currently active step
let currentActiveStepId = null;

function addReasoningStep(stepId, message, status) {
    let iconClass;
    if (status === 'active') {
        // Only show spinner if this is the first active step or if we're switching to a new step
        if (currentActiveStepId === null || currentActiveStepId !== stepId) {
            // Clear any existing active spinner
            if (currentActiveStepId) {
                const existingStepEl = $(`#step-${currentActiveStepId}`);
                if (existingStepEl.length > 0) {
                    const existingIconEl = existingStepEl.find('.step-icon');
                    existingIconEl.removeClass('fa-spinner fa-spin text-blue-500');
                    existingIconEl.addClass('fa-circle text-gray-400');
                }
            }
            iconClass = 'fa-spinner fa-spin text-blue-500';
            currentActiveStepId = stepId;
        } else {
            iconClass = 'fa-circle text-gray-400';
        }
    } else if (status === 'completed') {
        iconClass = 'fa-check-circle text-green-500';
        // Clear active step if this step was active
        if (currentActiveStepId === stepId) {
            currentActiveStepId = null;
        }
    } else if (status === 'retrying') {
        iconClass = 'fa-sync-alt text-yellow-500';
    } else if (status === 'warning') {
        iconClass = 'fa-exclamation-triangle text-yellow-500';
    } else {
        iconClass = 'fa-exclamation-triangle text-red-500';
    }

    let messageClass = 'font-medium text-sm step-title';
    let iconSize = 'w-5 text-center mt-1';

    if (message.includes('💭')) {
        messageClass = 'font-normal text-sm step-title text-blue-600 dark:text-blue-300 italic';
        iconSize = 'w-4 text-center mt-1';
    } else if (message.includes('📋') || message.includes('⚠️')) {
        messageClass = 'font-medium text-sm step-title text-slate-700 dark:text-slate-300';
    }

    const stepHtml = `
        <div id="step-${stepId}" class="reasoning-step p-1 opacity-0 transform translate-y-2 transition-all duration-300 ${status === 'warning' ? 'warning' : ''}">
            <div class="flex items-start space-x-3">
                <i class="step-icon fas ${iconClass} ${iconSize}"></i>
                <div class="flex-1">
                    <p class="${messageClass}">${message}</p>
                    <div class="step-details text-xs text-text-secondary mt-0.5 space-y-0.5"></div>
                </div>
            </div>
        </div>`;

    $('#reasoningStepsContainer').append(stepHtml);

    // Animate in
    setTimeout(() => {
        $(`#step-${stepId}`).removeClass('opacity-0 translate-y-2');
    }, 50);

    const container = $('#reasoningStepsContainer')[0];
    container.scrollTo({
        top: container.scrollHeight,
        behavior: 'smooth'
    });
}

function updateReasoningStep(stepId, message, status) {
    let stepEl = $(`#step-${stepId}`);
    if (stepEl.length === 0) {
        addReasoningStep(stepId, message, status);
        return;
    }

    const iconEl = stepEl.find('.step-icon');

    // Clear existing classes
    iconEl.removeClass('fa-spinner fa-spin text-blue-500 fa-check-circle text-green-500 fa-exclamation-triangle text-red-500 fa-sync-alt text-yellow-500 fa-circle text-gray-400');

    // Set appropriate icon and color based on status
    if (status === 'completed') {
        iconEl.addClass('fa-check-circle text-green-500');
        // Clear active step if this step was active
        if (currentActiveStepId === stepId) {
            currentActiveStepId = null;
        }
    } else if (status === 'failed') {
        iconEl.addClass('fa-exclamation-triangle text-red-500');
        // Clear active step if this step was active
        if (currentActiveStepId === stepId) {
            currentActiveStepId = null;
        }
    } else if (status === 'retrying') {
        iconEl.addClass('fa-sync-alt text-yellow-500');
    } else if (status === 'warning') {
        iconEl.addClass('fa-exclamation-triangle text-yellow-500');
    } else if (status === 'active') {
        // Only show spinner if this is the first active step or if we're switching to a new step
        if (currentActiveStepId === null || currentActiveStepId !== stepId) {
            // Clear any existing active spinner
            if (currentActiveStepId) {
                const existingStepEl = $(`#step-${currentActiveStepId}`);
                if (existingStepEl.length > 0) {
                    const existingIconEl = existingStepEl.find('.step-icon');
                    existingIconEl.removeClass('fa-spinner fa-spin text-blue-500');
                    existingIconEl.addClass('fa-circle text-gray-400');
                }
            }
            iconEl.addClass('fa-spinner fa-spin text-blue-500');
            currentActiveStepId = stepId;
        } else {
            iconEl.addClass('fa-circle text-gray-400');
        }
    } else {
        // Default to inactive state
        iconEl.addClass('fa-circle text-gray-400');
    }

    if (message) {
        stepEl.find('.step-title').text(message);
    }

    // Remove streaming indicator when step completes
    stepEl.find('.streaming-indicator').fadeOut(300);

    // Add/remove warning class based on status
    if (status === 'warning') {
        stepEl.addClass('warning');
    } else {
        stepEl.removeClass('warning');
    }

    if (status === 'completed') {
        stepEl.addClass('animate-pulse');
        setTimeout(() => {
            stepEl.removeClass('animate-pulse');
        }, 1000);
    } else if (status === 'retrying') {
        // Add a subtle animation for retrying status
        stepEl.addClass('animate-pulse');
        setTimeout(() => {
            stepEl.removeClass('animate-pulse');
        }, 2000);
    }
}

function addReasoningInfo(stepId, message) {
    let stepEl = $(`#step-${stepId}`);
    if(stepEl.length === 0) {
        return;
    }
    const detailsContainer = stepEl.find('.step-details');

    let infoClass = 'text-slate-600 dark:text-slate-400';
    if (message.includes('💭')) {
        infoClass = 'text-blue-600 dark:text-blue-300 italic';
    } else if (message.includes('📋')) {
        infoClass = 'text-slate-700 dark:text-slate-300 font-medium';
    } else if (message.includes('⚠️')) {
        infoClass = 'text-yellow-600 dark:text-yellow-400';
    }

    detailsContainer.append(`<div class="${infoClass}">- ${message}</div>`);
    const container = $('#reasoningStepsContainer')[0];
    setTimeout(() => {
        container.scrollTo({
            top: container.scrollHeight,
            behavior: 'smooth'
        });
    }, 100);
}

function updatePanelStatus(status, message) {
    const iconContainer = $('#aiPanelIconContainer');
    const icon = $('#aiPanelIcon');
    const statusDot = $('#aiPanelStatusDot');
    const statusText = $('#aiPanelStatusText');
    const title = $('#aiPanelTitle');
    const subtitle = $('#aiPanelSubtitle');

    // Clear all existing classes
    iconContainer.removeClass('bg-blue-100 bg-green-100 bg-red-100 dark:bg-blue-900 dark:bg-green-900/50 dark:bg-red-900/50');
    icon.removeClass('fa-brain fa-check-circle fa-exclamation-triangle fa-spinner fa-spin text-blue-600 text-green-500 text-red-500 dark:text-blue-300 dark:text-green-300 dark:text-red-300');
    statusDot.removeClass('pulse-dot bg-blue-500 bg-green-500 bg-red-500');

    if (status === 'processing') {
        iconContainer.addClass('bg-blue-100 dark:bg-blue-900');
        // Remove spinner from header - use professional chart icon
        icon.addClass('fas fa-chart-line text-blue-600 dark:text-blue-300');
        title.text('AI Analysis in Progress');
        subtitle.text(message || 'Processing your query...');
        statusText.text('Processing...');
        statusDot.show().addClass('pulse-dot bg-blue-500');
    } else if (status === 'completed') {
        iconContainer.addClass('bg-green-100 dark:bg-green-900/50');
        icon.addClass('fas fa-check-circle text-green-500 dark:text-green-300');
        title.text('Analysis Complete');
        subtitle.text(message || 'Successfully processed your query');
        statusText.text('Completed');
        statusDot.show().removeClass('pulse-dot').addClass('bg-green-500');

        // Hide status dot after animation
        setTimeout(() => {
            statusDot.fadeOut(500);
        }, 2000);
    } else if (status === 'error') {
        iconContainer.addClass('bg-red-100 dark:bg-red-900/50');
        icon.addClass('fas fa-exclamation-triangle text-red-500 dark:text-red-300');
        title.text('Analysis Failed');
        subtitle.text(message || 'An error occurred during processing');
        statusText.text('Error');
        statusDot.show().removeClass('pulse-dot').addClass('bg-red-500');
    }
}

// --- DATA FETCHING (for pagination) ---
async function fetchPage(page) {
    $('#singleSheetTableBody').html(`<tr><td colspan="99" class="text-center p-8"><i class="fas fa-spinner fa-spin text-2xl text-text-tertiary"></i></td></tr>`);

    try {
        const token = localStorage.getItem('authToken');
        if (!token) { logout(); return null; }

        // Use the dedicated pagination endpoint instead of the full query endpoint
        const response = await fetch(`${CONFIG.apiBaseUrl}/screener/query/paginate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({
                question: currentQuery,
                page: page,
                page_size: CONFIG.pageSize
            })
        });

        if (response.status === 401) { logout(); return null; }
        const result = await response.json();

        if (!response.ok || !result.success) {
            throw new Error(result.error || `HTTP error! status: ${response.status}`);
        }

        // Handle pagination result properly - don't call handleFinalResult for pagination
        handlePaginationResult(result, page);
        return result;

    } catch (error) {
        const errorMessage = error.message.includes('Failed to fetch')
            ? "Could not connect to the backend. Please ensure the server is running."
            : `Pagination failed: ${error.message}`;
        showError(errorMessage);
        return null;
    }
}

// Handle pagination results specifically
function handlePaginationResult(result, page) {

    // Update the lastApiResponse with the new page data
    lastApiResponse = result;

    // Update current data and pagination info
    currentData = result.data_rows || [];
    const pagination = result.pagination_info;

    if (pagination) {
        totalRecords = pagination.total_records;
        totalPages = pagination.total_pages;
        currentPage = pagination.current_page;
    } else {
        const dataRowCount = result.data_rows ? result.data_rows.length : 0;
        totalRecords = dataRowCount;
        totalPages = 1;
        currentPage = page;
    }

    // Sorting state persists across pages - no need to clear it

    if (totalRecords === 0) {
        showNoResults();
    } else {
        // Re-render the results with the new page data
        renderSingleSheetResults(result);
        hideAllResultContainers();
        $('#singleSheetContainer').removeClass('hidden');
    }
}


// --- UI RENDERING ---
function setLoadingState(isLoading) {
    if (isLoading) {
        // Hide entire suggestions section, show AI reasoning panel
        $('.screener-categories').addClass('hidden');
        $('#expandSuggestionsBtn').addClass('hidden');
        $('#aiReasoningPanel').removeClass('hidden');
        
        // Hide other result containers
        $('#singleSheetContainer, #noResultsContainer, #errorContainer, #loadingContainer').addClass('hidden');
        
        updateSearchButtonState(true);

        // Update sidebar search indicator to loading
        updateSidebarSearchIndicator('loading');

        // Safety timeout to prevent loading state from getting stuck (5 minutes)
        clearTimeout(window.loadingSafetyTimeout);
        window.loadingSafetyTimeout = setTimeout(() => {
            forceClearLoadingState();
        }, 300000); // 5 minutes
    } else {
        // Hide loading container but keep reasoning panel visible
        $('#loadingContainer').addClass('hidden');
        updateSearchButtonState(false);

        // Clear safety timeout
        clearTimeout(window.loadingSafetyTimeout);

        // Don't clear sidebar here - it will be set to 'completed' in handleFinalResult
    }
}

// Global function to force clear loading state (safety function)
function forceClearLoadingState() {
    $('#loadingContainer').addClass('hidden');
    updateSearchButtonState(false);
    updateSidebarSearchIndicator('completed');
}

function updateSidebarSearchIndicator(state) {

    if (state === 'loading' || state === true) {
        // LOADING STATE - Transform search icon into spinner (keep same background)

        $('.sidebar-menu-item[data-section="search"]').each(function() {
            const $item = $(this);
            const $icon = $item.find('i').first(); // Get the first icon element

            // Transform the existing icon into a spinner (keep original icon color scheme)
            $icon.removeClass('fa-filter fa-check-circle text-green-500')
                 .addClass('fa-spinner fa-spin');

            // DON'T add search-active class - keep original styling
        });


    } else if (state === 'completed') {
        // COMPLETED STATE - Transform icon into checkmark (keep same background)

        $('.sidebar-menu-item[data-section="search"]').each(function() {
            const $item = $(this);
            const $icon = $item.find('i').first(); // Get the first icon element

            // Transform the existing icon into a blue checkmark (same as active tab color)
            $icon.removeClass('fa-filter fa-spinner fa-spin')
                 .addClass('fa-check-circle')
                 .css('color', '#3b82f6'); // Same blue as active tab color

            // DON'T add search-active class - keep original styling
        });


    } else {
        // CLEARING ALL STATES - Restore normal search icon (keep same background)

        $('.sidebar-menu-item[data-section="search"]').each(function() {
            const $item = $(this);
            const $icon = $item.find('i').first(); // Get the first icon element

            // Restore the original search icon and remove any color overrides
            $icon.removeClass('fa-spinner fa-spin fa-check-circle text-green-500')
                 .addClass('fa-filter')
                 .css('color', ''); // Remove any inline color styling

            // DON'T remove any classes - keep original styling

        });

    }
}

function renderPagination(paginationInfo) {
    if (!paginationInfo || paginationInfo.total_pages <= 1) {
        $('#singleSheetPaginationContainer').addClass('hidden');
        return;
    }
    const { current_page, total_pages, total_records, showing_from, showing_to } = paginationInfo;
    $('#singleSheetPaginationInfo').text(`Showing ${showing_from.toLocaleString()}-${showing_to.toLocaleString()} of ${total_records.toLocaleString()}`);
    $('#pageJumpInput').val(current_page).attr('max', total_pages);
    $('#pageTotal').text(`of ${total_pages.toLocaleString()}`);
    $('#firstPageBtn, #prevPageBtn').prop('disabled', current_page === 1);
    $('#lastPageBtn, #nextPageBtn').prop('disabled', current_page === total_pages);
    $('#singleSheetPaginationContainer').removeClass('hidden');
}

function showNoResults() {
    hideAllResultContainers();
    $('#noResultsContainer').removeClass('hidden');

    // Ensure loading state is cleared
    setLoadingState(false);

    // Don't auto-clear sidebar indicator - let user see completed state until they click search tab
}

function showError(message, showRetry = false) {

    // Check if this is a rate limit error
    const isRateLimitError = message.toLowerCase().includes('rate limit') ||
                            message.toLowerCase().includes('daily limit') ||
                            message.toLowerCase().includes('too many requests') ||
                            message.toLowerCase().includes('429');

    if (isRateLimitError) {
        showRateLimitError(message);
        return;
    }

    // Reset error container to original structure for non-rate-limit errors
    resetErrorContainer();

    $('#errorMessage').text(message);
    hideAllResultContainers();
    $('#errorContainer').removeClass('hidden');

    // Ensure loading state is cleared
    setLoadingState(false);

    // Always show retry button for search errors
    $('#retryButton').removeClass('hidden');

    // Don't auto-clear sidebar indicator - let user see completed state until they click search tab
}

// Function to reset error container to original structure
function resetErrorContainer() {
    const originalErrorContainer = `
        <div class="mx-auto w-12 h-12 rounded-full bg-red-100 dark:bg-red-900/50 flex items-center justify-center mb-4">
            <i class="fas fa-exclamation-triangle text-2xl text-red-500 dark:text-red-400"></i>
        </div>
        <h3 class="text-lg font-semibold text-text-primary mb-1">Search Failed</h3>
        <p id="errorMessage" class="text-sm text-text-secondary whitespace-pre-wrap mb-4"></p>
        <div class="flex flex-col sm:flex-row gap-3 justify-center items-center">
            <button id="retryButton" class="btn btn-primary" onclick="retryLastQuery()">
                <i class="fas fa-redo mr-2"></i>
                <span>Try Again</span>
            </button>
            <button class="btn btn-secondary text-sm help-btn" onclick="openErrorReportingModal()">
                <i class="fas fa-life-ring mr-2"></i>
                <span>Need Help?</span>
            </button>
        </div>
    `;

    $('#errorContainer').html(originalErrorContainer);
}

function showRateLimitError(message) {

    // Hide all result containers
    hideAllResultContainers();

    // Clear loading state
    setLoadingState(false);

    // Update panel status
    updatePanelStatus('error', 'Monthly limit reached');

    // Hide the usage indicator completely when rate limit is reached
    $('#usageIndicator').addClass('hidden');

    // Show rate limit specific error container
    $('#errorContainer').removeClass('hidden');

    // Create a professional rate limit error message
    const isMonthlyLimit = message.toLowerCase().includes('monthly') || message.toLowerCase().includes('month');
    const title = isMonthlyLimit ? 'Monthly Query Limit Reached' : 'Query Limit Reached';
    const description = isMonthlyLimit
        ? 'You\'ve used all 20 queries for this month. Your limit will reset at the beginning of next month.'
        : 'You\'ve reached your query limit. Please try again later.';

    const rateLimitMessage = `
        <div class="rate-limit-error-container">
            <div class="rate-limit-header">
                <div class="rate-limit-icon">
                    <i class="fas fa-hourglass-end"></i>
                </div>
                <h3 class="rate-limit-title">${title}</h3>
                <p class="rate-limit-description">${description}</p>
            </div>

            <div class="rate-limit-details">
                <div class="rate-limit-card">
                    <div class="rate-limit-card-icon">
                        <i class="fas fa-calendar-alt"></i>
                    </div>
                    <div class="rate-limit-card-content">
                        <h4>Monthly Limit</h4>
                        <p>20 queries per month</p>
                    </div>
                </div>

                <div class="rate-limit-card">
                    <div class="rate-limit-card-icon">
                        <i class="fas fa-clock"></i>
                    </div>
                    <div class="rate-limit-card-content">
                        <h4>Reset Date</h4>
                        <p>Beginning of next month</p>
                    </div>
                </div>

                <div class="rate-limit-card">
                    <div class="rate-limit-card-icon">
                        <i class="fas fa-chart-line"></i>
                    </div>
                    <div class="rate-limit-card-content">
                        <h4>Current Usage</h4>
                        <p>20/20 queries used</p>
                    </div>
                </div>
            </div>

            <div class="rate-limit-actions">
                <div class="rate-limit-info">
                    <i class="fas fa-info-circle"></i>
                    <span>Your query limit will automatically reset at the beginning of next month.</span>
                </div>

                <div class="rate-limit-buttons">
                    <button onclick="openUsageModal()" class="btn btn-secondary">
                        <i class="fas fa-chart-line mr-2"></i>
                        <span>View Usage Details</span>
                    </button>
                    <button onclick="openErrorReportingModal()" class="btn btn-primary">
                        <i class="fas fa-envelope mr-2"></i>
                        <span>Contact Support</span>
                    </button>
                </div>
            </div>
        </div>
    `;

    // Replace the entire error container content
    $('#errorContainer').html(rateLimitMessage);

    // Hide retry button for rate limit errors (it's now part of the new content)
}

// Function to update usage indicator (simplified - only handles limit reached)
function updateUsageIndicator(monthlyUsed = null, monthlyLimit = 20) {
    if (monthlyUsed === null) {
        return;
    }

    const remaining = monthlyLimit - monthlyUsed;

    if (remaining <= 0) {
        // Limit reached - disable search button
        disableSearchButton();
    } else {
        // Enable search button
        enableSearchButton();
    }
}

// Function to disable search button when limit reached
function disableSearchButton() {
    const searchBtn = $('#searchBtn');
    searchBtn.prop('disabled', true);
    searchBtn.addClass('opacity-50 cursor-not-allowed');
    searchBtn.removeClass('hover:bg-accent-secondary');
}

// Function to enable search button
function enableSearchButton() {
    const searchBtn = $('#searchBtn');
    searchBtn.prop('disabled', false);
    searchBtn.removeClass('opacity-50 cursor-not-allowed');
    searchBtn.addClass('hover:bg-accent-secondary');
}

// Function to show limit reached warning (removed - only error messages shown when limits reached)
function showLimitReachedWarning() {
    // This function is intentionally empty - usage display removed
}

// Function to hide limit reached warning (removed - only error messages shown when limits reached)
function hideLimitReachedWarning() {
    // This function is intentionally empty - usage display removed
}

window.goToPage = function(page) {
    if (page >= 1 && page <= totalPages && page !== currentPage) {
        // Check if we're viewing a loaded screen (has complete data)
        // For loaded screens, we have more data than what's currently displayed
        if (isViewingSavedScreen && currentData && currentData.length > 0 &&
            lastApiResponse && lastApiResponse.pagination_info &&
            currentData.length > lastApiResponse.data_rows.length) {
            // For loaded screens, navigate through the loaded data
            navigateLoadedScreenPage(page);
        } else {
            // For regular queries, fetch from API
            fetchPage(page);
        }
    }
}

// Navigate through loaded screen data without making API calls
function navigateLoadedScreenPage(page) {
    const pageSize = lastApiResponse.pagination_info.page_size;
    const startIndex = (page - 1) * pageSize;
    const endIndex = startIndex + pageSize;

    // Get the page data from the loaded screen
    const pageData = currentData.slice(startIndex, endIndex);

    // Update current page
    currentPage = page;

    // Update lastApiResponse with the new page data
    lastApiResponse.data_rows = pageData;
    lastApiResponse.pagination_info = {
        ...lastApiResponse.pagination_info,
        current_page: page,
        has_next: page < totalPages,
        has_previous: page > 1,
        showing_from: startIndex + 1,
        showing_to: Math.min(endIndex, totalRecords)
    };

    // Re-render the results with the new page data
    renderSingleSheetResults({
        columns: lastApiResponse.columns,
        data_rows: pageData,
        friendly_columns: lastApiResponse.friendly_columns,
        pagination_info: lastApiResponse.pagination_info
    });
}

// --- EXPORT FUNCTIONS ---

// Single sheet export
function exportSingleSheetData() {
    if (!lastApiResponse || !lastApiResponse.data_rows || lastApiResponse.data_rows.length === 0) {
        showToast('No data to export.', 'warning');
        return;
    }

    // Check if we're viewing a loaded screen with complete data
    if (currentData && currentData.length > 0 && currentData.length > lastApiResponse.data_rows.length) {
        // For loaded screens, export the complete dataset
        const { columns } = lastApiResponse;
        exportToCSV(columns, currentData, 'stratalens_screen_complete');
        showToast('Exported complete screen data!', 'success');
    } else {
        // For regular queries or single-page screens, export current page
        const { columns, data_rows } = lastApiResponse;
        exportToCSV(columns, data_rows, 'stratalens_query');
    }
}



// Helper function to export to CSV
function exportToCSV(columns, data_rows, filename) {
    let csvContent = "data:text/csv;charset=utf-8,";
    csvContent += columns.join(",") + "\r\n";
    data_rows.forEach(row => {
        const rowArray = columns.map(col => {
            let cell = row[col] === null || row[col] === undefined ? '' : String(row[col]);
            if (cell.includes('"')) {
                cell = cell.replace(/"/g, '""');
            }
            if (cell.includes(',')) {
                cell = `"${cell}"`;
            }
            return cell;
        });
        csvContent += rowArray.join(",") + "\r\n";
    });

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `${filename}_${new Date().toISOString().split('T')[0]}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    showToast('Data exported successfully!', 'success');
}

// --- CHARTING (ENHANCED FOR MULTI-SHEET SUPPORT) ---

// Missing chart functions that were referenced in event listeners
function updateChart() {

    const xCol = $('#chartXAxis').val();
    const yCol = $('#chartYAxis').val();

    // Track chart updates
    trackEvent('update_chart', {
        x_axis: xCol,
        y_axis: yCol,
        chart_type: $('.chart-type-toggle button.active').data('type') || 'bar',
        data_source: $('#chartSheetSelector').val() || 'single_sheet'
    });
    const chartType = $('.chart-type-toggle button.active').data('type') || 'bar';

    if (!xCol || !yCol) {
        $('#chartError').removeClass('hidden').find('p').text('Please select both X and Y axis columns');
        return;
    }

    $('#chartError').addClass('hidden');

    // Determine data source
    let dataSource, friendlyColumns;
    const selectedSheet = $('#chartSheetSelector').val();

    if (currentData && currentData.length > 0) {
        dataSource = currentData;
        friendlyColumns = lastApiResponse?.friendly_columns || {};
    } else {
        $('#chartError').removeClass('hidden').find('p').text('No data available for charting');
        return;
    }

    // Create chart based on type and data
    try {
        chartInstance = ChartFunctions.destroyExistingChart(chartInstance);

        let chartConfig;

        // Single-sheet data - allow both bar and line charts
        if (chartType === 'line' && ChartUtils.isTimeSeriesData(dataSource, xCol)) {
            // Check for multiple series (segments/companies)
            const seriesInfo = ChartUtils.hasMultipleSeries(dataSource, xCol);
            if (seriesInfo.hasMultiple) {
                chartConfig = ChartFunctions.createSegmentTimeSeriesChart(dataSource, xCol, yCol, friendlyColumns, seriesInfo);
            } else {
                chartConfig = ChartFunctions.createTimeSeriesChart(dataSource, xCol, yCol, friendlyColumns);
            }
        } else if (dataSource.some(row => row._company)) {
            // Multi-company data
            chartConfig = ChartFunctions.createMultiCompanyBarChart(dataSource, xCol, yCol, friendlyColumns);
        } else {
            // Standard single-company chart
            chartConfig = ChartFunctions.createStandardChart(dataSource, xCol, yCol, chartType, friendlyColumns);
        }

        // Create the chart with the proper chart type from config
        const ctx = document.getElementById('interactiveChart').getContext('2d');
        chartInstance = new Chart(ctx, {
            type: chartType,
            data: chartConfig.data,
            options: chartConfig.options
        });


    } catch (error) {
        $('#chartError').removeClass('hidden').find('p').text('Error creating chart: ' + error.message);
    }
}

// Note: populateChartSelectors is available as ChartFunctions.populateChartSelectors

// Note: ChartFunctions options are available as ChartFunctions.createTimeSeriesChartOptions, ChartFunctions.createBarChartOptions, etc.

// Use generateColorPalette from ChartUtils

function openChartModal() {
    if (!currentData || currentData.length === 0) {
        showToast("No data available", "warning");
        return;
    }

    // Track chart modal open
    trackEvent('open_chart_modal', {
        data_rows: currentData.length,
        chart_type: 'single_sheet'
    });

    // Ensure we have fresh data
    if (!lastApiResponse || !lastApiResponse.data_rows) {
        showToast("No valid data", "warning");
        return;
    }

    // View toggle buttons removed
    $('body').addClass('overflow-hidden');
    $('#chartModal').removeClass('hidden').addClass('flex');

    // Hide sheet selector for single sheet
    $('#chartSheetSelector').addClass('hidden');

    // 🚨 SINGLE-SHEET: Only bar charts allowed - hide line chart option
    $('.chart-type-toggle button[data-type="line"]').hide();
    $('.chart-type-toggle button[data-type="bar"]').show().addClass('active').siblings().removeClass('active');

    // Set up for bar chart comparison
    $('#chartDataPointsLabel').text('Show Companies');
    $('#chartDataPointsSelect').empty().append(`
        <option value="all">All Companies</option>
        <option value="5">Top 5</option>
        <option value="8">Top 8</option>
        <option value="10" selected>Top 10</option>
        <option value="15">Top 15</option>
        <option value="20">Top 20</option>
        <option value="50">Top 50</option>
    `).val('10');

    // Clear any existing chart before creating new one
    chartInstance = ChartFunctions.destroyExistingChart(chartInstance);

    // Initialize controls and populate selectors with CURRENT data
    populateChartSelectors({
        lastApiResponse: lastApiResponse,
        currentData: currentData
    });

    // Small delay to ensure modal is visible before creating chart
    setTimeout(() => {
        updateChart();
    }, 100);

}

function closeChartModal() {
    // View toggle buttons removed

    // Properly destroy chart before closing
    chartInstance = ChartFunctions.destroyExistingChart(chartInstance);

    // RESET CHART TYPE VISIBILITY - Show both buttons again for next use
    $('.chart-type-toggle button').show();
    $('.chart-type-toggle button[data-type="bar"]').addClass('active').siblings().removeClass('active'); // Default to bar

    // Clear chart selectors
    $('#chartXAxis, #chartYAxis').empty().append('<option value="">No data available</option>');
    $('#chartSheetSelector').empty().addClass('hidden');

    // Reset chart type toggle to default
    $('#chartModal .chart-type-toggle button[data-type="bar"]').addClass('active');
    $('#chartModal .chart-type-toggle button[data-type="line"]').removeClass('active');

    // Clear error messages
    $('#chartError').addClass('hidden');

    $('body').removeClass('overflow-hidden');
    $('#chartModal').addClass('hidden').removeClass('flex');

}


// Function moved to charts.js - using ChartFunctions.isMultiSheetTimeSeriesData

// Function moved to charts.js - using ChartFunctions.prepareCombinedChartData

// Use isTimeSeriesData from ChartUtils
// Note: isTimeSeriesData is available as ChartUtils.isTimeSeriesData

// Function moved to charts.js - using ChartFunctions.createSegmentTimeSeriesChart

// Function moved to charts.js - using ChartFunctions.createTimeSeriesChart

// Function moved to charts.js - using ChartFunctions.createMultiCompanyBarChart


// Use formatFinancialValue from ChartUtils
// Note: formatFinancialValue is available as ChartUtils.formatFinancialValue

// Note: createStandardChartOptions is available as ChartFunctions.createStandardChartOptions

// Note: updateDataPointsLabel function removed - handled directly in chart type toggle

// --- HELPERS & UTILITIES ---
// Use parseFinancialNumber from ChartUtils
// Note: parseFinancialNumber is available as ChartUtils.parseFinancialNumber

// Use detectColumnFormat from ChartUtils
// Note: detectColumnFormat is available as ChartUtils.detectColumnFormat

// Use getSectorClass from ChartUtils
// Note: getSectorClass is available as ChartUtils.getSectorClass

// Toast functions
window.showToast = function(message, type = 'info', duration = CONFIG.toastDuration) {
    // Track toast notifications
    trackEvent('show_toast', {
        message_type: type,
        duration: duration
    });

    clearTimeout(toastTimeout);
    const icons = {
        'success': { c: 'fa-check-circle', cl: 'text-green-500', bg: 'bg-green-100 dark:bg-green-500/10' },
        'error': { c: 'fa-times-circle', cl: 'text-red-500', bg: 'bg-red-100 dark:bg-red-500/10' },
        'warning': { c: 'fa-exclamation-triangle', cl: 'text-yellow-500', bg: 'bg-yellow-100 dark:bg-yellow-500/10' },
        'info': { c: 'fa-info-circle', cl: 'text-blue-500', bg: 'bg-blue-100 dark:bg-blue-500/10' }
    };
    const config = icons[type] || icons.info;
    $('#toastIcon').attr('class', `fas ${config.c} ${config.cl}`);
    $('#toastIconContainer').attr('class', `flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${config.bg}`);
    $('#toastMessage').text(message);
    $('#toast').removeClass('translate-x-[120%]').addClass('translate-x-0');
    toastTimeout = setTimeout(() => hideToast(), duration);
}

window.hideToast = function() {
    $('#toast').removeClass('translate-x-0').addClass('translate-x-[120%]');
}

// Global export functions for window
// Note: exportCurrentSheet, exportAllSheetsCSV, exportAllSheetsExcel functions are not implemented

// Auto-collapse timing control
window.reasoningPanelTimeout = null;

function scheduleReasoningPanelCollapse(delay = 3000) {
    clearTimeout(window.reasoningPanelTimeout);
    window.reasoningPanelTimeout = setTimeout(() => {
        $('#aiReasoningPanel').removeClass('expanded');
    }, delay);
}

// =============================================================================
// NAVIGATION STATE MANAGEMENT
// =============================================================================

let currentSection = 'chat';
let navigationState = {
    search: {
        // Search state is already managed by existing variables
    },
    screens: {
        data: [],
        page: 1,
        limit: 20,
        totalPages: 1,
        filter: '',
        lastUpdated: null
    }
};

// Collections state
let activeCollectionType = 'screens';

// Main section switching function
function switchToSection(sectionName) {
    // Track section change
    trackEvent('section_change', {
        from_section: currentSection,
        to_section: sectionName
    });

    // Hide all sections
    $('.main-section').removeClass('active');

    // Show target section
    $(`#${sectionName}Section`).addClass('active');

    // Update sidebar active state
    $('.sidebar-menu-item').removeClass('active');
    $(`.sidebar-menu-item[data-section="${sectionName}"]`).addClass('active');

    // Show/hide chat controls in navbar based on section
    if (sectionName === 'chat') {
        $('#chatNavControls').removeClass('hidden');
    } else {
        $('#chatNavControls').addClass('hidden');
    }

    // Update current section
    currentSection = sectionName;

    // Handle section-specific initialization
    if (sectionName === 'screens') {
        initializeScreensSection();
    } else if (sectionName === 'companies') {
        initializeCompaniesSection();
    } else if (sectionName === 'charting') {
        initializeChartingSection();
    } else if (sectionName === 'chat') {
        // Scroll to bottom when chat section is opened
        setTimeout(() => {
            const chatSection = document.getElementById('chatSection');
            if (chatSection) {
                chatSection.scrollTo({
                    top: chatSection.scrollHeight,
                    behavior: 'smooth'
                });
            }
        }, 100);
    }

}

// Initialize screens section when first accessed
function initializeScreensSection() {
    // Restore filter state
    if (navigationState.screens.filter) {
        $('#screensFilterType').val(navigationState.screens.filter);
    }

    // Restore page state
    screensPage = navigationState.screens.page || 1;
    screensLimit = navigationState.screens.limit || 20;

    // Set default collection type first
    activeCollectionType = 'screens';
    $('.collection-nav-btn').removeClass('active');
    $('.collection-nav-btn[data-collection-type="screens"]').addClass('active');

    if (!navigationState.screens.lastUpdated) {
        // First time - load fresh data
        loadUserScreens();
    } else {
        // Restore cached data if available and recent (within last 5 minutes)
        const lastUpdate = new Date(navigationState.screens.lastUpdated);
        const now = new Date();
        const timeDiff = (now - lastUpdate) / (1000 * 60); // minutes

        if (timeDiff < 5 && navigationState.screens.data.length > 0) {
            currentScreensData = navigationState.screens.data;
            $('#activeCollectionCount').text(`${currentScreensData.length} screens`);
            $('#screensNavCount').text(currentScreensData.length);
            $('#screensLastUpdated').text(moment(lastUpdate).format('HH:mm:ss'));

            if (currentScreensData.length === 0) {
                $('#screensEmptyContainer').removeClass('hidden');
            } else {
                $('#screensList').removeClass('hidden');
                renderScreensList(currentScreensData);
            }
        } else {
            // Data is stale - refresh
            loadUserScreens();
        }
    }
}

// =============================================================================
// SCREENS FUNCTIONALITY
// =============================================================================

// Use navigationState.screens for persistence
let currentScreensData = [];
let screensPage = 1;
let screensLimit = 20;
let screensTotalPages = 1;

// Open Save Screen Modal
function openSaveScreenModal() {
    // Check if we have data to save
    if (!lastApiResponse) {
        showToast('No query results to save. Please run a query first.', 'warning');
        return;
    }

    // Auto-suggest a screen name based on the query
    const query = currentQuery || 'Untitled Query';
    const suggestedName = query.length > 30 ? query.substring(0, 30) + '...' : query;
    $('#screenName').val(suggestedName);
    $('#screenDescription').val('');

    $('#saveScreenModal').removeClass('hidden').addClass('flex');
    $('body').addClass('overflow-hidden');

    // Focus on name input
    setTimeout(() => {
        $('#screenName').focus().select();
    }, 100);
}

// Close Save Screen Modal
function closeSaveScreenModal() {
    $('#saveScreenModal').addClass('hidden').removeClass('flex');
    $('body').removeClass('overflow-hidden');
    $('#saveScreenForm')[0].reset();
}

// Handle Save Screen Form Submission
async function handleSaveScreen(e) {
    e.preventDefault();

    const screenName = $('#screenName').val().trim();
    const screenDescription = $('#screenDescription').val().trim();

    // Track screen save attempt
    trackEvent('save_screen_attempt', {
        screen_name_length: screenName.length,
        has_description: !!screenDescription,
        query_type: 'single_sheet'
    });

    if (!screenName) {
        showToast('Please enter a screen name', 'warning');
        return;
    }

    // Show loading state
    $('#saveScreenBtnText').text('Saving...');
    $('#saveScreenSpinner').removeClass('hidden');
    $('#saveScreenSubmitBtn').prop('disabled', true);

    try {
        const token = localStorage.getItem('authToken');
        if (!token) {
            logout();
            return;
        }

        // Prepare screen data based on current results
        let screenData = {
            screen_name: screenName,
            description: screenDescription || null,
            query: currentQuery,
            tables_used: []
        };

        if (lastApiResponse) {
            // For single sheet data, we need to get the complete dataset from cache
            // instead of just the current page's data
            let completeDataRows = lastApiResponse.data_rows || [];
            let completeTotalRows = lastApiResponse.data_rows ? lastApiResponse.data_rows.length : 0;

            // If we have pagination info, we need to fetch the complete dataset
            if (lastApiResponse.pagination_info && lastApiResponse.pagination_info.total_records > lastApiResponse.data_rows.length) {
                try {
                    const completeResponse = await fetch(`${CONFIG.apiBaseUrl}/screener/query/complete-dataset`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${token}`
                        },
                        body: JSON.stringify({
                            question: currentQuery
                        })
                    });

                    if (completeResponse.ok) {
                        const completeResult = await completeResponse.json();
                        if (completeResult.success && completeResult.data_rows) {
                            completeDataRows = completeResult.data_rows;
                            completeTotalRows = completeResult.data_rows.length;
                        }
                    }
                } catch (error) {
                }
            }

            // Single sheet data with complete dataset
            screenData = {
                ...screenData,
                query_type: 'single_sheet',
                columns: lastApiResponse.columns || [],
                friendly_columns: lastApiResponse.friendly_columns || {},
                data_rows: completeDataRows,
                total_rows: completeTotalRows,
                tables_used: lastApiResponse.tables_used || []
            };
        } else {
            throw new Error('No valid data to save');
        }

        const response = await fetch(`${CONFIG.apiBaseUrl}/screens/save`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify(screenData)
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || 'Failed to save screen');
        }

        // Track successful screen save
        trackEvent('save_screen_success', {
            screen_name: screenName,
            query_type: screenData.query_type,
            total_rows: screenData.total_rows
        });

        showToast(result.message || 'Screen saved successfully!', 'success');
        closeSaveScreenModal();
        
        // Refresh the screens list if we're currently viewing screens
        if (currentSection === 'screens') {
            loadUserScreens();
        }

    } catch (error) {
        // Track screen save error
        trackEvent('save_screen_error', {
            error: error.message
        });

        showToast(error.message || 'Failed to save screen', 'error');
    } finally {
        // Reset loading state
        $('#saveScreenBtnText').text('Save Screen');
        $('#saveScreenSpinner').addClass('hidden');
        $('#saveScreenSubmitBtn').prop('disabled', false);
    }
}

// Screens pagination functions
function goToScreensPage(page) {
    if (page >= 1 && page <= screensTotalPages && page !== screensPage) {
        screensPage = page;
        navigationState.screens.page = page;
        loadUserScreens();
    }
}

function updateScreensPagination(paginationInfo) {
    if (!paginationInfo || paginationInfo.pages <= 1) {
        $('#screensPaginationContainer').addClass('hidden');
        return;
    }

    const { page, total, pages } = paginationInfo;
    screensTotalPages = pages;

    $('#screensPaginationInfo').text(`Showing page ${page} of ${pages} (${total} total screens)`);
    $('#screensPageJumpInput').val(page).attr('max', pages);
    $('#screensPageTotal').text(`of ${pages}`);

    // Update button states
    $('#screensFirstPageBtn, #screensPrevPageBtn').prop('disabled', page === 1);
    $('#screensLastPageBtn, #screensNextPageBtn').prop('disabled', page === pages);

    $('#screensPaginationContainer').removeClass('hidden');
}

// Load User's Saved Screens
async function loadUserScreens() {

    // Show loading state
    $('#screensLoadingContainer').removeClass('hidden');
    $('#screensErrorContainer, #screensEmptyContainer, #watchlistsEmptyContainer, #portfoliosEmptyContainer').addClass('hidden');
    $('#screensList').empty();
    $('#screensPaginationContainer').addClass('hidden');

    try {
        const token = localStorage.getItem('authToken');
        if (!token) {
            logout();
            return;
        }

        const queryType = $('#screensFilterType').val();
        const params = new URLSearchParams({
            page: screensPage,
            limit: screensLimit
        });

        if (queryType) {
            params.append('query_type', queryType);
        }

        const response = await fetch(`${CONFIG.apiBaseUrl}/screens/list?${params.toString()}`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (response.status === 401) {
            logout();
            return;
        }

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || 'Failed to load screens');
        }

        currentScreensData = result.screens || [];

        // Update navigation state
        navigationState.screens.data = currentScreensData;
        navigationState.screens.page = screensPage;
        navigationState.screens.filter = queryType;
        navigationState.screens.lastUpdated = new Date().toISOString();

        // Update UI
        $('#activeCollectionCount').text(`${result.summary?.total_screens || 0} screens`);
        $('#screensNavCount').text(result.summary?.total_screens || 0);
        $('#screensLastUpdated').text(moment().format('HH:mm:ss'));

        if (currentScreensData.length === 0) {
            $('#screensEmptyContainer').removeClass('hidden');
        } else {
            $('#screensList').removeClass('hidden');
            renderScreensList(currentScreensData);
            if (result.pagination) {
                updateScreensPagination(result.pagination);
            }
        }

    } catch (error) {
        $('#screensErrorMessage').text(error.message || 'Failed to load screens');
        $('#screensErrorContainer').removeClass('hidden');
    } finally {
        $('#screensLoadingContainer').addClass('hidden');
    }
}

// Render Screens List
function renderScreensList(screens) {
    const container = $('#screensList').empty();

    if (screens.length === 0) {
        return;
    }

    // Create professional list table
    const table = $(`
        <div class="screens-table-wrapper">
            <table class="screens-table w-full">
                <thead>
                    <tr>
                        <th class="screen-name-col">Name</th>
                        <th class="screen-type-col">Type</th>
                        <th class="screen-labels-col">Labels</th>
                        <th class="screen-updated-col">Last Updated</th>
                        <th class="screen-actions-col">Actions</th>
                    </tr>
                </thead>
                <tbody>
                </tbody>
            </table>
        </div>
    `);

    const tbody = table.find('tbody');

    screens.forEach((screen, index) => {
        const row = $(`
            <tr class="screen-row" data-screen-id="${screen.id}">
                <td class="screen-name-cell">
                    <div class="flex items-center gap-3">
                        <div class="screen-icon">
                            <i class="fas fa-chart-bar text-accent-primary"></i>
                        </div>
                        <div>
                            <div class="screen-name font-medium text-text-primary">${ChartUtils.escapeHtml(screen.screen_name)}</div>
                            ${screen.description ? `<div class="screen-description text-xs text-text-tertiary mt-0.5">${ChartUtils.escapeHtml(screen.description)}</div>` : ''}
                        </div>
                    </div>
                </td>
                <td class="screen-type-cell">
                    <div class="type-info">
                        <span class="screen-type-badge single-sheet">
                            <i class="fas fa-table mr-1"></i>
                            Single Sheet
                        </span>
                        <div class="type-details text-xs text-text-tertiary mt-1">
                            ${screen.total_rows.toLocaleString()} rows
                            ${screen.companies ? ` • ${screen.companies.length} companies` : ''}
                        </div>
                    </div>
                </td>
                <td class="screen-labels-cell">
                    <div class="labels-container">
                        <div class="screen-labels">
                            <!-- TODO: Implement labels functionality -->
                            <span class="add-label-btn" title="Add labels to organize your screens">
                                <i class="fas fa-tag text-text-tertiary"></i>
                                <span class="text-xs text-text-tertiary ml-1">Add labels</span>
                            </span>
                        </div>
                    </div>
                </td>
                <td class="screen-updated-cell">
                    <div class="text-xs text-text-tertiary">
                        <div class="updated-time">${moment(screen.updated_at).fromNow()}</div>
                        <div class="updated-date text-text-quaternary">${moment(screen.updated_at).format('MMM DD, YYYY')}</div>
                    </div>
                </td>
                <td class="screen-actions-cell">
                    <div class="flex items-center gap-1">
                        <button class="load-screen-btn action-btn primary" data-screen-id="${screen.id}" title="Load this screen">
                            <i class="fas fa-play"></i>
                            <span class="action-text">Load</span>
                        </button>
                        <button class="plot-screen-stocks-btn action-btn charting" data-screen-id="${screen.id}" title="Plot stocks from this screen">
                            <i class="fas fa-chart-line"></i>
                            <span class="action-text">Plot</span>
                        </button>
                        <button class="delete-screen-btn action-btn secondary" data-screen-id="${screen.id}" title="Delete this screen">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </td>
            </tr>
        `);

        tbody.append(row);
    });

    container.append(table);

    // Add event listeners
    $('.load-screen-btn').on('click', function(e) {
        e.stopPropagation();
        const screenId = $(this).data('screen-id');
        loadScreen(screenId);
    });

    $('.plot-screen-stocks-btn').on('click', function(e) {
        e.stopPropagation();
        const screenId = $(this).data('screen-id');
        openPlotScreenStocksModal(screenId);
    });


    $('.delete-screen-btn').on('click', function(e) {
        e.stopPropagation();
        const screenId = $(this).data('screen-id');
        deleteScreen(screenId);
    });

    // Add row click handler for loading
    $('.screen-row').on('click', function(e) {
        // Don't trigger if clicking on action buttons or labels
        if (!$(e.target).closest('.screen-actions-cell, .add-label-btn').length) {
            const screenId = $(this).data('screen-id');
            loadScreen(screenId);
        }
    });
}

// Load a Saved Screen
async function loadScreen(screenId) {
    try {
        // Track screen load attempt
        trackEvent('load_screen', {
            screen_id: screenId
        });

        const token = localStorage.getItem('authToken');
        if (!token) {
            logout();
            return;
        }

        const response = await fetch(`${CONFIG.apiBaseUrl}/screens/${screenId}`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (response.status === 401) {
            logout();
            return;
        }

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || 'Failed to load screen');
        }

        // Switch back to search section
        switchToSection('search');

        // Set current query and screen tracking
        currentQuery = result.screen.query;
        currentQuestion = result.screen.query;
        currentScreenId = screenId;
        isViewingSavedScreen = true;
        $('#queryInput').val(currentQuery);

        // Clear previous data
        clearPreviousQueryData();

        // Load single-sheet data
        currentData = result.data_rows || [];
            totalRecords = result.data_rows ? result.data_rows.length : 0;

            // Create proper pagination info for loaded screen data
            const defaultPageSize = 20;
            totalPages = Math.ceil(totalRecords / defaultPageSize);
            currentPage = 1;

            // Create pagination info for the first page
            const paginationInfo = {
                current_page: 1,
                page_size: defaultPageSize,
                total_records: totalRecords,
                total_pages: totalPages,
                has_next: totalPages > 1,
                has_previous: false,
                showing_from: 1,
                showing_to: Math.min(defaultPageSize, totalRecords)
            };

            // Get first page of data
            const firstPageData = result.data_rows ? result.data_rows.slice(0, defaultPageSize) : [];

            lastApiResponse = {
                columns: result.columns || [],
                data_rows: firstPageData,
                friendly_columns: result.friendly_columns || {},
                tables_used: result.screen.tables_used || [],
                pagination_info: paginationInfo
            };

            renderSingleSheetResults({
                columns: result.columns,
                data_rows: firstPageData,
                friendly_columns: result.friendly_columns,
                pagination_info: paginationInfo
            });
        hideAllResultContainers();
        $('#singleSheetContainer').removeClass('hidden');

        // Track successful screen load
        trackEvent('load_screen_success', {
            screen_id: screenId,
            screen_name: result.screen.screen_name,
            query_type: result.screen.query_type
        });

        showToast(`Screen "${result.screen.screen_name}" loaded successfully!`, 'success');

    } catch (error) {
        // Track screen load error
        trackEvent('load_screen_error', {
            screen_id: screenId,
            error: error.message
        });

        showToast(error.message || 'Failed to load screen', 'error');
    }
}


// Delete a Saved Screen
async function deleteScreen(screenId) {
    if (!confirm('Are you sure you want to delete this screen? This action cannot be undone.')) {
        return;
    }

    try {
        const token = localStorage.getItem('authToken');
        if (!token) {
            logout();
            return;
        }

        const response = await fetch(`${CONFIG.apiBaseUrl}/screens/${screenId}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (response.status === 401) {
            logout();
            return;
        }

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || 'Failed to delete screen');
        }

        showToast('Screen deleted successfully', 'success');
        // Refresh the list if we're on the screens section
        if (currentSection === 'screens') {
            loadUserScreens();
        }

    } catch (error) {
        showToast(error.message || 'Failed to delete screen', 'error');
    }
}

// =============================================================================
// PLOT STOCKS FROM SCREEN FUNCTIONALITY
// =============================================================================

let currentPlotScreenData = null;

// Open Plot Stocks from Screen Modal
async function openPlotScreenStocksModal(screenId) {
    try {
        // Track plot screen attempt
        trackEvent('plot_screen_stocks_attempt', {
            screen_id: screenId
        });

        const token = localStorage.getItem('authToken');
        if (!token) {
            logout();
            return;
        }

        // Show loading state
        $('#plotScreenStocksBtnText').text('Loading...');
        $('#plotScreenStocksSpinner').removeClass('hidden');
        $('#plotScreenStocksBtn').prop('disabled', true);

        const response = await fetch(`${CONFIG.apiBaseUrl}/screens/${screenId}`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (response.status === 401) {
            logout();
            return;
        }

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || 'Failed to load screen data');
        }

        // Store screen data for later use
        currentPlotScreenData = result;

        // Extract stocks from screen data
        const stocks = extractStocksFromScreenData(result);

        if (stocks.length === 0) {
            showToast('No stocks found in this screen', 'warning');
            return;
        }

        // Populate modal
        populatePlotScreenStocksModal(result.screen, stocks);

        // Show modal
        $('#plotScreenStocksModal').removeClass('hidden').addClass('flex');
        $('body').addClass('overflow-hidden');

        // Track successful modal open
        trackEvent('plot_screen_stocks_modal_opened', {
            screen_id: screenId,
            stock_count: stocks.length
        });

    } catch (error) {
        showToast(error.message || 'Failed to load screen data', 'error');
    } finally {
        // Reset loading state
        $('#plotScreenStocksBtnText').text('Plot Selected Stocks');
        $('#plotScreenStocksSpinner').addClass('hidden');
        $('#plotScreenStocksBtn').prop('disabled', false);
    }
}

// Extract stocks from screen data
function extractStocksFromScreenData(screenData) {
    const stocks = [];
    const seenSymbols = new Set();

    try {
        if (screenData.screen.query_type === 'multi_sheet') {
            // Multi-sheet data - extract from companies array and sheet data
            if (screenData.screen.companies && screenData.screen.companies.length > 0) {
                screenData.screen.companies.forEach(company => {
                    if (company && typeof company === 'string' && !seenSymbols.has(company)) {
                        stocks.push({
                            symbol: company,
                            companyName: company // Use symbol as fallback name
                        });
                        seenSymbols.add(company);
                    }
                });
            }

            // Also extract from sheet data if available
            if (screenData.sheets && screenData.sheets.length > 0) {
                screenData.sheets.forEach(sheet => {
                    if (sheet.data && sheet.data.length > 0) {
                        sheet.data.forEach(row => {
                            const symbol = extractSymbolFromRow(row);
                            if (symbol && !seenSymbols.has(symbol)) {
                                const companyName = extractCompanyNameFromRow(row) || symbol;
                                stocks.push({
                                    symbol: symbol,
                                    companyName: companyName
                                });
                                seenSymbols.add(symbol);
                            }
                        });
                    }
                });
            }
        } else {
            // Single sheet data - extract from data rows
            if (screenData.data_rows && screenData.data_rows.length > 0) {
                screenData.data_rows.forEach(row => {
                    const symbol = extractSymbolFromRow(row);
                    if (symbol && !seenSymbols.has(symbol)) {
                        const companyName = extractCompanyNameFromRow(row) || symbol;
                        stocks.push({
                            symbol: symbol,
                            companyName: companyName
                        });
                        seenSymbols.add(symbol);
                    }
                });
            }
        }

        return stocks;

    } catch (error) {
        return [];
    }
}

// Extract symbol from a data row
function extractSymbolFromRow(row) {
    const symbolColumns = ['symbol', 'ticker', 'stock_symbol', 'company_symbol'];
    for (const col of symbolColumns) {
        if (row[col] && typeof row[col] === 'string' && row[col].trim()) {
            return row[col].trim().toUpperCase();
        }
    }
    return null;
}

// Extract company name from a data row
function extractCompanyNameFromRow(row) {
    const nameColumns = ['companyName', 'company_name', 'name', 'company'];
    for (const col of nameColumns) {
        if (row[col] && typeof row[col] === 'string' && row[col].trim()) {
            return row[col].trim();
        }
    }
    return null;
}

// Populate the plot screen stocks modal
function populatePlotScreenStocksModal(screen, stocks) {
    // Populate screen info
    $('#plotScreenInfo').html(`
        <div class="font-medium">${ChartUtils.escapeHtml(screen.screen_name)}</div>
        <div class="text-text-secondary text-xs mt-1">
            ${screen.query_type === 'multi_sheet' ? 'Multi Sheet' : 'Single Sheet'} •
            ${stocks.length} stocks found
        </div>
    `);

    // Populate stocks list
    const stocksList = $('#plotScreenStocksList');
    stocksList.empty();

    stocks.forEach((stock, index) => {
        const isSelected = index < 10; // First 10 selected by default
        const stockItem = $(`
            <div class="flex items-center gap-3 p-3 border-b border-border-primary last:border-b-0 hover:bg-bg-tertiary">
                <input type="checkbox"
                       id="stock-${stock.symbol}"
                       class="stock-checkbox w-4 h-4 text-accent-primary bg-bg-primary border-border-primary rounded focus:ring-accent-primary focus:ring-2"
                       data-symbol="${stock.symbol}"
                       data-company-name="${ChartUtils.escapeHtml(stock.companyName)}"
                       ${isSelected ? 'checked' : ''}>
                <label for="stock-${stock.symbol}" class="flex-1 cursor-pointer">
                    <div class="font-medium text-text-primary">${stock.symbol}</div>
                    <div class="text-sm text-text-secondary">${ChartUtils.escapeHtml(stock.companyName)}</div>
                </label>
            </div>
        `);
        stocksList.append(stockItem);
    });

    // Update counts
    updatePlotStocksCounts();

    // Add event listeners
    $('.stock-checkbox').on('change', updatePlotStocksCounts);
    $('#selectAllPlotStocksBtn').on('click', selectAllPlotStocks);
    $('#deselectAllPlotStocksBtn').on('click', deselectAllPlotStocks);
}

// Update the selected/total counts
function updatePlotStocksCounts() {
    const totalCheckboxes = $('.stock-checkbox').length;
    const selectedCheckboxes = $('.stock-checkbox:checked').length;

    $('#plotStocksTotalCount').text(totalCheckboxes);
    $('#plotStocksSelectedCount').text(selectedCheckboxes);

    // Enable/disable plot button based on selection
    $('#plotScreenStocksBtn').prop('disabled', selectedCheckboxes === 0);
}

// Select all stocks
function selectAllPlotStocks() {
    $('.stock-checkbox').prop('checked', true);
    updatePlotStocksCounts();
}

// Deselect all stocks
function deselectAllPlotStocks() {
    $('.stock-checkbox').prop('checked', false);
    updatePlotStocksCounts();
}

// Close Plot Screen Stocks Modal
function closePlotScreenStocksModal() {
    $('#plotScreenStocksModal').addClass('hidden').removeClass('flex');
    $('body').removeClass('overflow-hidden');
    currentPlotScreenData = null;

    // Clean up event listeners
    $('.stock-checkbox').off('change');
    $('#selectAllPlotStocksBtn').off('click');
    $('#deselectAllPlotStocksBtn').off('click');
}

// Handle Plot Screen Stocks button click
async function handlePlotScreenStocks() {
    const selectedStocks = [];

    $('.stock-checkbox:checked').each(function() {
        selectedStocks.push({
            symbol: $(this).data('symbol'),
            companyName: $(this).data('company-name')
        });
    });

    if (selectedStocks.length === 0) {
        showToast('Please select at least one stock to plot', 'warning');
        return;
    }

    // Track plot action
    trackEvent('plot_screen_stocks_executed', {
        screen_id: currentPlotScreenData?.screen?.id,
        stock_count: selectedStocks.length,
        stocks: selectedStocks.map(s => s.symbol)
    });

    // Close modal
    closePlotScreenStocksModal();

    // Switch to charting section
    switchToSection('charting');

    // Initialize charting section if needed
    if (!chartingSectionInitialized) {
        initializeChartingSection();
        chartingSectionInitialized = true;
    }

    // Clear existing stocks and add selected ones
    chartingSelectedStocks = [];
    selectedStocks.forEach(stock => {
        addStockToCharting(stock.symbol, stock.companyName);
    });

    // Add screen context to charting section
    if (currentPlotScreenData?.screen) {
        addScreenContextToCharting(currentPlotScreenData.screen);
    } else {
    }

    // Show success message
    showToast(`Added ${selectedStocks.length} stocks from "${currentPlotScreenData?.screen?.screen_name}" to charting`, 'success');

}

// Add screen context to charting section
function addScreenContextToCharting(screen) {

    // Create or update screen context display
    let screenContext = $('#chartingScreenContext');

    if (screenContext.length === 0) {
        // Create the screen context element if it doesn't exist
        screenContext = $(`
            <div id="chartingScreenContext" class="mb-4 p-3 bg-accent-primary/10 border border-accent-primary/20 rounded-lg">
                <div class="flex items-center justify-between">
                    <div class="flex items-center gap-2">
                        <i class="fas fa-chart-bar text-accent-primary"></i>
                        <div>
                            <div class="text-sm font-medium text-text-primary">Plotting from Screen</div>
                            <div class="text-xs text-text-secondary">${ChartUtils.escapeHtml(screen.screen_name)}</div>
                        </div>
                    </div>
                    <button id="clearScreenContextBtn" class="text-xs text-text-tertiary hover:text-text-primary" title="Clear screen context">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            </div>
        `);


        // Insert after the header section (before the chart configuration controls)
        const headerSection = $('#chartingSection .card .p-6.border-b.border-border-primary');
        headerSection.after(screenContext);

        // Add event listener for clear button
        $('#clearScreenContextBtn').on('click', clearScreenContext);
    } else {
        // Update existing screen context
        screenContext.find('.text-text-secondary').text(screen.screen_name);
    }
}

// Clear screen context
function clearScreenContext() {
    $('#chartingScreenContext').remove();
}


// Note: escapeHtml is available as ChartUtils.escapeHtml

// DEBUG: Manual sidebar indicator check function
window.debugSidebarIndicator = function() {
    const $searchMenuItem = $('.sidebar-menu-item[data-section="search"]');

    // Force clear it
    updateSidebarSearchIndicator(false);
};

// DEBUG: Enhanced functions for the new state system
window.forceClearSidebar = function() {
    updateSidebarSearchIndicator('cleared');
};

// DEBUG: Force show tab content
window.forceShowTabs = function() {

    // Force show all tab content
    $('.tab-content').css({
        'display': 'block',
        'visibility': 'visible',
        'opacity': '1',
        'height': 'auto',
        'min-height': '300px'
    });

    // Force show segment containers
    $('#productSegmentsData, #geographicSegmentsData').css({
        'display': 'block',
        'visibility': 'visible',
        'opacity': '1',
        'height': 'auto',
        'min-height': '300px',
        'max-height': 'none',
        'overflow': 'visible'
    });

    // Force show TTM containers
    $('#ttmProfitabilityData, #ttmEfficiencyData, #ttmLiquidityData').css({
        'display': 'grid',
        'visibility': 'visible',
        'opacity': '1',
        'height': 'auto',
        'min-height': '200px',
        'max-height': 'none',
        'overflow': 'visible'
    });

    // Force table visibility
    $('.segment-table').css({
        'display': 'table',
        'width': '100%',
        'height': 'auto',
        'min-height': '200px',
        'border-collapse': 'collapse'
    });

    $('.segment-table th, .segment-table td').css({
        'display': 'table-cell',
        'padding': '8px 12px',
        'border': '1px solid #e2e8f0'
    });

};

// Test the completed state
window.testCompleted = function() {
    updateSidebarSearchIndicator('completed');
};

// Test the loading state
window.testLoading = function() {
    updateSidebarSearchIndicator('loading');
};

// ULTIMATE NUCLEAR OPTION: Completely destroy all search indicators
window.nukeSearchIndicator = function() {

    // Find all search menu items
    const $allSearchItems = $('.sidebar-menu-item[data-section="search"]');

    $allSearchItems.each(function(index) {
        const $item = $(this);

        // Remove ALL indicator elements
        $item.find('.search-loading-icon, .search-completed-icon, .search-status-text').remove();
        $item.removeClass('search-active');

        // Ensure search icon is visible
        $item.find('.search-icon, i.fa-filter').removeClass('hidden').show();

    });

};

// SIMPLE TEST: Just call the regular function
window.testClear = function() {
    updateSidebarSearchIndicator('cleared');
};

// =============================================================================
// CHARTING FUNCTIONALITY
// =============================================================================

// Charting state variables
let chartingSelectedStocks = [];
let chartingMainChartInstance = null;
let chartingSearchTimeout = null;
let chartingSectionInitialized = false;

// Initialize charting section
function initializeChartingSection() {

    // Reset state
    chartingSelectedStocks = [];
    updateSelectedStocksDisplay();

    // Show empty state
    $('#chartingEmptyState').removeClass('hidden');
    $('#chartingChartArea').addClass('hidden');

    // Clear any existing chart
    if (chartingMainChartInstance) {
        chartingMainChartInstance.destroy();
        chartingMainChartInstance = null;
    }

    // Reset form controls
    $('#chartingStockSearch').val('');
    $('#chartingMetricSelect').val('');
    $('#chartingMetricTypeSelect').val('absolute');
    $('#chartingTimePeriodSelect').val('5');

    // Setup event listeners if not already done
    if (!chartingSectionInitialized) {
        setupChartingEventListeners();
    }
    
    chartingSectionInitialized = true;
}

// Setup charting event listeners
function setupChartingEventListeners() {
    // Stock search input
    $('#chartingStockSearch').on('input', function() {
        clearTimeout(chartingSearchTimeout);
        const query = $(this).val().trim();

        if (query.length >= 1) {
            chartingSearchTimeout = setTimeout(() => {
                searchChartingStocks(query);
            }, 300);
        } else {
            hideChartingStockSuggestions();
        }
    });

    // Hide suggestions when clicking outside
    $(document).on('click', function(e) {
        if (!$(e.target).closest('#chartingStockSearch, #chartingStockSuggestions').length) {
            hideChartingStockSuggestions();
        }
    });

    // Generate chart button
    $('#generateChartBtn').on('click', generateFinancialChart);

    // Clear all button
    $('#clearChartingDataBtn').on('click', clearChartingData);

    // Export chart button
    $('#exportChartBtn').on('click', exportFinancialChart);

    // Chart controls
    $('#chartingRefreshBtn').on('click', generateFinancialChart);
    $('#chartingFullscreenBtn').on('click', toggleChartFullscreen);

    // Zoom control event listeners
    $('#zoomInBtn').on('click', function() {
        if (chartingMainChartInstance) {
            chartingMainChartInstance.zoom(1.1);
            chartingMainChartInstance.update('none');
        }
    });

    $('#zoomOutBtn').on('click', function() {
        if (chartingMainChartInstance) {
            chartingMainChartInstance.zoom(0.9);
            chartingMainChartInstance.update('none');
        }
    });

    $('#resetZoomBtn').on('click', function() {
        if (chartingMainChartInstance) {
            chartingMainChartInstance.resetZoom();
        }
    });

    // Metric type selection
    $('#chartingMetricTypeSelect').on('change', function() {
        const metricType = $(this).val();
        handleMetricTypeChange(metricType);
    });

}

// Handle metric type change
function handleMetricTypeChange(metricType) {

    // Always show the KPIs & Segments dropdown
    $('#chartingFinancialMetrics').removeClass('hidden');
    $('#chartingSegmentInfo').addClass('hidden');
}

// Search for stocks
async function searchChartingStocks(query) {
    try {
        const token = localStorage.getItem('authToken');
        if (!token) {
            logout();
            return;
        }

        const response = await fetch(`${CONFIG.apiBaseUrl}/companies/search?query=${encodeURIComponent(query)}&limit=10`, {
            headers: { 'Authorization': `Bearer ${token}` }
        });

        if (response.ok) {
            const data = await response.json();
            showChartingStockSuggestions(data.companies || []);
        } else {
            hideChartingStockSuggestions();
        }
    } catch (error) {
        hideChartingStockSuggestions();
    }
}

// Show stock suggestions
function showChartingStockSuggestions(companies) {
    const container = $('#chartingStockSuggestions');

    if (companies.length === 0) {
        hideChartingStockSuggestions();
        return;
    }

    container.empty();

    companies.forEach(company => {
        const isSelected = chartingSelectedStocks.some(stock => stock.symbol === company.symbol);
        const item = $(`
            <div class="stock-suggestion-item p-3 hover:bg-bg-tertiary cursor-pointer border-b border-border-primary last:border-b-0 ${isSelected ? 'opacity-50' : ''}"
                 data-symbol="${company.symbol}" data-name="${company.companyName}">
                <div class="flex items-center justify-between">
                    <div>
                        <div class="font-semibold text-sm">${company.symbol}</div>
                        <div class="text-xs text-text-secondary">${company.companyName}</div>
                        ${company.sector ? `<div class="text-xs text-text-tertiary">${company.sector}</div>` : ''}
                    </div>
                    ${isSelected ? '<i class="fas fa-check text-green-500"></i>' : '<i class="fas fa-plus text-accent-primary"></i>'}
                </div>
            </div>
        `);

        if (!isSelected) {
            item.on('click', () => addStockToCharting(company.symbol, company.companyName));
        }

        container.append(item);
    });

    container.removeClass('hidden').show();
}

// Hide stock suggestions
function hideChartingStockSuggestions() {
    $('#chartingStockSuggestions').addClass('hidden').hide();
}

// Add stock to charting
function addStockToCharting(symbol, companyName) {
    // Check if already added
    if (chartingSelectedStocks.some(stock => stock.symbol === symbol)) {
        showToast('Company already added', 'warning');
        return;
    }

    // Add to selected stocks
    chartingSelectedStocks.push({
        symbol: symbol,
        companyName: companyName
    });

    // Update display
    updateSelectedStocksDisplay();

    // Clear search
    $('#chartingStockSearch').val('');
    hideChartingStockSuggestions();

    // Track event
    trackEvent('charting_add_stock', {
        symbol: symbol,
        total_stocks: chartingSelectedStocks.length
    });

}

// Remove stock from charting
function removeStockFromCharting(symbol) {
    chartingSelectedStocks = chartingSelectedStocks.filter(stock => stock.symbol !== symbol);
    updateSelectedStocksDisplay();

    // Track event
    trackEvent('charting_remove_stock', {
        symbol: symbol,
        remaining_stocks: chartingSelectedStocks.length
    });

}

// Update selected stocks display
function updateSelectedStocksDisplay() {
    const container = $('#chartingSelectedStocks');

    if (chartingSelectedStocks.length === 0) {
        container.html(`
            <div class="text-text-tertiary text-sm italic flex items-center">
                No companies selected. Search and add companies above.
            </div>
        `);
        return;
    }

    container.empty();

    chartingSelectedStocks.forEach(stock => {
        const stockTag = $(`
            <div class="stock-tag flex items-center gap-2 px-3 py-2 bg-accent-primary/10 border border-accent-primary/20 rounded-lg">
                <div class="flex-1">
                    <div class="font-semibold text-sm">${stock.symbol}</div>
                    <div class="text-xs text-text-secondary">${stock.companyName}</div>
                </div>
                <button class="remove-stock-btn w-5 h-5 flex items-center justify-center rounded-full hover:bg-red-100 dark:hover:bg-red-900/30 text-text-tertiary hover:text-red-500"
                        data-symbol="${stock.symbol}">
                    <i class="fas fa-times text-xs"></i>
                </button>
            </div>
        `);

        stockTag.find('.remove-stock-btn').on('click', () => removeStockFromCharting(stock.symbol));
        container.append(stockTag);
    });
}

// Generate financial chart
async function generateFinancialChart() {
    if (chartingSelectedStocks.length === 0) {
        showToast('Select companies', 'warning');
        return;
    }

    const metric = $('#chartingMetricSelect').val();
    const metricType = $('#chartingMetricTypeSelect').val();
    const timePeriod = $('#chartingTimePeriodSelect').val();

    // Show loading state
    showChartingLoading();

    try {
        let chartData = [];
        let chartType = 'financial';
        let segmentType = null;
        let metricLabel = metric;

        if (metric === 'product_segment_revenue' || metric === 'geographic_segment_revenue') {
            chartType = 'segment';
            segmentType = metric === 'product_segment_revenue' ? 'product' : 'geographic';
            metricLabel = segmentType === 'product' ? 'Product Segment Revenue' : 'Geographic Segment Revenue';
            // Track chart generation
            trackEvent('charting_generate_chart', {
                companies: chartingSelectedStocks.map(s => s.symbol),
                chart_type: chartType,
                segment_type: segmentType,
                metric_type: metricType,
                time_period: timePeriod
            });
            // Fetch segment data for all selected companies
            chartData = await fetchSegmentChartingData(chartingSelectedStocks, segmentType, timePeriod, metricType);
        } else {
            // Track chart generation
            trackEvent('charting_generate_chart', {
                companies: chartingSelectedStocks.map(s => s.symbol),
                chart_type: chartType,
                metric: metric,
                metric_type: metricType,
                time_period: timePeriod
            });
            // Fetch financial data for all selected companies
            chartData = await fetchChartingData(chartingSelectedStocks, metric, metricType, timePeriod);
        }

        if (chartData.length === 0) {
            showChartingError('No data available');
            return;
        }

        // Create and render chart
        if (chartType === 'segment') {
            renderSegmentChart(chartData, segmentType === 'product' ? 'product_segments' : 'geographic_segments', timePeriod, metricType);
        } else {
            renderFinancialChart(chartData, metric, metricType, timePeriod);
        }

    } catch (error) {
        showChartingError('Chart generation failed');
    }
}

// Update fetchChartingData to handle metric type and ignore segment metrics
async function fetchChartingData(stocks, metric, metricType, timePeriod) {
    if (metric === 'product_segment_revenue' || metric === 'geographic_segment_revenue') {
        // This is handled by fetchSegmentChartingData in generateFinancialChart
        return [];
    }
    const token = localStorage.getItem('authToken');
    if (!token) {
        logout();
        return [];
    }

    // Use the new charting endpoint - can handle up to 50 years
    const requestedYears = timePeriod === 'all' ? 50 : parseInt(timePeriod);


    // Determine which statement type based on metric
    const statementTypeMap = {
        // Income statement metrics
        'revenue': 'income',
        'netIncome': 'income',
        'grossProfit': 'income',
        'operatingIncome': 'income',
        'ebitda': 'income',
        'eps': 'income',
        'epsDiluted': 'income',
        'grossProfitRatio': 'income',
        'operatingIncomeRatio': 'income',
        'netIncomeRatio': 'income',
        'costOfRevenue': 'income',
        'operatingExpenses': 'income',
        'incomeBeforeTax': 'income',
        'incomeTaxExpense': 'income',
        'interestExpense': 'income',
        'interestIncome': 'income',
        'depreciationAndAmortization': 'income',
        'researchAndDevelopmentExpenses': 'income',
        'sellingAndMarketingExpenses': 'income',
        'generalAndAdministrativeExpenses': 'income',

        // Balance sheet metrics
        'totalAssets': 'balance',
        'totalEquity': 'balance',
        'totalLiabilities': 'balance',
        'totalDebt': 'balance',
        'cashAndCashEquivalents': 'balance',
        'cashAndShortTermInvestments': 'balance',
        'netReceivables': 'balance',
        'inventory': 'balance',
        'totalCurrentAssets': 'balance',
        'totalCurrentLiabilities': 'balance',
        'propertyPlantEquipmentNet': 'balance',
        'goodwill': 'balance',
        'intangibleAssets': 'balance',
        'longTermDebt': 'balance',
        'shortTermDebt': 'balance',
        'retainedEarnings': 'balance',
        'commonStock': 'balance',
        'netDebt': 'balance',

        // Cash flow metrics
        'operatingCashFlow': 'cashflow',
        'netCashProvidedByOperatingActivities': 'cashflow',
        'freeCashFlow': 'cashflow',
        'capitalExpenditure': 'cashflow',
        'investmentsInPropertyPlantAndEquipment': 'cashflow',
        'netCashUsedForInvestingActivites': 'cashflow',
        'netCashUsedProvidedByFinancingActivities': 'cashflow',
        'acquisitionsNet': 'cashflow',
        'purchasesOfInvestments': 'cashflow',
        'salesMaturitiesOfInvestments': 'cashflow',
        'commonStockIssued': 'cashflow',
        'commonStockRepurchased': 'cashflow',
        'netChangeInCash': 'cashflow',
        'changeInWorkingCapital': 'cashflow'
    };

    // Map frontend metric names to actual database column names
    const metricColumnMap = {
        // Income statement metrics (from income_statements table)
        'revenue': 'revenue',
        'netIncome': 'netIncome',
        'grossProfit': 'grossProfit',
        'operatingIncome': 'operatingIncome',
        'ebitda': 'ebitda',
        'eps': 'eps',
        'epsDiluted': 'epsdiluted', // Note: lowercase 'd' in DB
        'grossProfitRatio': 'grossProfitRatio',
        'operatingIncomeRatio': 'operatingIncomeRatio',
        'netIncomeRatio': 'netIncomeRatio',
        'costOfRevenue': 'costOfRevenue',
        'operatingExpenses': 'operatingExpenses',
        'incomeBeforeTax': 'incomeBeforeTax',
        'incomeTaxExpense': 'incomeTaxExpense',
        'interestExpense': 'interestExpense',
        'interestIncome': 'interestIncome',
        'depreciationAndAmortization': 'depreciationAndAmortization',
        'researchAndDevelopmentExpenses': 'researchAndDevelopmentExpenses',
        'sellingAndMarketingExpenses': 'sellingAndMarketingExpenses',
        'generalAndAdministrativeExpenses': 'generalAndAdministrativeExpenses',

        // Balance sheet metrics (from balance_sheets table)
        'totalAssets': 'totalAssets',
        'totalEquity': 'totalEquity', // Note: this is totalEquity, not totalStockholdersEquity
        'totalLiabilities': 'totalLiabilities',
        'totalDebt': 'totalDebt',
        'cashAndCashEquivalents': 'cashAndCashEquivalents',
        'cashAndShortTermInvestments': 'cashAndShortTermInvestments',
        'netReceivables': 'netReceivables',
        'inventory': 'inventory',
        'totalCurrentAssets': 'totalCurrentAssets',
        'totalCurrentLiabilities': 'totalCurrentLiabilities',
        'propertyPlantEquipmentNet': 'propertyPlantEquipmentNet',
        'goodwill': 'goodwill',
        'intangibleAssets': 'intangibleAssets',
        'longTermDebt': 'longTermDebt',
        'shortTermDebt': 'shortTermDebt',
        'retainedEarnings': 'retainedEarnings',
        'commonStock': 'commonStock',
        'netDebt': 'netDebt',

        // Cash flow metrics (from cash_flow_statements table)
        'operatingCashFlow': 'operatingCashFlow',
        'netCashProvidedByOperatingActivities': 'netCashProvidedByOperatingActivities',
        'freeCashFlow': 'freeCashFlow',
        'capitalExpenditure': 'capitalExpenditure',
        'investmentsInPropertyPlantAndEquipment': 'investmentsInPropertyPlantAndEquipment',
        'netCashUsedForInvestingActivites': 'netCashUsedForInvestingActivites',
        'netCashUsedProvidedByFinancingActivities': 'netCashUsedProvidedByFinancingActivities',
        'acquisitionsNet': 'acquisitionsNet',
        'purchasesOfInvestments': 'purchasesOfInvestments',
        'salesMaturitiesOfInvestments': 'salesMaturitiesOfInvestments',
        'commonStockIssued': 'commonStockIssued',
        'commonStockRepurchased': 'commonStockRepurchased',
        'netChangeInCash': 'netChangeInCash',
        'changeInWorkingCapital': 'changeInWorkingCapital',
        'depreciationAndAmortization': 'depreciationAndAmortization'
    };

    const statementType = statementTypeMap[metric] || 'income';
    const dbColumnName = metricColumnMap[metric] || metric;


    try {
        // Use the new multi-company charting endpoint
        const response = await fetch(`${CONFIG.apiBaseUrl}/charting/multi-company-time-series`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                symbols: stocks.map(s => s.symbol),
                metric: dbColumnName, // Use the correct DB column name
                metric_type: metricType, // Add metric type parameter
                statement_type: statementType,
                years: requestedYears
            })
        });

        if (response.ok) {
            const data = await response.json();

            if (data.debug_info) {
            }

            if (data.success && data.companies && data.companies.length > 0) {
                // Convert the API response to the format expected by the chart renderer
                const chartData = [];

                data.companies.forEach(company => {
                    const companyName = company.companyName || company.symbol;

                    company.data.forEach(dataPoint => {
                        if (dataPoint.value !== null && dataPoint.value !== undefined) {
                            chartData.push({
                                company: companyName,
                                symbol: company.symbol,
                                year: dataPoint.year,
                                value: dataPoint.value,
                                date: dataPoint.date
                            });
                        }
                    });
                });


                // Show detailed breakdown by company
                const companyCounts = {};
                chartData.forEach(point => {
                    if (!companyCounts[point.symbol]) {
                        companyCounts[point.symbol] = 0;
                    }
                    companyCounts[point.symbol]++;
                });

                return chartData;
            } else {
                return [];
            }
        } else {
            const errorText = await response.text();
            return [];
        }
    } catch (error) {
        return [];
    }
}

// Fetch segment data for charting
async function fetchSegmentChartingData(stocks, segmentType, timePeriod, metricType = 'absolute') {
    const token = localStorage.getItem('authToken');
    if (!token) {
        logout();
        return [];
    }

    // Use the new segment charting endpoint
    const requestedYears = timePeriod === 'all' ? 50 : parseInt(timePeriod);


    try {
        // Use the multi-company segment charting endpoint
        const response = await fetch(`${CONFIG.apiBaseUrl}/charting/multi-company-segments`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                symbols: stocks.map(s => s.symbol),
                segment_type: segmentType,
                metric_type: metricType,
                years: requestedYears
            })
        });

        if (response.ok) {
            const data = await response.json();

            if (data.success && data.chart_data && data.chart_data.length > 0) {
                // Convert the API response to the format expected by the chart renderer
                const chartData = [];

                data.chart_data.forEach(dataPoint => {
                    if (dataPoint.value !== null && dataPoint.value !== undefined) {
                        chartData.push({
                            company: dataPoint.company,
                            symbol: dataPoint.symbol,
                            segment: dataPoint.segment,
                            year: dataPoint.year,
                            value: dataPoint.value,
                            percentage: dataPoint.percentage
                        });
                    }
                });


                return chartData;
            } else {
                return [];
            }
        } else {
            const errorText = await response.text();
            return [];
        }
    } catch (error) {
        return [];
    }
}

// Render financial chart
function renderFinancialChart(data, metric, metricType, timePeriod) {
    // Destroy existing chart
    if (chartingMainChartInstance) {
        chartingMainChartInstance.destroy();
    }

    // Group data by company
    const companiesData = {};
    data.forEach(item => {
        if (!companiesData[item.company]) {
            companiesData[item.company] = [];
        }
        companiesData[item.company].push(item);
    });

    // Sort each company's data by year
    Object.keys(companiesData).forEach(company => {
        companiesData[company].sort((a, b) => a.year - b.year);
    });

    // Get all unique years and sort them
    const allYears = [...new Set(data.map(item => item.year))].sort((a, b) => a - b);

    // Create datasets for each company
    const colors = ChartUtils.generateColorPalette(Object.keys(companiesData).length);
    const datasets = Object.entries(companiesData).map(([company, companyData], index) => {
        const yearlyData = allYears.map(year => {
            const dataPoint = companyData.find(item => item.year === year);
            return dataPoint ? dataPoint.value : null;
        });

        return {
            label: company,
            data: yearlyData,
            borderColor: colors[index],
            backgroundColor: colors[index] + '15',
            borderWidth: 2.5,
            fill: false,
            tension: 0.3,
            pointRadius: 4,
            pointHoverRadius: 7,
            pointBackgroundColor: colors[index],
            pointBorderColor: $('html').hasClass('dark') ? '#1e293b' : '#ffffff',
            pointBorderWidth: 2,
            spanGaps: false,
            // Enhanced styling for elegance
            pointStyle: 'circle',
            pointHitRadius: 10,
            pointHoverBorderWidth: 3
        };
    });

    // Create chart
    const ctx = document.getElementById('chartingMainChart').getContext('2d');
    chartingMainChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: allYears,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 800, easing: 'easeOutQuart' },
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: $('html').hasClass('dark') ? '#cbd5e1' : '#475569',
                        padding: 15,
                        usePointStyle: true,
                        font: { size: 12, family: 'Inter', weight: '600' }
                    }
                },
                datalabels: {
                    display: false, // Hide all data labels by default
                },
                tooltip: {
                    backgroundColor: $('html').hasClass('dark') ? 'rgba(15, 23, 42, 0.95)' : 'rgba(255, 255, 255, 0.95)',
                    titleColor: $('html').hasClass('dark') ? '#f1f5f9' : '#0f172a',
                    bodyColor: $('html').hasClass('dark') ? '#e2e8f0' : '#334155',
                    borderColor: $('html').hasClass('dark') ? '#334155' : '#e2e8f0',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                    titleFont: { size: 13, weight: 'bold', family: 'Inter' },
                    bodyFont: { size: 12, family: 'Inter' },
                    callbacks: {
                        title: (context) => `Year: ${context[0].label}`,
                        label: (ctx) => {
                            const value = ctx.parsed.y;
                            if (value === null || value === undefined || isNaN(value)) {
                                return `${ctx.dataset.label}: N/A`;
                            }

                            if (metricType === 'growth') {
                                return `${ctx.dataset.label}: ${value.toFixed(1)}%`;
                            }

                            // Format large amounts with M/B/T notation
                            const absValue = Math.abs(value);
                            const sign = value < 0 ? '-' : '';

                            // Handle per-share metrics (EPS)
                            if (metric && (metric.includes('eps') || metric.includes('EPS') || metric.toLowerCase().includes('per share'))) {
                                return `${ctx.dataset.label}: ${sign}$${absValue.toFixed(2)}`;
                            }

                            // Format large amounts with M/B/T notation
                            if (absValue >= 1e12) {
                                return `${ctx.dataset.label}: ${sign}$${(absValue / 1e12).toFixed(1)}T`;
                            } else if (absValue >= 1e9) {
                                return `${ctx.dataset.label}: ${sign}$${(absValue / 1e9).toFixed(1)}B`;
                            } else if (absValue >= 1e6) {
                                return `${ctx.dataset.label}: ${sign}$${(absValue / 1e6).toFixed(1)}M`;
                            } else if (absValue >= 1e3) {
                                return `${ctx.dataset.label}: ${sign}$${(absValue / 1e3).toFixed(1)}K`;
                            } else {
                                return `${ctx.dataset.label}: ${sign}$${absValue.toFixed(0)}`;
                            }
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Year',
                        color: $('html').hasClass('dark') ? '#94a3b8' : '#64748b',
                        font: { size: 12, weight: '600', family: 'Inter' }
                    },
                    grid: {
                        color: $('html').hasClass('dark') ? '#334155' : '#e2e8f0'
                    },
                    ticks: {
                        color: $('html').hasClass('dark') ? '#94a3b8' : '#64748b',
                        font: { size: 11, family: 'Inter' }
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: metricType === 'growth' ? `${getMetricDisplayName(metric)} Growth Rate (%)` : getMetricDisplayName(metric),
                        color: $('html').hasClass('dark') ? '#94a3b8' : '#64748b',
                        font: { size: 12, weight: '600', family: 'Inter' }
                    },
                    grid: {
                        color: $('html').hasClass('dark') ? 'rgba(51, 65, 85, 0.3)' : 'rgba(226, 232, 240, 0.6)',
                        lineWidth: 1,
                        drawBorder: false
                    },
                    ticks: {
                        color: $('html').hasClass('dark') ? '#94a3b8' : '#64748b',
                        font: { size: 11, family: 'Inter' },
                        padding: 8,
                        callback: function(value) {
                            // Custom Y-axis formatter with M/B/T notation
                            if (value === null || value === undefined || isNaN(value)) {
                                return '0';
                            }

                            const absValue = Math.abs(value);
                            const sign = value < 0 ? '-' : '';

                            // Handle percentage metrics and growth rates
                            if (metricType === 'growth' || (metric && (metric.includes('Ratio') || metric.includes('Margin') || metric.toLowerCase().includes('return')))) {
                                return `${sign}${absValue.toFixed(1)}%`;
                            }

                            // Handle per-share metrics (EPS)
                            if (metric && (metric.includes('eps') || metric.includes('EPS') || metric.toLowerCase().includes('per share'))) {
                                return `${sign}$${absValue.toFixed(2)}`;
                            }

                            // Format large amounts with M/B/T notation
                            if (absValue >= 1e12) {
                                return `${sign}$${(absValue / 1e12).toFixed(1)}T`;
                            } else if (absValue >= 1e9) {
                                return `${sign}$${(absValue / 1e9).toFixed(1)}B`;
                            } else if (absValue >= 1e6) {
                                return `${sign}$${(absValue / 1e6).toFixed(1)}M`;
                            } else if (absValue >= 1e3) {
                                return `${sign}$${(absValue / 1e3).toFixed(1)}K`;
                            } else {
                                return `${sign}$${absValue.toFixed(0)}`;
                            }
                        }
                    }
                }
            }
        }
    });

    // Show chart area and update title
    showChartingChart(metric, metricType, chartingSelectedStocks.length, timePeriod);

}

// Render segment chart
function renderSegmentChart(data, chartType, timePeriod, metricType = 'absolute') {
    // Destroy existing chart
    if (chartingMainChartInstance) {
        chartingMainChartInstance.destroy();
    }

    const segmentType = chartType === 'product_segments' ? 'Product' : 'Geographic';

    // Group data by company and segment, ensuring unique segment names
    const segmentsData = {};
    const seenSegments = new Set();


    data.forEach(item => {
        // Extract the actual company name (before the dash)
        const companyName = item.company.split(' - ')[0].trim();
        const segmentName = item.segment.trim();


        // Create a clean key with just company and segment
        const key = `${companyName} - ${segmentName}`;


        // Only add if we haven't seen this exact combination before
        if (!seenSegments.has(key)) {
            seenSegments.add(key);
            segmentsData[key] = [];
        }

        segmentsData[key].push(item);
    });


    // Sort each segment's data by year
    Object.keys(segmentsData).forEach(segment => {
        segmentsData[segment].sort((a, b) => a.year - b.year);
    });

    // Get all unique years and sort them
    const allYears = [...new Set(data.map(item => item.year))].sort((a, b) => a - b);

    // Create datasets for each company-segment combination
    const colors = ChartUtils.generateColorPalette(Object.keys(segmentsData).length);
    const datasets = Object.entries(segmentsData).map(([segmentName, segmentData], index) => {
        const yearlyData = allYears.map(year => {
            const dataPoint = segmentData.find(item => item.year === year);
            return dataPoint ? dataPoint.value : null;
        });

        return {
            label: segmentName,
            data: yearlyData,
            borderColor: colors[index],
            backgroundColor: colors[index] + '20',
            borderWidth: 3,
            fill: false,
            tension: 0.4,
            pointRadius: 5,
            pointHoverRadius: 8,
            pointBackgroundColor: colors[index],
            pointBorderColor: '#ffffff',
            pointBorderWidth: 2,
            spanGaps: false
        };
    });

    // Create chart
    const ctx = document.getElementById('chartingMainChart').getContext('2d');
    chartingMainChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: allYears,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 800, easing: 'easeOutQuart' },
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: $('html').hasClass('dark') ? '#cbd5e1' : '#475569',
                        padding: 15,
                        usePointStyle: true,
                        font: { size: 12, family: 'Inter', weight: '600' }
                    }
                },
                datalabels: {
                    display: false, // Hide all data labels by default
                },
                tooltip: {
                    backgroundColor: $('html').hasClass('dark') ? 'rgba(15, 23, 42, 0.95)' : 'rgba(255, 255, 255, 0.95)',
                    titleColor: $('html').hasClass('dark') ? '#f1f5f9' : '#0f172a',
                    bodyColor: $('html').hasClass('dark') ? '#e2e8f0' : '#334155',
                    borderColor: $('html').hasClass('dark') ? '#334155' : '#e2e8f0',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                    titleFont: { size: 13, weight: 'bold', family: 'Inter' },
                    bodyFont: { size: 12, family: 'Inter' },
                    callbacks: {
                        title: (context) => `Year: ${context[0].label}`,
                        label: (ctx) => {
                            const value = ctx.parsed.y;
                            if (value === null || value === undefined || isNaN(value)) {
                                return `${ctx.dataset.label}: N/A`;
                            }

                            if (metricType === 'growth') {
                                return `${ctx.dataset.label}: ${value.toFixed(1)}%`;
                            }

                            // Format large amounts with M/B/T notation
                            const absValue = Math.abs(value);
                            const sign = value < 0 ? '-' : '';

                            // Format large amounts with M/B/T notation for segment revenue
                            if (absValue >= 1e12) {
                                return `${ctx.dataset.label}: ${sign}$${(absValue / 1e12).toFixed(1)}T`;
                            } else if (absValue >= 1e9) {
                                return `${ctx.dataset.label}: ${sign}$${(absValue / 1e9).toFixed(1)}B`;
                            } else if (absValue >= 1e6) {
                                return `${ctx.dataset.label}: ${sign}$${(absValue / 1e6).toFixed(1)}M`;
                            } else if (absValue >= 1e3) {
                                return `${ctx.dataset.label}: ${sign}$${(absValue / 1e3).toFixed(1)}K`;
                            } else {
                                return `${ctx.dataset.label}: ${sign}$${absValue.toFixed(0)}`;
                            }
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Year',
                        color: $('html').hasClass('dark') ? '#94a3b8' : '#64748b',
                        font: { size: 12, weight: '600', family: 'Inter' }
                    },
                    grid: {
                        color: $('html').hasClass('dark') ? '#334155' : '#e2e8f0'
                    },
                    ticks: {
                        color: $('html').hasClass('dark') ? '#94a3b8' : '#64748b',
                        font: { size: 11, family: 'Inter' }
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: metricType === 'growth' ? `${segmentType} Segment Revenue Growth (%)` : `${segmentType} Segment Revenue ($)`,
                        color: $('html').hasClass('dark') ? '#94a3b8' : '#64748b',
                        font: { size: 12, weight: '600', family: 'Inter' }
                    },
                    grid: {
                        color: $('html').hasClass('dark') ? '#334155' : '#e2e8f0'
                    },
                    ticks: {
                        color: $('html').hasClass('dark') ? '#94a3b8' : '#64748b',
                        font: { size: 11, family: 'Inter' },
                        callback: function(value) {
                            // Custom Y-axis formatter
                            if (value === null || value === undefined || isNaN(value)) {
                                return '0';
                            }

                            const absValue = Math.abs(value);
                            const sign = value < 0 ? '-' : '';

                            // Handle growth rates for segments
                            if (metricType === 'growth') {
                                return `${sign}${absValue.toFixed(1)}%`;
                            }

                            // Format large amounts with M/B/T notation for absolute values
                            if (absValue >= 1e12) {
                                return `${sign}$${(absValue / 1e12).toFixed(1)}T`;
                            } else if (absValue >= 1e9) {
                                return `${sign}$${(absValue / 1e9).toFixed(1)}B`;
                            } else if (absValue >= 1e6) {
                                return `${sign}$${(absValue / 1e6).toFixed(1)}M`;
                            } else if (absValue >= 1e3) {
                                return `${sign}$${(absValue / 1e3).toFixed(1)}K`;
                            } else {
                                return `${sign}$${absValue.toFixed(0)}`;
                            }
                        }
                    }
                }
            }
        }
    });

    // Show chart area and update title
    const periodText = timePeriod === 'all' ? 'All Years' : `${timePeriod}Y`;
    const uniqueCompanies = [...new Set(data.map(item => item.company))];

    $('#chartingEmptyState').addClass('hidden');
    $('#chartingChartArea').removeClass('hidden');
    $('#chartingChartError').addClass('hidden');

    const titleText = metricType === 'growth' ? `${segmentType} Segment Revenue Growth` : `${segmentType} Segment Revenue Comparison`;
    $('#chartingChartTitle').text(titleText);
    $('#chartingChartSubtitle').text(`${Object.keys(segmentsData).length} segments • ${periodText}`);

}

// Chart.js watermark plugin
const chartWatermarkPlugin = {
    id: 'watermark',
    beforeDraw: function(chart) {
        try {
            // Comprehensive defensive checks
            if (!chart) return;
            if (!chart.ctx) return;
            if (!chart.canvas) return;
            if (!chart.options) return;

            const ctx = chart.ctx;
            const canvas = chart.canvas;

            // Save the current context state
            ctx.save();

            // Get theme colors safely with fallbacks
            let isDark = false;
            try {
                isDark = (typeof $ !== 'undefined') && $('html').hasClass('dark');
            } catch (e) {
                isDark = false;
            }

            const primaryColor = '#0070d8'; // StrataLens blue
            const textColor = isDark ? '#f1f5f9' : '#475569';
            const bgColor = isDark ? 'rgba(15, 23, 42, 0.9)' : 'rgba(248, 250, 252, 0.9)';

            // Check for export mode with comprehensive safety
            let isExport = false;
            try {
                if (chart.options &&
                    chart.options.plugins &&
                    chart.options.plugins.export &&
                    chart.options.plugins.export.active === true) {
                    isExport = true;
                }
            } catch (e) {
                isExport = false;
            }

            if (isExport) {
                // Professional corner watermark for exports
                try {
                    ctx.save();

                    // Position in bottom-right corner with adequate padding
                    const padding = 20;
                    const watermarkWidth = 160;
                    const watermarkHeight = 40;
                    const startX = canvas.width - watermarkWidth - padding;
                    const startY = canvas.height - watermarkHeight - padding;

                    // Draw semi-transparent background box
                    ctx.globalAlpha = 0.85;
                    ctx.fillStyle = isDark ? 'rgba(15, 23, 42, 0.95)' : 'rgba(255, 255, 255, 0.95)';
                    ctx.roundRect = ctx.roundRect || function(x, y, w, h, r) {
                        ctx.beginPath();
                        ctx.moveTo(x + r, y);
                        ctx.lineTo(x + w - r, y);
                        ctx.quadraticCurveTo(x + w, y, x + w, y + r);
                        ctx.lineTo(x + w, y + h - r);
                        ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
                        ctx.lineTo(x + r, y + h);
                        ctx.quadraticCurveTo(x, y + h, x, y + h - r);
                        ctx.lineTo(x, y + r);
                        ctx.quadraticCurveTo(x, y, x + r, y);
                    };
                    ctx.roundRect(startX, startY, watermarkWidth, watermarkHeight, 6);
                    ctx.fill();

                    // Add subtle border
                    ctx.globalAlpha = 0.3;
                    ctx.strokeStyle = primaryColor;
                    ctx.lineWidth = 1;
                    ctx.stroke();

                    // Reset alpha for text and logo
                    ctx.globalAlpha = 1.0;

                    // Draw StrataLens logo (better S design)
                    const logoX = startX + 15;
                    const logoY = startY + 20;
                    const logoSize = 12;

                    ctx.fillStyle = primaryColor;
                    ctx.strokeStyle = primaryColor;
                    ctx.lineWidth = 2;

                    // Create a more visible "S" logo
                    ctx.beginPath();
                    // Top curve
                    ctx.arc(logoX, logoY - 4, logoSize/2, 0, Math.PI, false);
                    // Bottom curve
                    ctx.arc(logoX, logoY + 4, logoSize/2, Math.PI, 0, false);
                    ctx.lineWidth = 3;
                    ctx.stroke();

                    // Company name
                    ctx.fillStyle = isDark ? '#f1f5f9' : '#1e293b';
                    ctx.font = 'bold 16px Inter, -apple-system, BlinkMacSystemFont, sans-serif';
                    ctx.textAlign = 'left';
                    ctx.textBaseline = 'middle';
                    ctx.fillText('StrataLens', logoX + logoSize + 8, logoY - 2);

                    // Subtitle
                    ctx.fillStyle = isDark ? '#94a3b8' : '#64748b';
                    ctx.font = '12px Inter, -apple-system, BlinkMacSystemFont, sans-serif';
                    ctx.fillText('Financial Analytics', logoX + logoSize + 8, logoY + 12);

                    ctx.restore();
                } catch (exportError) {
                }
            } else {
                // Standard subtle watermark for charts
                try {
                    const padding = 15;
                    const baseY = canvas.height - padding;
                    const baseX = canvas.width - padding;

                    // Subtle background for better readability
                    ctx.globalAlpha = 0.1;
                    ctx.fillStyle = isDark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)';
                    ctx.fillRect(baseX - 85, baseY - 18, 85, 18);

                    // Draw "StrataLens" text
                    ctx.globalAlpha = 0.25;
                    ctx.fillStyle = isDark ? '#94a3b8' : '#64748b';
                    ctx.font = 'bold 11px Inter, -apple-system, BlinkMacSystemFont, sans-serif';
                    ctx.textAlign = 'right';
                    ctx.textBaseline = 'bottom';
                    ctx.fillText('StrataLens', baseX, baseY);

                    // Draw small logo icon
                    const iconSize = 8;
                    const iconX = baseX - 75;
                    const iconY = baseY - 6;

                    ctx.globalAlpha = 0.2;
                    ctx.fillStyle = primaryColor;
                    ctx.strokeStyle = primaryColor;
                    ctx.lineWidth = 1.5;

                    // Simple S shape
                    ctx.beginPath();
                    ctx.arc(iconX, iconY - 2, iconSize/3, 0, Math.PI, false);
                    ctx.arc(iconX, iconY + 2, iconSize/3, Math.PI, 0, false);
                    ctx.stroke();
                } catch (standardError) {
                }
            }

            // Restore the context state
            ctx.restore();
        } catch (error) {
            // Don't let watermark errors break the chart
        }
    }
};

// Chart.js background plugin for exports
const chartBackgroundPlugin = {
    id: 'background',
    beforeDraw: function(chart) {
        try {
            // Comprehensive defensive checks
            if (!chart) return;
            if (!chart.ctx) return;
            if (!chart.options) return;
            if (!chart.width) return;
            if (!chart.height) return;

            let backgroundColor = null;
            try {
                if (chart.options &&
                    chart.options.plugins &&
                    chart.options.plugins.background &&
                    chart.options.plugins.background.color) {
                    backgroundColor = chart.options.plugins.background.color;
                }
            } catch (e) {
                backgroundColor = null;
            }

            if (backgroundColor) {
                const ctx = chart.ctx;
                try {
                    ctx.save();
                    ctx.fillStyle = backgroundColor;
                    ctx.fillRect(0, 0, chart.width, chart.height);
                    ctx.restore();
                } catch (drawError) {
                    try {
                        ctx.restore();
                    } catch (restoreError) {
                        // Silent fail on restore
                    }
                }
            }
        } catch (error) {
            // Always try to restore context even on error
            try {
                if (chart && chart.ctx) {
                    chart.ctx.restore();
                }
            } catch (restoreError) {
                // Silent fail on restore error
            }
        }
    }
};

// Register the plugins globally with error handling
try {
    Chart.register(chartWatermarkPlugin);
    Chart.register(chartBackgroundPlugin);
} catch (registrationError) {
}

// Using ChartUtils.formatFinancialValue for consistent formatting across all charts

// Get metric display name
function getMetricDisplayName(metric) {
    const displayNames = {
        // Income Statement
        'revenue': 'Revenue ($)',
        'netIncome': 'Net Income ($)',
        'grossProfit': 'Gross Profit ($)',
        'operatingIncome': 'Operating Income ($)',
        'ebitda': 'EBITDA ($)',
        'eps': 'Earnings Per Share ($)',
        'epsDiluted': 'Diluted EPS ($)',
        'costOfRevenue': 'Cost of Revenue ($)',
        'operatingExpenses': 'Operating Expenses ($)',
        'incomeBeforeTax': 'Income Before Tax ($)',
        'incomeTaxExpense': 'Income Tax Expense ($)',
        'interestExpense': 'Interest Expense ($)',
        'interestIncome': 'Interest Income ($)',
        'depreciationAndAmortization': 'Depreciation & Amortization ($)',
        'researchAndDevelopmentExpenses': 'R&D Expenses ($)',
        'sellingAndMarketingExpenses': 'Selling & Marketing Expenses ($)',
        'generalAndAdministrativeExpenses': 'G&A Expenses ($)',

        // Balance Sheet
        'totalAssets': 'Total Assets ($)',
        'totalEquity': 'Total Equity ($)',
        'totalLiabilities': 'Total Liabilities ($)',
        'totalDebt': 'Total Debt ($)',
        'cashAndCashEquivalents': 'Cash & Equivalents ($)',
        'cashAndShortTermInvestments': 'Cash & Short-term Investments ($)',
        'netReceivables': 'Net Receivables ($)',
        'inventory': 'Inventory ($)',
        'totalCurrentAssets': 'Total Current Assets ($)',
        'totalCurrentLiabilities': 'Total Current Liabilities ($)',
        'propertyPlantEquipmentNet': 'Property, Plant & Equipment ($)',
        'goodwill': 'Goodwill ($)',
        'intangibleAssets': 'Intangible Assets ($)',
        'longTermDebt': 'Long-term Debt ($)',
        'shortTermDebt': 'Short-term Debt ($)',
        'retainedEarnings': 'Retained Earnings ($)',
        'commonStock': 'Common Stock ($)',
        'netDebt': 'Net Debt ($)',

        // Cash Flow
        'operatingCashFlow': 'Operating Cash Flow ($)',
        'netCashProvidedByOperatingActivities': 'Net Cash from Operations ($)',
        'freeCashFlow': 'Free Cash Flow ($)',
        'capitalExpenditure': 'Capital Expenditure ($)',
        'investmentsInPropertyPlantAndEquipment': 'Investments in PP&E ($)',
        'netCashUsedForInvestingActivites': 'Net Cash from Investing ($)',
        'netCashUsedProvidedByFinancingActivities': 'Net Cash from Financing ($)',
        'acquisitionsNet': 'Acquisitions (Net) ($)',
        'purchasesOfInvestments': 'Purchases of Investments ($)',
        'salesMaturitiesOfInvestments': 'Sales/Maturities of Investments ($)',
        'commonStockIssued': 'Common Stock Issued ($)',
        'commonStockRepurchased': 'Common Stock Repurchased ($)',
        'netChangeInCash': 'Net Change in Cash ($)',
        'changeInWorkingCapital': 'Change in Working Capital ($)',

        // Ratios
        'grossProfitRatio': 'Gross Profit Margin (%)',
        'operatingIncomeRatio': 'Operating Margin (%)',
        'netIncomeRatio': 'Net Margin (%)',

        // Segments
        'product_segment_revenue': 'Product Segment Revenue ($)',
        'geographic_segment_revenue': 'Geographic Segment Revenue ($)'
    };

    return displayNames[metric] || metric;
}

// Show charting loading state
function showChartingLoading() {
    $('#chartingEmptyState').addClass('hidden');
    $('#chartingChartArea').removeClass('hidden');
            $('#chartingChartTitle').text('Generating Chart...');
        $('#chartingChartSubtitle').text('Loading data...');

    // Show loading spinner on canvas
    const ctx = document.getElementById('chartingMainChart').getContext('2d');
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.fillStyle = $('html').hasClass('dark') ? '#64748b' : '#94a3b8';
    ctx.font = '14px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('Loading chart data...', ctx.canvas.width / 2, ctx.canvas.height / 2);
}

// Show charting chart
function showChartingChart(metric, metricType, companyCount, timePeriod) {
    $('#chartingEmptyState').addClass('hidden');
    $('#chartingChartArea').removeClass('hidden');
    $('#chartingChartError').addClass('hidden');

    const metricName = getMetricDisplayName(metric);
    const periodText = timePeriod === 'all' ? 'All Years' : `${timePeriod}Y`;
    const typeText = metricType === 'growth' ? 'Growth' : '';

    $('#chartingChartTitle').text(typeText ? `${metricName} ${typeText}` : metricName);
    $('#chartingChartSubtitle').text(periodText);
}

// Show charting error
function showChartingError(message) {
    $('#chartingEmptyState').addClass('hidden');
    $('#chartingChartArea').removeClass('hidden');
    $('#chartingChartError').removeClass('hidden');
    $('#chartingChartErrorMessage').text(message);
}

// Clear charting data
function clearChartingData() {
    chartingSelectedStocks = [];
    updateSelectedStocksDisplay();

    // Destroy chart
    if (chartingMainChartInstance) {
        chartingMainChartInstance.destroy();
        chartingMainChartInstance = null;
    }

    // Show empty state
    $('#chartingEmptyState').removeClass('hidden');
    $('#chartingChartArea').addClass('hidden');

    // Reset form
    $('#chartingStockSearch').val('');
    $('#chartingMetricSelect').val('');
    $('#chartingMetricTypeSelect').val('absolute');
    hideChartingStockSuggestions();

    // Clear screen context
    clearScreenContext();

    // Track event
    trackEvent('charting_clear_all');

    showToast('Cleared', 'success');
}

// Export financial chart
function exportFinancialChart() {
    if (!chartingMainChartInstance) {
        showToast('No chart available', 'warning');
        return;
    }

    try {
        // Track event
        trackEvent('charting_export_chart', {
            companies: chartingSelectedStocks.map(s => s.symbol),
            metric: $('#chartingMetricSelect').val()
        });

        // Store original export state safely
        let originalExportActive = false;
        let originalBackgroundColor = null;

        try {
            if (chartingMainChartInstance.options &&
                chartingMainChartInstance.options.plugins &&
                chartingMainChartInstance.options.plugins.export) {
                originalExportActive = chartingMainChartInstance.options.plugins.export.active || false;
            }

            if (chartingMainChartInstance.options &&
                chartingMainChartInstance.options.plugins &&
                chartingMainChartInstance.options.plugins.background) {
                originalBackgroundColor = chartingMainChartInstance.options.plugins.background.color || null;
            }
        } catch (e) {
            // Use defaults if access fails
            originalExportActive = false;
            originalBackgroundColor = null;
        }

        // Get theme colors
        const isDark = $('html').hasClass('dark');
        const bgColor = isDark ? '#0f172a' : '#ffffff';

        // Safely set export mode
        if (!chartingMainChartInstance.options) {
            chartingMainChartInstance.options = {};
        }
        if (!chartingMainChartInstance.options.plugins) {
            chartingMainChartInstance.options.plugins = {};
        }
        if (!chartingMainChartInstance.options.plugins.export) {
            chartingMainChartInstance.options.plugins.export = {};
        }
        if (!chartingMainChartInstance.options.plugins.background) {
            chartingMainChartInstance.options.plugins.background = {};
        }

        // Set export flags without triggering recursion
        chartingMainChartInstance.options.plugins.export.active = true;
        chartingMainChartInstance.options.plugins.background.color = bgColor;

        // Force render without update to avoid recursion
        chartingMainChartInstance.render();

        // Create download link with high quality
        const url = chartingMainChartInstance.toBase64Image('image/png', 1.0);
        const link = document.createElement('a');
        const timestamp = new Date().toISOString().slice(0, 10);
        const metric = $('#chartingMetricSelect').val() || 'chart';
        const companies = chartingSelectedStocks.map(s => s.symbol).join('-');
        link.download = `stratalens-${metric}-${companies}-${timestamp}.png`;
        link.href = url;
        link.click();

        // Restore original state safely
        chartingMainChartInstance.options.plugins.export.active = originalExportActive;
        if (originalBackgroundColor) {
            chartingMainChartInstance.options.plugins.background.color = originalBackgroundColor;
        } else {
            chartingMainChartInstance.options.plugins.background.color = null;
        }

        // Force render to restore original appearance
        chartingMainChartInstance.render();

        showToast('Chart exported with StrataLens branding!', 'success');
    } catch (error) {
        showToast('Failed to export chart', 'error');
    }
}

// Export modal chart
function exportModalChart() {
    if (!chartInstance) {
        showToast('No chart available', 'warning');
        return;
    }

    try {
        // Track event
        trackEvent('export_modal_chart', {
            chart_type: $('.chart-type-toggle button.active').data('type') || 'bar',
            data_source: $('#chartSheetSelector').val() || 'single_sheet'
        });

        // Store original export state safely
        let originalExportActive = false;
        let originalBackgroundColor = null;

        try {
            if (chartInstance.options &&
                chartInstance.options.plugins &&
                chartInstance.options.plugins.export) {
                originalExportActive = chartInstance.options.plugins.export.active || false;
            }

            if (chartInstance.options &&
                chartInstance.options.plugins &&
                chartInstance.options.plugins.background) {
                originalBackgroundColor = chartInstance.options.plugins.background.color || null;
            }
        } catch (e) {
            // Use defaults if access fails
            originalExportActive = false;
            originalBackgroundColor = null;
        }

        // Get theme colors
        const isDark = $('html').hasClass('dark');
        const bgColor = isDark ? '#0f172a' : '#ffffff';

        // Safely set export mode
        if (!chartInstance.options) {
            chartInstance.options = {};
        }
        if (!chartInstance.options.plugins) {
            chartInstance.options.plugins = {};
        }
        if (!chartInstance.options.plugins.export) {
            chartInstance.options.plugins.export = {};
        }
        if (!chartInstance.options.plugins.background) {
            chartInstance.options.plugins.background = {};
        }

        // Set export flags without triggering recursion
        chartInstance.options.plugins.export.active = true;
        chartInstance.options.plugins.background.color = bgColor;

        // Force render without update to avoid recursion
        chartInstance.render();

        // Create download link with high quality
        const url = chartInstance.toBase64Image('image/png', 1.0);
        const link = document.createElement('a');
        const timestamp = new Date().toISOString().slice(0, 10);
        const xAxis = $('#chartXAxis').val() || 'chart';
        const yAxis = $('#chartYAxis').val() || 'data';
        link.download = `stratalens-${xAxis}-vs-${yAxis}-${timestamp}.png`;
        link.href = url;
        link.click();

        // Restore original state safely
        chartInstance.options.plugins.export.active = originalExportActive;
        if (originalBackgroundColor) {
            chartInstance.options.plugins.background.color = originalBackgroundColor;
        } else {
            chartInstance.options.plugins.background.color = null;
        }

        // Force render to restore original appearance
        chartInstance.render();

        showToast('Chart exported with StrataLens branding!', 'success');
    } catch (error) {
        showToast('Failed to export chart', 'error');
    }
}

// Toggle chart fullscreen
function toggleChartFullscreen() {
    const chartArea = $('#chartingChartArea');

    if (chartArea.hasClass('fullscreen-chart')) {
        chartArea.removeClass('fullscreen-chart');
        $('#chartingFullscreenBtn i').removeClass('fa-compress').addClass('fa-expand');
    } else {
        chartArea.addClass('fullscreen-chart');
        $('#chartingFullscreenBtn i').removeClass('fa-expand').addClass('fa-compress');
    }

    // Resize chart after fullscreen toggle
    setTimeout(() => {
        if (chartingMainChartInstance) {
            chartingMainChartInstance.resize();
        }
    }, 100);
}

// Note: Application initialization is handled inside the document ready function


window.retryLastQuery = function() {
    if (!lastApiResponse || !currentQuery) {
        showToast('No previous query to retry.', 'warning');
        return;
    }
    // Hide error and retry button
    $('#errorContainer').addClass('hidden');
    $('#retryButton').addClass('hidden');
    // Re-run the last query
    performSearchWithStreaming();
};

// =============================================================================
// SCREENER SUGGESTIONS COLLAPSE/EXPAND FUNCTIONALITY
// =============================================================================

// Collapse screener suggestions when search starts
function collapseScreenerSuggestions() {
    const categoriesSection = $('.screener-categories');
    if (categoriesSection.length && !categoriesSection.hasClass('collapsed')) {
        
        // Add collapsed class and animate
        categoriesSection.addClass('collapsed');
        
        // Add a toggle button to expand again
        if (!$('#expandSuggestionsBtn').length) {
            const expandButton = $(`
                <div class="text-center mb-4">
                    <button id="expandSuggestionsBtn" class="btn btn-outline">
                        <i class="fas fa-chevron-down mr-2"></i>
                        <span>Show Query Suggestions</span>
                    </button>
                </div>
            `);
            categoriesSection.before(expandButton);
            
            // Add click handler for expand button
            $('#expandSuggestionsBtn').on('click', expandScreenerSuggestions);
        }
    }
}

// Expand screener suggestions
function expandScreenerSuggestions() {
    const categoriesSection = $('.screener-categories');
    const expandButton = $('#expandSuggestionsBtn');
    
    
    // Remove collapsed class and animate
    categoriesSection.removeClass('collapsed');
    
    // Remove the expand button
    expandButton.parent().remove();
    
    // Scroll to suggestions for better UX
    $('html, body').animate({
        scrollTop: categoriesSection.offset().top - 100
    }, 500);
}

// Clear search and show suggestions again
function resetScreenerToSuggestions() {
    // Clear search input
    $('#queryInput').val('');
    
    // Hide all result containers but keep AI panel visible
    $('#singleSheetContainer, #noResultsContainer, #errorContainer, #loadingContainer').addClass('hidden');
    
    // Show entire suggestions section
    $('.screener-categories').removeClass('hidden');
    $('#suggestionsContent').removeClass('hidden');
    $('#suggestionsChevron').removeClass('fa-chevron-down').addClass('fa-chevron-up');
    
    // Reset search button state
    updateSearchButtonState(false);
    
    // Focus on search input
    $('#queryInput').focus();
    
}

// Make functions globally available
window.expandScreenerSuggestions = expandScreenerSuggestions;
window.resetScreenerToSuggestions = resetScreenerToSuggestions;

// =============================================================================
// EXPAND TRUNCATED VALUE FUNCTIONALITY
// =============================================================================

// Global variables for expand functionality
let currentQuestion = '';
let currentScreenId = null; // Track if we're viewing a saved screen
let isViewingSavedScreen = false; // Flag to indicate if we're viewing a saved screen

// Function to expand truncated values
async function expandTruncatedValue(columnName, rowIndex, sheetIndex) {
    try {
        // Show loading state
        $('#expandValueContent').html('<div class="flex items-center justify-center py-4"><i class="fas fa-spinner fa-spin mr-2"></i>Loading full value...</div>');
        $('#expandValueColumn').text(columnName);
        $('#expandValueModal').removeClass('hidden').addClass('flex');

        let response;
        let requestBody;

        // Check if we're viewing a saved screen
        if (isViewingSavedScreen && currentScreenId) {
            // Use the screens expand endpoint
            requestBody = {
                screen_id: currentScreenId,
                row_index: rowIndex,
                column_name: columnName,
                sheet_index: sheetIndex
            };


            response = await fetch(`${CONFIG.apiBaseUrl}/screens/expand-value`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('authToken')}`
                },
                body: JSON.stringify(requestBody)
            });
        } else {
            // Use the regular query expand endpoint
            const question = currentQuestion || $('#queryInput').val();
            if (!question) {
                showToast('No current query found', 'error');
                return;
            }

            requestBody = {
                question: question,
                row_index: rowIndex,
                column_name: columnName,
                sheet_index: sheetIndex
            };


            response = await fetch(`${CONFIG.apiBaseUrl}/screener/query/expand-value`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('authToken')}`
                },
                body: JSON.stringify(requestBody)
            });
        }

        const result = await response.json();

        if (!response.ok || !result.success) {
            const errorMsg = result.error || 'Failed to retrieve full value';
            $('#expandValueContent').html(`<div class="text-red-600"><i class="fas fa-exclamation-triangle mr-2"></i>${errorMsg}</div>`);
            return;
        }

        // Display the full value
        $('#expandValueContent').text(result.full_value || 'No value available');

    } catch (error) {

        // Show error with fallback option
        $('#expandValueContent').html(`
            <div class="text-red-600 mb-4">
                <i class="fas fa-exclamation-triangle mr-2"></i>Error: ${error.message}
            </div>
            <div class="text-sm text-text-secondary">
                <p>This might be due to:</p>
                <ul class="list-disc list-inside mt-2 space-y-1">
                    <li>Server not running</li>
                    <li>Network connectivity issues</li>
                    <li>Authentication problems</li>
                </ul>
            </div>
        `);
    }
}

// Make expandTruncatedValue available globally
window.expandTruncatedValue = expandTruncatedValue;

// Setup expand value modal event listeners
function setupExpandValueModal() {
    // Close modal with X button
    $('#closeExpandValueModalBtn').on('click', function() {
        $('#expandValueModal').removeClass('flex').addClass('hidden');
    });

    // Close modal with footer button
    $('#closeExpandValueFooterBtn').on('click', function() {
        $('#expandValueModal').removeClass('flex').addClass('hidden');
    });

    // Close modal when clicking backdrop
    $('#expandValueModal').on('click', function(e) {
        if (e.target === this) {
            $(this).removeClass('flex').addClass('hidden');
        }
    });

    // Close modal with Escape key
    $(document).on('keydown', function(e) {
        if (e.key === 'Escape' && !$('#expandValueModal').hasClass('hidden')) {
            $('#expandValueModal').removeClass('flex').addClass('hidden');
        }
    });
}

// Initialize expand value modal
setupExpandValueModal();

// =============================================================================
// USAGE MODAL FUNCTIONALITY
// =============================================================================

// Usage Modal Functions
function openUsageModal() {
    trackEvent('open_usage_modal');
    $('#usageModal').removeClass('hidden').addClass('flex');
    loadUsageData();
}

function closeUsageModal() {
    $('#usageModal').removeClass('flex').addClass('hidden');
}

async function loadUsageData() {
    try {
        // Show loading state
        $('#usageLoading').removeClass('hidden');
        $('#usageContent').addClass('hidden');
        $('#usageError').addClass('hidden');


        // Fetch usage data
        const response = await fetch(`${CONFIG.apiBaseUrl}/user/usage`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('authToken')}`
            }
        });


        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
        }

        const usageData = await response.json();

        // Track usage data view
        trackEvent('view_usage_data', {
            total_requests: usageData.total_requests,
            monthly_requests: usageData.monthly_requests
        });

        // Update UI with usage data
        $('#totalRequests').text(usageData.total_requests.toLocaleString());
        $('#monthlyRequests').text(usageData.monthly_requests.toLocaleString());

        // Remove usage indicator display (keeping rate limiting logic intact)
        $('#usageIndicator').addClass('hidden');

        // Calculate reset time for minute limit (removed - only error messages shown when limits reached)
        const now = new Date();
        const resetTime = new Date(now.getTime() + 60 * 1000); // 60 seconds from now
        const secondsRemaining = Math.max(0, Math.floor((resetTime - now) / 1000));
        // Reset time display removed - only error messages shown when limits reached

        // Show content
        $('#usageLoading').addClass('hidden');
        $('#usageContent').removeClass('hidden');

    } catch (error) {
        // Track usage data error
        trackEvent('usage_data_error', {
            error: error.message
        });


        // Show error state
        $('#usageLoading').addClass('hidden');
        $('#usageContent').addClass('hidden');
        $('#usageError').removeClass('hidden');

        // Update error message
        $('#usageError .text-red-600').text(`Failed to load usage data: ${error.message}`);
    }
}

// Load usage data for profile modal
async function loadProfileUsageData() {
    try {
        // Show loading state
        $('#profileUsageLoading').removeClass('hidden');
        $('#profileUsageContent').addClass('hidden');
        $('#profileUsageError').addClass('hidden');


        // Fetch usage data
        const response = await fetch(`${CONFIG.apiBaseUrl}/user/usage`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('authToken')}`
            }
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
        }

        const usageData = await response.json();

        // Update UI with usage data
        $('#profileTotalRequests').text(usageData.total_requests.toLocaleString());
        $('#profileMonthlyRequests').text(usageData.monthly_requests.toLocaleString());

        // Update usage indicator for rate limiting
        updateUsageIndicator(usageData.monthly_requests, 20);

        // Show content
        $('#profileUsageLoading').addClass('hidden');
        $('#profileUsageContent').removeClass('hidden');

    } catch (error) {

        // Show error state
        $('#profileUsageLoading').addClass('hidden');
        $('#profileUsageContent').addClass('hidden');
        $('#profileUsageError').removeClass('hidden');
    }
}

function setupUsageModal() {
    // Usage modal functionality removed - now integrated into profile modal
}

function setupOnboardingModal() {
    // Setup onboarding modal event listeners
    $('#closeOnboardingModal').on('click', hideOnboardingModal);
    $('#completeOnboarding').on('click', completeOnboarding);

    // Close onboarding modal when clicking backdrop
    $('#onboardingModal').on('click', function(e) {
        if (e.target === this) {
            hideOnboardingModal();
        }
    });

    // Close onboarding modal with Escape key
    $(document).on('keydown', function(e) {
        if (e.key === 'Escape' && !$('#onboardingModal').hasClass('hidden')) {
            hideOnboardingModal();
        }
    });
}

// Profile Management Functions
function setupProfileModal() {
    // Setup profile modal event listeners
    $('#profileBtn').on('click', showProfileModal);
    $('#closeProfileModal').on('click', hideProfileModal);
    $('#cancelProfileBtn').on('click', hideProfileModal);
    $('#saveProfileBtn').on('click', saveProfile);
    $('#changePasswordBtn').on('click', showChangePasswordModal);

    // Profile tab switching
    $(document).on('click', '.profile-tab-btn', function() {
        const tabName = $(this).data('profile-tab');
        switchProfileTab(tabName);
    });

    // Close profile modal when clicking backdrop
    $('#profileModal').on('click', function(e) {
        if (e.target === this) {
            hideProfileModal();
        }
    });

    // Close profile modal with Escape key
    $(document).on('keydown', function(e) {
        if (e.key === 'Escape' && !$('#profileModal').hasClass('hidden')) {
            hideProfileModal();
        }
    });
}

function setupChangePasswordModal() {
    // Setup change password modal event listeners
    $('#closeChangePasswordModal').on('click', hideChangePasswordModal);
    $('#cancelChangePasswordBtn').on('click', hideChangePasswordModal);
    $('#savePasswordBtn').on('click', changePassword);

    // Close change password modal when clicking backdrop
    $('#changePasswordModal').on('click', function(e) {
        if (e.target === this) {
            hideChangePasswordModal();
        }
    });

    // Close change password modal with Escape key
    $(document).on('keydown', function(e) {
        if (e.key === 'Escape' && !$('#changePasswordModal').hasClass('hidden')) {
            hideChangePasswordModal();
        }
    });
}

// Profile tab switching function
function switchProfileTab(tabName) {
    // Update tab button states
    $('.profile-tab-btn').removeClass('active');
    $(`.profile-tab-btn[data-profile-tab="${tabName}"]`).addClass('active');

    // Update tab content visibility
    $('.profile-tab-content').removeClass('active').addClass('hidden');
    $(`#${tabName}TabContent`).addClass('active').removeClass('hidden');

    // Load data based on tab
    if (tabName === 'usage') {
        loadProfileUsageData();
    }
}

function showProfileModal() {
    trackEvent('open_profile_modal');
    $('#profileModal').removeClass('hidden').addClass('flex');
    $('body').addClass('overflow-hidden');
    
    // Reset to profile tab
    switchProfileTab('profile');
    loadUserProfile();
}

function hideProfileModal() {
    $('#profileModal').addClass('hidden').removeClass('flex');
    $('body').removeClass('overflow-hidden');
    resetProfileForm();
}

function showChangePasswordModal() {
    $('#changePasswordModal').removeClass('hidden').addClass('flex');
    $('body').addClass('overflow-hidden');
    $('#changePasswordForm')[0].reset();
}

function hideChangePasswordModal() {
    $('#changePasswordModal').addClass('hidden').removeClass('flex');
    $('body').removeClass('overflow-hidden');
}

async function loadUserProfile() {
    try {
        const response = await fetch(`${CONFIG.apiBaseUrl}/user/profile`, {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('authToken')}`,
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error('Failed to load profile');
        }

        const profile = await response.json();
        populateProfileForm(profile);
    } catch (error) {
        showToast('Error loading profile', 'error');
    }
}

function populateProfileForm(profile) {
    $('#profileUsername').val(profile.username || '');
    $('#profileEmail').val(profile.email || '');
    $('#profileFullName').val(profile.full_name || '');
    $('#profileCompany').val(profile.company || '');
}

function resetProfileForm() {
    $('#profileForm')[0].reset();
}

async function saveProfile() {
    const saveBtn = $('#saveProfileBtn');
    const saveBtnText = $('#saveProfileBtnText');
    const saveSpinner = $('#saveProfileSpinner');

    // Show loading state
    saveBtn.prop('disabled', true);
    saveBtnText.text('Saving...');
    saveSpinner.removeClass('hidden');

    try {
        const formData = {
            username: $('#profileUsername').val().trim(),
            email: $('#profileEmail').val().trim() || null,
            full_name: $('#profileFullName').val().trim(),
            company: $('#profileCompany').val().trim() || null
        };

        // Validate required fields
        if (!formData.username || !formData.full_name) {
            throw new Error('Username and Full Name are required');
        }

        const response = await fetch(`${CONFIG.apiBaseUrl}/user/profile`, {
            method: 'PUT',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('authToken')}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to update profile');
        }

        const updatedProfile = await response.json();
        populateProfileForm(updatedProfile);
        showToast('Profile updated successfully', 'success');

        // Update the username in the header if it was changed
        if (updatedProfile.username) {
            $('#userName').text(updatedProfile.username);
        }
    } catch (error) {
        showToast(error.message, 'error');
    } finally {
        // Reset loading state
        saveBtn.prop('disabled', false);
        saveBtnText.text('Save Changes');
        saveSpinner.addClass('hidden');
    }
}

async function changePassword() {
    const saveBtn = $('#savePasswordBtn');
    const saveBtnText = $('#savePasswordBtnText');
    const saveSpinner = $('#savePasswordSpinner');

    // Show loading state
    saveBtn.prop('disabled', true);
    saveBtnText.text('Changing...');
    saveSpinner.removeClass('hidden');

    try {
        const currentPassword = $('#currentPassword').val();
        const newPassword = $('#newPassword').val();
        const confirmPassword = $('#confirmPassword').val();

        // Validate passwords
        if (!currentPassword || !newPassword || !confirmPassword) {
            throw new Error('All password fields are required');
        }

        if (newPassword !== confirmPassword) {
            throw new Error('New passwords do not match');
        }

        if (newPassword.length < 8) {
            throw new Error('New password must be at least 8 characters long');
        }

        const response = await fetch(`${CONFIG.apiBaseUrl}/auth/change-password`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('authToken')}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                current_password: currentPassword,
                new_password: newPassword
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to change password');
        }

        showToast('Password changed successfully', 'success');
        hideChangePasswordModal();
        $('#changePasswordForm')[0].reset();
    } catch (error) {
        showToast(error.message, 'error');
    } finally {
        // Reset loading state
        saveBtn.prop('disabled', false);
        saveBtnText.text('Change Password');
        saveSpinner.addClass('hidden');
    }
}

// Test function for expand functionality
window.testExpandValue = function() {

    // Test with dummy data
    expandTruncatedValue('test_column', 0, 0);
};


// Test function for onboarding modal
window.testOnboarding = function() {
    showOnboardingModal();
};

function formatLargeNumberFixed2(value) {
    if (value === null || value === undefined || isNaN(value)) return '--';
    const num = Number(value);
    if (Math.abs(num) >= 1e12) return (num / 1e12).toFixed(2) + 'T';
    if (Math.abs(num) >= 1e9) return (num / 1e9).toFixed(2) + 'B';
    if (Math.abs(num) >= 1e6) return (num / 1e6).toFixed(2) + 'M';
    if (Math.abs(num) >= 1e3) return (num / 1e3).toFixed(2) + 'K';
    return num.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

// Patch usages in the company profile section (lines ~3129-3207)
// Replace formatLargeNumber and formatPercentage with the new fixed2 versions
// For direct number rendering, use .toFixed(2) or toLocaleString with 2 decimals

// Force activate search section as default
function forceActivateSearchSection() {
    
    // Use the proper section switching function instead of manual CSS manipulation
    switchToSection('chat');
    
}


// Error reporting modal functions
function openErrorReportingModal() {
    trackEvent('open_error_reporting');
    $('#errorReportingModal').removeClass('hidden').addClass('flex');
    $('body').addClass('overflow-hidden');
}

function closeErrorReportingModal() {
    $('#errorReportingModal').addClass('hidden').removeClass('flex');
    $('body').removeClass('overflow-hidden');
}

function openEmailClient() {
    const subject = encodeURIComponent('StrataLens Support Request');
    const body = encodeURIComponent(`Hi Hrishi,\n\nI'm experiencing an issue with StrataLens and would appreciate your help.\n\nWhat I was trying to do:\n[Please describe what you were trying to accomplish]\n\nError message I encountered:\n[Please paste the error message here]\n\nAdditional details:\n[Any other relevant information]\n\nThank you for your help!\n\nBest regards,\n[Your name]`);
            const mailtoLink = `mailto:hrishi@stratalens.ai?subject=${subject}&body=${body}`;
    const win = window.open(mailtoLink);
    setTimeout(() => {
        closeErrorReportingModal();
        if (!win || win.closed || typeof win.closed === 'undefined') {
            if (typeof showToast === 'function') {
                showToast('Could not open your email client. Please email hrishi@stratalens.ai manually.', 'warning');
            }
        }
    }, 500);
}



// Expose error reporting modal functions to the global window object
window.openErrorReportingModal = openErrorReportingModal;
window.closeErrorReportingModal = closeErrorReportingModal;
window.openEmailClient = openEmailClient;

// Old chat function window exports removed - using new chat.js


// Expose other functions to the global window object
window.loadUserScreens = loadUserScreens;
window.switchToSection = switchToSection;
window.openUsageModal = openUsageModal;
window.loadUsageData = loadUsageData;
window.loadProfileUsageData = loadProfileUsageData;
window.switchProfileTab = switchProfileTab;

// Expose chat functions
window.startNewChat = startNewChat;
window.loadConversation = loadConversation;
window.loadChatHistory = loadChatHistory;
window.displayChatHistory = displayChatHistory;
window.openChatHistoryModal = openChatHistoryModal;
window.closeChatHistoryModal = closeChatHistoryModal;

// Expose transcript functions
window.viewTranscript = viewTranscript;
window.openTranscriptModal = openTranscriptModal;
window.closeTranscriptModal = closeTranscriptModal;

// ================================================
// CHAT FUNCTIONALITY
// ================================================

// Global conversation state
window.currentConversationId = null;  // Track active conversation

// ================================================
// CHAT HISTORY AND STATS FUNCTIONALITY
// ================================================

function setupChatHistoryAndStats() {
    
    // Chat History Button (both old and new IDs for compatibility)
    $('#chatHistoryBtn, #chatHistoryNavBtn').on('click', function() {
        openChatHistoryModal();
    });
    
    // New Chat Button (both old and new IDs for compatibility)
    $('#newChatBtn, #newChatNavBtn').on('click', function() {
        startNewChat();
    });
    
    // Close buttons
    $('#closeChatHistoryModalBtn, #chatHistoryClearBtn').on('click', function() {
        closeChatHistoryModal();
    });
    
}

function openChatHistoryModal() {
    
    // Reset pagination state
    conversationPagination = {
        currentPage: 0,
        hasMore: true,
        loading: false,
        conversations: []
    };
    
    $('body').addClass('modal-open');
    $('#chatHistoryModal').removeClass('hidden').addClass('flex');
    loadChatHistory();
}

function closeChatHistoryModal() {
    $('body').removeClass('modal-open');
    $('#chatHistoryModal').removeClass('flex').addClass('hidden');
}

function openChatStatsModal() {
    $('body').addClass('modal-open'); // Lower chat elements z-index
    $('#chatStatsModal').removeClass('hidden').addClass('flex');
    loadChatStats();
}

function closeChatStatsModal() {
    $('body').removeClass('modal-open'); // Restore chat elements z-index
    $('#chatStatsModal').removeClass('flex').addClass('hidden');
}

// Global pagination state
let conversationPagination = {
    currentPage: 0,
    hasMore: true,
    loading: false,
    conversations: []
};

async function loadChatHistory(loadMore = false) {
    try {
        if (conversationPagination.loading) {
            return;
        }
        
        conversationPagination.loading = true;
        
        if (!loadMore) {
            $('#chatHistoryLoading').removeClass('hidden');
            $('#chatHistoryList').empty();
            conversationPagination.currentPage = 0;
            conversationPagination.conversations = [];
        } else {
            $('#loadMoreConversationsBtn').prop('disabled', true).html('<i class="fas fa-spinner fa-spin mr-2"></i>Loading...');
        }
        
        const limit = 20; // Load 20 conversations at a time
        const offset = conversationPagination.currentPage * limit;
        
        const response = await fetch(`${CONFIG.apiBaseUrl}/chat/conversations?limit=${limit}&offset=${offset}`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('authToken')}`
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        
        if (data.success && data.conversations) {
            // Add new conversations to our list
            conversationPagination.conversations.push(...data.conversations);
            conversationPagination.currentPage++;
            conversationPagination.hasMore = data.conversations.length === limit; // Has more if we got a full page
            
            // Update conversation count info
            $('#conversationCount').text(`${conversationPagination.conversations.length} conversation${conversationPagination.conversations.length !== 1 ? 's' : ''}`);
            
            // Display all conversations
            displayChatHistory(conversationPagination.conversations, conversationPagination.hasMore);
        } else {
            if (!loadMore) {
                displayChatHistory([]);
            }
            conversationPagination.hasMore = false;
        }
        
        $('#chatHistoryLoading').addClass('hidden');
        conversationPagination.loading = false;
        
    } catch (error) {
        conversationPagination.loading = false;
        
        if (!loadMore) {
            $('#chatHistoryLoading').addClass('hidden');
            $('#chatHistoryList').html(`
                <div class="text-center py-8">
                    <div class="text-red-500 mb-2">
                        <i class="fas fa-exclamation-triangle text-2xl"></i>
                    </div>
                    <p class="text-text-secondary mb-4">Failed to load conversations</p>
                    <button onclick="loadChatHistory()" class="btn btn-primary btn-sm">
                        <i class="fas fa-redo mr-2"></i>Try Again
                    </button>
                </div>
            `);
        } else {
            $('#loadMoreConversationsBtn').prop('disabled', false).html('<i class="fas fa-chevron-down mr-2"></i>Load More Conversations');
            showToast('Failed to load more conversations', 'error');
        }
    }
}

function displayChatHistory(conversations, hasMore = false) {
    const container = $('#chatHistoryList');
    
    if (!conversations || conversations.length === 0) {
        container.html(`
            <div class="text-center py-12">
                <div class="w-16 h-16 bg-gradient-to-br from-blue-100 to-purple-100 dark:from-blue-900/50 dark:to-purple-900/50 rounded-full flex items-center justify-center mx-auto mb-4">
                    <i class="fas fa-comments text-blue-600 dark:text-blue-400 text-2xl"></i>
                </div>
                <h3 class="text-xl font-semibold text-text-primary mb-3">No Conversations Yet</h3>
                <p class="text-text-secondary mb-6 max-w-sm mx-auto leading-relaxed">
                    Start chatting to see your history here.
                </p>
                <button onclick="closeChatHistoryModal(); startNewChat();" class="btn btn-primary">
                    <i class="fas fa-plus mr-2"></i>Start Chatting
                </button>
            </div>
        `);
        $('#chatHistoryLoadMore').addClass('hidden');
        return;
    }
    
    
    // Clear container only if this is a fresh load
    if (conversationPagination.currentPage <= 1) {
        container.empty();
    }
    
    // Add conversations to the list
    conversations.forEach((conversation) => {
        // Skip if already displayed (prevent duplicates)
        if (container.find(`[data-conversation-id="${conversation.id}"]`).length > 0) {
            return;
        }
        
        const conversationItem = $(`
            <div class="chat-conversation-item" data-conversation-id="${conversation.id}">
                <div class="conversation-header">
                    <div class="conversation-title">${escapeHtml(conversation.title)}</div>
                    <div class="conversation-date">${formatDate(conversation.updated_at)}</div>
                </div>
                <div class="conversation-meta">
                    <span class="conversation-created">Created ${formatDate(conversation.created_at)}</span>
                </div>
            </div>
        `);
        
        // Add click handler to restore conversation
        conversationItem.on('click', function() {
            const conversationId = $(this).data('conversation-id');
            loadConversation(conversationId, conversation.title);
        });
        
        container.append(conversationItem);
    });
    
    // Handle "Load More" button
    if (hasMore && conversationPagination.hasMore) {
        $('#chatHistoryLoadMore').removeClass('hidden');
        $('#loadMoreConversationsBtn').prop('disabled', false).html('<i class="fas fa-chevron-down mr-2"></i>Load More Conversations');
        
        // Set up load more button event listener
        $('#loadMoreConversationsBtn').off('click').on('click', function() {
            loadChatHistory(true);
        });
    } else {
        $('#chatHistoryLoadMore').addClass('hidden');
    }
}

async function loadChatStats() {
    try {
        $('#chatStatsLoading').removeClass('hidden');
        $('#chatStatsContent').addClass('hidden');
        
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Mock stats data
        const mockStats = {
            totalMessages: 147,
            activeDays: 23,
            avgQuestionLength: 42,
            topConversations: [
                { preview: 'Tech companies performance analysis', messages: 8, date: '2024-01-15' },
                { preview: 'Retail profit margin comparison', messages: 6, date: '2024-01-14' },
                { preview: 'R&D spending research', messages: 4, date: '2024-01-12' }
            ]
        };
        
        renderChatStats(mockStats);
        $('#chatStatsLoading').addClass('hidden');
        $('#chatStatsContent').removeClass('hidden');
        
    } catch (error) {
        $('#chatStatsLoading').addClass('hidden');
        $('#chatStatsError').removeClass('hidden');
    }
}

function renderChatStats(stats) {
    $('#totalChatMessages').text(stats.totalMessages);
    $('#activeChatDays').text(stats.activeDays);
    $('#avgQuestionLength').text(stats.avgQuestionLength + ' chars');
    
    // Render top conversations
    const conversationsContainer = $('#topConversations');
    conversationsContainer.empty();
    
    stats.topConversations.forEach((conv, index) => {
        const convItem = $(`
            <div class="conversation-item">
                <div class="conversation-preview">${conv.preview}</div>
                <div class="conversation-meta">
                    <span>${conv.messages} messages</span>
                    <span>${formatDate(conv.date)}</span>
                </div>
            </div>
        `);
        conversationsContainer.append(convItem);
    });
}

function performChatHistorySearch() {
    const query = $('#chatHistorySearch').val().trim();
    const dateFrom = $('#chatHistoryDateFrom').val();
    const dateTo = $('#chatHistoryDateTo').val();
    
    // Implement search functionality
}

// Removed copyChatItem and shareChatItem - no longer needed for simplified history

async function loadConversation(conversationId, conversationTitle) {
    
    try {
        // Close history modal
        closeChatHistoryModal();
        
        // Switch to chat section
        if (!$('#chatSection').hasClass('active')) {
            switchToSection('chat');
        }
        
        // Show loading state
        $('#chatMessages').html(`
            <div class="chat-message system">
                <div class="message-avatar">
                    <i class="fas fa-spinner fa-spin text-blue-500"></i>
                </div>
                <div class="chat-bubble">
                    <div class="message-content">
                        <p><strong>🔄 Loading "${escapeHtml(conversationTitle)}"</strong></p>
                        <p class="text-sm opacity-75 mt-1">Loading...</p>
                    </div>
                </div>
            </div>
        `);
        
        // Fetch full conversation from API
        const response = await fetch(`${CONFIG.apiBaseUrl}/chat/conversations/${conversationId}`, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('authToken')}`
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        if (!data.success || !data.conversation) {
            throw new Error('No conversation data received');
        }
        
        const conversation = data.conversation;
        
        // Clear and rebuild chat
        $('#chatMessages').empty();
        
        // Add conversation header
        $('#chatMessages').append(`
                <div class="chat-message system">
                <div class="message-avatar">
                    <i class="fas fa-comments text-blue-500"></i>
                </div>
                    <div class="chat-bubble">
                    <div class="message-content">
                        <p><strong>💬 "${escapeHtml(conversation.title)}"</strong></p>
                        <p class="text-sm opacity-75 mt-1">${conversation.messages.length} messages • Started ${formatDate(conversation.created_at)}</p>
                    </div>
                    </div>
                </div>
            `);
        
        // Add all messages with proper markdown rendering
        conversation.messages.forEach((msg) => {
            if (msg.role === 'user') {
                $('#chatMessages').append(`
                    <div class="chat-message user">
                        <div class="message-avatar">
                            <i class="fas fa-user-circle"></i>
                        </div>
                        <div class="chat-bubble">
                            <div class="message-content">${escapeHtml(msg.content)}</div>
                        </div>
                    </div>
                `);
            } else if (msg.role === 'assistant') {
                // Render assistant message with markdown
                const renderedContent = renderMarkdown(msg.content);
                
                const assistantMessage = $(`
                    <div class="chat-message assistant">
                        <div class="chat-bubble">
                            <div class="message-content">${renderedContent}</div>
                        </div>
                    </div>
                `);
                
                // Add citations if available
                if (msg.citations && msg.citations.length > 0) {
                    const citationsHtml = createCitationsHtml(msg.citations);
                    assistantMessage.find('.chat-bubble').append(citationsHtml);
                }
                
                $('#chatMessages').append(assistantMessage);
            }
        });
        
        // Continuation notice removed for cleaner chat interface
        
        // Set active conversation ID
        window.currentConversationId = conversationId;
        
        // Scroll to bottom and focus
        setTimeout(() => {
            const chatSection = document.getElementById('chatSection');
            if (chatSection) {
                chatSection.scrollTop = chatSection.scrollHeight;
            }
        $('#chatInput').focus();
        }, 100);
        
        showToast(`Loaded: "${conversation.title}"`, 'success');
        
    } catch (error) {
        
        $('#chatMessages').html(`
            <div class="chat-message system">
                <div class="message-avatar">
                    <i class="fas fa-exclamation-triangle text-red-500"></i>
                </div>
                <div class="chat-bubble">
                    <div class="message-content">
                        <p><strong>❌ Failed to load conversation</strong></p>
                        <p class="text-sm opacity-75 mt-1">${error.message}</p>
                        <div class="mt-3 flex gap-2">
                            <button onclick="loadConversation('${conversationId}', '${escapeHtml(conversationTitle)}')" class="btn btn-primary btn-sm">
                                <i class="fas fa-redo mr-1"></i>Try Again
                            </button>
                            <button onclick="startNewChat()" class="btn btn-secondary btn-sm">
                                <i class="fas fa-plus mr-1"></i>New Chat
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `);
        
        showToast('Failed to load conversation', 'error');
    }
}

// Start a new chat
function startNewChat() {
    
    // Clear current conversation ID
    window.currentConversationId = null;
    
    // Clear chat messages
    $('#chatMessages').empty();
    
    // Add fresh welcome message
    const welcomeMessage = `
        <div class="chat-message assistant">
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="chat-bubble">
                <div class="message-content">
                    <p>👋 Hello! I'm your AI financial assistant. I can help you analyze financial data and earnings transcripts.</p>
                    <p class="text-sm opacity-75 mt-2">Ask me anything about companies, financial metrics, or market trends!</p>
            </div>
        </div>
        </div>
    `;
    $('#chatMessages').append(welcomeMessage);
    
    
    // Switch to chat section if not already there
    if (!$('#chatSection').hasClass('active')) {
        switchToSection('chat');
    }
    
    // Focus on input
    $('#chatInput').focus();
    
    // Show success message
    showToast('New conversation started', 'success');
    
}

// Clean chat system - no legacy code

// Helper function to create citations HTML
function createCitationsHtml(citations) {
    return `
            <div class="citations">
                <div class="citations-header">
                <span class="sources-count">${citations.length} source${citations.length !== 1 ? 's' : ''}</span>
                <button class="toggle-sources-btn" onclick="toggleCitations(this)">
                        <i class="fas fa-chevron-down"></i>
                    </button>
                </div>
            <div class="citations-content" style="display: none;">
                    <div class="citations-list">
                    ${citations.map((citation, index) => `
                        <div class="citation-item" data-citation-index="${index}">
                                <div class="citation-header">
                                    <div class="citation-header-left">
                                        <span class="citation-company">${escapeHtml(citation.company || 'Unknown')}</span>
                                        <span class="citation-quarter">${escapeHtml(citation.quarter || 'N/A')}</span>
                                        ${citation.transcript_available ? '<span class="transcript-badge">📄 Transcript</span>' : ''}
                                    </div>
                                    ${citation.transcript_available ? `<button class="view-transcript-btn" onclick="viewTranscript('${citation.company}', '${citation.quarter}', ${index})">
                                        <i class="fas fa-file-alt mr-1"></i>View Full Transcript
                                    </button>` : ''}
                                </div>
                            <div class="citation-preview">${escapeHtml(citation.chunk_text || citation.content || citation.text || 'No preview available')}</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
}

// Helper function to toggle citations visibility
function toggleCitations(button) {
    const citationsContent = $(button).closest('.citations').find('.citations-content');
    const icon = $(button).find('i');
    
    if (citationsContent.is(':visible')) {
        citationsContent.slideUp(200);
        icon.removeClass('fa-chevron-up').addClass('fa-chevron-down');
    } else {
        citationsContent.slideDown(200);
        icon.removeClass('fa-chevron-down').addClass('fa-chevron-up');
    }
}

// Helper function to escape HTML
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Helper function to render markdown (same logic as ChatInterface)
function renderMarkdown(content) {
    if (typeof marked !== 'undefined') {
        try {
            // Configure marked.js for better rendering (same as ChatInterface)
            marked.setOptions({
                breaks: true,
                gfm: true,
                headerIds: false,
                mangle: false
            });
            return marked.parse(content);
        } catch (error) {
            return escapeHtml(content);
        }
    } else {
        return escapeHtml(content);
    }
}

// Get relevant chunks for a specific company and quarter
function getRelevantChunksForCompany(company, quarter) {
    if (!window.lastCitations || window.lastCitations.length === 0) {
        return [];
    }
    
    // Convert quarter format for comparison
    const quarterForComparison = quarter.replace(' ', '_').toLowerCase();
    
    return window.lastCitations
        .filter(citation => citation.company === company && citation.quarter === quarterForComparison)
        .map(citation => ({
            chunk_text: citation.chunk_text,
            chunk_id: citation.chunk_id || '',
            relevance_score: citation.relevance_score || 0
        }));
}

// Format transcript with proper speaker separation
function formatTranscriptWithSpeakers(transcriptText, relevantChunks = []) {
    if (!transcriptText) return 'No transcript available';
    
    // Enhanced speaker patterns to better match earnings call transcripts
    const speakerPatterns = [
        // Pattern for full names like "Kenneth J. Dorell:" or "Mark Elliot Zuckerberg:"
        /^([A-Z][a-zA-Z\s]+[A-Za-z]):\s/gm,
        // Pattern for names with middle initials like "Kenneth J. Dorell:"
        /^([A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+):\s/gm,
        // Pattern for names with periods like "Mr. John Pitzer:"
        /^([A-Z][a-z]*\.?\s*[A-Za-z\s]+[A-Za-z]):\s/gm,
        // Pattern for "Operator:" style single words
        /^([A-Z][a-z]+):\s/gm,
        // Pattern for names with hyphens like "Lip-Bu Tan:"
        /^([A-Za-z]+-[A-Za-z\s]+[A-Za-z]):\s/gm,
    ];
    
    let formattedText = transcriptText;
    let hasSpeakers = false;
    
    // Apply formatting for each speaker pattern
    speakerPatterns.forEach(pattern => {
        formattedText = formattedText.replace(pattern, (match, speaker) => {
            const cleanSpeaker = speaker.trim();
            // Skip if it's too short, contains numbers, or is a common word
            const commonWords = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'all', 'this', 'that'];
            if (cleanSpeaker.length < 3 || /\d/.test(cleanSpeaker) || commonWords.includes(cleanSpeaker.toLowerCase())) {
                return match;
            }
            hasSpeakers = true;
            return `<div class="speaker-section mb-4"><div class="speaker-name font-semibold text-accent-primary mb-2">${cleanSpeaker}:</div>`;
        });
    });
    
    // Close any unclosed speaker sections and wrap content
    if (hasSpeakers) {
        formattedText = formattedText.replace(/(<div class="speaker-section mb-4"><div class="speaker-name font-semibold text-accent-primary mb-2">[^<]*<\/div>)([^<]*?)(?=<div class="speaker-section mb-4">|$)/gs, 
            (match, speakerTag, content) => {
                // Clean up the content and wrap it properly
                const cleanContent = content.trim().replace(/\n\s*\n/g, '\n');
                return speakerTag + `<div class="speaker-content text-text-primary leading-relaxed">${cleanContent}</div></div>`;
            });
    } else {
        // If no speakers were detected, wrap the entire content
        formattedText = `<div class="speaker-section mb-4"><div class="speaker-content text-text-primary leading-relaxed">${formattedText}</div></div>`;
    }
    
    // Apply chunk highlighting if relevant chunks are provided
    if (relevantChunks && relevantChunks.length > 0) {
        formattedText = highlightRelevantChunks(formattedText, relevantChunks);
    }
    
    return formattedText;
}

// Highlight relevant chunks in the transcript
function highlightRelevantChunks(formattedText, relevantChunks) {
    let highlightedText = formattedText;
    
    relevantChunks.forEach((chunk, index) => {
        if (chunk.chunk_text && chunk.chunk_text.trim().length > 0) {
            // Escape special regex characters in the chunk text
            const escapedChunkText = chunk.chunk_text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            
            // Create a regex that matches the chunk text (case-insensitive, flexible whitespace)
            const chunkRegex = new RegExp(escapedChunkText.replace(/\s+/g, '\\s+'), 'gi');
            
            // Replace matches with highlighted version
            highlightedText = highlightedText.replace(chunkRegex, (match) => {
                const relevanceScore = chunk.relevance_score || 0;
                const highlightIntensity = Math.min(Math.max(relevanceScore * 100, 20), 100); // Scale to 20-100%
                
                return `<mark class="chunk-highlight" style="background-color: rgba(59, 130, 246, ${highlightIntensity / 100}); padding: 2px 4px; border-radius: 3px; transition: all 0.2s ease;" title="Relevance: ${(relevanceScore * 100).toFixed(1)}%">${match}</mark>`;
            });
        }
    });
    
    return highlightedText;
}

// Function to view full transcript from citation
function viewTranscript(company, quarter, citationIndex) {
    
    try {
        // Parse quarter to extract year and quarter number
        const quarterMatch = quarter.match(/(\d{4})_Q(\d)/);
        if (!quarterMatch) {
            showToast('Invalid quarter format', 'error');
            return;
        }
        
        const year = parseInt(quarterMatch[1]);
        const quarterNum = parseInt(quarterMatch[2]);
        
        // Show loading toast
        showToast(`Loading ${company} ${quarter} transcript...`, 'info');
        
        // Use unified transcript endpoint (works for both authenticated and demo users)
        const endpoint = `${CONFIG.apiBaseUrl}/transcript/${company}/${year}/${quarterNum}`;
        
        // Prepare headers - include auth token if available
        const headers = {};
        const authToken = localStorage.getItem('authToken');
        if (authToken) {
            headers['Authorization'] = `Bearer ${authToken}`;
        }
        
        // Fetch transcript from API
        fetch(endpoint, { headers })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.success && data.transcript_text) {
                // Get relevant chunks for highlighting
                const relevantChunks = getRelevantChunksForCompany(company, quarter);
                
                // Open transcript in a modal or new window
                openTranscriptModal(company, quarter, data.transcript_text, data.metadata, relevantChunks);
            } else {
                throw new Error('Transcript data not available');
            }
        })
        .catch(error => {
            
            // Check if it's a 404 error with helpful message
            if (error.message.includes('404')) {
                if (error.message.includes('Chunks are available for search')) {
                    showToast(`Transcript viewing unavailable for ${company} ${quarter}. You can still ask questions about this company's earnings - the AI has access to relevant information from this quarter.`, 'warning');
                } else {
                    showToast(`No transcript data available for ${company} ${quarter}. This transcript may not be in our database.`, 'error');
                }
            } else {
            showToast(`Failed to load ${company} transcript: ${error.message}`, 'error');
            }
        });
        
    } catch (error) {
        showToast('Failed to view transcript', 'error');
    }
}

// Function to open transcript in a modal
function openTranscriptModal(company, quarter, transcriptText, metadata, relevantChunks = []) {
    
    // Remove any existing transcript modal first
    $('#transcriptModal').remove();
    
    // Create modal HTML with maximum z-index to appear above everything
    const modalHtml = `
        <div id="transcriptModal" style="position: fixed !important; top: 0 !important; left: 0 !important; width: 100vw !important; height: 100vh !important; z-index: 2147483647 !important; display: flex !important; align-items: center !important; justify-content: center !important; background: rgba(0, 0, 0, 0.75) !important; backdrop-filter: blur(4px) !important; padding: 1rem !important; margin: 0 !important; box-sizing: border-box !important;">
            <div class="card max-w-4xl w-full max-h-[90vh] bg-bg-secondary dark:bg-bg-primary" style="position: relative !important; z-index: 2147483647 !important; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5) !important;">
                <div class="px-6 py-4 border-b border-border-primary">
                    <div class="flex items-center justify-between">
                        <h3 class="text-lg font-semibold flex items-center gap-2">
                            <i class="fas fa-file-contract text-accent-primary"></i>
                            ${escapeHtml(company)} ${escapeHtml(quarter)} Earnings Transcript
                            ${relevantChunks.length > 0 ? `<span class="text-sm font-normal text-text-secondary">(${relevantChunks.length} relevant sections highlighted)</span>` : ''}
                        </h3>
                        <button id="transcriptModalCloseBtn" class="w-10 h-10 flex items-center justify-center text-text-tertiary hover:text-text-primary rounded-full hover:bg-slate-100 dark:hover:bg-slate-700/50 transition-all">
                            <i class="fas fa-times text-lg"></i>
                        </button>
                    </div>
                </div>
                <div class="p-6 overflow-y-auto max-h-[70vh]">
                    <div class="mb-4">
                        <div class="flex items-center gap-4 text-sm text-text-secondary">
                            <span><i class="fas fa-building mr-1"></i>${escapeHtml(company)}</span>
                            <span><i class="fas fa-calendar mr-1"></i>${escapeHtml(quarter)}</span>
                            ${metadata?.date ? `<span><i class="fas fa-clock mr-1"></i>${escapeHtml(metadata.date)}</span>` : ''}
                            ${relevantChunks.length > 0 ? `<span><i class="fas fa-highlighter mr-1"></i>${relevantChunks.length} relevant sections</span>` : ''}
                        </div>
                    </div>
                    <div class="prose prose-sm max-w-none bg-bg-primary p-4 rounded-lg border border-border-primary">
                        <div class="text-sm leading-relaxed">
                            ${formatTranscriptWithSpeakers(transcriptText, relevantChunks)}
                        </div>
                    </div>
                </div>
                <div class="px-6 py-4 border-t border-border-primary bg-bg-tertiary">
                    <div class="flex justify-between items-center">
                        <div class="text-sm text-text-secondary">
                            <i class="fas fa-info-circle mr-1"></i>
                            Press Escape or click outside to close
                        </div>
                        <button id="transcriptModalFooterCloseBtn" class="btn btn-secondary">
                            <i class="fas fa-times mr-2"></i>Close
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Add modal to body using vanilla JS to ensure proper placement
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = modalHtml;
    const modalElement = tempDiv.firstElementChild;
    document.body.appendChild(modalElement);
    
    // Set up event listeners after modal is added to DOM
    setTimeout(() => {
        // Close button in header
        $('#transcriptModalCloseBtn').on('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            closeTranscriptModal();
        });
        
        // Close button in footer
        $('#transcriptModalFooterCloseBtn').on('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            closeTranscriptModal();
        });
        
        // Close on backdrop click
        $('#transcriptModal').on('click', function(e) {
            if (e.target === this) {
                closeTranscriptModal();
            }
        });
        
        // Close on Escape key
        $(document).on('keydown.transcriptModal', function(e) {
            if (e.key === 'Escape') {
                closeTranscriptModal();
            }
        });
        
    }, 100);
}

// Function to close transcript modal
function closeTranscriptModal() {
    
    try {
        // Remove escape key listener
        $(document).off('keydown.transcriptModal');
        
        // Remove modal immediately (no animation to avoid issues)
        const modal = $('#transcriptModal');
        if (modal.length) {
            modal.remove();
        } else {
        }
        
        // Show confirmation
        showToast('Transcript closed', 'info');
        
    } catch (error) {
        // Force remove any transcript modals
        $('[id*="transcript"]').remove();
    }
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
        year: 'numeric', 
        month: 'short', 
        day: 'numeric' 
    });
}

// Setup chat visibility handling for background processing
function setupChatVisibilityHandling() {
    
    // Handle page visibility changes
    document.addEventListener('visibilitychange', function() {
        if (document.hidden) {
        } else {
            // When user returns to page, clear any notification indicators
            if (pendingChatNotifications > 0) {
                pendingChatNotifications = 0;
                updateChatTabIndicator();
            }
        }
    });
    
    // Handle tab switching within the application
    $(document).on('click', '[data-tab="chat"]', function() {
        chatTabVisible = true;
        pendingChatNotifications = 0;
        updateChatTabIndicator();
    });
    
    // Track when user switches away from chat tab
    $(document).on('click', '[data-tab]:not([data-tab="chat"])', function() {
        chatTabVisible = false;
    });
}

// Update chat tab indicator for background notifications
function updateChatTabIndicator() {
    const chatTabElement = $('[data-tab="chat"]');
    if (chatTabElement.length) {
        if (pendingChatNotifications > 0) {
            // Add notification indicator
            if (!chatTabElement.find('.notification-badge').length) {
                chatTabElement.append(`<span class="notification-badge">${pendingChatNotifications}</span>`);
            } else {
                chatTabElement.find('.notification-badge').text(pendingChatNotifications);
            }
        } else {
            // Remove notification indicator
            chatTabElement.find('.notification-badge').remove();
        }
    }
}

async function testChatFunction() {
    
    try {
        // Test the debug endpoint first
        const response = await fetch(`${CONFIG.apiBaseUrl}/debug/test-chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Add the test message to chat
        // addChatMessage('assistant', data.answer, 'normal', data.citations || []); // DISABLED
        
        return true;
    } catch (error) {
        // addChatMessage('assistant', `Test failed: ${error.message}`, 'error'); // DISABLED
        return false;
    }
}

// Initialize chat functionality - DISABLED (using new chat.js)
function initializeChat() {
    return; // Disabled - using new ChatGPT-style interface
    
    // Prevent multiple initializations
    if (window.chatInitialized) {
        return;
    }
    
    // ✅ BACKGROUND PROCESSING: Handle page visibility changes
    setupChatVisibilityHandling();
    
    // ✅ CHAT MODALS: Setup history and stats functionality
    setupChatHistoryAndStats();
    
    window.testChat = testChatFunction;
    
    window.testChatMessage = () => {
        // addChatMessage('assistant', 'This is a test message to verify chat display is working!', 'normal', []); // DISABLED
    };
    
    window.testDuplicateCall = () => {
        // sendChatMessage(); // DISABLED
        // sendChatMessage(); // This should be ignored // DISABLED
    };
    
    const chatInput = document.getElementById('chatInput');
    const sendButton = document.getElementById('sendChatButton');
    const messagesContainer = document.getElementById('chatMessages');

    // Check if marked library is available
    if (typeof marked === 'undefined') {
        window.markdownEnabled = false;
    } else {
        window.markdownEnabled = true;
    }


    if (chatInput && sendButton) {
        // Send message on Enter key - DISABLED (using new chat.js)
        chatInput.addEventListener('keydown', function(e) {
            return; // Disabled - using new ChatGPT-style interface
        });

        // Auto-resize textarea - DISABLED (using new chat.js)
        chatInput.addEventListener('input', function() {
            return; // Disabled - using new ChatGPT-style interface
        });

        // Send message on button click - DISABLED (using new chat.js)
        sendButton.addEventListener('click', function(e) {
            return; // Disabled - using new ChatGPT-style interface
        });

        // Clear chat button
        const clearButton = document.getElementById('clearChatBtn');
        if (clearButton) {
            clearButton.addEventListener('click', clearChatHistory);
        } else {
        }

        // Export chat button
        const exportButton = document.getElementById('exportChatBtn');
        if (exportButton) {
            exportButton.addEventListener('click', function() {
                // TODO: Implement export functionality
                showToast('Export functionality coming soon!', 'info');
            });
        } else {
        }


        // Clear any existing messages and load chat history
        if (messagesContainer) {
            messagesContainer.innerHTML = '';
        }
        loadChatHistory();

        // Focus on input for better UX
        setTimeout(() => {
            chatInput.focus();
        }, 100);
        
        // Mark chat as initialized
        window.chatInitialized = true;
    } else {
    }
}

// Insert quick message into chat input
// Old insertQuickMessage function removed - using new chat.js

// Show typing indicator
function showTypingIndicator() {
    const messagesContainer = document.getElementById('chatMessages');
    if (!messagesContainer) return;

    // Remove any existing typing indicator
    hideTypingIndicator();

    const typingDiv = document.createElement('div');
    typingDiv.id = 'typingIndicator';
    typingDiv.className = 'typing-indicator';
    typingDiv.innerHTML = `
        <div class="chat-avatar assistant">
            <i class="fas fa-robot"></i>
        </div>
        <div class="typing-bubble">
            <span style="margin-right: 0.5rem;">AI is typing</span>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;

    messagesContainer.appendChild(typingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Hide typing indicator
function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Old isProcessingSendMessage variable removed - using new chat.js

// Send chat message - DISABLED (using new chat.js)
// Old sendChatMessage function removed - using new chat.js

// Add chat message to UI
// Old addChatMessage function completely removed - using new chat.js
// Old createCitationsElement function completely removed - using new chat.js  
// Old expandCitation function completely removed - using new chat.js

// Old loadChatHistory function removed - using new chat.js

// Old displayChatHistory function removed - using new chat.js

// Old clearChatHistory function removed - using new chat.js

// =============================================================================
// REGISTRATION FUNCTIONALITY
// =============================================================================

async function handleRegister(e) {
    e.preventDefault();
    
    const username = $('#regUsername').val().trim();
    const email = $('#regEmail').val().trim();
    const password = $('#regPassword').val();
    const passwordConfirm = $('#regPasswordConfirm').val();
    
    // Validate required fields
    if (!username || !email || !password || !passwordConfirm) {
        showToast('Please fill in all required fields', 'warning');
        return;
    }
    
    // Validate password match
    if (password !== passwordConfirm) {
        showToast('Passwords do not match', 'warning');
        return;
    }
    
    // Username is already captured from the form field above
    
    setRegisterLoading(true);
    
    try {
        const response = await fetch(`${CONFIG.apiBaseUrl}/auth/register`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
            username: username,
            email: email,
            full_name: username,
            password: password
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Switch to login tab
            showLoginTab();
            showToast('Registration successful! Please sign in with your credentials.', 'success');
            
            // Clear the form
            $('#registerForm')[0].reset();
            
            // Track successful registration
            trackEvent('user_registration_success', {
                username: username,
                has_email: !!email,
                has_reason: !!reason
            });
            
        } else {
            showToast(data.detail || 'Registration failed', 'error');
        }
        
    } catch (error) {
        showToast('Registration failed. Please try again.', 'error');
    } finally {
        setRegisterLoading(false);
    }
}

function setRegisterLoading(loading) {
    const btnText = $('#registerBtnText');
    const spinner = $('#registerSpinner');
    
    if (loading) {
        btnText.hide();
        spinner.show();
    } else {
        btnText.show();
        spinner.hide();
    }
}

// =============================================================================
// MAGIC LINK AUTHENTICATION
// =============================================================================

async function handleMagicLink(e) {
    e.preventDefault();
    
    const email = $('#magicLinkEmail').val().trim();
    
    if (!email) {
        showToast('Please enter your email address', 'warning');
        return;
    }
    
    const btnText = $('#magicLinkBtnText');
    const spinner = $('#magicLinkSpinner');
    
    try {
        btnText.hide();
        spinner.show();
        
        const response = await fetch(`${CONFIG.apiBaseUrl}/auth/magic-link`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ email: email })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showToast('Magic link sent! Check your email and click the link to sign in.', 'success');
            $('#magicLinkForm')[0].reset();
            
            // Track magic link request
            trackEvent('magic_link_requested', { email: email });
        } else {
            showToast(data.detail || 'Failed to send magic link', 'error');
        }
        
    } catch (error) {
        showToast('Network error. Please try again.', 'error');
    } finally {
        btnText.show();
        spinner.hide();
    }
}

// =============================================================================
// GOOGLE OAUTH AUTHENTICATION
// =============================================================================

async function handleGoogleAuth() {
    try {
        // Track Google auth attempt
        trackEvent('google_auth_initiated');
        
        // Redirect to Google OAuth endpoint
        window.location.href = `${CONFIG.apiBaseUrl}/auth/google`;
        
    } catch (error) {
        showToast('Failed to initiate Google authentication', 'error');
    }
}

// Authentication event listeners removed - using request access modal only
$(document).ready(function() {
    // All authentication handlers removed - only request access modal is available
});

