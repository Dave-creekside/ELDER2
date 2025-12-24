// ELDER Dashboard JavaScript

// Global state
let socket = null;
let currentTab = 'graph';

// DOM elements cache
const elements = {
    tabs: null,
    tabContents: null,
    connectionStatus: null,
    connectionText: null,
    chatInput: null,
    sendButton: null,
    messagesContainer: null,
    thinkingIndicator: null
};

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    console.log('Dashboard initializing...');
    cacheElements();
    setupTabSwitching();
    
    // Force initial state - hide all tabs first
    const allTabs = document.querySelectorAll('.tab-content');
    allTabs.forEach(tab => {
        tab.classList.remove('active');
        tab.style.display = 'none';
    });
    
    // Show Graph tab - call without event on startup
    showTab('graph', null);
    console.log('Dashboard initialized with Graph tab');
});

// Cache DOM elements
function cacheElements() {
    elements.tabs = document.querySelectorAll('.tab');
    elements.tabContents = document.querySelectorAll('.tab-content');
    elements.connectionStatus = document.getElementById('connection-status');
    elements.connectionText = document.getElementById('connection-text');
    elements.chatInput = document.getElementById('chat-input');
    elements.sendButton = document.getElementById('send-button');
    elements.messagesContainer = document.getElementById('messages');
    elements.messagesScrollContainer = document.getElementById('messages-container');
    elements.thinkingIndicator = document.getElementById('thinking');
}

// Setup tab switching
function setupTabSwitching() {
    elements.tabs.forEach(tab => {
        tab.addEventListener('click', (event) => {
            const targetTab = tab.dataset.tab;
            showTab(targetTab, event);
        });
    });
}

// Show specific tab
    function showTab(tabName, event) {
        console.log(`showTab called with: ${tabName}`);
        
        // Debug: List all tab-content elements
        console.log('Available tab contents:', Array.from(document.querySelectorAll('.tab-content')).map(el => el.id));
        
        // Hide all tab contents
        const contents = document.querySelectorAll('.tab-content');
        contents.forEach(content => {
            content.classList.remove('active');
            content.style.display = 'none';
            
            // Notify iframe it's being hidden
            const iframe = content.querySelector('iframe');
            if (iframe && iframe.contentWindow) {
                try {
                    iframe.contentWindow.postMessage({ type: 'visibility', visible: false }, '*');
                } catch (e) {
                    // Iframe might not be ready yet
                }
            }
        });
        
        // Remove active from all tabs
        const tabs = document.querySelectorAll('.tab');
        tabs.forEach(tab => {
            tab.classList.remove('active');
        });
        
        // Show selected content
        const selectedContent = document.getElementById(`${tabName}-tab`);
        if (selectedContent) {
            selectedContent.classList.add('active');
            selectedContent.style.display = 'block';
        } else {
            console.error(`Tab content not found: ${tabName}-tab`);
            // Try to list what IDs are actually available
            console.log('Available element IDs:', Array.from(document.querySelectorAll('[id]')).map(el => el.id));
            return; // Exit early if tab not found
        }
        
        // Mark tab as active
        if (event && event.target) {
            event.target.classList.add('active');
        } else {
            // Find the tab button for this tabName
            const tabButton = document.querySelector(`.tab[data-tab="${tabName}"]`);
            if (tabButton) {
                tabButton.classList.add('active');
            }
        }
        
        // Notify the newly visible iframe
        if (selectedContent) {
            const activeIframe = selectedContent.querySelector('iframe');
            if (activeIframe && activeIframe.contentWindow) {
                // Small delay to ensure iframe is ready
                setTimeout(() => {
                    activeIframe.contentWindow.postMessage({ 
                        type: 'visibility', 
                        visible: true,
                        tabName: tabName 
                    }, '*');
                }, 100);
            }
        }
        
        // Save current tab
        currentTab = tabName;
        
        // Initialize terminal if needed
        if (tabName === 'terminal' && !socket) {
            initializeSocket();
            initializeTerminal();
        }
    }

// Initialize Socket.IO connection
function initializeSocket() {
    if (socket) return; // Already initialized
    
    // Use relative URL for socket connection
    socket = io({
        reconnectionDelay: 1000,
        reconnection: true,
        reconnectionAttempts: 10,
        transports: ['websocket', 'polling'],
        agent: false,
        upgrade: false,
        rejectUnauthorized: false
    });
    
    socket.on('connect', () => {
        console.log('Connected to ELDER');
        elements.connectionStatus.classList.add('connected');
        elements.connectionText.textContent = 'Connected';
        // Request model info on connect
        socket.emit('get_available_models');
    });

    socket.on('model_info', (data) => {
        console.log('Received model info:', data);
        const selector = document.getElementById('model-selector');
        if (!selector) return;
        
        selector.innerHTML = '';
        
        Object.entries(data.providers).forEach(([provider, info]) => {
            const option = document.createElement('option');
            option.value = provider;
            option.textContent = `${info.name} (${info.model})`;
            if (provider === data.current_provider) {
                option.selected = true;
            }
            // Mark unconfigured options visually or disable them
            if (!info.configured) {
                option.textContent += ' [No Key]';
                // option.disabled = true; // Let user try so they get the error message
                option.style.color = '#ff4040';
            }
            selector.appendChild(option);
        });
        
        // Remove old listener if any to avoid duplicates (clone node trick)
        const newSelector = selector.cloneNode(true);
        selector.parentNode.replaceChild(newSelector, selector);
        
        newSelector.addEventListener('change', (e) => {
            const provider = e.target.value;
            console.log('Switching model to:', provider);
            // Show loading state
            newSelector.disabled = true;
            newSelector.style.opacity = '0.5';
            socket.emit('switch_model', { provider });
        });
    });

    socket.on('model_switched', (data) => {
        const selector = document.getElementById('model-selector');
        if (selector) {
            selector.disabled = false;
            selector.style.opacity = '1';
        }
        
        if (data.success) {
            console.log('Model switched:', data.message);
            // Flash success
            selector.style.borderColor = '#00ff88';
            setTimeout(() => { selector.style.borderColor = ''; }, 1000);
        } else {
            console.error('Failed to switch model:', data.error);
            // Flash error
            selector.style.borderColor = '#ff4040';
            setTimeout(() => { selector.style.borderColor = ''; }, 1000);
            // Reset selection if possible (would need to track previous)
        }
    });

    socket.on('model_switch_error', (data) => {
        console.error('Model switch error:', data.error);
        const selector = document.getElementById('model-selector');
        if (selector) {
            selector.disabled = false;
            selector.style.opacity = '1';
            selector.style.borderColor = '#ff4040';
            setTimeout(() => { selector.style.borderColor = ''; }, 1000);
        }
        alert(data.error);
    });
    
    socket.on('disconnect', () => {
        console.log('Disconnected from ELDER');
        elements.connectionStatus.classList.remove('connected');
        elements.connectionText.textContent = 'Disconnected';
    });
    
    socket.on('error', (error) => {
        console.error('Socket error:', error);
    });
}

// Initialize terminal functionality
function initializeTerminal() {
    // Clear any existing listeners
    elements.sendButton.replaceWith(elements.sendButton.cloneNode(true));
    elements.chatInput.replaceWith(elements.chatInput.cloneNode(true));
    
    // Re-cache after cloning
    elements.chatInput = document.getElementById('chat-input');
    elements.sendButton = document.getElementById('send-button');
    
    // Setup event listeners
    elements.sendButton.addEventListener('click', sendMessage);
    elements.chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Focus input
    elements.chatInput.focus();
    
    // Setup socket handlers if connected
    if (socket) {
        // Remove old listeners
        socket.off('chat_response');
        socket.off('chat_error');
        
        socket.on('chat_response', handleChatResponse);
        socket.on('chat_error', handleChatError);
        
        // Also listen for thinking status
        socket.on('thinking_status', (data) => {
            if (!data.active) {
                hideThinking();
            }
        });
    }
}

// Send message to ELDER
function sendMessage() {
    const message = elements.chatInput.value.trim();
    if (!message) return;
    
    console.log('Sending message:', message);
    
    // Add user message to display
    addMessage('user', message);
    
    // Clear input
    elements.chatInput.value = '';
    elements.chatInput.focus();
    
    // Show thinking indicator
    showThinking();
    
    // Send to server
    if (socket && socket.connected) {
        console.log('Emitting chat_message to socket');
        socket.emit('chat_message', { message });
    } else {
        console.log('Socket not connected');
        addMessage('system', 'Not connected to ELDER. Please wait...');
        hideThinking();
    }
}

// Handle chat response
function handleChatResponse(data) {
    console.log('Received chat response:', data);
    hideThinking();
    addMessage('elder', data.message);
}

// Handle chat error
function handleChatError(data) {
    console.log('Received chat error:', data);
    hideThinking();
    addMessage('system', `Error: ${data.error}`);
}

// Add message to chat
function addMessage(type, content) {
    console.log(`Adding message: type=${type}, content=${content.substring(0, 50)}...`);
    
    const message = document.createElement('div');
    message.className = `message ${type}`;
    
    const time = document.createElement('div');
    time.className = 'message-time';
    const sender = type === 'user' ? 'You' : type === 'elder' ? 'Elder' : 'System';
    time.textContent = `${sender} • ${new Date().toLocaleTimeString()}`;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.textContent = content;
    
    message.appendChild(time);
    message.appendChild(messageContent);
    
    // Ensure the thinking indicator is not in the way
    const thinkingEl = document.querySelector('.thinking-indicator');
    if (thinkingEl && thinkingEl.parentNode === elements.messagesContainer) {
        elements.messagesContainer.insertBefore(message, thinkingEl);
    } else {
        elements.messagesContainer.appendChild(message);
    }
    
    console.log(`Messages in container: ${elements.messagesContainer.children.length}`);
    scrollToBottom();
}

// Show thinking indicator
function showThinking() {
    console.log('Showing thinking indicator');
    elements.thinkingIndicator.classList.add('active');
    // Don't append to messages container, it might be causing issues
    // Just show it in place
    scrollToBottom();
}

// Hide thinking indicator
function hideThinking() {
    elements.thinkingIndicator.classList.remove('active');
}

// Scroll chat to bottom
function scrollToBottom() {
    const container = elements.messagesScrollContainer || document.getElementById('messages-container');
    if (container) {
        // Use smooth scrolling for a better UX
        container.scrollTo({
            top: container.scrollHeight,
            behavior: 'smooth'
        });
    }
}

// Utility function to check if element is visible
function isVisible(element) {
    return element && element.offsetParent !== null;
}

// Debug function - can be called from console
window.debugTabs = function() {
    console.log('=== Tab Debug Info ===');
    console.log('Current tab:', currentTab);
    
    elements.tabContents.forEach(content => {
        console.log(`${content.id}:`, {
            hasActiveClass: content.classList.contains('active'),
            displayStyle: content.style.display,
            computedDisplay: window.getComputedStyle(content).display,
            isVisible: isVisible(content)
        });
    });
    
    console.log('Tab buttons:');
    elements.tabs.forEach(tab => {
        console.log(`${tab.dataset.tab} tab:`, {
            hasActiveClass: tab.classList.contains('active')
        });
    });
};

// Make showTab available globally for debugging
window.showTab = showTab;

// Debug function to test chat UI
window.testChat = function() {
    console.log('Testing chat UI...');
    
    // Switch to terminal tab
    showTab('terminal', null);
    
    // Add test messages
    setTimeout(() => {
        console.log('Adding test user message...');
        addMessage('user', 'Test message from user');
    }, 500);
    
    setTimeout(() => {
        console.log('Showing thinking...');
        showThinking();
    }, 1000);
    
    setTimeout(() => {
        console.log('Adding Elder response...');
        hideThinking();
        addMessage('elder', 'This is a test response from Elder. The chat UI is working correctly!');
    }, 2000);
    
    setTimeout(() => {
        console.log('Test complete. Check if messages are visible.');
        console.log('Messages container children:', elements.messagesContainer.children.length);
        const scrollContainer = document.getElementById('messages-container');
        console.log('Container scroll height:', scrollContainer ? scrollContainer.scrollHeight : 'N/A');
    }, 2500);
};
