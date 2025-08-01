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
    
    // Show Graph tab
    showTab('graph');
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
    elements.thinkingIndicator = document.getElementById('thinking');
}

// Setup tab switching
function setupTabSwitching() {
    elements.tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetTab = tab.dataset.tab;
            showTab(targetTab);
        });
    });
}

// Show specific tab
function showTab(tabName) {
    console.log(`Switching to tab: ${tabName}`);
    
    // Update active states on tabs
    elements.tabs.forEach(tab => {
        if (tab.dataset.tab === tabName) {
            tab.classList.add('active');
        } else {
            tab.classList.remove('active');
        }
    });
    
    // Update active states on content and explicitly set display
    elements.tabContents.forEach(content => {
        if (content.id === `${tabName}-tab`) {
            content.classList.add('active');
            // Special handling for terminal tab which needs flex
            if (content.id === 'terminal-tab') {
                content.style.display = 'flex';
            } else {
                content.style.display = 'block';
            }
            console.log(`Showing ${content.id}`);
        } else {
            content.classList.remove('active');
            content.style.display = 'none';
            console.log(`Hiding ${content.id}`);
        }
    });
    
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
    
    socket = io('http://localhost:5000', {
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
    }
}

// Send message to ELDER
function sendMessage() {
    const message = elements.chatInput.value.trim();
    if (!message) return;
    
    // Add user message to display
    addMessage('user', message);
    
    // Clear input
    elements.chatInput.value = '';
    elements.chatInput.focus();
    
    // Show thinking indicator
    showThinking();
    
    // Send to server
    if (socket && socket.connected) {
        socket.emit('chat_message', { message });
    } else {
        addMessage('system', 'Not connected to ELDER. Please wait...');
        hideThinking();
    }
}

// Handle chat response
function handleChatResponse(data) {
    hideThinking();
    addMessage('elder', data.message);
}

// Handle chat error
function handleChatError(data) {
    hideThinking();
    addMessage('system', `Error: ${data.error}`);
}

// Add message to chat
function addMessage(type, content) {
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
    
    elements.messagesContainer.appendChild(message);
    scrollToBottom();
}

// Show thinking indicator
function showThinking() {
    elements.thinkingIndicator.classList.add('active');
    elements.messagesContainer.appendChild(elements.thinkingIndicator);
    scrollToBottom();
}

// Hide thinking indicator
function hideThinking() {
    elements.thinkingIndicator.classList.remove('active');
}

// Scroll chat to bottom
function scrollToBottom() {
    const container = document.querySelector('.terminal-container');
    if (container) {
        container.scrollTop = container.scrollHeight;
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
