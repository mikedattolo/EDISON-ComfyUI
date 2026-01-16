// EDISON Web UI - Main Application
class EdisonApp {
    constructor() {
        this.currentChatId = null;
        this.chats = this.loadChats();
        this.currentMode = 'auto';
        this.settings = this.loadSettings();
        this.isStreaming = false;
        
        this.initializeElements();
        this.attachEventListeners();
        this.checkSystemStatus();
        this.loadCurrentChat();
    }

    initializeElements() {
        // Main elements
        this.messagesContainer = document.getElementById('messagesContainer');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.charCount = document.getElementById('charCount');
        this.rememberCheckbox = document.getElementById('rememberCheckbox');
        this.welcomeScreen = document.getElementById('welcomeScreen');
        
        // Sidebar elements
        this.chatHistory = document.getElementById('chatHistory');
        this.newChatBtn = document.getElementById('newChatBtn');
        
        // Mode selector
        this.modeButtons = document.querySelectorAll('.mode-btn');
        
        // Settings modal
        this.settingsModal = document.getElementById('settingsModal');
        this.settingsBtn = document.getElementById('settingsBtn');
        this.closeSettingsBtn = document.getElementById('closeSettingsBtn');
        this.saveSettingsBtn = document.getElementById('saveSettingsBtn');
        this.clearHistoryBtn = document.getElementById('clearHistoryBtn');
        this.apiEndpointInput = document.getElementById('apiEndpoint');
        this.defaultModeSelect = document.getElementById('defaultMode');
        this.systemStatus = document.getElementById('systemStatus');
    }

    attachEventListeners() {
        // Message input
        this.messageInput.addEventListener('input', () => this.handleInputChange());
        this.messageInput.addEventListener('keydown', (e) => this.handleKeyDown(e));
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        
        // Mode selector
        this.modeButtons.forEach(btn => {
            btn.addEventListener('click', () => this.setMode(btn.dataset.mode));
        });
        
        // Sidebar
        this.newChatBtn.addEventListener('click', () => this.createNewChat());
        
        // Settings
        this.settingsBtn.addEventListener('click', () => this.openSettings());
        this.closeSettingsBtn.addEventListener('click', () => this.closeSettings());
        this.saveSettingsBtn.addEventListener('click', () => this.saveSettings());
        this.clearHistoryBtn.addEventListener('click', () => this.clearHistory());
        
        // Modal backdrop click
        this.settingsModal.addEventListener('click', (e) => {
            if (e.target === this.settingsModal) this.closeSettings();
        });
    }

    handleInputChange() {
        const text = this.messageInput.value;
        const length = text.length;
        
        this.charCount.textContent = `${length} / 4000`;
        this.sendBtn.disabled = length === 0 || this.isStreaming;
        
        // Auto-resize textarea
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = this.messageInput.scrollHeight + 'px';
    }

    handleKeyDown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!this.sendBtn.disabled) {
                this.sendMessage();
            }
        }
    }

    setMode(mode) {
        this.currentMode = mode;
        this.modeButtons.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isStreaming) return;
        
        // Hide welcome screen
        if (this.welcomeScreen) {
            this.welcomeScreen.style.display = 'none';
        }
        
        // Create chat if needed
        if (!this.currentChatId) {
            this.createNewChat();
        }
        
        // Add user message to UI
        this.addMessage('user', message);
        
        // Clear input
        this.messageInput.value = '';
        this.handleInputChange();
        
        // Prepare request
        const remember = this.rememberCheckbox.checked;
        const mode = this.currentMode === 'auto' ? 'auto' : this.currentMode;
        
        // Add assistant message placeholder
        const assistantMessageEl = this.addMessage('assistant', '', true);
        
        try {
            this.isStreaming = true;
            this.sendBtn.disabled = true;
            
            const response = await this.callEdisonAPI(message, mode, remember);
            
            // Update assistant message
            this.updateMessage(assistantMessageEl, response.response, response.mode_used);
            
            // Save to chat history
            this.saveMessageToChat(message, response.response, response.mode_used);
            
        } catch (error) {
            console.error('Error sending message:', error);
            this.updateMessage(
                assistantMessageEl, 
                `⚠️ Error: ${error.message}. Please check your connection and API endpoint in settings.`,
                'error'
            );
        } finally {
            this.isStreaming = false;
            this.sendBtn.disabled = false;
        }
    }

    async callEdisonAPI(message, mode, remember) {
        const endpoint = `${this.settings.apiEndpoint}/chat`;
        
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                mode: mode,
                remember: remember
            })
        });
        
        if (!response.ok) {
            throw new Error(`API returned ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    }

    addMessage(role, content, isStreaming = false) {
        const messageEl = document.createElement('div');
        messageEl.className = `message ${role}`;
        if (isStreaming) messageEl.classList.add('streaming');
        
        const avatar = role === 'user' ? '👤' : '🤖';
        const roleName = role === 'user' ? 'You' : 'EDISON';
        
        messageEl.innerHTML = `
            <div class="message-header">
                <div class="message-avatar">${avatar}</div>
                <span class="message-role">${roleName}</span>
                <span class="message-mode" style="display: none;"></span>
            </div>
            <div class="message-content">${this.formatMessage(content)}</div>
        `;
        
        this.messagesContainer.appendChild(messageEl);
        this.scrollToBottom();
        
        return messageEl;
    }

    updateMessage(messageEl, content, mode) {
        messageEl.classList.remove('streaming');
        
        const contentEl = messageEl.querySelector('.message-content');
        contentEl.innerHTML = this.formatMessage(content);
        
        if (mode && mode !== 'error') {
            const modeEl = messageEl.querySelector('.message-mode');
            modeEl.textContent = mode.toUpperCase();
            modeEl.style.display = 'inline-block';
        }
        
        this.scrollToBottom();
    }

    formatMessage(content) {
        if (!content) return '';
        
        // Basic markdown-like formatting
        let formatted = content
            // Code blocks
            .replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
                return `<pre><code class="language-${lang || 'text'}">${this.escapeHtml(code.trim())}</code></pre>`;
            })
            // Inline code
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            // Bold
            .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
            // Italic
            .replace(/\*([^*]+)\*/g, '<em>$1</em>')
            // Line breaks
            .replace(/\n/g, '<br>');
        
        return formatted;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }

    createNewChat() {
        const chatId = Date.now().toString();
        const chat = {
            id: chatId,
            title: 'New Chat',
            messages: [],
            createdAt: new Date().toISOString()
        };
        
        this.chats.unshift(chat);
        this.currentChatId = chatId;
        this.saveChats();
        this.renderChatHistory();
        this.clearMessages();
    }

    loadCurrentChat() {
        if (this.chats.length === 0) {
            this.createNewChat();
            return;
        }
        
        if (!this.currentChatId) {
            this.currentChatId = this.chats[0].id;
        }
        
        const chat = this.chats.find(c => c.id === this.currentChatId);
        if (chat) {
            this.renderMessages(chat.messages);
        }
        
        this.renderChatHistory();
    }

    switchChat(chatId) {
        this.currentChatId = chatId;
        const chat = this.chats.find(c => c.id === chatId);
        if (chat) {
            this.clearMessages();
            this.renderMessages(chat.messages);
            this.renderChatHistory();
        }
    }

    clearMessages() {
        this.messagesContainer.innerHTML = '';
        this.welcomeScreen = document.createElement('div');
        this.welcomeScreen.className = 'welcome-screen';
        this.welcomeScreen.id = 'welcomeScreen';
        this.welcomeScreen.innerHTML = `
            <h2>Welcome to EDISON</h2>
            <p>Your offline AI assistant for conversations, code, and creative tasks</p>
            
            <div class="capabilities">
                <div class="capability-card">
                    <div class="capability-icon">💬</div>
                    <h3>Natural Conversations</h3>
                    <p>Chat naturally with fast responses using the 14B model</p>
                </div>
                <div class="capability-card">
                    <div class="capability-icon">🧠</div>
                    <h3>Deep Thinking</h3>
                    <p>Get detailed analysis with the powerful 72B model</p>
                </div>
                <div class="capability-card">
                    <div class="capability-icon">💻</div>
                    <h3>Code Assistant</h3>
                    <p>Write, debug, and explain code in multiple languages</p>
                </div>
                <div class="capability-card">
                    <div class="capability-icon">🤖</div>
                    <h3>Agent Mode</h3>
                    <p>Execute complex tasks with tool-using capabilities</p>
                </div>
            </div>
        `;
        this.messagesContainer.appendChild(this.welcomeScreen);
    }

    renderMessages(messages) {
        if (messages.length === 0) return;
        
        if (this.welcomeScreen) {
            this.welcomeScreen.style.display = 'none';
        }
        
        messages.forEach(msg => {
            const messageEl = this.addMessage(msg.role, msg.content);
            if (msg.mode && msg.role === 'assistant') {
                const modeEl = messageEl.querySelector('.message-mode');
                modeEl.textContent = msg.mode.toUpperCase();
                modeEl.style.display = 'inline-block';
            }
        });
    }

    saveMessageToChat(userMessage, assistantMessage, mode) {
        const chat = this.chats.find(c => c.id === this.currentChatId);
        if (!chat) return;
        
        chat.messages.push(
            { role: 'user', content: userMessage },
            { role: 'assistant', content: assistantMessage, mode: mode }
        );
        
        // Update chat title with first message
        if (chat.messages.length === 2) {
            chat.title = userMessage.substring(0, 50) + (userMessage.length > 50 ? '...' : '');
        }
        
        this.saveChats();
        this.renderChatHistory();
    }

    renderChatHistory() {
        this.chatHistory.innerHTML = '';
        
        this.chats.forEach(chat => {
            const item = document.createElement('div');
            item.className = 'chat-history-item';
            if (chat.id === this.currentChatId) {
                item.classList.add('active');
            }
            
            item.innerHTML = `<div class="chat-history-text">${chat.title}</div>`;
            item.addEventListener('click', () => this.switchChat(chat.id));
            
            this.chatHistory.appendChild(item);
        });
    }

    openSettings() {
        this.apiEndpointInput.value = this.settings.apiEndpoint;
        this.defaultModeSelect.value = this.settings.defaultMode;
        this.settingsModal.classList.add('open');
        this.checkSystemStatus();
    }

    closeSettings() {
        this.settingsModal.classList.remove('open');
    }

    saveSettings() {
        this.settings.apiEndpoint = this.apiEndpointInput.value.trim();
        this.settings.defaultMode = this.defaultModeSelect.value;
        
        localStorage.setItem('edison_settings', JSON.stringify(this.settings));
        
        // Update current mode if needed
        this.setMode(this.settings.defaultMode);
        
        this.closeSettings();
    }

    async checkSystemStatus() {
        try {
            const response = await fetch(`${this.settings.apiEndpoint}/health`, {
                method: 'GET',
                signal: AbortSignal.timeout(5000)
            });
            
            if (response.ok) {
                const data = await response.json();
                this.systemStatus.className = 'status-indicator online';
                this.systemStatus.innerHTML = `
                    <span class="status-dot"></span>
                    <span class="status-text">Connected - ${data.service || 'EDISON'}</span>
                `;
            } else {
                throw new Error('Service unavailable');
            }
        } catch (error) {
            this.systemStatus.className = 'status-indicator offline';
            this.systemStatus.innerHTML = `
                <span class="status-dot"></span>
                <span class="status-text">Offline - ${error.message}</span>
            `;
        }
    }

    clearHistory() {
        if (confirm('Are you sure you want to clear all chat history? This cannot be undone.')) {
            this.chats = [];
            this.currentChatId = null;
            this.saveChats();
            this.createNewChat();
            this.closeSettings();
        }
    }

    loadChats() {
        const saved = localStorage.getItem('edison_chats');
        return saved ? JSON.parse(saved) : [];
    }

    saveChats() {
        localStorage.setItem('edison_chats', JSON.stringify(this.chats));
    }

    loadSettings() {
        const saved = localStorage.getItem('edison_settings');
        const defaults = {
            apiEndpoint: 'http://192.168.1.26:8811',
            defaultMode: 'auto',
            streamResponses: true,
            syntaxHighlight: true
        };
        
        return saved ? { ...defaults, ...JSON.parse(saved) } : defaults;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.edisonApp = new EdisonApp();
});
