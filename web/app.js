// EDISON Web UI - Main Application
class EdisonApp {
    constructor() {
        this.currentChatId = null;
        this.userId = this.getOrCreateUserId();  // Get persistent user ID for cross-network sync
        this.chats = this.loadChats();
        this.currentMode = 'auto';
        this.settings = this.loadSettings();
        this.isStreaming = false;
        
        this.initializeElements();
        this.attachEventListeners();
        this.checkSystemStatus();
        this.loadCurrentChat();
    }

    getOrCreateUserId() {
        // Use a persistent user ID stored in localStorage for cross-network access
        let userId = localStorage.getItem('edison_user_id');
        if (!userId) {
            // Generate a simple ID based on timestamp and random string
            userId = 'user_' + Date.now().toString(36) + '_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('edison_user_id', userId);
        }
        return userId;
    }

    initializeElements() {
        // Main elements
        this.messagesContainer = document.getElementById('messagesContainer');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.charCount = document.getElementById('charCount');
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
        
        // Prepare request - remember is now auto-detected by backend
        const mode = this.currentMode === 'auto' ? 'auto' : this.currentMode;
        
        // Add assistant message placeholder
        const assistantMessageEl = this.addMessage('assistant', '', true);
        
        try {
            this.isStreaming = true;
            this.sendBtn.disabled = true;
            
            const response = await this.callEdisonAPI(message, mode);
            
            // Update assistant message
            this.updateMessage(assistantMessageEl, response.response, response.mode_used);
            
            // Handle work mode - display task steps
            if (response.mode_used === 'work' && response.work_steps) {
                this.displayWorkSteps(response.work_steps, assistantMessageEl);
                
                // Update work desktop if visible
                if (window.workModeActive) {
                    window.updateWorkDesktop(
                        message, 
                        response.search_results || [], 
                        [], 
                        `Task broken into ${response.work_steps.length} steps`
                    );
                }
            }
            
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

    async callEdisonAPI(message, mode) {
        const endpoint = `${this.settings.apiEndpoint}/chat`;
        
        // Get recent conversation history for context
        const conversationHistory = this.getRecentMessages(5);
        
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                mode: mode,
                remember: null,  // Auto-detected by backend
                conversation_history: conversationHistory
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
    
    getRecentMessages(count = 5) {
        /**Get recent messages from current chat for conversation context*/
        if (!this.currentChatId) return [];
        
        const currentChat = this.chats.find(c => c.id === this.currentChatId);
        if (!currentChat || !currentChat.messages) return [];
        
        // Get last N messages (excluding the current one being sent)
        const recentMessages = currentChat.messages.slice(-count).map(msg => ({
            role: msg.role,
            content: msg.content
        }));
        
        return recentMessages;
    }
    
    displayWorkSteps(steps, messageEl) {
        /**Display work mode task breakdown steps in the message*/
        if (!steps || steps.length === 0) return;
        
        const contentEl = messageEl.querySelector('.message-content');
        const stepsHTML = `
            <div class="work-steps">
                <div class="work-steps-header">📋 Task Breakdown</div>
                <ol class="work-steps-list">
                    ${steps.map(step => `<li>${this.escapeHtml(step)}</li>`).join('')}
                </ol>
            </div>
        `;
        
        // Prepend steps before the main response
        contentEl.innerHTML = stepsHTML + contentEl.innerHTML;
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
            
            <h3 class="welcome-section-title">💡 Modes</h3>
            <div class="capabilities">
                <div class="capability-card">
                    <div class="capability-icon">🔮</div>
                    <h3>Auto</h3>
                    <p>Automatically detects your intent and picks the best mode</p>
                </div>
                <div class="capability-card">
                    <div class="capability-icon">⚡</div>
                    <h3>Instant</h3>
                    <p>Ultra-fast responses for quick questions and simple tasks</p>
                </div>
                <div class="capability-card">
                    <div class="capability-icon">💬</div>
                    <h3>Chat</h3>
                    <p>Natural conversations with fast responses using the 14B model</p>
                </div>
                <div class="capability-card">
                    <div class="capability-icon">🧠</div>
                    <h3>Reasoning</h3>
                    <p>Deep analysis and step-by-step reasoning with the QwQ-32B model</p>
                </div>
                <div class="capability-card">
                    <div class="capability-icon">🧩</div>
                    <h3>Thinking</h3>
                    <p>Extended reasoning with the powerful 72B model for complex problems</p>
                </div>
                <div class="capability-card">
                    <div class="capability-icon">💻</div>
                    <h3>Code</h3>
                    <p>Write, debug, and explain code in multiple languages</p>
                </div>
                <div class="capability-card">
                    <div class="capability-icon">🤖</div>
                    <h3>Agent</h3>
                    <p>Execute complex multi-step tasks with tool-using capabilities</p>
                </div>
                <div class="capability-card">
                    <div class="capability-icon">🕸️</div>
                    <h3>Swarm</h3>
                    <p>Parallel specialized agents collaborate on complex tasks</p>
                </div>
                <div class="capability-card">
                    <div class="capability-icon">🖥️</div>
                    <h3>Work</h3>
                    <p>Task visualization and structured project workflows</p>
                </div>
            </div>

            <h3 class="welcome-section-title">🚀 Features</h3>
            <div class="capabilities">
                <div class="capability-card">
                    <div class="capability-icon">🎨</div>
                    <h3>Image Generation</h3>
                    <p>Create images from text prompts using FLUX models</p>
                </div>
                <div class="capability-card">
                    <div class="capability-icon">📐</div>
                    <h3>3D Models</h3>
                    <p>Generate 3D models from text descriptions</p>
                </div>
                <div class="capability-card">
                    <div class="capability-icon">🖼️</div>
                    <h3>Gallery</h3>
                    <p>Browse and manage all your generated images</p>
                </div>
                <div class="capability-card">
                    <div class="capability-icon">📁</div>
                    <h3>File Manager</h3>
                    <p>Upload, browse, and manage files on the server</p>
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
        // Load from localStorage first for immediate display
        const saved = localStorage.getItem('edison_chats');
        const localChats = saved ? JSON.parse(saved) : [];
        
        // Then sync with server in background
        this.syncChatsFromServer();
        
        return localChats;
    }

    async syncChatsFromServer() {
        try {
            const response = await fetch(`${this.settings.apiEndpoint}/chats/sync`, {
                method: 'GET',
                credentials: 'include',  // Include cookies for user ID
                headers: { 
                    'Content-Type': 'application/json',
                    'X-Edison-User-ID': this.userId  // Send user ID for cross-network access
                }
            });
            
            if (response.ok) {
                const data = await response.json();
                const serverChats = data.chats || [];
                
                if (serverChats.length > 0) {
                    // Merge server chats with local chats (server takes priority for same IDs)
                    const merged = this.mergeChats(this.chats, serverChats);
                    this.chats = merged;
                    localStorage.setItem('edison_chats', JSON.stringify(this.chats));
                    this.renderChatHistory();
                    console.log(`Synced ${serverChats.length} chats from server`);
                }
            }
        } catch (error) {
            console.warn('Could not sync chats from server:', error.message);
        }
    }

    mergeChats(localChats, serverChats) {
        // Create a map of all chats by ID
        const chatMap = new Map();
        
        // Add local chats first
        for (const chat of localChats) {
            chatMap.set(chat.id, chat);
        }
        
        // Server chats take priority (they may have newer messages)
        for (const chat of serverChats) {
            const existing = chatMap.get(chat.id);
            if (existing) {
                // Keep the one with more messages or newer timestamp
                const existingMsgCount = existing.messages?.length || 0;
                const serverMsgCount = chat.messages?.length || 0;
                if (serverMsgCount >= existingMsgCount) {
                    chatMap.set(chat.id, chat);
                }
            } else {
                chatMap.set(chat.id, chat);
            }
        }
        
        // Sort by most recent first
        return Array.from(chatMap.values()).sort((a, b) => {
            const aTime = a.timestamp || 0;
            const bTime = b.timestamp || 0;
            return bTime - aTime;
        });
    }

    saveChats() {
        // Save locally first for immediate feedback
        localStorage.setItem('edison_chats', JSON.stringify(this.chats));
        
        // Then sync to server in background
        this.syncChatsToServer();
    }

    async syncChatsToServer() {
        try {
            await fetch(`${this.settings.apiEndpoint}/chats/sync`, {
                method: 'POST',
                credentials: 'include',  // Include cookies for user ID
                headers: { 
                    'Content-Type': 'application/json',
                    'X-Edison-User-ID': this.userId  // Send user ID for cross-network access
                },
                body: JSON.stringify({ chats: this.chats })
            });
        } catch (error) {
            console.warn('Could not sync chats to server:', error.message);
        }
    }

    loadSettings() {
        const saved = localStorage.getItem('edison_settings');
        const protocol = window.location.protocol || 'http:';
        const host = window.location.hostname || 'localhost';
        const defaults = {
            apiEndpoint: `${protocol}//${host}:8811`,
            defaultMode: 'auto',
            streamResponses: true,
            syntaxHighlight: true
        };
        
        const settings = saved ? { ...defaults, ...JSON.parse(saved) } : defaults;
        // Migrate away from hardcoded legacy IP
        if (settings.apiEndpoint && settings.apiEndpoint.includes('192.168.1.26')) {
            settings.apiEndpoint = defaults.apiEndpoint;
        }
        return settings;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.edisonApp = new EdisonApp();
});
