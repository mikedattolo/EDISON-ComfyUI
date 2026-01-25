// EDISON Web UI - Enhanced Application with Advanced Features
class EdisonApp {
    constructor() {
        this.currentChatId = null;
        this.chats = this.loadChats();
        this.currentMode = 'auto';
        this.settings = this.loadSettings();
        this.isStreaming = false;
        this.abortController = null;
        this.sidebarCollapsed = false;
        
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
        this.stopBtn = document.getElementById('stopBtn');
        this.charCount = document.getElementById('charCount');
        this.welcomeScreen = document.getElementById('welcomeScreen');
        
        // Sidebar elements
        this.sidebar = document.getElementById('sidebar');
        this.sidebarToggle = document.getElementById('sidebarToggle');
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
        this.stopBtn.addEventListener('click', () => this.stopGeneration());
        
        // Mode selector
        this.modeButtons.forEach(btn => {
            btn.addEventListener('click', () => this.setMode(btn.dataset.mode));
        });
        
        // Sidebar
        this.newChatBtn.addEventListener('click', () => this.createNewChat());
        this.sidebarToggle.addEventListener('click', () => this.toggleSidebar());
        
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

    toggleSidebar() {
        this.sidebarCollapsed = !this.sidebarCollapsed;
        this.sidebar.classList.toggle('collapsed');
        document.querySelector('.main-content').classList.toggle('sidebar-collapsed');
    }

    handleInputChange() {
        const text = this.messageInput.value;
        const length = text.length;
        
        this.charCount.textContent = `${length} / 4000`;
        this.sendBtn.disabled = length === 0 || this.isStreaming;
        
        // Auto-resize textarea
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 200) + 'px';
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

    async sendMessage(editingMessageId = null) {
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
        
        // Get uploaded files if any
        const attachedFiles = window.uploadedFiles || [];
        
        // Add user message to UI (with file indicators)
        let displayMessage = message;
        if (attachedFiles.length > 0) {
            displayMessage += `\n\n📎 Files attached: ${attachedFiles.map(f => f.name).join(', ')}`;
        }
        const userMessageEl = this.addMessage('user', displayMessage);
        userMessageEl.dataset.messageId = Date.now();
        
        // Clear input
        this.messageInput.value = '';
        this.handleInputChange();
        
        // Prepare request
        const mode = this.currentMode === 'auto' ? 'auto' : this.currentMode;
        
        // Add assistant message placeholder
        const assistantMessageEl = this.addMessage('assistant', '', true);
        
        // Show stop button, hide send button
        this.sendBtn.style.display = 'none';
        this.stopBtn.style.display = 'flex';
        
        try {
            this.isStreaming = true;
            this.abortController = new AbortController();
            
            const response = await this.callEdisonAPI(message, mode);
            
            // Update assistant message
            this.updateMessage(assistantMessageEl, response.response, response.mode_used);
            
            // Save to chat history
            this.saveMessageToChat(message, response.response, response.mode_used);
            
            // Generate smart title if first message
            const chat = this.chats.find(c => c.id === this.currentChatId);
            if (chat && chat.messages.length === 2) {
                this.generateChatTitle(chat, message, response.response);
            }
            
        } catch (error) {
            if (error.name === 'AbortError') {
                this.updateMessage(assistantMessageEl, '⚠️ Response generation stopped by user.', 'stopped');
            } else {
                console.error('Error sending message:', error);
                this.updateMessage(
                    assistantMessageEl, 
                    `⚠️ Error: ${error.message}. Please check your connection and API endpoint in settings.`,
                    'error'
                );
            }
        } finally {
            this.isStreaming = false;
            this.abortController = null;
            this.sendBtn.style.display = 'flex';
            this.stopBtn.style.display = 'none';
            this.sendBtn.disabled = false;
        }
    }

    stopGeneration() {
        if (this.abortController) {
            this.abortController.abort();
        }
    }

    async callEdisonAPI(message, mode) {
        const endpoint = `${this.settings.apiEndpoint}/chat`;
        
        // Get recent conversation history for context
        const conversationHistory = this.getRecentMessages(5);
        // Include uploaded files if any
        console.log('📤 Checking for attached files...');
        console.log('window.uploadedFiles:', window.uploadedFiles);
        const attachedFiles = window.uploadedFiles || [];
        console.log('📤 Attached files:', attachedFiles.length, attachedFiles.map(f => f.name));
        
        let enhancedMessage = message;
        let images = [];
        
        if (attachedFiles.length > 0) {
            console.log('📤 Processing files...');
            
            // Separate images from text files
            const textFiles = attachedFiles.filter(f => !f.isImage);
            const imageFiles = attachedFiles.filter(f => f.isImage);
            
            // Add text files to message
            if (textFiles.length > 0) {
                console.log('📤 Including text files in message');
                enhancedMessage += '\n\n[Attached files:]\n';
                textFiles.forEach(file => {
                    console.log(`📤 Adding file: ${file.name}, content length: ${file.content?.length || 0}`);
                    enhancedMessage += `\n--- File: ${file.name} ---\n${file.content}\n`;
                });
            }
            
            // Collect images
            if (imageFiles.length > 0) {
                console.log('📤 Including images for vision');
                images = imageFiles.map(f => f.content);
            }
            
            console.log('📤 Enhanced message length:', enhancedMessage.length, 'Images:', images.length);
            
            // Clear files after preparing
            window.uploadedFiles.length = 0;
            const attachedFilesDiv = document.getElementById('attachedFiles');
            if (attachedFilesDiv) attachedFilesDiv.style.display = 'none';
            // Reset file input
            const fileInput = document.getElementById('fileInput');
            if (fileInput) fileInput.value = '';
            console.log('✅ Files cleared after sending');
        } else {
            console.log('📤 No files to attach');
        }
        
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: enhancedMessage,
                mode: mode,
                remember: null,  // Auto-detected by backend
                conversation_history: conversationHistory,
                images: images.length > 0 ? images : undefined
            }),
            signal: this.abortController?.signal
        });
        
        if (!response.ok) {
            throw new Error(`API returned ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    }

    getRecentMessages(count = 5) {
        const messages = Array.from(this.messagesContainer.querySelectorAll('.message:not(.streaming)'));
        const recent = messages.slice(-count * 2); // Get last N exchanges (user + assistant pairs)
        
        return recent.map(msg => ({
            role: msg.classList.contains('user') ? 'user' : 'assistant',
            content: msg.querySelector('.message-content')?.textContent || ''
        }));
    }

    addMessage(role, content, isStreaming = false) {
        const messageEl = document.createElement('div');
        messageEl.className = `message ${role}`;
        if (isStreaming) messageEl.classList.add('streaming');
        
        const avatar = role === 'user' ? '👤' : '🤖';
        const roleName = role === 'user' ? 'You' : 'EDISON';
        
        const actionButtons = role === 'user' ? `
            <div class="message-actions">
                <button class="action-btn edit-btn" title="Edit and resubmit">
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                        <path d="M12.146.146a.5.5 0 0 1 .708 0l3 3a.5.5 0 0 1 0 .708l-10 10a.5.5 0 0 1-.168.11l-5 2a.5.5 0 0 1-.65-.65l2-5a.5.5 0 0 1 .11-.168l10-10zM11.207 2.5 13.5 4.793 14.793 3.5 12.5 1.207 11.207 2.5zm1.586 3L10.5 3.207 4 9.707V10h.5a.5.5 0 0 1 .5.5v.5h.5a.5.5 0 0 1 .5.5v.5h.293l6.5-6.5zm-9.761 5.175-.106.106-1.528 3.821 3.821-1.528.106-.106A.5.5 0 0 1 5 12.5V12h-.5a.5.5 0 0 1-.5-.5V11h-.5a.5.5 0 0 1-.468-.325z"/>
                    </svg>
                </button>
            </div>
        ` : `
            <div class="message-actions">
                <button class="action-btn copy-btn" title="Copy to clipboard">
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                        <path d="M4 1.5H3a2 2 0 0 0-2 2V14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V3.5a2 2 0 0 0-2-2h-1v1h1a1 1 0 0 1 1 1V14a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V3.5a1 1 0 0 1 1-1h1v-1z"/>
                        <path d="M9.5 1a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5v-1a.5.5 0 0 1 .5-.5h3zm-3-1A1.5 1.5 0 0 0 5 1.5v1A1.5 1.5 0 0 0 6.5 4h3A1.5 1.5 0 0 0 11 2.5v-1A1.5 1.5 0 0 0 9.5 0h-3z"/>
                    </svg>
                </button>
                <button class="action-btn regenerate-btn" title="Regenerate response">
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                        <path fill-rule="evenodd" d="M8 3a5 5 0 1 0 4.546 2.914.5.5 0 0 1 .908-.417A6 6 0 1 1 8 2v1z"/>
                        <path d="M8 4.466V.534a.25.25 0 0 1 .41-.192l2.36 1.966c.12.1.12.284 0 .384L8.41 4.658A.25.25 0 0 1 8 4.466z"/>
                    </svg>
                </button>
            </div>
        `;
        
        messageEl.innerHTML = `
            <div class="message-header">
                <div class="message-avatar">${avatar}</div>
                <span class="message-role">${roleName}</span>
                <span class="message-mode" style="display: none;"></span>
            </div>
            <div class="message-content">${this.formatMessage(content)}</div>
            ${actionButtons}
        `;
        
        // Attach event listeners to action buttons
        if (role === 'user') {
            const editBtn = messageEl.querySelector('.edit-btn');
            editBtn.addEventListener('click', () => this.editMessage(messageEl, content));
        } else {
            const copyBtn = messageEl.querySelector('.copy-btn');
            const regenerateBtn = messageEl.querySelector('.regenerate-btn');
            
            copyBtn.addEventListener('click', () => this.copyToClipboard(content));
            regenerateBtn.addEventListener('click', () => this.regenerateResponse(messageEl));
        }
        
        this.messagesContainer.appendChild(messageEl);
        this.scrollToBottom();
        
        return messageEl;
    }

    editMessage(messageEl, originalContent) {
        this.messageInput.value = originalContent;
        this.messageInput.focus();
        this.handleInputChange();
        this.scrollToBottom();
    }

    async regenerateResponse(assistantMessageEl) {
        // Find the previous user message
        let userMessageEl = assistantMessageEl.previousElementSibling;
        while (userMessageEl && !userMessageEl.classList.contains('user')) {
            userMessageEl = userMessageEl.previousElementSibling;
        }
        
        if (!userMessageEl) return;
        
        const userContent = userMessageEl.querySelector('.message-content').textContent;
        
        // Remove the current assistant message
        assistantMessageEl.remove();
        
        // Resend with the same user message
        this.messageInput.value = userContent;
        await this.sendMessage();
    }

    async copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            // Show a brief notification
            this.showNotification('Copied to clipboard!');
        } catch (err) {
            console.error('Failed to copy:', err);
        }
    }

    showNotification(message) {
        const notification = document.createElement('div');
        notification.className = 'notification';
        notification.textContent = message;
        document.body.appendChild(notification);
        
        setTimeout(() => notification.classList.add('show'), 10);
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 2000);
    }

    updateMessage(messageEl, content, mode) {
        messageEl.classList.remove('streaming');
        
        const contentEl = messageEl.querySelector('.message-content');
        contentEl.innerHTML = this.formatMessage(content);
        
        if (mode && mode !== 'error' && mode !== 'stopped') {
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
            // Links
            .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>')
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
        requestAnimationFrame(() => {
            this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        });
    }

    async generateChatTitle(chat, firstMessage, firstResponse) {
        try {
            // Use a simple heuristic to generate a short title
            const combined = firstMessage + ' ' + firstResponse;
            const words = combined.split(/\s+/).slice(0, 6);
            chat.title = words.join(' ').substring(0, 40) + '...';
            
            this.saveChats();
            this.renderChatHistory();
        } catch (error) {
            console.error('Error generating title:', error);
        }
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

    deleteChat(chatId, event) {
        event.stopPropagation();
        
        if (confirm('Delete this chat?')) {
            this.chats = this.chats.filter(c => c.id !== chatId);
            this.saveChats();
            
            if (this.currentChatId === chatId) {
                if (this.chats.length > 0) {
                    this.switchChat(this.chats[0].id);
                } else {
                    this.createNewChat();
                }
            } else {
                this.renderChatHistory();
            }
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
            
            item.innerHTML = `
                <div class="chat-history-text">${this.escapeHtml(chat.title)}</div>
                <button class="delete-chat-btn" title="Delete chat">
                    <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
                        <path d="M5.5 5.5A.5.5 0 0 1 6 6v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm2.5 0a.5.5 0 0 1 .5.5v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm3 .5a.5.5 0 0 0-1 0v6a.5.5 0 0 0 1 0V6z"/>
                        <path fill-rule="evenodd" d="M14.5 3a1 1 0 0 1-1 1H13v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V4h-.5a1 1 0 0 1-1-1V2a1 1 0 0 1 1-1H6a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1h3.5a1 1 0 0 1 1 1v1zM4.118 4 4 4.059V13a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V4.059L11.882 4H4.118zM2.5 3V2h11v1h-11z"/>
                    </svg>
                </button>
            `;
            
            item.querySelector('.chat-history-text').addEventListener('click', () => this.switchChat(chat.id));
            item.querySelector('.delete-chat-btn').addEventListener('click', (e) => this.deleteChat(chat.id, e));
            
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
