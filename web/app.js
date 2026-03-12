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
        
        // Collect attached files before clearing
        const attachedFiles = (window.uploadedFiles || []).slice();
        
        // Add user message to UI (with image thumbnails if present)
        const userMsgEl = this.addMessage('user', message);
        if (attachedFiles.length > 0) {
            const contentEl = userMsgEl.querySelector('.message-content');
            const thumbsHtml = attachedFiles
                .filter(f => f.isImage)
                .map(f => `<img src="${this.escapeAttr(f.content)}" class="msg-thumb" alt="${this.escapeHtml(f.name)}" />`)
                .join('');
            const fileNames = attachedFiles
                .filter(f => !f.isImage)
                .map(f => `<span class="msg-file-chip">📄 ${this.escapeHtml(f.name)}</span>`)
                .join('');
            if (thumbsHtml || fileNames) {
                contentEl.innerHTML += `<div class="msg-attachments">${thumbsHtml}${fileNames}</div>`;
            }
        }
        
        // Clear input and files
        this.messageInput.value = '';
        this.handleInputChange();
        window.uploadedFiles = [];
        if (typeof updateAttachedFilesUI === 'function') updateAttachedFilesUI();
        
        // Prepare request
        const mode = this.currentMode === 'auto' ? 'auto' : this.currentMode;
        
        // Add assistant message placeholder
        const assistantMessageEl = this.addMessage('assistant', '', true);
        const contentEl = assistantMessageEl.querySelector('.message-content');
        
        // Show stop button
        this.showStopButton();
        
        try {
            this.isStreaming = true;
            this.sendBtn.disabled = true;
            this.currentAbortController = new AbortController();
            
            const response = await this.callEdisonStreamAPI(message, mode, attachedFiles, assistantMessageEl, contentEl);
            
            // Update final message with proper formatting
            this.updateMessage(assistantMessageEl, response.response, response.mode_used);
            
            // Handle work mode
            if (response.mode_used === 'work' && response.work_steps) {
                this.displayWorkSteps(response.work_steps, assistantMessageEl);
            }
            
            // Save to chat history
            this.saveMessageToChat(message, response.response, response.mode_used);
            
        } catch (error) {
            if (error.name === 'AbortError') {
                this.updateMessage(assistantMessageEl, contentEl.textContent || '*(generation stopped)*', 'stopped');
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
            this.sendBtn.disabled = false;
            this.currentAbortController = null;
            this.hideStopButton();
        }
    }

    showStopButton() {
        if (!this._stopBtn) {
            this._stopBtn = document.createElement('button');
            this._stopBtn.className = 'stop-btn';
            this._stopBtn.textContent = '■ Stop';
            this._stopBtn.title = 'Stop generation';
            this._stopBtn.addEventListener('click', () => this.stopGeneration());
        }
        this.sendBtn.parentElement.insertBefore(this._stopBtn, this.sendBtn);
        this._stopBtn.style.display = 'inline-flex';
        this.sendBtn.style.display = 'none';
    }

    hideStopButton() {
        if (this._stopBtn) this._stopBtn.style.display = 'none';
        this.sendBtn.style.display = '';
    }

    stopGeneration() {
        if (this.currentAbortController) {
            this.currentAbortController.abort();
        }
    }

    escapeAttr(text) {
        return text.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    async callEdisonStreamAPI(message, mode, attachedFiles, msgEl, contentEl) {
        const endpoint = `${this.settings.apiEndpoint}/chat/stream`;
        const conversationHistory = this.getRecentMessages(5);
        
        // Collect images and file contents from attachments
        const images = [];
        let fileContext = '';
        for (const file of attachedFiles) {
            if (file.isImage) {
                images.push(file.content);
            } else {
                // Include text file content as context
                const preview = typeof file.content === 'string' ? file.content.slice(0, 8000) : '';
                fileContext += `\n\n[Attached file: ${file.name}]\n${preview}`;
            }
        }
        
        const fullMessage = fileContext ? `${message}\n${fileContext}` : message;

        const body = {
            message: fullMessage,
            mode: mode,
            remember: null,
            conversation_history: conversationHistory,
            chat_id: this.currentChatId,
        };
        if (images.length > 0) {
            body.images = images;
        }

        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
            signal: this.currentAbortController?.signal,
        });

        if (!response.ok) {
            throw new Error(`API returned ${response.status}: ${response.statusText}`);
        }

        // Parse SSE stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let accumulated = '';
        let finalPayload = null;
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || ''; // Keep incomplete line in buffer

            for (const line of lines) {
                if (line.startsWith('event: ')) {
                    this._sseEventType = line.slice(7).trim();
                } else if (line.startsWith('data: ')) {
                    const dataStr = line.slice(6);
                    try {
                        const data = JSON.parse(dataStr);
                        if (this._sseEventType === 'token') {
                            const token = data.t || data.token || '';
                            accumulated += token;
                            contentEl.innerHTML = this.formatMessage(accumulated);
                            this.scrollToBottom();
                        } else if (this._sseEventType === 'done') {
                            finalPayload = data;
                        } else if (this._sseEventType === 'status') {
                            // Show status steps like "Searching web", "Using memory"
                            if (data.steps) {
                                const statusHtml = data.steps.map(s => 
                                    `<span class="status-step">${s.stage}${s.detail ? ': ' + this.escapeHtml(s.detail) : ''}</span>`
                                ).join(' → ');
                                contentEl.innerHTML = `<div class="stream-status">${statusHtml}</div>` + this.formatMessage(accumulated);
                            }
                        } else if (this._sseEventType === 'error') {
                            throw new Error(data.error || data.detail || 'Stream error');
                        }
                    } catch (parseErr) {
                        if (parseErr.message && !parseErr.message.includes('JSON')) throw parseErr;
                    }
                }
            }
        }

        // Use final payload response if available, otherwise accumulated text
        return {
            response: finalPayload?.response || accumulated,
            mode_used: finalPayload?.mode_used || mode,
            work_steps: finalPayload?.work_steps || [],
            search_results: finalPayload?.search_results || [],
            artifact: finalPayload?.artifact || null,
        };
    }

    async callEdisonAPI(message, mode) {
        // Fallback non-streaming endpoint
        const endpoint = `${this.settings.apiEndpoint}/chat`;
        const conversationHistory = this.getRecentMessages(5);
        
        // Include any attached images
        const images = (window.uploadedFiles || [])
            .filter(f => f.isImage)
            .map(f => f.content);
        
        const body = {
            message: message,
            mode: mode,
            remember: null,
            conversation_history: conversationHistory,
            chat_id: this.currentChatId,
        };
        if (images.length > 0) body.images = images;

        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
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
        
        let formatted = content;

        // Code blocks with copy button
        formatted = formatted.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
            const langLabel = lang || 'text';
            const escaped = this.escapeHtml(code.trim());
            const copyId = 'code_' + Math.random().toString(36).slice(2, 8);
            return `<div class="code-block-wrapper"><div class="code-block-header"><span class="code-lang">${langLabel}</span><button class="copy-code-btn" onclick="window.edisonApp.copyCode('${copyId}')">📋 Copy</button></div><pre id="${copyId}"><code class="language-${langLabel}">${escaped}</code></pre></div>`;
        });

        // Tables: detect markdown tables and render as HTML
        formatted = formatted.replace(/((?:^\|.+\|$\n?)+)/gm, (tableBlock) => {
            const rows = tableBlock.trim().split('\n').filter(r => r.trim());
            if (rows.length < 2) return tableBlock;
            const parseRow = (row) => row.split('|').slice(1, -1).map(c => c.trim());
            const headerCells = parseRow(rows[0]);
            // Check if row 2 is separator (---|----|---)
            const isSep = /^\|[\s\-:|]+\|$/.test(rows[1]);
            const dataStart = isSep ? 2 : 1;
            let html = '<table class="md-table"><thead><tr>' + headerCells.map(c => `<th>${c}</th>`).join('') + '</tr></thead><tbody>';
            for (let i = dataStart; i < rows.length; i++) {
                const cells = parseRow(rows[i]);
                html += '<tr>' + cells.map(c => `<td>${c}</td>`).join('') + '</tr>';
            }
            html += '</tbody></table>';
            return html;
        });

        // Headers
        formatted = formatted.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
        formatted = formatted.replace(/^### (.+)$/gm, '<h3>$1</h3>');
        formatted = formatted.replace(/^## (.+)$/gm, '<h2>$1</h2>');
        formatted = formatted.replace(/^# (.+)$/gm, '<h1>$1</h1>');

        // Horizontal rules
        formatted = formatted.replace(/^---+$/gm, '<hr>');

        // Blockquotes
        formatted = formatted.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');

        // Unordered lists
        formatted = formatted.replace(/((?:^[\t ]*[-*] .+$\n?)+)/gm, (listBlock) => {
            const items = listBlock.trim().split('\n')
                .filter(l => l.trim())
                .map(l => `<li>${l.replace(/^[\t ]*[-*] /, '').trim()}</li>`)
                .join('');
            return `<ul>${items}</ul>`;
        });

        // Ordered lists
        formatted = formatted.replace(/((?:^[\t ]*\d+\. .+$\n?)+)/gm, (listBlock) => {
            const items = listBlock.trim().split('\n')
                .filter(l => l.trim())
                .map(l => `<li>${l.replace(/^[\t ]*\d+\.\s*/, '').trim()}</li>`)
                .join('');
            return `<ol>${items}</ol>`;
        });

        // Links
        formatted = formatted.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

        // Inline code (after code blocks are handled)
        formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Bold
        formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

        // Italic
        formatted = formatted.replace(/\*([^*]+)\*/g, '<em>$1</em>');

        // Strikethrough
        formatted = formatted.replace(/~~([^~]+)~~/g, '<del>$1</del>');

        // Line breaks (but not inside pre/table blocks)
        formatted = formatted.replace(/\n/g, '<br>');

        // Clean up extra <br> after block elements
        formatted = formatted.replace(/<\/(h[1-4]|ul|ol|table|blockquote|hr|pre|div)><br>/g, '</$1>');
        formatted = formatted.replace(/<br><(h[1-4]|ul|ol|table|blockquote|hr)/g, '<$1');

        return formatted;
    }

    copyCode(id) {
        const el = document.getElementById(id);
        if (el) {
            const text = el.textContent;
            navigator.clipboard.writeText(text).then(() => {
                const btn = el.parentElement.querySelector('.copy-code-btn');
                if (btn) {
                    btn.textContent = '✓ Copied';
                    setTimeout(() => { btn.textContent = '📋 Copy'; }, 2000);
                }
            });
        }
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
