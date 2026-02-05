// EDISON Web UI - Enhanced Application with Advanced Features
class EdisonApp {
    constructor() {
        this.currentChatId = null;
        this.userId = this.getOrCreateUserId();  // Get persistent user ID for cross-network sync
        this.settings = this.loadSettings();  // Load settings FIRST
        this.chats = this.loadChats();  // Then load chats (needs settings)
        this.currentMode = 'auto';
        this.isStreaming = false;
        this.abortController = null;
        this.currentRequestId = null;  // Track current streaming request for cancellation
        this.sidebarCollapsed = false;
        this.lastImagePrompt = null; // Track last image generation prompt for regeneration
        this.projectId = localStorage.getItem('edison_project_id') || 'default';
        this.sessionId = localStorage.getItem('edison_session_id') || `session_${Date.now()}`;
        localStorage.setItem('edison_session_id', this.sessionId);
        this.availableModels = [];
        this.selectedModel = localStorage.getItem('edison_selected_model') || 'auto';
        this.modelSelector = null;
        this.modelSelect = null;
        this.bestVlm = null;
        
        this.initializeElements();
        this.attachEventListeners();
        this.checkSystemStatus();
        this.loadCurrentChat();
        this.setMode(this.settings.defaultMode);
        this.loadAvailableModels();
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
        this.comfyuiEndpointInput = document.getElementById('comfyuiEndpoint');
        this.defaultModeSelect = document.getElementById('defaultMode');
        this.systemStatus = document.getElementById('systemStatus');

        // Model selector
        this.modelSelector = document.getElementById('modelSelector');
        this.modelSelect = document.getElementById('modelSelect');
        this.voiceBtn = document.getElementById('voiceBtn');
        this.voiceEndpointInput = document.getElementById('voiceEndpoint');
        
        // User ID sync controls
        this.userIdInput = document.getElementById('userIdInput');
        this.copyUserIdBtn = document.getElementById('copyUserIdBtn');
        this.importUserIdInput = document.getElementById('importUserIdInput');
        this.importUserIdBtn = document.getElementById('importUserIdBtn');
        this.userSelect = document.getElementById('userSelect');
        this.refreshUsersBtn = document.getElementById('refreshUsersBtn');
        this.newUserNameInput = document.getElementById('newUserNameInput');
        this.addUserBtn = document.getElementById('addUserBtn');
        this.renameUserInput = document.getElementById('renameUserInput');
        this.renameUserBtn = document.getElementById('renameUserBtn');
        this.deleteUserBtn = document.getElementById('deleteUserBtn');
        
        // Theme controls (in settings modal)
        this.themeButtons = document.querySelectorAll('.theme-btn');
        this.colorButtons = document.querySelectorAll('.color-btn');

        // Artifacts panel
        this.artifactsPanel = document.getElementById('artifactsPanel');
        this.artifactTitle = document.getElementById('artifactTitle');
        this.artifactCode = document.getElementById('artifactCode');
        this.artifactCopyBtn = document.getElementById('artifactCopyBtn');
        this.artifactDownloadBtn = document.getElementById('artifactDownloadBtn');
        this.artifactCloseBtn = document.getElementById('artifactCloseBtn');
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

        // Model selector
        if (this.modelSelect) {
            this.modelSelect.addEventListener('change', (e) => {
                this.setSelectedModel(e.target.value);
            });
        }

        if (this.voiceBtn) {
            this.voiceBtn.addEventListener('click', () => this.startVoiceInput());
        }
        
        // Sidebar
        this.newChatBtn.addEventListener('click', () => this.createNewChat());
        this.sidebarToggle.addEventListener('click', () => this.toggleSidebar());
        
        // Settings
        this.settingsBtn.addEventListener('click', () => this.openSettings());
        this.closeSettingsBtn.addEventListener('click', () => this.closeSettings());
        this.saveSettingsBtn.addEventListener('click', () => this.saveSettings());
        this.clearHistoryBtn.addEventListener('click', () => this.clearHistory());
        
        // User ID sync controls
        if (this.copyUserIdBtn) {
            this.copyUserIdBtn.addEventListener('click', () => this.copyUserId());
        }
        if (this.importUserIdBtn) {
            this.importUserIdBtn.addEventListener('click', () => this.importUserId());
        }
        if (this.userSelect) {
            this.userSelect.addEventListener('change', () => this.switchUserFromSelect());
        }
        if (this.refreshUsersBtn) {
            this.refreshUsersBtn.addEventListener('click', () => this.loadUsers());
        }
        if (this.addUserBtn) {
            this.addUserBtn.addEventListener('click', () => this.addUser());
        }
        if (this.renameUserBtn) {
            this.renameUserBtn.addEventListener('click', () => this.renameUser());
        }
        if (this.deleteUserBtn) {
            this.deleteUserBtn.addEventListener('click', () => this.deleteUser());
        }
        
        // Theme controls (in settings modal)
        this.themeButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                if (window.themeManager) {
                    window.themeManager.setTheme(btn.dataset.theme);
                }
            });
        });
        
        this.colorButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                if (window.themeManager) {
                    window.themeManager.setColor(btn.dataset.color);
                }
            });
        });

        if (this.artifactCopyBtn) {
            this.artifactCopyBtn.addEventListener('click', () => this.copyArtifact());
        }
        if (this.artifactDownloadBtn) {
            this.artifactDownloadBtn.addEventListener('click', () => this.downloadArtifact());
        }
        if (this.artifactCloseBtn) {
            this.artifactCloseBtn.addEventListener('click', () => this.hideArtifactPanel());
        }
        
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

        if (this.modelSelector) {
            this.modelSelector.style.display = mode === 'auto' ? 'none' : 'flex';
        }
    }

    setSelectedModel(modelPath) {
        this.selectedModel = modelPath || 'auto';
        localStorage.setItem('edison_selected_model', this.selectedModel);
        console.log(`Model changed to: ${this.selectedModel}`);
    }

    async loadAvailableModels() {
        if (!this.modelSelect) return;

        try {
            const response = await fetch(`${this.settings.apiEndpoint}/models/list`);
            if (!response.ok) throw new Error('Failed to load models');

            const data = await response.json();
            this.availableModels = data.models || [];
            this.bestVlm = data.vision_default || null;
            this.populateModelSelector();
            console.log(`Loaded ${this.availableModels.length} available models`);
        } catch (error) {
            console.warn('Failed to load available models:', error.message);
        }
    }

    populateModelSelector() {
        if (!this.modelSelect) return;

        this.modelSelect.innerHTML = '';
        const autoOption = document.createElement('option');
        autoOption.value = 'auto';
        autoOption.textContent = this.bestVlm
            ? `Auto (System Default · VLM: ${this.bestVlm})`
            : 'Auto (System Default)';
        this.modelSelect.appendChild(autoOption);

        const isVlmName = (name) => /llava|vision|vlm|mmproj/i.test(name || '');

        this.availableModels.forEach(model => {
            const option = document.createElement('option');
            option.value = model.path;
            option.textContent = model.name;
            const vlm = model.is_vlm || isVlmName(model.name) || (this.bestVlm && model.filename === this.bestVlm);
            if (vlm) {
                option.disabled = true;
                option.textContent = `${model.name} (VLM - auto)`;
            }
            this.modelSelect.appendChild(option);
        });

        // Reset selection if it points to a VLM
        const selectedIsVlm = this.availableModels.some(m => m.path === this.selectedModel && (m.is_vlm || isVlmName(m.name)));
        if (selectedIsVlm) {
            this.selectedModel = 'auto';
            localStorage.setItem('edison_selected_model', 'auto');
        }

        if (this.selectedModel) {
            this.modelSelect.value = this.selectedModel;
        }
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
            
            // Check if this is an image generation or regeneration request
            const isImageRequest = this.detectImageGenerationRequest(message);
            const isRegenerateRequest = this.detectImageRegenerationRequest(message);
            
            if (isImageRequest || isRegenerateRequest) {
                // Determine the prompt to use
                let promptToUse;
                
                if (isRegenerateRequest && this.lastImagePrompt) {
                    // Regenerating with modifications
                    promptToUse = this.combinePrompts(this.lastImagePrompt, message);
                } else if (isImageRequest && this.hasRecentImageContext()) {
                    // New image request with context - build prompt from recent conversation
                    promptToUse = this.buildPromptFromContext(message);
                } else {
                    // Direct image generation request
                    promptToUse = message;
                }
                
                await this.handleImageGeneration(promptToUse, assistantMessageEl);
            } else if (this.settings.streamResponses) {
                // Use streaming endpoint
                await this.callEdisonAPIStream(message, mode, assistantMessageEl);
            } else {
                // Use non-streaming endpoint
                const response = await this.callEdisonAPI(message, mode);
                
                // Check if response contains image generation trigger from backend
                if (response.image_generation && response.image_generation.prompt) {
                    console.log('🎨 Backend triggered image generation via Coral');
                    await this.handleImageGeneration(response.image_generation.prompt, assistantMessageEl);
                } else {
                    // Update assistant message
                    this.updateMessage(assistantMessageEl, response.response, response.mode_used);
                    
                    // Save to chat history
                    this.saveMessageToChat(message, response.response, response.mode_used);
                    
                    // Generate smart title if first message
                    const chat = this.chats.find(c => c.id === this.currentChatId);
                    if (chat && chat.messages.length === 2) {
                        this.generateChatTitle(chat, message, response.response);
                    }
                }
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

    async stopGeneration() {
        // Cancel server-side generation if we have a request ID
        if (this.currentRequestId) {
            try {
                await fetch(`${this.settings.apiEndpoint}/chat/cancel`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ request_id: this.currentRequestId })
                });
                console.log(`✅ Cancelled request ${this.currentRequestId} on server`);
            } catch (error) {
                console.warn('Failed to cancel request on server:', error);
            }
            this.currentRequestId = null;
        }
        
        // Also abort the client-side fetch
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
        
        const isPdfFile = (file) => {
            const name = (file?.name || '').toLowerCase();
            const type = (file?.type || '').toLowerCase();
            const content = file?.content || '';
            return type === 'application/pdf' || name.endsWith('.pdf') || content.startsWith('data:application/pdf');
        };

        if (attachedFiles.length > 0) {
            console.log('📤 Processing files...');
            
            // Separate images from text files
            const textFiles = attachedFiles.filter(f => !f.isImage && !isPdfFile(f));
            const imageFiles = attachedFiles.filter(f => f.isImage);
            const pdfFiles = attachedFiles.filter(f => isPdfFile(f));
            
            // Add text files to message
            if (textFiles.length > 0) {
                console.log('📤 Including text files in message');
                enhancedMessage += '\n\n[Attached files:]\n';
                textFiles.forEach(file => {
                    console.log(`📤 Adding file: ${file.name}, content length: ${file.content?.length || 0}`);
                    const maxFileChars = 6000;
                    const fileContent = file.content || '';
                    const truncated = fileContent.length > maxFileChars
                        ? `${fileContent.slice(0, maxFileChars)}\n\n[TRUNCATED FILE: ${fileContent.length} chars total]`
                        : fileContent;
                    enhancedMessage += `\n--- File: ${file.name} ---\n${truncated}\n`;
                });
            }

            if (pdfFiles.length > 0) {
                console.log('📤 Uploading PDFs for extraction');
                for (const file of pdfFiles) {
                    await this.uploadDocument(file);
                    enhancedMessage += `\n[Attached PDF: ${file.name} uploaded for extraction]\n`;
                }
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
                project_id: this.projectId,
                session_id: this.sessionId,
                images: images.length > 0 ? images : undefined,
                selected_model: this.selectedModel !== 'auto' ? this.selectedModel : undefined
            }),
            signal: this.abortController?.signal
        });
        
        if (!response.ok) {
            throw new Error(`API returned ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    }

    async callEdisonAPIStream(message, mode, assistantMessageEl) {
        const endpoint = `${this.settings.apiEndpoint}/chat/stream`;
        
        // Get recent conversation history for context
        const conversationHistory = this.getRecentMessages(5);
        
        // Include uploaded files if any
        console.log('📤 Checking for attached files...');
        console.log('window.uploadedFiles:', window.uploadedFiles);
        const attachedFiles = window.uploadedFiles || [];
        console.log('📤 Attached files:', attachedFiles.length, attachedFiles.map(f => f.name));
        
        let enhancedMessage = message;
        let images = [];
        
        const isPdfFile = (file) => {
            const name = (file?.name || '').toLowerCase();
            const type = (file?.type || '').toLowerCase();
            const content = file?.content || '';
            return type === 'application/pdf' || name.endsWith('.pdf') || content.startsWith('data:application/pdf');
        };

        if (attachedFiles.length > 0) {
            console.log('📤 Processing files...');
            
            // Separate images from text files
            const textFiles = attachedFiles.filter(f => !f.isImage && !isPdfFile(f));
            const imageFiles = attachedFiles.filter(f => f.isImage);
            const pdfFiles = attachedFiles.filter(f => isPdfFile(f));
            
            // Add text files to message
            if (textFiles.length > 0) {
                console.log('📤 Including text files in message');
                enhancedMessage += '\n\n[Attached files:]\n';
                textFiles.forEach(file => {
                    console.log(`📤 Adding file: ${file.name}, content length: ${file.content?.length || 0}`);
                    const maxFileChars = 6000;
                    const fileContent = file.content || '';
                    const truncated = fileContent.length > maxFileChars
                        ? `${fileContent.slice(0, maxFileChars)}\n\n[TRUNCATED FILE: ${fileContent.length} chars total]`
                        : fileContent;
                    enhancedMessage += `\n--- File: ${file.name} ---\n${truncated}\n`;
                });
            }

            if (pdfFiles.length > 0) {
                console.log('📤 Uploading PDFs for extraction');
                for (const file of pdfFiles) {
                    await this.uploadDocument(file);
                    enhancedMessage += `\n[Attached PDF: ${file.name} uploaded for extraction]\n`;
                }
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
                project_id: this.projectId,
                session_id: this.sessionId,
                images: images.length > 0 ? images : undefined,
                selected_model: this.selectedModel !== 'auto' ? this.selectedModel : undefined
            }),
            signal: this.abortController?.signal
        });
        
        if (!response.ok) {
            throw new Error(`API returned ${response.status}: ${response.statusText}`);
        }
        
        // Read SSE stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let accumulatedResponse = '';
        let currentEventType = null;
        let swarmInserted = false;
        
        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || ''; // Keep incomplete line in buffer
                
                for (const line of lines) {
                    if (!line.trim()) continue;
                    
                    if (line.startsWith('event: ')) {
                        currentEventType = line.substring(7).trim();
                        continue;
                    }
                    
                    if (line.startsWith('data: ')) {
                        const dataStr = line.substring(6).trim();
                        try {
                            const data = JSON.parse(dataStr);
                            
                            if (data.request_id) {
                                // Init event with request_id
                                this.currentRequestId = data.request_id;
                                console.log(`📡 Streaming request started: ${this.currentRequestId}`);
                            } else if (currentEventType === 'swarm' && data.swarm_agents) {
                                if (!swarmInserted && data.swarm_agents.length > 0) {
                                    this.insertSwarmConversation(assistantMessageEl, data.swarm_agents);
                                    swarmInserted = true;
                                }
                            } else if (currentEventType === 'status' && data.stage) {
                                this.updateStatus(assistantMessageEl, data);
                            } else if (data.t) {
                                // Token event
                                accumulatedResponse += data.t;
                                this.updateMessage(assistantMessageEl, accumulatedResponse, mode);
                            } else if (data.ok !== undefined) {
                                // Done event
                                this.currentRequestId = null;  // Clear request ID
                                if (data.ok) {
                                    // Success
                                    this.updateMessage(assistantMessageEl, accumulatedResponse, data.mode_used || mode);
                                    
                                    // Display swarm agent conversation if not already inserted
                                    if (!swarmInserted && data.swarm_agents && data.swarm_agents.length > 0) {
                                        this.insertSwarmConversation(assistantMessageEl, data.swarm_agents);
                                        swarmInserted = true;
                                    }

                                    // Display generated files if available
                                    if (data.files && data.files.length > 0) {
                                        this.displayGeneratedFiles(assistantMessageEl, data.files);
                                    }
                                    if (data.artifact && data.artifact.content) {
                                        this.showArtifactPanel(data.artifact);
                                    }
                                    this.clearStatus(assistantMessageEl);
                                    
                                    assistantMessageEl.classList.remove('streaming');
                                    
                                    // Save to chat history
                                    this.saveMessageToChat(message, accumulatedResponse, data.mode_used || mode);
                                    
                                    // Generate smart title if first message
                                    const chat = this.chats.find(c => c.id === this.currentChatId);
                                    if (chat && chat.messages.length === 2) {
                                        this.generateChatTitle(chat, message, accumulatedResponse);
                                    }
                                } else if (data.error) {
                                    // Error
                                    throw new Error(data.error);
                                } else if (data.stopped) {
                                    // Stopped by user
                                    this.updateMessage(assistantMessageEl, '⚠️ Response generation stopped by user.', 'stopped');
                                    assistantMessageEl.classList.remove('streaming');
                                }
                                return; // Exit stream processing
                            }
                        } catch (e) {
                            console.warn('Failed to parse SSE data:', dataStr, e);
                        }
                    }
                }
            }
        } finally {
            reader.releaseLock();
            this.currentRequestId = null;  // Always clear request ID when done
        }
    }

    getRecentMessages(count = 5) {
        const messages = Array.from(this.messagesContainer.querySelectorAll('.message:not(.streaming):not(.swarm-agent)'));
        const recent = messages.slice(-count * 2); // Get last N exchanges (user + assistant pairs)
        
        return recent.map(msg => ({
            role: msg.classList.contains('user') ? 'user' : 'assistant',
            content: msg.querySelector('.message-content')?.textContent || ''
        }));
    }

    addMessage(role, content, isStreaming = false, isHtml = false) {
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
            <div class="message-content">${isHtml ? content : this.formatMessage(content)}</div>
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

    insertSwarmConversation(assistantMessageEl, swarmAgents) {
        // Insert swarm agent messages as separate chat entries before the main response
        const fragment = document.createDocumentFragment();

        swarmAgents.forEach(agent => {
            const agentMessageEl = document.createElement('div');
            agentMessageEl.className = 'message assistant swarm-agent';

            agentMessageEl.innerHTML = `
                <div class="message-header">
                    <div class="message-avatar">${agent.icon || '🐝'}</div>
                    <span class="message-role">${agent.agent || 'Agent'}</span>
                    <span class="message-mode">${(agent.model || 'Unknown Model').toUpperCase()}</span>
                </div>
                <div class="message-content">${this.formatMessage(agent.response || '')}</div>
                <div class="message-actions">
                    <button class="action-btn copy-btn" title="Copy to clipboard">
                        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                            <path d="M4 1.5H3a2 2 0 0 0-2 2V14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V3.5a2 2 0 0 0-2-2h-1v1h1a1 1 0 0 1 1 1V14a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V3.5a1 1 0 0 1 1-1h1v-1z"/>
                            <path d="M9.5 1a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5v-1a.5.5 0 0 1 .5-.5h3zm-3-1A1.5 1.5 0 0 0 5 1.5v1A1.5 1.5 0 0 0 6.5 4h3A1.5 1.5 0 0 0 11 2.5v-1A1.5 1.5 0 0 0 9.5 0h-3z"/>
                        </svg>
                    </button>
                </div>
            `;

            const copyBtn = agentMessageEl.querySelector('.copy-btn');
            if (copyBtn) {
                copyBtn.addEventListener('click', () => this.copyToClipboard(agent.response || ''));
            }

            fragment.appendChild(agentMessageEl);
        });

        // Insert all agent messages before the main assistant response
        this.messagesContainer.insertBefore(fragment, assistantMessageEl);
        this.scrollToBottom();
    }

    updateStatus(assistantMessageEl, data) {
        const header = assistantMessageEl.querySelector('.message-header');
        let statusEl = assistantMessageEl.querySelector('.message-status');
        if (!statusEl) {
            statusEl = document.createElement('div');
            statusEl.className = 'message-status';
            header.appendChild(statusEl);
        }
        const progress = data.total ? Math.round((data.current / data.total) * 100) : 0;
        const detail = data.detail ? ` — ${data.detail}` : '';
        statusEl.innerHTML = `
            <span class="status-ring" style="--progress:${progress}%"></span>
            <span class="status-text">${data.stage}${detail}</span>
        `;
    }

    clearStatus(assistantMessageEl) {
        const statusEl = assistantMessageEl.querySelector('.message-status');
        if (statusEl) statusEl.remove();
    }

    async uploadDocument(file) {
        try {
            const response = await fetch(`${this.settings.apiEndpoint}/upload-document`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: file.name,
                    content_base64: file.content
                })
            });
            if (!response.ok) {
                console.warn('PDF upload failed:', file.name);
            }
        } catch (error) {
            console.warn('PDF upload error:', error.message);
        }
    }

    displayGeneratedFiles(assistantMessageEl, files) {
        const contentEl = assistantMessageEl.querySelector('.message-content');
        const fileSection = document.createElement('div');
        fileSection.className = 'generated-files';
        fileSection.innerHTML = `
            <div class="generated-files-header">📁 Generated Files</div>
            <ul class="generated-files-list">
                ${files.map(file => `
                    <li>
                        <a href="${this.settings.apiEndpoint}${file.url}" target="_blank" rel="noopener" download>${file.name}</a>
                        <span class="file-meta">${file.type?.toUpperCase() || 'FILE'} · ${this.formatFileSize(file.size || 0)}</span>
                    </li>
                `).join('')}
            </ul>
        `;
        contentEl.appendChild(fileSection);
    }

    formatFileSize(bytes) {
        if (!bytes) return '0 B';
        const units = ['B', 'KB', 'MB', 'GB'];
        const idx = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
        const size = bytes / Math.pow(1024, idx);
        return `${size.toFixed(size >= 10 || idx === 0 ? 0 : 1)} ${units[idx]}`;
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

        let reasoningHtml = '';
        const thinkingMatch = content.match(/<thinking>([\s\S]*?)<\/thinking>/i);
        if (thinkingMatch) {
            const thinkingText = this.escapeHtml(thinkingMatch[1].trim()).replace(/\n/g, '<br>');
            reasoningHtml = `
                <details class="reasoning-section">
                    <summary>🔍 Show Reasoning Process</summary>
                    <div class="thinking-content">${thinkingText}</div>
                </details>
            `;
            content = content.replace(/<thinking>[\s\S]*?<\/thinking>/i, '').trim();
        }
        
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
        
        return reasoningHtml + formatted;
    }

    showArtifactPanel(artifact) {
        if (!this.artifactsPanel || !artifact) return;
        const title = artifact.title || `${(artifact.type || 'artifact').toUpperCase()} Artifact`;
        const code = artifact.code || '';
        this.artifactTitle.textContent = title;
        this.artifactCode.textContent = code;
        this.currentArtifact = { title, code, type: artifact.type || 'txt' };

        this.artifactsPanel.style.display = 'flex';
        const container = document.querySelector('.chat-container');
        if (container) container.classList.add('with-artifacts');
    }

    hideArtifactPanel() {
        if (!this.artifactsPanel) return;
        this.artifactsPanel.style.display = 'none';
        const container = document.querySelector('.chat-container');
        if (container) container.classList.remove('with-artifacts');
        this.currentArtifact = null;
    }

    copyArtifact() {
        if (!this.currentArtifact) return;
        navigator.clipboard.writeText(this.currentArtifact.code || '').catch(() => {
            alert('Copy failed');
        });
    }

    downloadArtifact() {
        if (!this.currentArtifact) return;
        const extMap = {
            html: 'html',
            react: 'jsx',
            svg: 'svg',
            mermaid: 'html',
            code: 'txt'
        };
        const ext = extMap[this.currentArtifact.type] || 'txt';
        const blob = new Blob([this.currentArtifact.code], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `artifact.${ext}`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    }

    async startVoiceInput() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert('Voice input not supported in this browser.');
            return;
        }
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream);
            const chunks = [];
            mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
            mediaRecorder.onstop = async () => {
                const blob = new Blob(chunks, { type: 'audio/wav' });
                const formData = new FormData();
                formData.append('audio', blob, 'voice.wav');
                const endpoint = this.settings.voiceEndpoint || 'http://localhost:8809';
                const response = await fetch(`${endpoint}/voice-to-text`, {
                    method: 'POST',
                    body: formData
                });
                if (!response.ok) throw new Error('Voice transcription failed');
                const data = await response.json();
                this.messageInput.value = (data.text || '').trim();
                this.handleInputChange();
            };
            mediaRecorder.start();
            setTimeout(() => mediaRecorder.stop(), 5000);
        } catch (error) {
            alert(error.message || 'Voice input failed');
        }
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
            // Call backend API to generate a smart, concise title
            const response = await fetch(`${this.settings.apiEndpoint}/generate-title`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: firstMessage,
                    response: firstResponse
                })
            });
            
            if (response.ok) {
                const data = await response.json();
                chat.title = data.title || firstMessage.substring(0, 40);
            } else {
                // Fallback to simple truncation
                chat.title = firstMessage.substring(0, 40);
            }
            
            this.saveChats();
            this.renderChatHistory();
        } catch (error) {
            console.error('Error generating title:', error);
            // Fallback to simple truncation
            chat.title = firstMessage.substring(0, 40);
            this.saveChats();
            this.renderChatHistory();
        }
    }

    detectImageGenerationRequest(message) {
        const lowerMessage = message.toLowerCase().trim();
        
        // Check for explicit image generation keywords
        const imageKeywords = [
            'generate image', 'create image', 'make image', 'draw', 'generate a picture',
            'create a picture', 'make a picture', 'generate an image', 'create an image',
            'make an image', 'paint', 'illustrate', 'visualize', 'generate art', 'create art',
            'make art', 'flux', 'stable diffusion', 'text to image', 'text2img'
        ];
        
        if (imageKeywords.some(keyword => lowerMessage.includes(keyword))) {
            return true;
        }
        
        // Check for contextual follow-up commands when there's image context
        if (this.hasRecentImageContext()) {
            const followUpKeywords = [
                'generate it', 'create it', 'make it', 'do it', 'go ahead',
                'yes', 'yeah', 'sure', 'okay', 'ok', 'both', 'all of them',
                'that one', 'those', 'them'
            ];
            
            // For very short messages, be more aggressive with context
            if (lowerMessage.length <= 10) {
                return followUpKeywords.some(keyword => lowerMessage === keyword || lowerMessage.includes(keyword));
            }
        }
        
        return false;
    }

    hasRecentImageContext() {
        // Check if any of the last 3 user messages mentioned images
        const messages = Array.from(this.messagesContainer.querySelectorAll('.message.user:not(.streaming)'));
        const recentMessages = messages.slice(-3);
        
        return recentMessages.some(msg => {
            const content = msg.querySelector('.message-content')?.textContent || '';
            const lowerContent = content.toLowerCase();
            return lowerContent.includes('image') || 
                   lowerContent.includes('picture') || 
                   lowerContent.includes('draw') ||
                   lowerContent.includes('heart') ||
                   lowerContent.includes('cartoon') ||
                   lowerContent.includes('generate');
        });
    }

    buildPromptFromContext(message) {
        // Build a prompt by combining recent conversation context
        const messages = Array.from(this.messagesContainer.querySelectorAll('.message:not(.streaming):not(.swarm-agent)'));
        const recentMessages = messages.slice(-6); // Last 3 exchanges
        
        let promptParts = [];
        
        // Extract relevant details from recent messages
        recentMessages.forEach(msg => {
            const content = msg.querySelector('.message-content')?.textContent || '';
            const lowerContent = content.toLowerCase();
            
            // Skip messages that are just responses or questions
            if (msg.classList.contains('assistant')) return;
            if (lowerContent.includes('?') && lowerContent.length < 50) return;
            
            // Look for descriptive content
            if (lowerContent.includes('heart') || 
                lowerContent.includes('cartoon') ||
                lowerContent.includes('cute') ||
                lowerContent.includes('name')) {
                promptParts.push(content.trim());
            }
            
            // Extract names mentioned
            const nameMatches = content.match(/\b([A-Z][a-z]+)\b/g);
            if (nameMatches) {
                nameMatches.forEach(name => {
                    if (name.length > 2 && !['You', 'EDISON', 'The', 'For', 'Image'].includes(name)) {
                        promptParts.push(name);
                    }
                });
            }
        });
        
        // If the current message is just a command like "generate it", use the context
        const simpleCommands = ['generate it', 'create it', 'make it', 'do it', 'both', 'yes', 'okay', 'sure'];
        const lowerMessage = message.toLowerCase().trim();
        
        if (simpleCommands.some(cmd => lowerMessage.includes(cmd) || lowerMessage === cmd)) {
            // Build from context only
            const uniqueParts = [...new Set(promptParts)];
            if (uniqueParts.length > 0) {
                return uniqueParts.join(', ');
            }
        }
        
        // Otherwise, combine message with context
        return message + (promptParts.length > 0 ? ', ' + promptParts.join(', ') : '');
    }

    detectImageRegenerationRequest(message) {
        // Only consider regeneration if we have a previous image prompt
        if (!this.lastImagePrompt) return false;
        
        const lowerMessage = message.toLowerCase().trim();
        const regenerateKeywords = [
            'try again', 'regenerate', 'make another', 'do it again', 'retry',
            'redo', 'one more', 'again', 'remake', 'recreate', 'generate another',
            'make a new one', 'new one', 'another one'
        ];
        
        // Also check if message is modifying the image ("but with", "except", "instead", etc.)
        const modificationKeywords = [
            'but with', 'except', 'instead', 'change', 'modify', 'different',
            'with', 'add', 'remove', 'without', 'replace', 'this time'
        ];
        
        // Check for simple follow-ups that suggest regeneration
        const simpleRegenerateWords = ['again', 'retry', 'redo', 'another'];
        if (simpleRegenerateWords.some(word => lowerMessage === word)) {
            return true;
        }
        
        return regenerateKeywords.some(keyword => lowerMessage.includes(keyword)) ||
               modificationKeywords.some(keyword => lowerMessage.includes(keyword));
    }

    combinePrompts(originalPrompt, modificationMessage) {
        // If the modification message contains explicit prompt details, combine them
        const lowerMessage = modificationMessage.toLowerCase().trim();
        
        // If message is very short and doesn't contain modifications, just use original
        const simpleCommands = ['again', 'retry', 'redo', 'another', 'try again', 'do it again'];
        if (simpleCommands.includes(lowerMessage)) {
            return originalPrompt;
        }
        
        // Extract any specific modifications from the message
        if (lowerMessage.includes('with') || lowerMessage.includes('but') || 
            lowerMessage.includes('instead') || lowerMessage.includes('change') ||
            lowerMessage.includes('add') || lowerMessage.includes('both')) {
            // Try to extract the modification part
            let modifiers = modificationMessage
                .replace(/^(try again|regenerate|remake|redo|again|please)/i, '')
                .replace(/^(but|with|except|instead|change to|modify to|and|also)/i, '')
                .trim();
            
            // Handle "both" as a special case to include previous context
            if (lowerMessage === 'both' || lowerMessage.includes('both')) {
                // Get the last few messages to understand context
                const recentMsgs = this.getRecentMessages(3);
                const contextNames = [];
                recentMsgs.forEach(msg => {
                    // Extract potential names or subjects from context
                    const matches = msg.content.match(/\b[A-Z][a-z]+\b/g);
                    if (matches) contextNames.push(...matches);
                });
                if (contextNames.length > 0) {
                    modifiers = contextNames.join(' and ');
                }
            }
            
            if (modifiers.length > 0) {
                return `${this.extractImagePrompt(originalPrompt)}, ${modifiers}`;
            }
        }
        
        // Otherwise, just use the original prompt
        return originalPrompt;
    }

    async handleImageGeneration(message, assistantMessageEl) {
        try {
            // Extract the actual image prompt from the message
            const imagePrompt = this.extractImagePrompt(message);
            
            this.updateMessage(assistantMessageEl, '🎨 Generating image, please wait...', 'image');
            
            // Call image generation API with ComfyUI endpoint
            const response = await fetch(`${this.settings.apiEndpoint}/generate-image`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: imagePrompt,
                    width: 1024,
                    height: 1024,
                    comfyui_url: this.settings.comfyuiEndpoint
                })
            });
            
            if (!response.ok) {
                throw new Error(`Image generation failed: ${response.statusText}`);
            }
            
            const result = await response.json();
            const promptId = result.prompt_id;
            
            // Show loading animation
            const loadingHtml = `
                <div style="text-align: center; padding: 20px;">
                    <div style="display: inline-block; width: 300px; height: 300px; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 12px; position: relative; overflow: hidden;">
                        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center;">
                            <div style="width: 60px; height: 60px; border: 4px solid rgba(102, 126, 234, 0.3); border-top-color: #667eea; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 16px;"></div>
                            <div style="font-size: 16px; color: #667eea; font-weight: 500;">Generating Image...</div>
                            <div id="progress-text" style="font-size: 14px; color: #888; margin-top: 8px;">Initializing...</div>
                        </div>
                    </div>
                    <p style="margin-top: 16px; color: #888;"><strong>Prompt:</strong> ${imagePrompt}</p>
                </div>
            `;
            this.updateMessage(assistantMessageEl, loadingHtml, 'image');
            
            // Poll for completion
            let attempts = 0;
            const maxAttempts = 120; // 2 minutes timeout for first generation
            let lastStatus = 'queued';
            
            while (attempts < maxAttempts) {
                await new Promise(resolve => setTimeout(resolve, 1000));
                
                const statusResponse = await fetch(`${this.settings.apiEndpoint}/image-status/${promptId}`);
                const status = await statusResponse.json();
                
                if (status.status === 'completed') {
                    // Store the prompt for regeneration
                    this.lastImagePrompt = message;
                    
                    // Display the generated image with download and regenerate buttons
                    const fullImageUrl = `${this.settings.apiEndpoint}${status.image_url}`;
                    const imageHtml = `
                        <p>✅ Image generated successfully!</p>
                        <div class="generated-image">
                            <img src="${fullImageUrl}" alt="Generated image" style="max-width: 100%; border-radius: 8px; margin-top: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                        </div>
                        <div style="margin-top: 10px; display: flex; align-items: center; gap: 12px;">
                            <button onclick="downloadImage('${fullImageUrl}', 'EDISON_${Date.now()}.png')" style="padding: 10px 14px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 20px; box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3); transition: transform 0.2s; display: flex; align-items: center; justify-content: center; line-height: 1;" onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'" title="Download Image">
                                📥
                            </button>
                            <button onclick="regenerateImage('${imagePrompt.replace(/'/g, "\\'")}')
" style="padding: 10px 14px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 18px; box-shadow: 0 2px 8px rgba(240, 147, 251, 0.3); transition: transform 0.2s; display: flex; align-items: center; justify-content: center; line-height: 1;" onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'" title="Regenerate Image">
                                🔄
                            </button>
                            <span style="color: #888; font-size: 14px;"><strong>Prompt:</strong> ${imagePrompt}</span>
                        </div>
                    `;
                    this.updateMessage(assistantMessageEl, imageHtml, 'image');
                    
                    // Save to chat history with HTML so it persists after refresh
                    this.saveMessageToChat(message, imageHtml, 'image');
                    
                    break;
                } else if (status.status === 'error') {
                    throw new Error(status.message || 'Image generation failed');
                } else if (status.status === 'not_found') {
                    throw new Error('Generation job not found');
                }
                
                // Update progress text
                const progressEl = document.getElementById('progress-text');
                if (progressEl) {
                    let progressMsg = 'Processing...';
                    if (attempts < 5) {
                        progressMsg = 'Loading models...';
                    } else if (attempts < 30) {
                        progressMsg = 'Generating image (4 steps)...';
                    } else {
                        progressMsg = `Still generating... (${attempts}s)`;
                    }
                    progressEl.textContent = progressMsg;
                }
                attempts++;
            }
            
            if (attempts >= maxAttempts) {
                throw new Error('Image generation timed out');
            }
            
        } catch (error) {
            console.error('Image generation error:', error);
            this.updateMessage(
                assistantMessageEl,
                `⚠️ Error generating image: ${error.message}. Make sure ComfyUI is running and FLUX model is installed.`,
                'error'
            );
        }
    }

    extractImagePrompt(message) {
        // Remove common prefixes to extract the actual prompt
        let prompt = message
            .replace(/^(generate|create|make|draw|paint|illustrate|visualize)\s+(an?\s+)?(image|picture|art|photo)\s+(of\s+)?/i, '')
            .replace(/^(generate|create|make)\s+/i, '')
            .trim();
        
        if (!prompt) {
            prompt = message; // Fallback to original message
        }
        
        return prompt;
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
            // Check if this is an image message with HTML content
            const isImageMessage = msg.mode === 'image' && msg.content.includes('<img src=');
            const messageEl = this.addMessage(msg.role, msg.content, false, isImageMessage);
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
        this.comfyuiEndpointInput.value = this.settings.comfyuiEndpoint;
        if (this.voiceEndpointInput) {
            this.voiceEndpointInput.value = this.settings.voiceEndpoint || '';
        }
        this.defaultModeSelect.value = this.settings.defaultMode;
        
        // Populate user ID field
        if (this.userIdInput) {
            this.userIdInput.value = this.userId;
        }
        this.loadUsers();
        
        this.settingsModal.classList.add('open');
        this.checkSystemStatus();
    }

    closeSettings() {
        this.settingsModal.classList.remove('open');
    }

    copyUserId() {
        if (this.userId) {
            navigator.clipboard.writeText(this.userId).then(() => {
                // Show feedback
                const btn = this.copyUserIdBtn;
                const originalText = btn.textContent;
                btn.textContent = '✓ Copied!';
                setTimeout(() => { btn.textContent = originalText; }, 2000);
            }).catch(err => {
                console.error('Failed to copy user ID:', err);
                alert('Failed to copy. Please copy manually: ' + this.userId);
            });
        }
    }

    importUserId() {
        const newUserId = this.importUserIdInput?.value?.trim();
        if (!newUserId) {
            alert('Please paste a User ID first');
            return;
        }
        
        if (confirm(`This will replace your current User ID and sync chats from the imported account. Continue?`)) {
            // Save the new user ID
            this.userId = newUserId;
            localStorage.setItem('edison_user_id', newUserId);
            
            // Update display
            if (this.userIdInput) {
                this.userIdInput.value = newUserId;
            }
            
            // Clear import field
            if (this.importUserIdInput) {
                this.importUserIdInput.value = '';
            }
            
            // Sync chats from server with new user ID
            this.syncChatsFromServer().then(() => {
                alert('User ID imported successfully! Your chats have been synced.');
            });
        }
    }

    async loadUsers() {
        if (!this.userSelect) return;
        try {
            const response = await fetch(`${this.settings.apiEndpoint}/users`);
            if (!response.ok) throw new Error('Failed to load users');
            const data = await response.json();
            const users = data.users || [];
            this.userSelect.innerHTML = users.map(u => {
                const selected = u.id === this.userId ? 'selected' : '';
                return `<option value="${u.id}" ${selected}>${u.name}</option>`;
            }).join('');
        } catch (error) {
            console.warn('Could not load users:', error.message);
        }
    }

    async addUser() {
        const name = this.newUserNameInput?.value?.trim();
        if (!name) {
            alert('Please enter a user name');
            return;
        }
        try {
            const response = await fetch(`${this.settings.apiEndpoint}/users`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name })
            });
            if (!response.ok) throw new Error('Failed to create user');
            this.newUserNameInput.value = '';
            await this.loadUsers();
        } catch (error) {
            alert(error.message);
        }
    }

    async renameUser() {
        const name = this.renameUserInput?.value?.trim();
        const userId = this.userSelect?.value;
        if (!name || !userId) {
            alert('Select a user and enter a new name');
            return;
        }
        try {
            const response = await fetch(`${this.settings.apiEndpoint}/users/${userId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name })
            });
            if (!response.ok) throw new Error('Failed to rename user');
            this.renameUserInput.value = '';
            await this.loadUsers();
        } catch (error) {
            alert(error.message);
        }
    }

    async deleteUser() {
        const userId = this.userSelect?.value;
        if (!userId) return;
        if (!confirm('Delete this user and all chats? This cannot be undone.')) return;
        try {
            const response = await fetch(`${this.settings.apiEndpoint}/users/${userId}`, {
                method: 'DELETE'
            });
            if (!response.ok) throw new Error('Failed to delete user');
            if (userId === this.userId) {
                // Reset to a new local ID
                this.userId = this.getOrCreateUserId();
                if (this.userIdInput) this.userIdInput.value = this.userId;
                this.syncChatsFromServer();
            }
            await this.loadUsers();
        } catch (error) {
            alert(error.message);
        }
    }

    async switchUserFromSelect() {
        const userId = this.userSelect?.value;
        if (!userId || userId === this.userId) return;
        this.userId = userId;
        localStorage.setItem('edison_user_id', userId);
        if (this.userIdInput) this.userIdInput.value = userId;
        await this.syncChatsFromServer();
        this.renderChatHistory();
    }

    saveSettings() {
        this.settings.apiEndpoint = this.apiEndpointInput.value.trim();
        this.settings.comfyuiEndpoint = this.comfyuiEndpointInput.value.trim();
        if (this.voiceEndpointInput) {
            this.settings.voiceEndpoint = this.voiceEndpointInput.value.trim();
        }
        this.settings.defaultMode = this.defaultModeSelect.value;
        
        localStorage.setItem('edison_settings', JSON.stringify(this.settings));
        
        // Update current mode if needed
        this.setMode(this.settings.defaultMode);
        this.loadAvailableModels();
        
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
        const defaults = {
            apiEndpoint: 'http://192.168.1.26:8811',
            comfyuiEndpoint: 'http://192.168.1.26:8188',
            voiceEndpoint: 'http://192.168.1.26:8809',
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
