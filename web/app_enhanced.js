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
        this.lastGeneratedImage = null;
        this.projectId = localStorage.getItem('edison_project_id') || 'default';
        this.sessionId = localStorage.getItem('edison_session_id') || `session_${Date.now()}`;
        localStorage.setItem('edison_session_id', this.sessionId);
        this.availableModels = [];
        this.selectedModel = localStorage.getItem('edison_selected_model') || 'auto';
        this.readiness = { overall: 'unknown', components: [], unavailable: [] };
        this.modelSelector = null;
        this.modelSelect = null;
        this.bestVlm = null;
        this.supportedModes = new Set(['auto', 'instant', 'thinking', 'chat', 'reasoning', 'code', 'agent', 'swarm', 'work']);
        this.modeDescriptions = {
            auto: 'Auto picks the best strategy for your request.',
            instant: 'Instant is optimized for quick answers with minimal latency.',
            thinking: 'Thinking uses deeper reasoning for complex multi-step problems.',
            chat: 'Chat is balanced for natural conversation and everyday help.',
            reasoning: 'Reasoning focuses on explicit step-by-step analysis.',
            code: 'Code is specialized for coding, debugging, and implementation tasks.',
            agent: 'Agent can use tools for multi-step task execution.',
            swarm: 'Swarm coordinates multiple specialist agents in parallel.',
            work: 'Work mode structures progress and task-oriented workflows.',
        };

        // Swarm collaboration state
        this.swarmSessionId = null;
        this.swarmAgentCatalog = [];
        this.swarmAutocompleteVisible = false;
        
        this.initializeElements();
        this.attachEventListeners();
        this.checkSystemStatus();
        this.refreshReadiness();
        this.loadCurrentChat();
        this.setMode(localStorage.getItem('edison_last_mode') || this.settings.defaultMode);
        this.loadAvailableModels();
        this.loadSwarmAgentCatalog();  // Preload swarm agent names for @mention
        this.loadUsers();  // Populate user dropdowns on startup
        this.handleViewportChange();
        window.addEventListener('resize', () => this.handleViewportChange());
        // Re-sync chats when tab regains focus (cross-browser sync)
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden) {
                if (this._needsUserBootstrap) {
                    this._checkAndRestoreNamedUser();
                } else {
                    this.syncChatsFromServer();
                }
            }
        });
        // Periodic sync every 45 seconds
        this._syncInterval = setInterval(() => {
            if (!document.hidden && !this.isStreaming) {
                if (this._needsUserBootstrap) {
                    this._checkAndRestoreNamedUser();
                } else {
                    this.syncChatsFromServer();
                }
            }
        }, 45000);
        // On startup, check if we should resume a known named user
        this._checkAndRestoreNamedUser();
    }

    getOrCreateUserId() {
        // Use a persistent user ID stored in localStorage for cross-network access
        let userId = localStorage.getItem('edison_user_id');
        if (!userId) {
            this._needsUserBootstrap = true;
            // Generate a simple ID based on timestamp and random string
            userId = 'user_' + Date.now().toString(36) + '_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('edison_user_id', userId);
        } else {
            this._needsUserBootstrap = false;
        }
        return userId;
    }

    async _checkAndRestoreNamedUser() {
        // If we already confirmed this browser identity and don't need bootstrap, skip this check
        if (!this._needsUserBootstrap && localStorage.getItem('edison_user_id_confirmed')) return;

        try {
            const response = await fetch(`${this.settings.apiEndpoint}/users`);
            if (!response.ok) return;
            const data = await response.json();
            const users = (data.users || []).filter(u => {
                const name = u.name || '';
                // Only show users with custom (non-auto-generated) names
                return !(/^User-[a-f0-9\-]{4,}$/.test(name) || /^User-user_/.test(name));
            });

            if (users.length === 0) {
                // No named users yet — confirm current auto-generated one is fine
                this._needsUserBootstrap = false;
                localStorage.setItem('edison_user_id_confirmed', '1');
                await this.syncChatsFromServer();
                return;
            }

            // Check if current userId is already one of the named users
            const alreadyNamed = users.find(u => u.id === this.userId);
            if (alreadyNamed) {
                this._needsUserBootstrap = false;
                localStorage.setItem('edison_user_id_confirmed', '1');
                await this.syncChatsFromServer();
                return;
            }

            // We have named users but current session is auto-generated — offer a quick restore
            const suggested = users[0];
            const options = users.map(u => `- ${u.name}`).join('\n');
            const choice = window.confirm(
                `Welcome back! Found existing users:\n${options}\n\nClick OK to switch to "${suggested.name}" now, or Cancel to stay on a new user.`
            );
            if (choice) {
                this.setActiveUser(suggested.id, true);
                await this.syncChatsFromServer();
                this.loadCurrentChat();
            } else {
                await this.syncChatsFromServer();
            }
            this._needsUserBootstrap = false;
            // In either case, mark as confirmed so this doesn't prompt again
            localStorage.setItem('edison_user_id_confirmed', '1');
        } catch (e) {
            // Server not available yet, skip
        }
    }

    getChatsStorageKey(userId = null) {
        const resolvedUserId = userId || this.userId || 'default';
        return `edison_chats_${resolvedUserId}`;
    }

    getDefaultEndpoints() {
        // All API calls go through the web server's reverse proxy (/api/*)
        // so the browser only needs one HTTPS origin (port 8080).
        const origin = window.location.origin;  // e.g. https://host:8080
        const host = window.location.hostname || 'localhost';
        return {
            apiEndpoint: `${origin}/api`,
            comfyuiEndpoint: `http://${host}:8188`,
            voiceEndpoint: `${origin}/api`
        };
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
        this.mobileHeader = document.getElementById('mobileHeader');
        this.mobileMenuBtn = document.getElementById('mobileMenuBtn');
        this.mobileBackdrop = document.getElementById('mobileBackdrop');
        
        // Mode selector
        this.modeButtons = document.querySelectorAll('.mode-btn');
        this.modeHelp = document.getElementById('modeHelp');
        this.moduleAvailabilityBar = document.getElementById('moduleAvailabilityBar');
        
        // Settings modal
        this.settingsModal = document.getElementById('settingsModal');
        this.settingsBtn = document.getElementById('settingsBtn');
        this.closeSettingsBtn = document.getElementById('closeSettingsBtn');
        this.saveSettingsBtn = document.getElementById('saveSettingsBtn');
        this.clearHistoryBtn = document.getElementById('clearHistoryBtn');
        this.apiEndpointInput = document.getElementById('apiEndpoint');
        this.comfyuiEndpointInput = document.getElementById('comfyuiEndpoint');
        this.defaultModeSelect = document.getElementById('defaultMode');
        this.assistantProfileSelect = document.getElementById('assistantProfileId');
        this.systemStatus = document.getElementById('systemStatus');
        this.setupCheckBtn = document.getElementById('setupCheckBtn');
        this.setupCheckModal = document.getElementById('setupCheckModal');
        this.setupCheckList = document.getElementById('setupCheckList');
        this.setupCheckCloseBtn = document.getElementById('setupCheckCloseBtn');
        this.setupCheckRefreshBtn = document.getElementById('setupCheckRefreshBtn');

        // Model selector
        this.modelSelector = document.getElementById('modelSelector');
        this.modelSelect = document.getElementById('modelSelect');
        this.voiceBtn = document.getElementById('voiceBtn');
        this.voiceEndpointInput = document.getElementById('voiceEndpoint');
        this.modelNameInput = document.getElementById('modelNameInput');
        this.modelPathInput = document.getElementById('modelPathInput');
        this.modelCtxInput = document.getElementById('modelCtxInput');
        this.modelTensorInput = document.getElementById('modelTensorInput');
        this.loadModelBtn = document.getElementById('loadModelBtn');
        this.unloadModelBtn = document.getElementById('unloadModelBtn');
        
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

        // Chat header user switcher
        this.chatUserSelect = document.getElementById('chatUserSelect');
        this.chatAddUserBtn = document.getElementById('chatAddUserBtn');
        this.chatDeleteUserBtn = document.getElementById('chatDeleteUserBtn');
        this.chatCleanUsersBtn = document.getElementById('chatCleanUsersBtn');
        this.sandboxAllowAnyHostInput = document.getElementById('sandboxAllowAnyHost');
        this.sandboxAllowedHostsInput = document.getElementById('sandboxAllowedHosts');
        this.discordWebhookInput = document.getElementById('discordWebhook');
        this.slackWebhookInput = document.getElementById('slackWebhook');
        this.defaultPrinterIdInput = document.getElementById('defaultPrinterId');
        this.installedSkillsInput = document.getElementById('installedSkills');
        
        // Theme controls (in settings modal)
        this.themeButtons = document.querySelectorAll('.theme-btn');
        this.colorButtons = document.querySelectorAll('.color-btn');

        // Artifacts panel
        this.artifactsPanel = document.getElementById('artifactsPanel');
        this.artifactTitle = document.getElementById('artifactTitle');
        this.artifactMeta = document.getElementById('artifactMeta');
        this.artifactCode = document.getElementById('artifactCode');
        this.artifactPreviewPane = document.getElementById('artifactPreviewPane');
        this.artifactCodePane = document.getElementById('artifactCodePane');
        this.artifactPreviewFrame = document.getElementById('artifactPreviewFrame');
        this.artifactPreviewEmpty = document.getElementById('artifactPreviewEmpty');
        this.artifactPreviewTab = document.getElementById('artifactPreviewTab');
        this.artifactCodeTab = document.getElementById('artifactCodeTab');
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

        // Setup Check modal
        if (this.setupCheckBtn) {
            this.setupCheckBtn.addEventListener('click', () => this.openSetupCheck());
        }
        if (this.setupCheckCloseBtn) {
            this.setupCheckCloseBtn.addEventListener('click', () => this.closeSetupCheck());
        }
        if (this.setupCheckRefreshBtn) {
            this.setupCheckRefreshBtn.addEventListener('click', () => this.runSetupCheck());
        }
        if (this.setupCheckModal) {
            this.setupCheckModal.addEventListener('click', (e) => {
                if (e.target === this.setupCheckModal) this.closeSetupCheck();
            });
        }

        // Model selector
        if (this.modelSelect) {
            this.modelSelect.addEventListener('change', (e) => {
                this.setSelectedModel(e.target.value);
            });
        }

        if (this.voiceBtn) {
            this.voiceBtn.addEventListener('click', () => this.startVoiceInput());
        }
        if (this.loadModelBtn) {
            this.loadModelBtn.addEventListener('click', () => this.loadModelHotSwap());
        }
        if (this.unloadModelBtn) {
            this.unloadModelBtn.addEventListener('click', () => this.unloadModelHotSwap());
        }
        
        // Sidebar
        this.newChatBtn.addEventListener('click', () => this.createNewChat());
        this.sidebarToggle.addEventListener('click', () => this.toggleSidebar());
        if (this.mobileMenuBtn) {
            this.mobileMenuBtn.addEventListener('click', () => this.toggleSidebar());
        }
        if (this.mobileBackdrop) {
            this.mobileBackdrop.addEventListener('click', () => this.closeMobileSidebar());
        }
        
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

        // Chat header user switcher
        if (this.chatUserSelect) {
            this.chatUserSelect.addEventListener('change', () => this.switchUserFromChatHeader());
        }
        if (this.chatAddUserBtn) {
            this.chatAddUserBtn.addEventListener('click', () => this.addUserFromChatHeader());
        }
        if (this.chatDeleteUserBtn) {
            this.chatDeleteUserBtn.addEventListener('click', () => this.deleteUserFromChatHeader());
        }
        if (this.chatCleanUsersBtn) {
            this.chatCleanUsersBtn.addEventListener('click', () => this.cleanupUsers());
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
        if (this.artifactPreviewTab) {
            this.artifactPreviewTab.addEventListener('click', () => this.setArtifactView('preview'));
        }
        if (this.artifactCodeTab) {
            this.artifactCodeTab.addEventListener('click', () => this.setArtifactView('code'));
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
        if (this.isMobileView()) {
            if (this.sidebar.classList.contains('mobile-open')) {
                this.closeMobileSidebar();
            } else {
                this.openMobileSidebar();
            }
            return;
        }
        this.sidebarCollapsed = !this.sidebarCollapsed;
        this.sidebar.classList.toggle('collapsed');
        document.querySelector('.main-content').classList.toggle('sidebar-collapsed');
    }

    isMobileView() {
        return window.matchMedia('(max-width: 768px)').matches;
    }

    openMobileSidebar() {
        this.sidebar.classList.add('mobile-open');
        if (this.mobileBackdrop) this.mobileBackdrop.classList.add('active');
    }

    closeMobileSidebar() {
        this.sidebar.classList.remove('mobile-open');
        if (this.mobileBackdrop) this.mobileBackdrop.classList.remove('active');
    }

    handleViewportChange() {
        if (!this.isMobileView()) {
            this.closeMobileSidebar();
        }
    }

    handleInputChange() {
        const text = this.messageInput.value;
        const length = text.length;
        
        this.charCount.textContent = `${length} / 4000`;
        this.sendBtn.disabled = length === 0 || this.isStreaming;
        
        // Auto-resize textarea
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 200) + 'px';

        // @Agent mention autocomplete (only in swarm-related context)
        this.handleSwarmAutocomplete(text);
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
        let safeMode = this.supportedModes.has(mode) ? mode : 'auto';
        
        // Gate unavailable modes
        if (!this.isModeAvailable(safeMode)) {
            console.warn(`Mode '${safeMode}' is unavailable, falling back to 'auto'`);
            safeMode = 'auto';
        }
        
        this.currentMode = safeMode;
        this.modeButtons.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === safeMode);
        });

        if (this.modelSelector) {
            this.modelSelector.style.display = safeMode === 'auto' ? 'none' : 'flex';
        }
        
        // Save to localStorage
        localStorage.setItem('edison_last_mode', safeMode);
        
        // Apply readiness UI updates
        this.applyReadinessToUI();
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

        // Connect agent live view to track backend activity
        if (window.edisonAgentLive) {
            window.edisonAgentLive.connect(this.sessionId || 'default');
        }
        this.sendBtn.style.display = 'none';
        this.stopBtn.style.display = 'flex';
        
        try {
            this.isStreaming = true;
            this.abortController = new AbortController();
            
            // Check for @Agent direct message in active swarm session
            const atMatch = message.match(/^@(\w+)\s+([\s\S]+)/);
            if (atMatch && this.swarmSessionId) {
                const agentName = atMatch[1];
                const dmMessage = atMatch[2].trim();
                const result = await this.sendSwarmDirectMessage(agentName, dmMessage);
                if (result && result.ok) {
                    const r = result.response;
                    const replyContent = `**@${r.agent || agentName}** (${r.model || 'agent'}):\n\n${r.response || 'No response'}`;
                    this.updateMessage(assistantMessageEl, replyContent, 'swarm-dm');
                    assistantMessageEl.classList.add('swarm-agent');
                    this.saveMessageToChat(message, replyContent, 'swarm-dm');
                } else {
                    // Fall through to normal chat if DM fails (agent not found, session expired, etc.)
                    // The error notification was already shown by sendSwarmDirectMessage
                    this.updateMessage(assistantMessageEl, '⚠️ Could not reach that agent. The swarm session may have expired. Sending as a normal message...', 'error');
                    await this.callEdisonAPIStream(message, mode, assistantMessageEl);
                }
                return; // skip the rest of sendMessage
            }

            // Check if this is an image generation or regeneration request
            const isImageRequest = this.detectImageGenerationRequest(message);
            const isImageEditRequest = this.detectImageEditRequest(message);
            const isRegenerateRequest = this.detectImageRegenerationRequest(message);
            
            const allowClientImage = ['auto', 'instant', 'chat'].includes(mode);
            if (isImageEditRequest && allowClientImage) {
                await this.handleImageEdit(message, assistantMessageEl);
            } else if ((isImageRequest || isRegenerateRequest) && allowClientImage) {
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
                } else if (response.music_generation && response.music_generation.prompt) {
                    console.log('🎵 Backend triggered music generation via Coral');
                    this.updateMessage(assistantMessageEl, response.response || '🎵 Generating music...', 'music');
                    await this.handleMusicGeneration(response.music_generation.prompt, assistantMessageEl);
                } else {
                    // Update assistant message
                    this.updateMessage(assistantMessageEl, response.response, response.mode_used);
                    this.finalizeToolTimeline(assistantMessageEl, response);
                    
                    // Save to chat history
                    const toolSummary = this.buildToolSummary(assistantMessageEl);
                    this.saveMessageToChat(message, response.response, response.mode_used, toolSummary ? { toolSummary } : null);
                    
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
            // Keep agent live view connected for a bit to catch final events
            if (window.edisonAgentLive) {
                setTimeout(() => window.edisonAgentLive.disconnect(), 5000);
            }
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
                selected_model: this.selectedModel !== 'auto' ? this.selectedModel : undefined,
                swarm_session_id: this.swarmSessionId || undefined,
                assistant_profile_id: this.settings.assistantProfileId || undefined
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
                selected_model: this.selectedModel !== 'auto' ? this.selectedModel : undefined,
                swarm_session_id: this.swarmSessionId || undefined,
                assistant_profile_id: this.settings.assistantProfileId || undefined
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
        let streamCompleted = false;
        
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
                            } else if (currentEventType === 'browser_view' && data.type === 'browser_view') {
                                // Browser card from agent tool loop
                                if (window.injectBrowserCard) {
                                    window.injectBrowserCard(data);
                                }
                            } else if (currentEventType === 'status' && data.stage) {
                                this.updateStatus(assistantMessageEl, data);
                            } else if (data.t) {
                                // Token event
                                accumulatedResponse += data.t;
                                this.updateMessage(assistantMessageEl, accumulatedResponse, mode);
                            } else if (data.ok !== undefined) {
                                // Done event
                                streamCompleted = true;
                                this.currentRequestId = null;  // Clear request ID
                                if (data.ok) {
                                    // Handle music generation trigger from backend
                                    if (data.music_generation && data.music_generation.prompt) {
                                        console.log('🎵 Backend triggered music generation');
                                        this.updateMessage(assistantMessageEl, data.response || '🎵 Generating music...', data.mode_used || 'music');
                                        this.saveMessageToChat(message, data.response || '🎵 Generating music...', 'music');
                                        await this.handleMusicGeneration(data.music_generation.prompt, assistantMessageEl);
                                        return;
                                    }
                                    // Handle image generation trigger from backend
                                    if (data.image_generation && data.image_generation.prompt) {
                                        console.log('🎨 Backend triggered image generation via SSE');
                                        await this.handleImageGeneration(data.image_generation.prompt, assistantMessageEl);
                                        return;
                                    }
                                    // Success — normal text response
                                    const finalResponse = data.response || accumulatedResponse;
                                    this.updateMessage(assistantMessageEl, finalResponse, data.mode_used || mode);
                                    
                                    // Track swarm session for follow-up DMs and intervention
                                    if (data.swarm_session_id) {
                                        this.swarmSessionId = data.swarm_session_id;
                                        console.log(`🐝 Swarm session active: ${this.swarmSessionId}`);
                                    }
                                    
                                    // Display swarm agent conversation if not already inserted
                                    if (!swarmInserted && data.swarm_agents && data.swarm_agents.length > 0) {
                                        this.insertSwarmConversation(assistantMessageEl, data.swarm_agents);
                                        swarmInserted = true;
                                    }

                                    // Display generated files if available
                                    if (data.files && data.files.length > 0) {
                                        this.displayGeneratedFiles(assistantMessageEl, data.files);
                                    }
                                    if (data.artifact && (data.artifact.code || data.artifact.html || data.artifact.content)) {
                                        this.showArtifactPanel(data.artifact);
                                    }
                                    this.finalizeToolTimeline(assistantMessageEl, data);
                                    this.clearStatus(assistantMessageEl);
                                    
                                    // Display trust signals (search, memory, browser, etc.)
                                    if (data.trust_signals && data.trust_signals.length > 0) {
                                        this.renderTrustSignals(assistantMessageEl, data.trust_signals);
                                    }
                                    
                                    assistantMessageEl.classList.remove('streaming');
                                    
                                    // Save to chat history
                                    const toolSummary = this.buildToolSummary(assistantMessageEl);
                                    const metadata = toolSummary ? { toolSummary } : null;
                                    this.saveMessageToChat(message, finalResponse, data.mode_used || mode, metadata);
                                    
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

            // Fallback: stream ended without explicit done event
            if (!streamCompleted) {
                this.currentRequestId = null;
                this.clearStatus(assistantMessageEl);
                assistantMessageEl.classList.remove('streaming');

                if (accumulatedResponse && accumulatedResponse.trim().length > 0) {
                    this.updateMessage(assistantMessageEl, accumulatedResponse, mode);
                    this.saveMessageToChat(message, accumulatedResponse, mode);
                } else {
                    this.updateMessage(assistantMessageEl, '⚠️ Stream ended before completion.', 'stopped');
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
                <button class="action-btn branch-btn" title="Branch from here">
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                        <path d="M10 2.5a2.5 2.5 0 1 1 1 1.997V6A2 2 0 0 1 9 8H6a1 1 0 0 0-1 1v2.503A2.5 2.5 0 1 1 4 11.5V9a2 2 0 0 1 2-2h3a1 1 0 0 0 1-1V4.497A2.5 2.5 0 0 1 10 2.5z"/>
                    </svg>
                </button>
                <button class="action-btn edit-btn" title="Edit and resubmit">
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                        <path d="M12.146.146a.5.5 0 0 1 .708 0l3 3a.5.5 0 0 1 0 .708l-10 10a.5.5 0 0 1-.168.11l-5 2a.5.5 0 0 1-.65-.65l2-5a.5.5 0 0 1 .11-.168l10-10zM11.207 2.5 13.5 4.793 14.793 3.5 12.5 1.207 11.207 2.5zm1.586 3L10.5 3.207 4 9.707V10h.5a.5.5 0 0 1 .5.5v.5h.5a.5.5 0 0 1 .5.5v.5h.293l6.5-6.5zm-9.761 5.175-.106.106-1.528 3.821 3.821-1.528.106-.106A.5.5 0 0 1 5 12.5V12h-.5a.5.5 0 0 1-.5-.5V11h-.5a.5.5 0 0 1-.468-.325z"/>
                    </svg>
                </button>
            </div>
        ` : `
            <div class="message-actions">
                <button class="action-btn branch-btn" title="Branch from here">
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                        <path d="M10 2.5a2.5 2.5 0 1 1 1 1.997V6A2 2 0 0 1 9 8H6a1 1 0 0 0-1 1v2.503A2.5 2.5 0 1 1 4 11.5V9a2 2 0 0 1 2-2h3a1 1 0 0 0 1-1V4.497A2.5 2.5 0 0 1 10 2.5z"/>
                    </svg>
                </button>
                <button class="action-btn copy-btn" title="Copy to clipboard">
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                        <path d="M4 1.5H3a2 2 0 0 0-2 2V14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V3.5a2 2 0 0 0-2-2h-1v1h1a1 1 0 0 1 1 1V14a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V3.5a1 1 0 0 1 1-1h1v-1z"/>
                        <path d="M9.5 1a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5v-1a.5.5 0 0 1 .5-.5h3zm-3-1A1.5 1.5 0 0 0 5 1.5v1A1.5 1.5 0 0 0 6.5 4h3A1.5 1.5 0 0 0 11 2.5v-1A1.5 1.5 0 0 0 9.5 0h-3z"/>
                    </svg>
                </button>
                <button class="action-btn speak-btn" title="Play audio">
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                        <path d="M3 6h2l3-3v10l-3-3H3z"/>
                        <path d="M10.5 5.5a3.5 3.5 0 0 1 0 5" fill="none" stroke="currentColor" stroke-width="1.2"/>
                        <path d="M12 3.5a5.5 5.5 0 0 1 0 9" fill="none" stroke="currentColor" stroke-width="1.2"/>
                    </svg>
                </button>
                <button class="action-btn regenerate-btn" title="Regenerate response">
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                        <path fill-rule="evenodd" d="M8 3a5 5 0 1 0 4.546 2.914.5.5 0 0 1 .908-.417A6 6 0 1 1 8 2v1z"/>
                        <path d="M8 4.466V.534a.25.25 0 0 1 .41-.192l2.36 1.966c.12.1.12.284 0 .384L8.41 4.658A.25.25 0 0 1 8 4.466z"/>
                    </svg>
                </button>
                <button class="action-btn like-btn" title="Like response">👍</button>
                <button class="action-btn dislike-btn" title="Dislike response">👎</button>
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

        this.decorateMessageContent(messageEl, content, isHtml);
        
        // Attach event listeners to action buttons
        const branchBtn = messageEl.querySelector('.branch-btn');
        if (branchBtn) {
            branchBtn.addEventListener('click', () => this.branchConversationFromMessage(messageEl));
        }
        if (role === 'user') {
            const editBtn = messageEl.querySelector('.edit-btn');
            editBtn.addEventListener('click', () => this.editMessage(messageEl, content));
        } else {
            const copyBtn = messageEl.querySelector('.copy-btn');
            const speakBtn = messageEl.querySelector('.speak-btn');
            const regenerateBtn = messageEl.querySelector('.regenerate-btn');
            const likeBtn = messageEl.querySelector('.like-btn');
            const dislikeBtn = messageEl.querySelector('.dislike-btn');
            
            copyBtn.addEventListener('click', () => this.copyToClipboard(content));
            if (speakBtn) {
                speakBtn.addEventListener('click', () => this.playTtsForMessage(messageEl));
            }
            regenerateBtn.addEventListener('click', () => this.regenerateResponse(messageEl));
            if (likeBtn) likeBtn.addEventListener('click', () => this.rateMessage(messageEl, true));
            if (dislikeBtn) dislikeBtn.addEventListener('click', () => this.rateMessage(messageEl, false));
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
            const agentName = agent.agent || 'Agent';
            const isBoss = agentName.toLowerCase() === 'boss';
            const extraClass = isBoss ? ' swarm-boss' : '';

            const agentMessageEl = document.createElement('div');
            agentMessageEl.className = `message assistant swarm-agent${extraClass}`;
            agentMessageEl.dataset.agentName = agentName;

            agentMessageEl.innerHTML = `
                <div class="message-header">
                    <div class="message-avatar">${agent.icon || '🐝'}</div>
                    <span class="message-role">${agentName}</span>
                    <span class="message-mode">${(agent.model || 'Unknown Model').toUpperCase()}</span>
                    ${isBoss ? '<span class="swarm-boss-badge">BOSS</span>' : ''}
                </div>
                <div class="message-content">${this.formatMessage(agent.response || '')}</div>
                <div class="message-actions">
                    <button class="action-btn copy-btn" title="Copy to clipboard">
                        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                            <path d="M4 1.5H3a2 2 0 0 0-2 2V14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V3.5a2 2 0 0 0-2-2h-1v1h1a1 1 0 0 1 1 1V14a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V3.5a1 1 0 0 1 1-1h1v-1z"/>
                            <path d="M9.5 1a.5.5 0 0 1 .5.5v1a.5.5 0 0 1-.5.5h-3a.5.5 0 0 1-.5-.5v-1a.5.5 0 0 1 .5-.5h3zm-3-1A1.5 1.5 0 0 0 5 1.5v1A1.5 1.5 0 0 0 6.5 4h3A1.5 1.5 0 0 0 11 2.5v-1A1.5 1.5 0 0 0 9.5 0h-3z"/>
                        </svg>
                    </button>
                    <button class="action-btn reply-agent-btn" title="Reply to ${agentName}" data-agent="${agentName}">
                        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                            <path d="M6.598 5.013a.144.144 0 0 1 .202.134V6.3a.5.5 0 0 0 .5.5c.667 0 2.013.005 3.3.822.984.624 1.99 1.76 2.595 3.876-1.02-.983-2.185-1.516-3.205-1.799a8.74 8.74 0 0 0-1.921-.306 7.404 7.404 0 0 0-.798.008h-.013l-.005.001h-.001L7.3 9.9H7.2a.5.5 0 0 0-.498.45v1.17a.147.147 0 0 1-.202.134L1.904 8.045a.16.16 0 0 1 0-.281l4.694-2.75Z"/>
                        </svg>
                    </button>
                </div>
            `;

            const copyBtn = agentMessageEl.querySelector('.copy-btn');
            if (copyBtn) {
                copyBtn.addEventListener('click', () => this.copyToClipboard(agent.response || ''));
            }
            
            const replyBtn = agentMessageEl.querySelector('.reply-agent-btn');
            if (replyBtn) {
                replyBtn.addEventListener('click', () => {
                    this.messageInput.value = `@${agentName} `;
                    this.messageInput.focus();
                    this.handleInputChange();
                });
            }

            fragment.appendChild(agentMessageEl);
        });

        // Add swarm interaction toolbar after agent messages
        if (this.swarmSessionId) {
            const toolbar = document.createElement('div');
            toolbar.className = 'swarm-toolbar';
            toolbar.innerHTML = `
                <div class="swarm-toolbar-label">🐝 Swarm Team — Session active</div>
                <div class="swarm-toolbar-actions">
                    <button class="swarm-feedback-btn" title="Give feedback to the whole team">
                        <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
                            <path d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16z"/>
                            <path d="M4.285 9.567a.5.5 0 0 1 .683.183A3.498 3.498 0 0 0 8 11.5a3.498 3.498 0 0 0 3.032-1.75.5.5 0 1 1 .866.5A4.498 4.498 0 0 1 8 12.5a4.498 4.498 0 0 1-3.898-2.25.5.5 0 0 1 .183-.683zM7 6.5C7 7.328 6.552 8 6 8s-1-.672-1-1.5S5.448 5 6 5s1 .672 1 1.5zm4 0c0 .828-.448 1.5-1 1.5s-1-.672-1-1.5S9.448 5 10 5s1 .672 1 1.5z"/>
                        </svg>
                        Give Feedback
                    </button>
                    <span class="swarm-session-id" title="Session ID">${this.swarmSessionId.slice(0, 8)}…</span>
                </div>
            `;
            
            toolbar.querySelector('.swarm-feedback-btn').addEventListener('click', () => {
                this.showSwarmFeedbackInput(assistantMessageEl);
            });
            
            fragment.appendChild(toolbar);
        }

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

        this.appendToolTimelineEvent(assistantMessageEl, {
            kind: data.kind || 'status',
            stage: data.stage,
            detail: data.detail || '',
            progress,
        });
    }

    clearStatus(assistantMessageEl) {
        const statusEl = assistantMessageEl.querySelector('.message-status');
        if (statusEl) statusEl.remove();
    }

    renderTrustSignals(assistantMessageEl, signals) {
        if (!assistantMessageEl || !signals || signals.length === 0) return;
        const contentEl = assistantMessageEl.querySelector('.message-content');
        if (!contentEl) return;

        // Remove any existing trust signal bar
        const existing = contentEl.querySelector('.trust-signals');
        if (existing) existing.remove();

        const bar = document.createElement('div');
        bar.className = 'trust-signals';

        const iconMap = {
            search: '🔍', memory: '🧠', browser: '🌐', artifact: '📎',
            code: '⚙️', uncertain: '⚠️',
        };

        signals.forEach(sig => {
            const badge = document.createElement('span');
            badge.className = `trust-badge trust-${sig.type || 'info'}`;
            badge.title = sig.detail || sig.label || '';
            badge.textContent = `${iconMap[sig.type] || 'ℹ️'} ${sig.label || sig.type}`;
            bar.appendChild(badge);
        });

        contentEl.appendChild(bar);
    }

    ensureToolTimeline(assistantMessageEl) {
        const contentEl = assistantMessageEl.querySelector('.message-content');
        if (!contentEl) return null;

        let timeline = contentEl.querySelector('.tool-timeline');
        if (timeline) return timeline;

        timeline = document.createElement('details');
        timeline.className = 'tool-timeline';
        timeline.open = true;
        timeline.innerHTML = `
            <summary>
                <span class="tool-timeline-title">Tool Activity</span>
                <span class="tool-timeline-summary">Waiting for steps</span>
            </summary>
            <div class="tool-timeline-list"></div>
        `;
        contentEl.prepend(timeline);
        return timeline;
    }

    appendToolTimelineEvent(assistantMessageEl, event) {
        if (!assistantMessageEl || !event?.stage) return;

        const timeline = this.ensureToolTimeline(assistantMessageEl);
        if (!timeline) return;

        const list = timeline.querySelector('.tool-timeline-list');
        const summary = timeline.querySelector('.tool-timeline-summary');
        if (!list || !summary) return;

        const signature = JSON.stringify({
            kind: event.kind || '',
            stage: event.stage,
            detail: event.detail || ''
        });
        const exists = Array.from(list.querySelectorAll('.tool-step')).some(item => item.dataset.signature === signature);
        if (exists) return;

        const step = document.createElement('div');
        step.className = 'tool-step';
        step.dataset.signature = signature;
        step.innerHTML = `
            <div class="tool-step-title-row">
                <span class="tool-step-kind">${this.escapeHtml((event.kind || 'step').toUpperCase())}</span>
                <span class="tool-step-title">${this.escapeHtml(event.stage)}${typeof event.progress === 'number' && event.progress > 0 ? ` · ${event.progress}%` : ''}</span>
            </div>
            ${event.detail ? `<div class="tool-step-detail">${this.escapeHtml(event.detail)}</div>` : ''}
        `;
        list.appendChild(step);
        summary.textContent = `${event.stage}${event.detail ? ` • ${event.detail}` : ''}`;
    }

    finalizeToolTimeline(assistantMessageEl, payload = {}) {
        if (Array.isArray(payload.work_step_results)) {
            payload.work_step_results.forEach(step => {
                if (step?.title) {
                    this.appendToolTimelineEvent(assistantMessageEl, {
                        kind: step.kind || 'work',
                        stage: step.title,
                        detail: step.result || ''
                    });
                }
            });
        }

        if (Array.isArray(payload.swarm_agents) && payload.swarm_agents.length > 0) {
            this.appendToolTimelineEvent(assistantMessageEl, {
                kind: 'swarm',
                stage: 'Swarm collaboration',
                detail: `${payload.swarm_agents.length} agents contributed`
            });
        }

        if (Array.isArray(payload.search_results) && payload.search_results.length > 0) {
            this.appendToolTimelineEvent(assistantMessageEl, {
                kind: 'search',
                stage: 'Search results collected',
                detail: `${payload.search_results.length} result${payload.search_results.length === 1 ? '' : 's'}`
            });
        }

        if (payload.artifact) {
            this.appendToolTimelineEvent(assistantMessageEl, {
                kind: 'artifact',
                stage: 'Artifact prepared',
                detail: payload.artifact.title || payload.artifact.type || 'ready'
            });
        }

        if (Array.isArray(payload.files) && payload.files.length > 0) {
            this.appendToolTimelineEvent(assistantMessageEl, {
                kind: 'download',
                stage: 'Files exported',
                detail: `${payload.files.length} file${payload.files.length === 1 ? '' : 's'} saved`
            });
        }
    }

    buildToolSummary(assistantMessageEl) {
        const timeline = assistantMessageEl?.querySelector('.tool-timeline');
        if (!timeline) return null;

        const steps = Array.from(timeline.querySelectorAll('.tool-step')).map(step => ({
            kind: step.querySelector('.tool-step-kind')?.textContent?.trim()?.toLowerCase() || 'step',
            stage: step.querySelector('.tool-step-title')?.textContent?.trim() || '',
            detail: step.querySelector('.tool-step-detail')?.textContent?.trim() || ''
        })).filter(step => step.stage);

        return steps.length > 0 ? { steps } : null;
    }

    restoreToolTimeline(messageEl, toolSummary) {
        if (!toolSummary || !Array.isArray(toolSummary.steps)) return;
        toolSummary.steps.forEach(step => this.appendToolTimelineEvent(messageEl, step));
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
        messageEl.dataset.rawText = content || '';
        
        const contentEl = messageEl.querySelector('.message-content');
        const toolSummary = this.buildToolSummary(messageEl);
        contentEl.innerHTML = this.formatMessage(content);
        this.decorateMessageContent(messageEl, content, false);
        this.restoreToolTimeline(messageEl, toolSummary);
        
        if (mode && mode !== 'error' && mode !== 'stopped') {
            const modeEl = messageEl.querySelector('.message-mode');
            modeEl.textContent = mode.toUpperCase();
            modeEl.style.display = 'inline-block';
        }
        
        this.scrollToBottom();
    }

    async loadModelHotSwap() {
        const name = this.modelNameInput?.value?.trim();
        const path = this.modelPathInput?.value?.trim();
        const nCtx = parseInt(this.modelCtxInput?.value || '0', 10);
        const tensor = this.modelTensorInput?.value?.trim();
        if (!name || !path) {
            alert('Model name and path are required');
            return;
        }
        const payload = {
            name,
            path
        };
        if (nCtx) payload.n_ctx = nCtx;
        if (tensor) {
            payload.tensor_split = tensor.split(',').map(v => parseFloat(v.trim())).filter(v => !Number.isNaN(v));
        }
        const response = await fetch(`${this.settings.apiEndpoint}/models/load`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!response.ok) {
            const text = await response.text();
            alert(`Load failed: ${text}`);
            return;
        }
        this.showNotification('Model loaded');
    }

    async unloadModelHotSwap() {
        const name = this.modelNameInput?.value?.trim();
        if (!name) {
            alert('Model name is required');
            return;
        }
        const response = await fetch(`${this.settings.apiEndpoint}/models/unload`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });
        if (!response.ok) {
            const text = await response.text();
            alert(`Unload failed: ${text}`);
            return;
        }
        this.showNotification('Model unloaded');
    }

    async playTtsForMessage(messageEl) {
        const text = messageEl?.dataset?.rawText || '';
        if (!text) return;
        const endpoint = this.settings.voiceEndpoint || 'http://localhost:8809';
        const response = await fetch(`${endpoint}/text-to-voice`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        if (!response.ok) {
            alert('Voice playback failed');
            return;
        }
        const data = await response.json();
        const audioBytes = atob(data.audio || '');
        const array = new Uint8Array(audioBytes.length);
        for (let i = 0; i < audioBytes.length; i++) {
            array[i] = audioBytes.charCodeAt(i);
        }
        const blob = new Blob([array], { type: 'audio/mpeg' });
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audio.onended = () => URL.revokeObjectURL(url);
        audio.play();
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
                const language = (lang || 'text').toLowerCase();
                const trimmedCode = code.trim();
                const escapedCode = this.escapeHtml(trimmedCode);
                const encodedCode = encodeURIComponent(trimmedCode);
                return `
                    <div class="code-block-wrapper">
                        <div class="code-block-header">
                            <span class="code-lang">${language}</span>
                            <button class="copy-code-btn" data-code="${encodedCode}" data-language="${language}" title="Copy ${language} code">Copy</button>
                        </div>
                        <pre><code class="language-${language}">${escapedCode}</code></pre>
                    </div>
                `;
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

    decorateMessageContent(messageEl, rawContent, isHtml = false) {
        if (!messageEl || isHtml) return;

        const contentEl = messageEl.querySelector('.message-content');
        if (!contentEl) return;

        const copyCodeButtons = contentEl.querySelectorAll('.copy-code-btn');
        copyCodeButtons.forEach(btn => {
            btn.addEventListener('click', async () => {
                const code = decodeURIComponent(btn.dataset.code || '');
                try {
                    await navigator.clipboard.writeText(code);
                    const originalLabel = btn.textContent;
                    btn.textContent = 'Copied';
                    btn.classList.add('copied');
                    setTimeout(() => {
                        btn.textContent = originalLabel;
                        btn.classList.remove('copied');
                    }, 1500);
                } catch (error) {
                    console.error('Failed to copy code block:', error);
                    this.showNotification('Copy failed');
                }
            });
        });
    }

    showArtifactPanel(artifact) {
        if (!this.artifactsPanel || !artifact) return;
        const title = artifact.title || `${(artifact.type || 'artifact').toUpperCase()} Artifact`;
        const code = artifact.code || artifact.content || '';
        const type = artifact.type || 'txt';
        const previewHtml = this.getArtifactPreviewHtml(artifact, code, type);
        const previewSupported = Boolean(previewHtml);

        this.artifactTitle.textContent = title;
        this.artifactCode.textContent = code;
        if (this.artifactMeta) {
            const previewLabel = previewSupported ? 'live preview' : 'code only';
            this.artifactMeta.textContent = `${type.toUpperCase()} • ${previewLabel}`;
        }

        if (this.artifactPreviewFrame) {
            this.artifactPreviewFrame.srcdoc = previewHtml || '';
        }
        if (this.artifactPreviewEmpty) {
            this.artifactPreviewEmpty.style.display = previewSupported ? 'none' : 'flex';
        }

        this.currentArtifact = {
            title,
            code,
            type,
            previewHtml,
            previewSupported
        };

        this.setArtifactView(previewSupported ? 'preview' : 'code');

        this.artifactsPanel.style.display = 'flex';
        const container = document.querySelector('.chat-container');
        if (container) container.classList.add('with-artifacts');
    }

    getArtifactPreviewHtml(artifact, code, type) {
        if (artifact.html && typeof artifact.html === 'string') {
            return artifact.html;
        }

        if (!code || typeof code !== 'string') {
            return null;
        }

        if (type === 'html' || type === 'javascript' || type === 'mermaid') {
            return code;
        }

        if (type === 'svg') {
            return `<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { margin: 0; padding: 24px; display: flex; align-items: center; justify-content: center; min-height: 100vh; background: #f8fafc; }
        svg { max-width: 100%; height: auto; }
    </style>
</head>
<body>
${code}
</body>
</html>`;
        }

        return null;
    }

    setArtifactView(view) {
        const showPreview = view === 'preview' && this.currentArtifact?.previewSupported;

        if (this.artifactPreviewPane) {
            this.artifactPreviewPane.style.display = showPreview ? 'flex' : 'none';
        }
        if (this.artifactCodePane) {
            this.artifactCodePane.style.display = showPreview ? 'none' : 'block';
        }
        if (this.artifactPreviewTab) {
            this.artifactPreviewTab.classList.toggle('active', showPreview);
            this.artifactPreviewTab.disabled = !this.currentArtifact?.previewSupported;
        }
        if (this.artifactCodeTab) {
            this.artifactCodeTab.classList.toggle('active', !showPreview);
        }
    }

    hideArtifactPanel() {
        if (!this.artifactsPanel) return;
        this.artifactsPanel.style.display = 'none';
        if (this.artifactPreviewFrame) {
            this.artifactPreviewFrame.srcdoc = '';
        }
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
            javascript: 'html',
            code: 'txt'
        };
        const ext = extMap[this.currentArtifact.type] || 'txt';
        const mimeMap = {
            html: 'text/html;charset=utf-8',
            react: 'text/jsx;charset=utf-8',
            svg: 'image/svg+xml;charset=utf-8',
            mermaid: 'text/html;charset=utf-8',
            javascript: 'text/html;charset=utf-8'
        };
        const blob = new Blob([this.currentArtifact.code], { type: mimeMap[this.currentArtifact.type] || 'text/plain;charset=utf-8' });
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
        // Delegate to the new EdisonVoiceAssistant overlay if available
        if (window.edisonVoice) {
            window.edisonVoice.toggle();
            return;
        }
        // Fallback: check secure context (Web Speech API requires HTTPS or localhost)
        const isSecure = location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1';
        if (!isSecure) {
            alert('Voice input requires HTTPS or localhost. You are on HTTP over LAN — use HTTPS or access via localhost.');
            return;
        }
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            alert('Voice input not supported in this browser.');
            return;
        }
        try {
            const recognition = new SpeechRecognition();
            recognition.lang = 'en-US';
            recognition.interimResults = false;
            recognition.onresult = (event) => {
                const transcript = event.results?.[0]?.[0]?.transcript || '';
                this.messageInput.value = transcript.trim();
                this.handleInputChange();
            };
            recognition.onerror = (e) => {
                console.warn('Voice recognition error:', e.error);
                if (e.error === 'network') {
                    alert('Voice input network error. Web Speech API requires internet access to reach speech recognition servers. Check your connection or try a different browser.');
                } else if (e.error === 'not-allowed') {
                    alert('Microphone access denied. Please allow microphone access in your browser settings.');
                } else {
                    alert('Voice input failed: ' + (e.error || 'unknown error'));
                }
            };
            recognition.start();
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

    resetImageConversationState() {
        this.lastImagePrompt = null;
        this.lastGeneratedImage = null;
    }

    normalizeImageContext(context) {
        if (!context || typeof context !== 'object') {
            return null;
        }

        const sourcePath = context.sourcePath || context.source_path;
        const imageUrl = context.imageUrl || context.image_url || context.galleryImageUrl || context.gallery_image_url;
        const prompt = context.prompt || context.userPrompt || context.user_prompt || '';

        if (!sourcePath && !imageUrl) {
            return null;
        }

        return {
            prompt,
            userPrompt: context.userPrompt || context.user_prompt || prompt,
            sourcePath,
            imageUrl,
            galleryImageId: context.galleryImageId || context.gallery_image_id || null,
            galleryFilename: context.galleryFilename || context.gallery_filename || null,
            origin: context.origin || 'generated',
        };
    }

    applyImageContext(context) {
        const normalized = this.normalizeImageContext(context);
        if (!normalized) {
            return;
        }

        this.lastGeneratedImage = normalized;
        this.lastImagePrompt = normalized.prompt || normalized.userPrompt || this.lastImagePrompt;
    }

    buildImageMessageHtml(imageUrl, prompt, sourceLabel = 'Prompt') {
        const safePrompt = (prompt || '').replace(/'/g, "\\'");
        return `
            <p>✅ Image ready!</p>
            <div class="generated-image">
                <img src="${imageUrl}" alt="Generated image" style="max-width: 100%; border-radius: 8px; margin-top: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
            </div>
            <div style="margin-top: 10px; padding: 10px 12px; border-radius: 10px; background: rgba(102, 126, 234, 0.08); color: #5b6581; font-size: 13px; line-height: 1.45;">
                <strong>Edit this image:</strong> reply with changes like “make the background red”, “remove the text”, or “try a flatter logo style”.
            </div>
            <div style="margin-top: 10px; display: flex; align-items: center; gap: 12px;">
                <button onclick="downloadImage('${imageUrl}', 'EDISON_${Date.now()}.png')" style="padding: 10px 14px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 20px; box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3); transition: transform 0.2s; display: flex; align-items: center; justify-content: center; line-height: 1;" onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'" title="Download Image">
                    📥
                </button>
                <button onclick="regenerateImage('${safePrompt}')" style="padding: 10px 14px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 18px; box-shadow: 0 2px 8px rgba(240, 147, 251, 0.3); transition: transform 0.2s; display: flex; align-items: center; justify-content: center; line-height: 1;" onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'" title="Regenerate Image">
                    🔄
                </button>
                <span style="color: #888; font-size: 14px;"><strong>${sourceLabel}:</strong> ${prompt}</span>
            </div>
        `;
    }

    detectImageEditRequest(message) {
        if (!this.lastGeneratedImage?.sourcePath) return false;

        const lowerMessage = message.toLowerCase().trim();
        if (!lowerMessage) return false;
        if (this.detectImageGenerationRequest(message)) return false;

        const regenerateOnlyKeywords = [
            'try again', 'regenerate', 'make another', 'do it again', 'retry',
            'redo', 'one more', 'again', 'remake', 'recreate', 'generate another',
            'make a new one', 'new one', 'another one'
        ];
        if (regenerateOnlyKeywords.some(keyword => lowerMessage === keyword || lowerMessage.includes(keyword))) {
            return false;
        }

        const directEditRequests = [
            'make the background', 'change the background', 'remove the background', 'change the color',
            'change the colours', 'change the colors', 'remove the text', 'change the text', 'update the logo',
            'make it look', 'make it feel', 'clean this up', 'fix the text', 'adjust the font'
        ];
        if (directEditRequests.some(keyword => lowerMessage.includes(keyword))) {
            return true;
        }

        const conversationalStarts = [
            /^can you\s+/, /^could you\s+/, /^please\s+/, /^let'?s\s+/, /^i want\s+/, /^i need\s+/, /^we need\s+/,
            /^it should\s+/, /^it needs\s+/, /^this needs\s+/, /^maybe\s+/, /^what about\s+/
        ];
        const editTargets = [
            'it', 'this', 'image', 'logo', 'background', 'text', 'font', 'layout', 'icon', 'design', 'mark', 'wordmark'
        ];
        const editActions = [
            'change', 'edit', 'modify', 'remove', 'add', 'replace', 'swap', 'update', 'fix', 'clean', 'refine', 'simplify',
            'flatten', 'brighten', 'darken', 'soften', 'sharpen', 'move', 'resize', 'make', 'turn', 'give', 'use'
        ];
        const descriptiveEditWords = [
            'bolder', 'simpler', 'cleaner', 'flatter', 'modern', 'minimal', 'warmer', 'cooler', 'brighter', 'darker',
            'bigger', 'smaller', 'thicker', 'thinner', 'centered', 'transparent', 'monochrome'
        ];

        const isConversationalEdit = conversationalStarts.some(pattern => pattern.test(lowerMessage)) &&
            editActions.some(action => lowerMessage.includes(action)) &&
            editTargets.some(target => lowerMessage.includes(target));
        if (isConversationalEdit) {
            return true;
        }

        const isShortDescriptorEdit = lowerMessage.length <= 120 &&
            editTargets.some(target => lowerMessage.includes(target)) &&
            descriptiveEditWords.some(word => lowerMessage.includes(word));
        if (isShortDescriptorEdit) {
            return true;
        }

        const editPatterns = [
            /^with\b/, /^without\b/, /^make it\b/, /^make the\b/, /^make this\b/, /^turn\b/, /^change\b/, /^edit\b/, /^modify\b/,
            /^remove\b/, /^add\b/, /^replace\b/, /^swap\b/, /^update\b/, /^fix\b/, /^clean up\b/, /^give it\b/, /^set the\b/
        ];
        const editKeywords = [
            'change the', 'change it', 'edit the', 'edit it', 'modify the', 'modify it', 'remove the', 'remove this',
            'add a', 'add an', 'replace the', 'replace this', 'background', 'logo', 'text', 'font', 'color', 'colour',
            'style', 'layout', 'bigger', 'smaller', 'more modern', 'cleaner', 'transparent', 'red background'
        ];

        return editPatterns.some(pattern => pattern.test(lowerMessage)) ||
            editKeywords.some(keyword => lowerMessage.includes(keyword));
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
            // Check ComfyUI readiness before attempting generation
            const comfyuiComponent = this.readiness.components?.find(c => c.key === 'comfyui');
            if (comfyuiComponent?.state !== 'green') {
                const errorMsg = this.formatSystemMessage(comfyuiComponent || {
                    title: 'ComfyUI',
                    key: 'comfyui',
                    state: 'red',
                    likely_cause: 'ComfyUI is not available',
                    next_step: 'Check ComfyUI connection using Setup Check'
                });
                this.updateMessage(assistantMessageEl, `⚠️ Image generation unavailable:\n\n${errorMsg}`, 'error');
                this.showNotification('ComfyUI not available - check Setup Check for details');
                return;
            }
            
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
                    auto_optimize: true,
                    comfyui_url: this.settings.comfyuiEndpoint
                })
            });
            
            if (!response.ok) {
                const errorPayload = await response.json().catch(() => ({}));
                const detail = errorPayload.detail || {};
                let message = detail.message || errorPayload.error || `Image generation failed: ${response.statusText}`;
                if (detail.suggested_profile) {
                    const suggestion = detail.suggested_profile;
                    message += ` Suggested fallback: ${suggestion.width}x${suggestion.height} @ ${suggestion.steps} steps.`;
                }
                throw new Error(message);
            }
            
            const result = await response.json();
            const promptId = result.prompt_id;
            const effective = result.effective_parameters || {};
            
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
                    ${effective.optimized ? `<p style="margin-top: 8px; color: #f59e0b;"><strong>Low-memory mode:</strong> Rendering at ${effective.width}x${effective.height}, ${effective.steps} steps (${effective.profile || 'optimized'} profile).</p>` : ''}
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
                    const fullImageUrl = `${this.settings.apiEndpoint}${status.image_url}`;
                    const imageContext = {
                        prompt: imagePrompt,
                        userPrompt: message,
                        sourcePath: status.source_path || null,
                        imageUrl: fullImageUrl,
                        galleryImageId: status.gallery_image_id || null,
                        galleryFilename: status.gallery_filename || null,
                        origin: 'generated'
                    };
                    const imageHtml = this.buildImageMessageHtml(fullImageUrl, imagePrompt, 'Prompt');

                    this.applyImageContext(imageContext);
                    this.updateMessage(assistantMessageEl, imageHtml, 'image');
                    
                    // Save to chat history with HTML so it persists after refresh
                    this.saveMessageToChat(message, imageHtml, 'image', { imageContext });
                    
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

    async handleImageEdit(message, assistantMessageEl) {
        if (!this.lastGeneratedImage?.sourcePath) {
            this.updateMessage(assistantMessageEl, '⚠️ There is no editable image in the current chat yet. Generate an image first, then ask for changes.', 'error');
            return;
        }

        try {
            this.updateMessage(assistantMessageEl, '🎨 Editing the last image, please wait...', 'image');

            const response = await fetch(`${this.settings.apiEndpoint}/images/edit`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    source_path: this.lastGeneratedImage.sourcePath,
                    prompt: message,
                    parameters: {
                        auto_mask: true,
                        auto_refine: true,
                        comfyui_url: this.settings.comfyuiEndpoint,
                    }
                })
            });

            if (!response.ok) {
                const errorPayload = await response.json().catch(() => ({}));
                throw new Error(errorPayload.detail || `Image edit failed: ${response.statusText}`);
            }

            const result = await response.json();
            const edit = result.edit || {};
            const imageUrl = edit.image_url ? `${this.settings.apiEndpoint}${edit.image_url}` : this.lastGeneratedImage.imageUrl;
            const imageContext = {
                prompt: message,
                userPrompt: message,
                sourcePath: edit.output_path || this.lastGeneratedImage.sourcePath,
                imageUrl,
                galleryImageId: edit.gallery_image_id || null,
                galleryFilename: edit.gallery_filename || null,
                origin: 'edited'
            };
            const imageHtml = this.buildImageMessageHtml(imageUrl, message, 'Edit');

            this.applyImageContext(imageContext);
            this.updateMessage(assistantMessageEl, imageHtml, 'image');
            this.saveMessageToChat(message, imageHtml, 'image', { imageContext });
        } catch (error) {
            console.error('Image edit error:', error);
            this.updateMessage(
                assistantMessageEl,
                `⚠️ Error editing image: ${error.message}.`,
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

    async handleMusicGeneration(prompt, assistantMessageEl) {
        try {
            this.updateMessage(assistantMessageEl, '🎵 Generating music, please wait...', 'music');

            const response = await fetch(`${this.settings.apiEndpoint}/generate-music`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt, duration: 15 })
            });

            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.detail || `Music generation failed: ${response.statusText}`);
            }

            const result = await response.json();

            // If instant result (e.g. generated synchronously)
            if (result.audio_url || result.url) {
                const audioUrl = `${this.settings.apiEndpoint}${result.audio_url || result.url}`;
                const html = `
                    <p>✅ Music generated!</p>
                    <audio controls style="width:100%; margin-top:10px;">
                        <source src="${audioUrl}" type="audio/wav">
                    </audio>
                    <p style="color:#888; margin-top:8px;"><strong>Prompt:</strong> ${prompt}</p>`;
                this.updateMessage(assistantMessageEl, html, 'music');
                this.saveMessageToChat(prompt, html, 'music');
                return;
            }

            // Poll if async
            const promptId = result.prompt_id || result.job_id;
            if (promptId) {
                let attempts = 0;
                const maxAttempts = 180;
                while (attempts < maxAttempts) {
                    await new Promise(r => setTimeout(r, 2000));
                    const statusResp = await fetch(`${this.settings.apiEndpoint}/music-status/${promptId}`).catch(() => null);
                    if (!statusResp || !statusResp.ok) { attempts++; continue; }
                    const status = await statusResp.json();
                    if (status.status === 'complete' && status.audio_url) {
                        const audioUrl = `${this.settings.apiEndpoint}${status.audio_url}`;
                        const html = `
                            <p>✅ Music generated!</p>
                            <audio controls style="width:100%; margin-top:10px;">
                                <source src="${audioUrl}" type="audio/wav">
                            </audio>
                            <p style="color:#888; margin-top:8px;"><strong>Prompt:</strong> ${prompt}</p>`;
                        this.updateMessage(assistantMessageEl, html, 'music');
                        this.saveMessageToChat(prompt, html, 'music');
                        return;
                    } else if (status.status === 'error') {
                        throw new Error(status.detail || 'Music generation failed');
                    }
                    attempts++;
                }
                throw new Error('Music generation timed out');
            }

            // If no prompt_id and no audio_url, show whatever we got
            this.updateMessage(assistantMessageEl, result.message || '🎵 Music generation request submitted.', 'music');
        } catch (error) {
            console.error('Music generation error:', error);
            this.updateMessage(assistantMessageEl, `⚠️ Error generating music: ${error.message}`, 'error');
        }
    }

    createNewChat() {
        this.closeMobileSidebar();
        this.resetImageConversationState();
        const chatId = this.generateId('chat');
        const chat = {
            id: chatId,
            title: 'New Chat',
            messages: [],
            createdAt: new Date().toISOString(),
            timestamp: Date.now()
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

        this.resetImageConversationState();
        
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
        this.closeMobileSidebar();
        this.currentChatId = chatId;
        this.resetImageConversationState();
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
        this.resetImageConversationState();
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
            const normalizedMessage = this.normalizeStoredMessage(msg);
            // Check if this is an image message with HTML content
            const isImageMessage = normalizedMessage.mode === 'image' && normalizedMessage.content.includes('<img src=');
            const messageEl = this.addMessage(normalizedMessage.role, normalizedMessage.content, false, isImageMessage);
            messageEl.dataset.messageId = normalizedMessage.id;
            if (normalizedMessage.parentId) {
                messageEl.dataset.parentId = normalizedMessage.parentId;
            }
            if (normalizedMessage.role === 'assistant' && normalizedMessage.mode === 'image' && normalizedMessage.metadata?.imageContext) {
                this.applyImageContext(normalizedMessage.metadata.imageContext);
                if (this.lastGeneratedImage?.sourcePath) {
                    messageEl.dataset.imageSourcePath = this.lastGeneratedImage.sourcePath;
                }
            }
            if (normalizedMessage.role === 'assistant' && normalizedMessage.metadata?.toolSummary) {
                this.restoreToolTimeline(messageEl, normalizedMessage.metadata.toolSummary);
            }
            if (normalizedMessage.mode && normalizedMessage.role === 'assistant') {
                const modeEl = messageEl.querySelector('.message-mode');
                modeEl.textContent = normalizedMessage.mode.toUpperCase();
                modeEl.style.display = 'inline-block';
            }
        });
    }

    saveMessageToChat(userMessage, assistantMessage, mode, metadata = null) {
        const chat = this.chats.find(c => c.id === this.currentChatId);
        if (!chat) return;

        const createdAt = new Date().toISOString();
        const userEntry = this.createStoredMessage({
            role: 'user',
            content: userMessage,
            createdAt
        });
        const assistantEntry = this.createStoredMessage({
            role: 'assistant',
            content: assistantMessage,
            mode,
            metadata: metadata || undefined,
            parentId: userEntry.id,
            createdAt
        });

        chat.messages.push(userEntry, assistantEntry);
        chat.timestamp = Date.now();
        
        this.saveChats();
        this.renderChatHistory();
    }

    createStoredMessage(message) {
        return {
            id: message.id || this.generateId('msg'),
            role: message.role,
            content: message.content,
            mode: message.mode,
            metadata: message.metadata,
            parentId: message.parentId || null,
            createdAt: message.createdAt || new Date().toISOString()
        };
    }

    normalizeStoredMessage(message) {
        if (!message || typeof message !== 'object') {
            return this.createStoredMessage({ role: 'assistant', content: '' });
        }
        if (message.id) {
            return message;
        }
        return this.createStoredMessage(message);
    }

    branchConversationFromMessage(messageEl) {
        if (!messageEl || !this.currentChatId) return;

        const chat = this.chats.find(c => c.id === this.currentChatId);
        if (!chat?.messages?.length) return;

        const threadMessages = Array.from(this.messagesContainer.querySelectorAll('.message.user, .message.assistant:not(.swarm-agent)'));
        const branchIndex = threadMessages.indexOf(messageEl);
        if (branchIndex < 0) return;

        const branchMessages = chat.messages.slice(0, branchIndex + 1).map(msg => {
            const normalized = this.normalizeStoredMessage(msg);
            return {
                ...normalized,
                id: this.generateId('msg')
            };
        });

        const branchChat = {
            id: this.generateId('chat'),
            title: `${chat.title} Branch`,
            messages: branchMessages,
            createdAt: new Date().toISOString(),
            timestamp: Date.now(),
            branchOf: chat.id
        };

        this.chats.unshift(branchChat);
        this.currentChatId = branchChat.id;
        this.saveChats();
        this.clearMessages();
        this.renderMessages(branchChat.messages);
        this.renderChatHistory();
        this.showNotification('Created branched conversation');
    }

    generateId(prefix) {
        return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
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
                <div class="chat-history-labels">
                    <div class="chat-history-text">${this.escapeHtml(chat.title)}</div>
                    ${chat.branchOf ? '<span class="chat-history-badge">Branch</span>' : ''}
                </div>
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

    async openSettings() {
        this.apiEndpointInput.value = this.settings.apiEndpoint;
        this.comfyuiEndpointInput.value = this.settings.comfyuiEndpoint;
        if (this.voiceEndpointInput) {
            this.voiceEndpointInput.value = this.settings.voiceEndpoint || '';
        }
        const safeDefaultMode = this.supportedModes.has(this.settings.defaultMode) ? this.settings.defaultMode : 'auto';
        this.defaultModeSelect.value = safeDefaultMode;
        
        // Populate user ID field
        if (this.userIdInput) {
            this.userIdInput.value = this.userId;
        }
        this.loadUsers();
        if (this.discordWebhookInput) this.discordWebhookInput.value = this.settings.discordWebhook || '';
        if (this.slackWebhookInput) this.slackWebhookInput.value = this.settings.slackWebhook || '';
        if (this.defaultPrinterIdInput) this.defaultPrinterIdInput.value = this.settings.defaultPrinterId || '';
        if (this.assistantProfileSelect) {
            await this.loadAssistantProfiles();
            this.assistantProfileSelect.value = this.settings.assistantProfileId || '';
        }
        if (this.sandboxAllowedHostsInput) this.sandboxAllowedHostsInput.value = (this.settings.sandboxAllowedHosts || []).join(', ');
        if (this.sandboxAllowAnyHostInput) this.sandboxAllowAnyHostInput.checked = !!this.settings.sandboxAllowAnyHost;
        this.loadRuntimeSettings();
        
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
            this.setActiveUser(newUserId, true);
            localStorage.setItem('edison_user_id_confirmed', '1');
            
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
                this.loadCurrentChat();
                alert('User ID imported successfully! Your chats have been synced.');
            });
        }
    }

    async loadUsers() {
        try {
            const response = await fetch(`${this.settings.apiEndpoint}/users`);
            if (!response.ok) throw new Error('Failed to load users');
            const data = await response.json();
            const users = data.users || [];
            const optionsHtml = users.map(u => {
                const selected = u.id === this.userId ? 'selected' : '';
                return `<option value="${u.id}" ${selected}>${u.name}</option>`;
            }).join('');
            // Populate both the settings modal and header dropdowns
            if (this.userSelect) this.userSelect.innerHTML = optionsHtml;
            if (this.chatUserSelect) this.chatUserSelect.innerHTML = optionsHtml;
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
                this.setActiveUser(this.getOrCreateUserId(), true);
                await this.syncChatsFromServer();
                this.loadCurrentChat();
            }
            await this.loadUsers();
        } catch (error) {
            alert(error.message);
        }
    }

    async switchUserFromSelect() {
        const userId = this.userSelect?.value;
        if (!userId || userId === this.userId) return;
        this.setActiveUser(userId, true);
        await this.syncChatsFromServer();
        this.loadCurrentChat();
        // Keep header dropdown in sync
        if (this.chatUserSelect) this.chatUserSelect.value = userId;
    }

    async switchUserFromChatHeader() {
        const userId = this.chatUserSelect?.value;
        if (!userId || userId === this.userId) return;
        this.setActiveUser(userId, true);
        await this.syncChatsFromServer();
        this.loadCurrentChat();
        // Keep settings dropdown in sync
        if (this.userSelect) this.userSelect.value = userId;
    }

    async addUserFromChatHeader() {
        const name = prompt('Enter new user name:');
        if (!name || !name.trim()) return;
        try {
            const response = await fetch(`${this.settings.apiEndpoint}/users`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name.trim() })
            });
            if (!response.ok) throw new Error('Failed to create user');
            await this.loadUsers();
        } catch (error) {
            alert(error.message);
        }
    }

    async deleteUserFromChatHeader() {
        const userId = this.chatUserSelect?.value;
        if (!userId) return;
        if (!confirm('Delete this user and all chats? This cannot be undone.')) return;
        try {
            const response = await fetch(`${this.settings.apiEndpoint}/users/${userId}`, {
                method: 'DELETE'
            });
            if (!response.ok) throw new Error('Failed to delete user');
            if (userId === this.userId) {
                this.setActiveUser(this.getOrCreateUserId(), true);
                await this.syncChatsFromServer();
                this.loadCurrentChat();
            }
            await this.loadUsers();
        } catch (error) {
            alert(error.message);
        }
    }

    async cleanupUsers() {
        if (!confirm('Remove all auto-generated users (User-user_*)? This cannot be undone.')) return;
        try {
            const response = await fetch(`${this.settings.apiEndpoint}/users/cleanup`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ keep_ids: [this.userId] })
            });
            if (!response.ok) throw new Error('Failed to cleanup users');
            const data = await response.json();
            const removed = data.removed ?? 0;
            if (removed > 0) {
                alert(`Removed ${removed} auto-generated user(s).`);
            } else {
                alert('No auto-generated users found to remove.');
            }
            await this.loadUsers();
        } catch (error) {
            alert(error.message);
        }
    }

    setActiveUser(userId, resetChats) {
        this.userId = userId;
        localStorage.setItem('edison_user_id_confirmed', '1');
        localStorage.setItem('edison_user_id', userId);
        if (this.userIdInput) this.userIdInput.value = userId;
        if (resetChats) {
            this.currentChatId = null;
            this.chats = this.loadChats({ sync: false });
            this.renderChatHistory();
            this.clearMessages();
        }
    }

    async saveSettings() {
        this.settings.apiEndpoint = this.apiEndpointInput.value.trim();
        this.settings.comfyuiEndpoint = this.comfyuiEndpointInput.value.trim();
        if (this.voiceEndpointInput) {
            this.settings.voiceEndpoint = this.voiceEndpointInput.value.trim();
        }
        if (this.discordWebhookInput) this.settings.discordWebhook = this.discordWebhookInput.value.trim();
        if (this.slackWebhookInput) this.settings.slackWebhook = this.slackWebhookInput.value.trim();
        if (this.defaultPrinterIdInput) this.settings.defaultPrinterId = this.defaultPrinterIdInput.value.trim();
        if (this.assistantProfileSelect) this.settings.assistantProfileId = this.assistantProfileSelect.value || '';
        if (this.sandboxAllowAnyHostInput) this.settings.sandboxAllowAnyHost = !!this.sandboxAllowAnyHostInput.checked;
        if (this.sandboxAllowedHostsInput) {
            this.settings.sandboxAllowedHosts = this.sandboxAllowedHostsInput.value
                .split(',')
                .map(v => v.trim())
                .filter(Boolean);
        }
        this.settings.defaultMode = this.defaultModeSelect.value;

        try {
            await fetch(`${this.settings.apiEndpoint}/sandbox/config`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sandbox_allow_any_host: !!this.settings.sandboxAllowAnyHost,
                    sandbox_allowed_hosts: this.settings.sandboxAllowedHosts || [],
                }),
            });
        } catch (e) {
            console.warn('Failed to update sandbox config:', e);
        }
        
        localStorage.setItem('edison_settings', JSON.stringify(this.settings));
        
        // Update current mode if needed
        this.setMode(this.settings.defaultMode);
        this.loadAvailableModels();
        
        this.closeSettings();
    }

    async loadRuntimeSettings() {
        try {
            const [sandboxRes, skillsRes, printersRes] = await Promise.all([
                fetch(`${this.settings.apiEndpoint}/sandbox/config`).catch(() => null),
                fetch(`${this.settings.apiEndpoint}/skills`).catch(() => null),
                fetch(`${this.settings.apiEndpoint}/printing/printers`).catch(() => null),
            ]);

            if (sandboxRes && sandboxRes.ok) {
                const sandbox = await sandboxRes.json();
                this.settings.sandboxAllowAnyHost = !!sandbox.sandbox_allow_any_host;
                this.settings.sandboxAllowedHosts = sandbox.sandbox_allowed_hosts || [];
                if (this.sandboxAllowAnyHostInput) this.sandboxAllowAnyHostInput.checked = this.settings.sandboxAllowAnyHost;
                if (this.sandboxAllowedHostsInput) this.sandboxAllowedHostsInput.value = this.settings.sandboxAllowedHosts.join(', ');
            }

            if (skillsRes && skillsRes.ok) {
                const data = await skillsRes.json();
                const names = (data.skills || []).map(s => s.name || s.module);
                if (this.installedSkillsInput) this.installedSkillsInput.value = names.join(', ');
            }

            if (printersRes && printersRes.ok) {
                const data = await printersRes.json();
                const first = (data.printers || [])[0];
                if (!this.settings.defaultPrinterId && first) this.settings.defaultPrinterId = first.id;
                if (this.defaultPrinterIdInput) this.defaultPrinterIdInput.value = this.settings.defaultPrinterId || '';
            }
        } catch (e) {
            console.warn('Failed to load runtime settings:', e);
        }
    }

    async rateMessage(messageEl, liked) {
        if (!messageEl) return;
        const content = messageEl.querySelector('.message-content')?.textContent || '';
        this.showNotification(liked ? 'Thanks for the feedback' : 'Feedback recorded');
        await this.sendWebhookFeedback({
            type: liked ? 'like' : 'dislike',
            content,
            mode: this.currentMode,
            chatId: this.currentChatId,
        });
    }

    async sendWebhookFeedback(payload) {
        const targets = [this.settings.discordWebhook, this.settings.slackWebhook].filter(Boolean);
        if (targets.length === 0) return;
        for (const url of targets) {
            try {
                await fetch(url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: `[EDISON] ${payload.type}: ${payload.content.slice(0, 300)}` }),
                });
            } catch (e) {
                console.warn('Webhook delivery failed:', e);
            }
        }
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

    loadChats({ sync = true } = {}) {
        // Load from localStorage first for immediate display
        const saved = localStorage.getItem(this.getChatsStorageKey());
        const localChats = saved ? JSON.parse(saved) : [];
        
        // Then sync with server in background
        if (sync && !this._needsUserBootstrap) {
            this.syncChatsFromServer();
        }
        
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
                
                // Merge server chats with local chats (server takes priority for same IDs)
                const merged = this.mergeChats(this.chats, serverChats);
                this.chats = merged;
                localStorage.setItem(this.getChatsStorageKey(), JSON.stringify(this.chats));
                this.renderChatHistory();
                console.log(`Synced ${serverChats.length} chats from server`);
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
        localStorage.setItem(this.getChatsStorageKey(), JSON.stringify(this.chats));
        
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

    // ==================== SWARM COLLABORATION UI ====================

    async loadSwarmAgentCatalog() {
        try {
            const resp = await fetch(`${this.settings.apiEndpoint}/swarm/agents`);
            if (resp.ok) {
                const data = await resp.json();
                this.swarmAgentCatalog = data.agents || [];
                console.log(`🐝 Loaded ${this.swarmAgentCatalog.length} swarm agents for @mention`);
            }
        } catch (e) {
            console.warn('Could not load swarm agent catalog:', e.message);
        }
    }

    handleSwarmAutocomplete(text) {
        // Detect @<partial> at end of text
        const atMatch = text.match(/@(\w*)$/);
        if (atMatch && this.swarmAgentCatalog.length > 0) {
            const partial = atMatch[1].toLowerCase();
            const matches = this.swarmAgentCatalog.filter(a =>
                a.name.toLowerCase().startsWith(partial)
            );
            if (matches.length > 0 && partial.length > 0) {
                this.showSwarmAutocomplete(matches, atMatch.index);
                return;
            }
        }
        this.hideSwarmAutocomplete();
    }

    showSwarmAutocomplete(matches, atIndex) {
        let dropdown = document.getElementById('swarmAutocomplete');
        if (!dropdown) {
            dropdown = document.createElement('div');
            dropdown.id = 'swarmAutocomplete';
            dropdown.className = 'swarm-autocomplete';
            this.messageInput.parentElement.appendChild(dropdown);
        }
        dropdown.innerHTML = matches.map(a => `
            <div class="swarm-autocomplete-item" data-name="${a.name}">
                <span class="swarm-ac-icon">${a.icon}</span>
                <span class="swarm-ac-name">${a.name}</span>
                <span class="swarm-ac-role">${a.role}</span>
            </div>
        `).join('');
        dropdown.style.display = 'block';
        this.swarmAutocompleteVisible = true;

        dropdown.querySelectorAll('.swarm-autocomplete-item').forEach(item => {
            item.addEventListener('click', () => {
                const name = item.dataset.name;
                const before = this.messageInput.value.slice(0, atIndex);
                this.messageInput.value = `${before}@${name} `;
                this.messageInput.focus();
                this.hideSwarmAutocomplete();
                this.handleInputChange();
            });
        });
    }

    hideSwarmAutocomplete() {
        const dropdown = document.getElementById('swarmAutocomplete');
        if (dropdown) {
            dropdown.style.display = 'none';
        }
        this.swarmAutocompleteVisible = false;
    }

    showSwarmFeedbackInput(contextEl) {
        // Show an inline feedback input below the swarm toolbar
        let existing = document.querySelector('.swarm-feedback-area');
        if (existing) { existing.remove(); }

        const area = document.createElement('div');
        area.className = 'swarm-feedback-area';
        area.innerHTML = `
            <div class="swarm-feedback-header">💬 Give feedback to the team</div>
            <textarea class="swarm-feedback-input" placeholder="Your feedback, direction, or question to all agents..." rows="2"></textarea>
            <div class="swarm-feedback-actions">
                <button class="swarm-feedback-send">Send to Team</button>
                <button class="swarm-feedback-cancel">Cancel</button>
            </div>
        `;
        
        // Insert after the context element
        if (contextEl.nextSibling) {
            this.messagesContainer.insertBefore(area, contextEl.nextSibling);
        } else {
            this.messagesContainer.appendChild(area);
        }
        
        const input = area.querySelector('.swarm-feedback-input');
        input.focus();
        
        area.querySelector('.swarm-feedback-send').addEventListener('click', async () => {
            const feedback = input.value.trim();
            if (!feedback) return;
            await this.sendSwarmFeedback(feedback, area);
        });
        
        area.querySelector('.swarm-feedback-cancel').addEventListener('click', () => {
            area.remove();
        });
        
        // Enter to send (shift+enter for newline)
        input.addEventListener('keydown', async (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                const feedback = input.value.trim();
                if (feedback) await this.sendSwarmFeedback(feedback, area);
            }
        });
        
        this.scrollToBottom();
    }

    async sendSwarmFeedback(feedback, feedbackAreaEl) {
        if (!this.swarmSessionId) {
            this.showNotification('No active swarm session');
            return;
        }

        // Replace input with loading state
        feedbackAreaEl.innerHTML = '<div class="swarm-feedback-loading">🐝 Agents are reviewing your feedback...</div>';

        try {
            const resp = await fetch(`${this.settings.apiEndpoint}/swarm/feedback`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.swarmSessionId,
                    message: feedback,
                }),
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || 'Feedback failed');
            }

            const data = await resp.json();
            feedbackAreaEl.remove();

            // Display each agent's response to the feedback
            if (data.responses && data.responses.length > 0) {
                const userFeedbackEl = this.addMessage('user', `🐝 [Team Feedback]: ${feedback}`);
                userFeedbackEl.classList.add('swarm-feedback-msg');
                
                const fragment = document.createDocumentFragment();
                data.responses.forEach(r => {
                    const el = document.createElement('div');
                    el.className = 'message assistant swarm-agent swarm-feedback-response';
                    el.innerHTML = `
                        <div class="message-header">
                            <div class="message-avatar">${r.icon || '🐝'}</div>
                            <span class="message-role">${r.agent || 'Agent'}</span>
                            <span class="message-mode">FEEDBACK RESPONSE</span>
                        </div>
                        <div class="message-content">${this.formatMessage(r.response || '')}</div>
                    `;
                    fragment.appendChild(el);
                });
                this.messagesContainer.appendChild(fragment);
                this.scrollToBottom();
            }
        } catch (e) {
            feedbackAreaEl.innerHTML = `<div class="swarm-feedback-error">❌ ${e.message}</div>`;
            setTimeout(() => feedbackAreaEl.remove(), 3000);
        }
    }

    async sendSwarmDirectMessage(agentName, message) {
        if (!this.swarmSessionId) {
            this.showNotification('No active swarm session');
            return null;
        }
        
        try {
            const resp = await fetch(`${this.settings.apiEndpoint}/swarm/dm`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.swarmSessionId,
                    agent_name: agentName,
                    message: message,
                }),
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || 'DM failed');
            }

            return await resp.json();
        } catch (e) {
            console.error('Swarm DM error:', e);
            this.showNotification(`Failed to message @${agentName}: ${e.message}`);
            return null;
        }
    }

    loadSettings() {
        const saved = localStorage.getItem('edison_settings');
        const defaults = {
            ...this.getDefaultEndpoints(),
            defaultMode: 'auto',
            streamResponses: true,
            syntaxHighlight: true,
            sandboxAllowAnyHost: false,
            sandboxAllowedHosts: [],
            discordWebhook: '',
            slackWebhook: '',
            defaultPrinterId: '',
            assistantProfileId: '',
        };

        const settings = saved ? { ...defaults, ...JSON.parse(saved) } : defaults;

        // Migrate old direct-to-core endpoints to use the /api proxy
        if (settings.apiEndpoint && settings.apiEndpoint.match(/:8811/)) {
            console.log('↻ Migrating apiEndpoint from :8811 to /api proxy');
            settings.apiEndpoint = defaults.apiEndpoint;
        }
        const legacyHost = 'http://192.168.1.26';
        if (settings.apiEndpoint && settings.apiEndpoint.startsWith(legacyHost)) {
            settings.apiEndpoint = defaults.apiEndpoint;
        }
        if (settings.comfyuiEndpoint && settings.comfyuiEndpoint.startsWith('https://')) {
            settings.comfyuiEndpoint = settings.comfyuiEndpoint.replace(/^https:\/\//, 'http://');
        }
        if (settings.comfyuiEndpoint && settings.comfyuiEndpoint.startsWith(legacyHost)) {
            settings.comfyuiEndpoint = defaults.comfyuiEndpoint;
        }
        if (settings.voiceEndpoint && (settings.voiceEndpoint.startsWith(legacyHost) || settings.voiceEndpoint.match(/:8809/))) {
            settings.voiceEndpoint = defaults.voiceEndpoint;
        }

        // Persist migrated settings
        localStorage.setItem('edison_settings', JSON.stringify(settings));
        return settings;
    }

    async loadAssistantProfiles() {
        if (!this.assistantProfileSelect) return;
        try {
            const response = await fetch(`${this.settings.apiEndpoint}/assistants`);
            if (!response.ok) throw new Error('Failed to load assistants');
            const data = await response.json();
            const assistants = data.assistants || [];
            this.assistantProfileSelect.innerHTML = '<option value="">Default EDISON</option>' + assistants
                .filter(item => item.enabled !== false)
                .map(item => `<option value="${item.id}">${item.name}</option>`)
                .join('');
        } catch (error) {
            console.warn('Failed to load assistant profiles:', error.message);
        }
    }

    async refreshReadiness() {
        try {
            const response = await fetch(`${this.settings.apiEndpoint}/system/readiness`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                signal: AbortSignal.timeout(8000)
            });
            if (response.ok) {
                this.readiness = await response.json();
                this.applyReadinessToUI();
            } else {
                console.warn('Readiness check failed:', response.status);
                this.readiness = { overall: 'unknown', components: [], unavailable: [] };
            }
        } catch (error) {
            console.warn('Readiness fetch error:', error);
            this.readiness = { overall: 'unknown', components: [], unavailable: [] };
        }
    }

    applyReadinessToUI() {
        // Update mode button availability
        ['agent', 'swarm'].forEach(mode => {
            const btn = document.querySelector(`[data-mode="${mode}"]`);
            if (!btn) return;
            const available = this.isModeAvailable(mode);
            if (available) {
                btn.classList.remove('unavailable');
            } else {
                btn.classList.add('unavailable');
            }
        });

        // Update module availability bar
        if (this.moduleAvailabilityBar) {
            const criticalComponents = ['comfyui', 'core_api'];
            const chipsHtml = criticalComponents.map(key => {
                const component = this.readiness.components?.find(c => c.key === key);
                const state = component?.state === 'green' ? 'green' : 'red';
                const title = component?.title || key;
                return `<span class="availability-chip ${state}" title="${title}">${key}</span>`;
            }).join('');
            this.moduleAvailabilityBar.innerHTML = chipsHtml;
        }

        // Update mode help text
        if (this.modeHelp) {
            const mode = this.currentMode || this.settings.defaultMode;
            const description = this.modeDescriptions[mode] || 'Mode description not available.';
            this.modeHelp.textContent = description;
        }
    }

    isModeAvailable(mode) {
        if (mode !== 'agent' && mode !== 'swarm') return true;
        
        const modeKeyMap = { agent: 'agent', swarm: 'swarm' };
        const key = modeKeyMap[mode];
        if (!key) return true;
        
        const component = this.readiness.components?.find(c => c.key === key);
        return component?.ready !== false;
    }

    openSetupCheck() {
        if (this.setupCheckModal) {
            this.setupCheckModal.style.display = 'flex';
            this.runSetupCheck();
        }
    }

    closeSetupCheck() {
        if (this.setupCheckModal) {
            this.setupCheckModal.style.display = 'none';
        }
    }

    async runSetupCheck() {
        if (!this.setupCheckList) return;
        
        this.setupCheckList.innerHTML = '<div style="padding:20px; text-align:center; color:#888;">🔍 Checking system...</div>';
        
        try {
            await this.refreshReadiness();
            
            const components = this.readiness.components || [];
            if (components.length === 0) {
                this.setupCheckList.innerHTML = '<div style="padding:20px; color:#888;">No components to check.</div>';
                return;
            }
            
            const html = components.map(comp => `
                <div class="setup-check-item">
                    <div class="setup-check-title">
                        <span class="setup-check-state ${comp.state === 'green' ? 'green' : 'red'}">
                            ${comp.state === 'green' ? '✓' : '✕'}
                        </span>
                        ${comp.title || comp.key}
                    </div>
                    ${comp.system ? `<div style="font-size:0.85em; color:#888; margin-top:4px;">System: ${comp.system}</div>` : ''}
                    ${comp.state !== 'green' && comp.likely_cause ? `
                        <div style="font-size:0.85em; color:#fca5a5; margin-top:6px;">
                            <strong>Issue:</strong> ${comp.likely_cause}
                        </div>
                    ` : ''}
                    ${comp.state !== 'green' && comp.next_step ? `
                        <div style="font-size:0.85em; color:#9ddf88; margin-top:4px;">
                            <strong>Next step:</strong> ${comp.next_step}
                        </div>
                    ` : ''}
                    ${comp.raw_detail && comp.state !== 'green' ? `
                        <div style="font-size:0.75em; color:#aaa; margin-top:4px; font-family:monospace; white-space:pre-wrap; word-break:break-word;">
                            ${comp.raw_detail.slice(0, 500)}${comp.raw_detail.length > 500 ? '...' : ''}
                        </div>
                    ` : ''}
                </div>
            `).join('');
            
            this.setupCheckList.innerHTML = html;
        } catch (error) {
            console.error('Setup check error:', error);
            this.setupCheckList.innerHTML = `<div style="padding:20px; color:#fca5a5;">Error running setup check: ${error.message}</div>`;
        }
    }

    formatSystemMessage(component) {
        let msg = `**${component.title || component.key}**\n`;
        if (component.system) msg += `System: ${component.system}\n`;
        if (component.likely_cause) msg += `Issue: ${component.likely_cause}\n`;
        if (component.next_step) msg += `Next step: ${component.next_step}\n`;
        if (component.raw_detail) msg += `\`\`\`\n${component.raw_detail}\n\`\`\``;
        return msg;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.edisonApp = new EdisonApp();
});
