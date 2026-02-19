/**
 * EDISON Voice Assistant + Agent Live View
 * 
 * Voice Mode:
 *   - Orb overlay that reacts to audio amplitude
 *   - Web Speech API for STT (default) with MediaRecorder fallback
 *   - Web Speech API for TTS (default)
 *   - Integrates with existing chat endpoint
 * 
 * Agent Live View:
 *   - Collapsible panel in chat showing real-time agent steps
 *   - SSE connection to /agent/stream
 *   - Browser previews, file diffs, log messages
 */

// ═══════════════════════════════════════════════════════════════════════
// VOICE ASSISTANT
// ═══════════════════════════════════════════════════════════════════════

class EdisonVoiceAssistant {
    constructor(app) {
        this.app = app;  // reference to EdisonApp
        this.isActive = false;
        this.isListening = false;
        this.isSpeaking = false;
        this.isMuted = false;
        this.recognition = null;
        this.mediaRecorder = null;
        this.audioContext = null;
        this.analyser = null;
        this.animFrame = null;
        this.overlay = null;
        this.canvas = null;
        this.ctx = null;
        this.currentTranscript = '';
        this.serverConfig = null;
        this.selectedVoice = null;
        this.voices = [];

        this._initUI();
        this._fetchServerConfig();
    }

    // ── UI Setup ──────────────────────────────────────────────────────

    _initUI() {
        // Create overlay
        this.overlay = document.createElement('div');
        this.overlay.id = 'voiceOverlay';
        this.overlay.className = 'voice-overlay';
        this.overlay.setAttribute('role', 'dialog');
        this.overlay.setAttribute('aria-label', 'Voice assistant');
        this.overlay.style.display = 'none';
        this.overlay.innerHTML = `
            <div class="voice-overlay-content">
                <canvas id="voiceOrbCanvas" width="300" height="300"></canvas>
                <div class="voice-status" id="voiceStatus">Press to start</div>
                <div class="voice-transcript" id="voiceTranscript"></div>
                <div class="voice-controls">
                    <button class="voice-ctrl-btn" id="voiceMuteBtn" title="Mute TTS" aria-label="Mute text to speech">
                        🔊
                    </button>
                    <button class="voice-ctrl-btn voice-ctrl-close" id="voiceCloseBtn" title="Close (ESC)" aria-label="Close voice mode">
                        ✕
                    </button>
                    <button class="voice-ctrl-btn" id="voiceSettingsBtn" title="Voice settings" aria-label="Voice settings">
                        ⚙
                    </button>
                </div>
                <select id="voiceSelect" class="voice-select" style="display:none;" aria-label="Select voice"></select>
            </div>
        `;
        document.body.appendChild(this.overlay);

        this.canvas = document.getElementById('voiceOrbCanvas');
        this.ctx = this.canvas.getContext('2d');

        // Bind controls
        document.getElementById('voiceCloseBtn').addEventListener('click', () => this.deactivate());
        document.getElementById('voiceMuteBtn').addEventListener('click', () => this.toggleMute());
        document.getElementById('voiceSettingsBtn').addEventListener('click', () => this._toggleVoiceSelect());

        // ESC to close
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isActive) this.deactivate();
        });

        // Populate voices
        if ('speechSynthesis' in window) {
            const loadVoices = () => {
                this.voices = speechSynthesis.getVoices();
                this._populateVoiceSelect();
            };
            speechSynthesis.onvoiceschanged = loadVoices;
            loadVoices();
        }

        // Hook the existing voice button in the UI
        const voiceBtn = document.getElementById('voiceBtn');
        if (voiceBtn) {
            // Remove any existing listeners by cloning
            const newBtn = voiceBtn.cloneNode(true);
            voiceBtn.parentNode.replaceChild(newBtn, voiceBtn);
            newBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.toggle();
            });
        }
    }

    _populateVoiceSelect() {
        const sel = document.getElementById('voiceSelect');
        if (!sel) return;
        sel.innerHTML = '';
        this.voices.forEach((v, i) => {
            const opt = document.createElement('option');
            opt.value = i;
            opt.textContent = `${v.name} (${v.lang})`;
            if (v.default) opt.selected = true;
            sel.appendChild(opt);
        });
        sel.addEventListener('change', () => {
            this.selectedVoice = this.voices[parseInt(sel.value)] || null;
        });
    }

    _toggleVoiceSelect() {
        const sel = document.getElementById('voiceSelect');
        if (sel) sel.style.display = sel.style.display === 'none' ? 'block' : 'none';
    }

    async _fetchServerConfig() {
        try {
            const endpoint = this.app?.settings?.apiEndpoint || `${location.origin}/api`;
            const resp = await fetch(`${endpoint}/voice/config`);
            if (resp.ok) this.serverConfig = await resp.json();
        } catch {
            this.serverConfig = { recommended_mode: 'web_speech_api' };
        }
    }

    // ── Activation ────────────────────────────────────────────────────

    toggle() {
        this.isActive ? this.deactivate() : this.activate();
    }

    activate() {
        this.isActive = true;
        this.overlay.style.display = 'flex';

        // Check secure context — Web Speech API + getUserMedia require HTTPS or localhost
        const isSecure = window.isSecureContext ||
                         location.protocol === 'https:' ||
                         location.hostname === 'localhost' ||
                         location.hostname === '127.0.0.1';
        if (!isSecure) {
            this._setStatus('⚠️ Voice requires HTTPS or localhost');
            document.getElementById('voiceTranscript').textContent =
                'You are on HTTP over LAN. Enable HTTPS on your EDISON server, ' +
                'or access via https://localhost to use voice input.';
            this._startOrb();
            return;
        }

        this._setStatus('Listening…');
        this._startOrb();
        this._startListening();
    }

    deactivate() {
        this.isActive = false;
        this.overlay.style.display = 'none';
        this._stopListening();
        this._stopOrb();
        this._stopSpeaking();
    }

    toggleMute() {
        this.isMuted = !this.isMuted;
        const btn = document.getElementById('voiceMuteBtn');
        if (btn) btn.textContent = this.isMuted ? '🔇' : '🔊';
        if (this.isMuted) this._stopSpeaking();
    }

    // ── STT ───────────────────────────────────────────────────────────

    _startListening() {
        // Abort any previous recognition to prevent parallel instances
        if (this.recognition) {
            try { this.recognition.abort(); } catch {}
            this.recognition = null;
        }

        // Initialize network error counter (persists across retries)
        if (this._networkErrorCount === undefined) this._networkErrorCount = 0;

        // Try Web Speech API first
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (SpeechRecognition) {
            this.recognition = new SpeechRecognition();
            this.recognition.continuous = false;
            this.recognition.interimResults = true;
            this.recognition.lang = 'en-US';

            // Flag: onerror sets this so onend won't double-retry
            let errorHandled = false;

            this.recognition.onstart = () => {
                this.isListening = true;
                // DON'T reset _networkErrorCount here — onstart fires even
                // when Google servers are unreachable (the start is local)
                this._setStatus('Listening…');
            };

            this.recognition.onresult = (event) => {
                // Got actual speech results — NOW we know it works, reset counter
                this._networkErrorCount = 0;
                let interim = '';
                let final_ = '';
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    if (event.results[i].isFinal) {
                        final_ += event.results[i][0].transcript;
                    } else {
                        interim += event.results[i][0].transcript;
                    }
                }
                this.currentTranscript = final_ || interim;
                document.getElementById('voiceTranscript').textContent = this.currentTranscript;
            };

            this.recognition.onend = () => {
                this.isListening = false;
                if (errorHandled) return;  // onerror already handled retry/stop
                if (this.currentTranscript.trim()) {
                    this._sendTranscript(this.currentTranscript.trim());
                } else if (this.isActive) {
                    // Restart listening if nothing was said
                    setTimeout(() => {
                        if (this.isActive) this._startListening();
                    }, 500);
                }
            };

            this.recognition.onerror = (e) => {
                errorHandled = true;
                console.warn('Speech recognition error:', e.error);
                if (e.error === 'not-allowed') {
                    this._setStatus('Microphone access denied');
                    return;
                }
                // Count network AND aborted errors (aborted cascades from network failures)
                if (e.error === 'network' || e.error === 'aborted') {
                    this._networkErrorCount++;
                    console.warn(`[Voice] Recognition error #${this._networkErrorCount} (${e.error})`);
                    if (this._networkErrorCount >= 2) {
                        // Stop retrying, show clear message with type-to-speak option
                        this._stopListening();
                        this._setStatus('Speech recognition unavailable');
                        document.getElementById('voiceTranscript').textContent =
                            'Browser speech recognition needs internet access to Google servers. ' +
                            'Type your message below instead, or try Chrome/Edge with a stable connection.';
                        // Show a text input inside the overlay as fallback
                        this._showVoiceTextInput();
                        return;  // Don't retry
                    }
                    if (this.isActive) {
                        setTimeout(() => { if (this.isActive) this._startListening(); }, 1500);
                    }
                } else if (this.isActive) {
                    this._networkErrorCount++;
                    if (this._networkErrorCount >= 3) {
                        this._stopListening();
                        this._setStatus('Speech recognition not working');
                        document.getElementById('voiceTranscript').textContent =
                            'Speech recognition failed repeatedly. Type your message below instead.';
                        this._showVoiceTextInput();
                        return;
                    }
                    setTimeout(() => { if (this.isActive) this._startListening(); }, 1000);
                }
            };

            try {
                this.recognition.start();
                this._connectMicAnalyser();
            } catch (e) {
                console.warn('Failed to start recognition:', e);
                this._fallbackRecording();
            }
        } else {
            this._fallbackRecording();
        }
    }

    _stopListening() {
        this.isListening = false;
        if (this.recognition) {
            try { this.recognition.abort(); } catch {}
            this.recognition = null;
        }
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
        }
    }

    async _fallbackRecording() {
        // MediaRecorder fallback → POST to /voice/stt
        this._setStatus('Recording…');
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this._connectMicAnalyserFromStream(stream);
            const chunks = [];
            this.mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
            this.mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
            this.mediaRecorder.onstop = async () => {
                stream.getTracks().forEach(t => t.stop());
                const blob = new Blob(chunks, { type: 'audio/webm' });
                await this._sendAudioToServer(blob);
            };
            this.mediaRecorder.start();
            // Auto-stop after 10 seconds
            setTimeout(() => {
                if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
                    this.mediaRecorder.stop();
                }
            }, 10000);
        } catch (e) {
            this._setStatus('Microphone not available');
            console.error('MediaRecorder error:', e);
        }
    }

    async _sendAudioToServer(blob) {
        this._setStatus('Transcribing…');
        try {
            const endpoint = this.app?.settings?.apiEndpoint || `${location.origin}/api`;
            const formData = new FormData();
            formData.append('file', blob, 'recording.webm');
            const resp = await fetch(`${endpoint}/voice/stt`, { method: 'POST', body: formData });
            if (resp.ok) {
                const data = await resp.json();
                if (data.text) {
                    this.currentTranscript = data.text;
                    this._sendTranscript(data.text);
                    return;
                }
            }
            this._setStatus('Server STT unavailable. Try a supported browser for Web Speech API.');
        } catch (e) {
            this._setStatus('STT failed');
            console.error('STT error:', e);
        }
        // Restart listening
        if (this.isActive) setTimeout(() => { if (this.isActive) this._startListening(); }, 2000);
    }

    // ── Send to Chat ──────────────────────────────────────────────────

    async _sendTranscript(text) {
        this._setStatus('Thinking…');
        this.currentTranscript = '';
        document.getElementById('voiceTranscript').textContent = '';
        console.log('[Voice] Sending transcript:', text);

        // Put text in the chat input and use the app's normal sendMessage flow
        // This way it goes through the streaming endpoint with proper UI updates
        if (this.app && this.app.messageInput && typeof this.app.sendMessage === 'function') {
            this.app.messageInput.value = text;
            this.app.handleInputChange();

            try {
                // Let the app handle sending — this uses the streaming endpoint
                await this.app.sendMessage();
                console.log('[Voice] Message sent via app.sendMessage()');

                // Speak the last assistant message
                const lastAssistant = document.querySelectorAll('.message.assistant .message-content');
                if (lastAssistant.length > 0) {
                    const responseText = lastAssistant[lastAssistant.length - 1].textContent || '';
                    this._speak(responseText);
                } else {
                    this._afterSpeak();
                }
            } catch (e) {
                console.error('[Voice] sendMessage error:', e);
                this._setStatus('Error — tap to retry');
                if (this.isActive) setTimeout(() => { if (this.isActive) this._startListening(); }, 2000);
            }
        } else {
            // Fallback: direct API call if app not available
            console.warn('[Voice] App not available, using direct API call');
            try {
                const endpoint = this.app?.settings?.apiEndpoint || `${location.origin}/api`;
                const mode = this.app?.currentMode || 'auto';
                const resp = await fetch(`${endpoint}/chat`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text, mode: mode }),
                });
                if (!resp.ok) throw new Error(`Chat failed: ${resp.status}`);
                const data = await resp.json();
                const responseText = data.response || '';
                if (this.app) {
                    this.app.addMessage('user', text);
                    this.app.addMessage('assistant', responseText);
                }
                this._speak(responseText);
            } catch (e) {
                console.error('[Voice] Direct API error:', e);
                this._setStatus('Error — tap to retry');
                if (this.isActive) setTimeout(() => { if (this.isActive) this._startListening(); }, 2000);
            }
        }
    }

    // ── TTS ───────────────────────────────────────────────────────────

    _speak(text) {
        if (this.isMuted || !text) {
            this._afterSpeak();
            return;
        }
        if (!('speechSynthesis' in window)) {
            this._afterSpeak();
            return;
        }

        // Cancel any in-progress speech
        speechSynthesis.cancel();

        // Clean text for speech (remove markdown, code blocks, etc.)
        const clean = text
            .replace(/```[\s\S]*?```/g, ' code block ')
            .replace(/`[^`]+`/g, '')
            .replace(/[#*_~]/g, '')
            .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
            .replace(/\n+/g, '. ')
            .slice(0, 2000);  // Limit length

        const utterance = new SpeechSynthesisUtterance(clean);
        if (this.selectedVoice) utterance.voice = this.selectedVoice;
        utterance.rate = 1.0;
        utterance.pitch = 1.0;

        this.isSpeaking = true;
        this._setStatus('Speaking…');

        utterance.onend = () => this._afterSpeak();
        utterance.onerror = () => this._afterSpeak();

        speechSynthesis.speak(utterance);
    }

    _stopSpeaking() {
        if ('speechSynthesis' in window) speechSynthesis.cancel();
        this.isSpeaking = false;
    }

    _afterSpeak() {
        this.isSpeaking = false;
        if (this.isActive) {
            this._setStatus('Listening…');
            this._startListening();
        }
    }

    // ── Orb Animation ─────────────────────────────────────────────────

    _startOrb() {
        if (!this.canvas || !this.ctx) return;
        const W = this.canvas.width;
        const H = this.canvas.height;
        const cx = W / 2;
        const cy = H / 2;
        let time = 0;
        let amplitude = 0;

        const draw = () => {
            if (!this.isActive) return;
            time += 0.02;

            // Get audio amplitude
            if (this.analyser) {
                const data = new Uint8Array(this.analyser.frequencyBinCount);
                this.analyser.getByteTimeDomainData(data);
                let sum = 0;
                for (let i = 0; i < data.length; i++) {
                    const v = (data[i] - 128) / 128;
                    sum += v * v;
                }
                amplitude = Math.sqrt(sum / data.length);
            } else {
                amplitude *= 0.95;  // Decay
            }

            // Clear
            this.ctx.clearRect(0, 0, W, H);

            // Draw orb with swaying effect
            const baseRadius = 60;
            const sway = amplitude * 40 + Math.sin(time * 2) * 5;
            const radius = baseRadius + sway;

            // Glow
            const gradient = this.ctx.createRadialGradient(cx, cy, radius * 0.3, cx, cy, radius * 1.5);
            const hue = this.isListening ? 210 : (this.isSpeaking ? 150 : 260);
            gradient.addColorStop(0, `hsla(${hue}, 80%, 60%, 0.9)`);
            gradient.addColorStop(0.5, `hsla(${hue}, 70%, 50%, 0.4)`);
            gradient.addColorStop(1, `hsla(${hue}, 60%, 40%, 0)`);

            this.ctx.beginPath();
            // Organic shape with noise
            const points = 64;
            for (let i = 0; i <= points; i++) {
                const angle = (i / points) * Math.PI * 2;
                const noise = Math.sin(angle * 3 + time * 3) * amplitude * 15
                            + Math.sin(angle * 5 + time * 2) * 3;
                const r = radius + noise;
                const x = cx + Math.cos(angle) * r;
                const y = cy + Math.sin(angle) * r;
                if (i === 0) this.ctx.moveTo(x, y);
                else this.ctx.lineTo(x, y);
            }
            this.ctx.closePath();
            this.ctx.fillStyle = gradient;
            this.ctx.fill();

            // Inner glow
            const innerGrad = this.ctx.createRadialGradient(cx, cy, 0, cx, cy, radius * 0.6);
            innerGrad.addColorStop(0, `hsla(${hue}, 90%, 80%, 0.6)`);
            innerGrad.addColorStop(1, `hsla(${hue}, 80%, 60%, 0)`);
            this.ctx.beginPath();
            this.ctx.arc(cx, cy, radius * 0.5, 0, Math.PI * 2);
            this.ctx.fillStyle = innerGrad;
            this.ctx.fill();

            this.animFrame = requestAnimationFrame(draw);
        };

        this.animFrame = requestAnimationFrame(draw);
    }

    _stopOrb() {
        if (this.animFrame) cancelAnimationFrame(this.animFrame);
        this.animFrame = null;
    }

    // ── Audio Analyser ────────────────────────────────────────────────

    async _connectMicAnalyser() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this._connectMicAnalyserFromStream(stream);
        } catch {}
    }

    _connectMicAnalyserFromStream(stream) {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = this.audioContext.createMediaStreamSource(stream);
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;
            source.connect(this.analyser);
        } catch (e) {
            console.warn('Audio analyser setup failed:', e);
        }
    }

    // ── Helpers ────────────────────────────────────────────────────────

    _showVoiceTextInput() {
        // Show a text input inside the voice overlay as fallback when speech recognition fails
        const existing = document.getElementById('voiceTextInput');
        if (existing) return;  // Already showing

        const container = document.createElement('div');
        container.id = 'voiceTextInput';
        container.style.cssText = 'margin-top: 16px; display: flex; gap: 8px; width: 80%; max-width: 400px;';
        container.innerHTML = `
            <input type="text" id="voiceTypeInput" placeholder="Type your message here..."
                   style="flex: 1; padding: 10px 14px; border-radius: 20px; border: 1px solid rgba(255,255,255,0.3);
                          background: rgba(255,255,255,0.1); color: white; font-size: 14px; outline: none;">
            <button id="voiceTypeSendBtn"
                    style="padding: 10px 18px; border-radius: 20px; border: none; background: #4a9eff;
                           color: white; cursor: pointer; font-size: 14px; font-weight: 600;">Send</button>
        `;

        const overlayContent = this.overlay.querySelector('.voice-overlay-content');
        if (overlayContent) {
            overlayContent.appendChild(container);
        }

        const input = document.getElementById('voiceTypeInput');
        const sendBtn = document.getElementById('voiceTypeSendBtn');

        const send = () => {
            const text = input.value.trim();
            if (text) {
                input.value = '';
                this._sendTranscript(text);
            }
        };

        sendBtn.addEventListener('click', send);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') { e.preventDefault(); send(); }
        });
        input.focus();
    }

    _setStatus(text) {
        const el = document.getElementById('voiceStatus');
        if (el) el.textContent = text;
    }
}


// ═══════════════════════════════════════════════════════════════════════
// AGENT LIVE VIEW
// ═══════════════════════════════════════════════════════════════════════

class EdisonAgentLiveView {
    constructor(app) {
        this.app = app;
        this.enabled = true;
        this.isOpen = false;
        this.eventSource = null;
        this.steps = [];
        this.panel = null;

        this._initUI();
        this._fetchConfig();
    }

    // ── UI Setup ──────────────────────────────────────────────────────

    _initUI() {
        // Create collapsible panel container (inserted before input area)
        this.panel = document.createElement('div');
        this.panel.id = 'agentLivePanel';
        this.panel.className = 'agent-live-panel';
        this.panel.style.display = 'none';
        this.panel.innerHTML = `
            <div class="agent-live-header" id="agentLiveHeader">
                <span class="agent-live-title">
                    <span class="agent-live-dot"></span>
                    Live Activity
                </span>
                <button class="agent-live-toggle" id="agentLiveToggle" aria-label="Toggle live panel">▼</button>
            </div>
            <div class="agent-live-body" id="agentLiveBody">
                <div class="agent-live-steps" id="agentLiveSteps"></div>
            </div>
        `;

        // Insert above the messages container
        const chatContainer = document.querySelector('.chat-container');
        if (chatContainer) {
            const messagesContainer = document.getElementById('messagesContainer');
            if (messagesContainer) {
                chatContainer.insertBefore(this.panel, messagesContainer.nextSibling);
            } else {
                chatContainer.appendChild(this.panel);
            }
        } else {
            document.body.appendChild(this.panel);
        }

        // Toggle collapse
        document.getElementById('agentLiveToggle')?.addEventListener('click', () => {
            const body = document.getElementById('agentLiveBody');
            const btn = document.getElementById('agentLiveToggle');
            if (body && btn) {
                const collapsed = body.style.display === 'none';
                body.style.display = collapsed ? 'block' : 'none';
                btn.textContent = collapsed ? '▼' : '▶';
            }
        });
    }

    async _fetchConfig() {
        try {
            const endpoint = this.app?.settings?.apiEndpoint || `${location.origin}/api`;
            const resp = await fetch(`${endpoint}/agent/live-config`);
            if (resp.ok) {
                const cfg = await resp.json();
                this.enabled = cfg.enabled !== false;
            }
        } catch {
            // Default to enabled
        }
    }

    // ── Connection ────────────────────────────────────────────────────

    connect(sessionId = 'default') {
        if (!this.enabled) return;
        this.disconnect();

        // Show panel when connecting
        if (this.panel) {
            this.panel.style.display = 'block';
            const stepsEl = document.getElementById('agentLiveSteps');
            if (stepsEl && stepsEl.children.length === 0) {
                stepsEl.innerHTML = '<div class="agent-step" style="opacity:0.6">Waiting for agent activity...</div>';
            }
        }

        const endpoint = this.app?.settings?.apiEndpoint || `${location.origin}/api`;
        const url = `${endpoint}/agent/stream?session_id=${encodeURIComponent(sessionId)}`;

        try {
            this.eventSource = new EventSource(url);
            this.eventSource.onmessage = (e) => {
                try {
                    const event = JSON.parse(e.data);
                    this._handleEvent(event);
                } catch {}
            };
            this.eventSource.onerror = () => {
                // Auto-reconnect is handled by EventSource
            };
        } catch (e) {
            console.warn('Agent live view SSE connection failed:', e);
        }
    }

    disconnect() {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
    }

    show() {
        if (this.panel) {
            this.panel.style.display = 'block';
            this.isOpen = true;
        }
    }

    hide() {
        if (this.panel) {
            this.panel.style.display = 'none';
            this.isOpen = false;
        }
    }

    clear() {
        this.steps = [];
        const container = document.getElementById('agentLiveSteps');
        if (container) container.innerHTML = '';
    }

    // ── Event Handling ────────────────────────────────────────────────

    _handleEvent(event) {
        this.show();

        switch (event.type) {
            case 'agent_step':
                this._addStep(event);
                break;
            case 'browser_open':
                this._addBrowserEvent(event);
                break;
            case 'browser_screenshot':
                this._addScreenshot(event);
                break;
            case 'file_diff':
                this._addFileDiff(event);
                break;
            case 'log':
                this._addLog(event);
                break;
            case 'keepalive':
                break;
            default:
                break;
        }
    }

    _addStep(event) {
        const container = document.getElementById('agentLiveSteps');
        if (!container) return;

        // Update existing step or create new
        let stepEl = container.querySelector(`[data-step-id="${event.step_id}"]`);
        if (!stepEl) {
            stepEl = document.createElement('div');
            stepEl.className = 'agent-step';
            stepEl.dataset.stepId = event.step_id;
            container.appendChild(stepEl);
        }

        const statusIcon = event.status === 'done' ? '✓' :
                          event.status === 'error' ? '✗' :
                          event.status === 'running' ? '⟳' : '○';

        stepEl.innerHTML = `
            <span class="agent-step-icon ${event.status}">${statusIcon}</span>
            <span class="agent-step-title">${this._escapeHtml(event.title)}</span>
        `;

        container.scrollTop = container.scrollHeight;
    }

    _addBrowserEvent(event) {
        const container = document.getElementById('agentLiveSteps');
        if (!container) return;

        const el = document.createElement('div');
        el.className = 'agent-step agent-browser-event';
        el.innerHTML = `
            <span class="agent-step-icon">🌐</span>
            <span class="agent-step-title">
                Opened: <a href="${this._escapeHtml(event.url)}" target="_blank" rel="noopener">${this._escapeHtml(event.url)}</a>
            </span>
        `;
        container.appendChild(el);
        container.scrollTop = container.scrollHeight;
    }

    _addScreenshot(event) {
        const container = document.getElementById('agentLiveSteps');
        if (!container || !event.png_base64) return;

        const el = document.createElement('div');
        el.className = 'agent-step agent-screenshot';
        el.innerHTML = `
            <span class="agent-step-icon">📸</span>
            <img src="data:image/png;base64,${event.png_base64}" 
                 class="agent-screenshot-img" 
                 alt="Browser screenshot"
                 loading="lazy" />
        `;
        container.appendChild(el);
        container.scrollTop = container.scrollHeight;
    }

    _addFileDiff(event) {
        const container = document.getElementById('agentLiveSteps');
        if (!container) return;

        const el = document.createElement('div');
        el.className = 'agent-step agent-file-diff';
        el.innerHTML = `
            <details>
                <summary>
                    <span class="agent-step-icon">📝</span>
                    <span class="agent-step-title">${this._escapeHtml(event.path)}</span>
                </summary>
                <pre class="agent-diff-code">${this._escapeHtml(event.diff)}</pre>
            </details>
        `;
        container.appendChild(el);
        container.scrollTop = container.scrollHeight;
    }

    _addLog(event) {
        const container = document.getElementById('agentLiveSteps');
        if (!container) return;

        const levelIcon = event.level === 'error' ? '❌' :
                         event.level === 'warning' ? '⚠️' : 'ℹ️';
        const el = document.createElement('div');
        el.className = `agent-step agent-log agent-log-${event.level || 'info'}`;
        el.innerHTML = `
            <span class="agent-step-icon">${levelIcon}</span>
            <span class="agent-step-title">${this._escapeHtml(event.message)}</span>
        `;
        container.appendChild(el);
        container.scrollTop = container.scrollHeight;
    }

    _escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}


// ═══════════════════════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════════════════════

// Wait for EdisonApp to be ready, then attach
document.addEventListener('DOMContentLoaded', () => {
    // Retry until app is ready (app may be at window.app or window.edisonApp)
    const waitForApp = setInterval(() => {
        const appRef = window.edisonApp || window.app;
        if (appRef) {
            clearInterval(waitForApp);
            // Ensure window.app is set for backward compatibility
            if (!window.app) window.app = appRef;
            // Voice Assistant
            window.edisonVoice = new EdisonVoiceAssistant(appRef);
            // Agent Live View
            window.edisonAgentLive = new EdisonAgentLiveView(appRef);
            console.log('✓ EDISON Voice + Agent Live View initialized');
        }
    }, 200);

    // Timeout after 10s
    setTimeout(() => clearInterval(waitForApp), 10000);
});
