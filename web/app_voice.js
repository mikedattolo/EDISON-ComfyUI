/**
 * EDISON Voice Mode - Real-time voice interaction
 */

class VoiceMode {
    constructor(apiEndpoint) {
        this.apiEndpoint = apiEndpoint;
        this.sessionId = null;
        this.websocket = null;
        this.audioContext = null;
        this.mediaStream = null;
        this.audioWorkletNode = null;
        this.isActive = false;
        this.isPushToTalk = false;
        this.audioQueue = [];
        this.isPlaying = false;
        
        // UI elements
        this.voiceBtn = document.getElementById('voiceBtn');
        this.voicePanel = document.getElementById('voicePanel');
        this.voiceIndicator = document.getElementById('voiceIndicator');
        this.voiceState = document.getElementById('voiceState');
        this.voiceTranscript = document.getElementById('voiceTranscript');
        this.voiceCancelBtn = document.getElementById('voiceCancelBtn');
        
        this.init();
    }
    
    init() {
        // Voice button - push to talk
        this.voiceBtn.addEventListener('mousedown', () => this.startVoice());
        this.voiceBtn.addEventListener('mouseup', () => this.stopVoice());
        this.voiceBtn.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.startVoice();
        });
        this.voiceBtn.addEventListener('touchend', (e) => {
            e.preventDefault();
            this.stopVoice();
        });
        
        // Cancel button
        this.voiceCancelBtn.addEventListener('click', () => this.cancel());
        
        // Keyboard shortcut: Space for push-to-talk
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Space' && !e.repeat && this.isFocusOnInput()) {
                return; // Don't interfere with typing
            }
            if (e.code === 'Space' && !e.repeat && !this.isActive) {
                e.preventDefault();
                this.startVoice();
            }
        });
        
        document.addEventListener('keyup', (e) => {
            if (e.code === 'Space' && this.isActive) {
                e.preventDefault();
                this.stopVoice();
            }
        });
    }
    
    isFocusOnInput() {
        const active = document.activeElement;
        return active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA');
    }
    
    async startVoice() {
        if (this.isActive) return;
        
        console.log('Starting voice mode...');
        this.isActive = true;
        this.voiceBtn.classList.add('active');
        this.voicePanel.style.display = 'flex';
        this.voiceTranscript.innerHTML = '';
        
        try {
            // Create session
            const response = await fetch(`${this.apiEndpoint}/voice/session`, {
                method: 'POST'
            });
            
            if (!response.ok) {
                throw new Error('Failed to create voice session');
            }
            
            const data = await response.json();
            this.sessionId = data.session_id;
            console.log('Voice session created:', this.sessionId);
            
            // Initialize audio
            await this.initAudio();
            
            // Connect WebSocket
            await this.connectWebSocket();
            
        } catch (error) {
            console.error('Failed to start voice mode:', error);
            this.showError('Voice mode failed to start. Check console for details.');
            this.stopVoice();
        }
    }
    
    async stopVoice() {
        if (!this.isActive) return;
        
        console.log('Stopping voice mode...');
        
        // Stop audio capture
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }
        
        if (this.audioWorkletNode) {
            this.audioWorkletNode.disconnect();
            this.audioWorkletNode = null;
        }
        
        // Keep panel open to show response
        this.voiceBtn.classList.remove('active');
    }
    
    async cancel() {
        console.log('Cancelling voice session...');
        
        // Cancel on server
        if (this.sessionId) {
            try {
                await fetch(`${this.apiEndpoint}/voice/cancel/${this.sessionId}`, {
                    method: 'POST'
                });
            } catch (error) {
                console.error('Failed to cancel session:', error);
            }
        }
        
        // Close WebSocket
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        
        // Stop audio
        this.stopVoice();
        
        // Stop playback
        this.stopPlayback();
        
        // Reset state
        this.isActive = false;
        this.sessionId = null;
        this.voicePanel.style.display = 'none';
        this.voiceTranscript.innerHTML = '';
    }
    
    async initAudio() {
        // Check if getUserMedia is available
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('Microphone access not supported. Voice mode requires HTTPS or localhost.');
        }
        
        try {
            // Request microphone access
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
        } catch (err) {
            if (err.name === 'NotAllowedError') {
                throw new Error('Microphone permission denied. Please allow microphone access.');
            } else if (err.name === 'NotFoundError') {
                throw new Error('No microphone found. Please connect a microphone.');
            } else {
                throw new Error(`Microphone error: ${err.message}`);
            }
        }
        
        // Create audio context
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 16000
        });
        
        // Create audio source
        const source = this.audioContext.createMediaStreamSource(this.mediaStream);
        
        // Create ScriptProcessor for audio capture (AudioWorklet would be better but more complex)
        const bufferSize = 4096;
        const processor = this.audioContext.createScriptProcessor(bufferSize, 1, 1);
        
        processor.onaudioprocess = (e) => {
            if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
                return;
            }
            
            const inputData = e.inputBuffer.getChannelData(0);
            
            // Convert float32 to int16 PCM
            const pcm16 = new Int16Array(inputData.length);
            for (let i = 0; i < inputData.length; i++) {
                const s = Math.max(-1, Math.min(1, inputData[i]));
                pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }
            
            // Send to server
            const base64 = this.arrayBufferToBase64(pcm16.buffer);
            this.websocket.send(JSON.stringify({
                type: 'audio',
                pcm16_base64: base64,
                sample_rate: 16000
            }));
        };
        
        source.connect(processor);
        processor.connect(this.audioContext.destination);
        
        this.audioWorkletNode = processor;
        
        console.log('Audio initialized');
    }
    
    async connectWebSocket() {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${this.apiEndpoint.replace(/^https?:\/\//, '')}/voice/ws/${this.sessionId}`;
        
        console.log('Connecting to WebSocket:', wsUrl);
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onopen = () => {
            console.log('WebSocket connected');
        };
        
        this.websocket.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                this.handleWebSocketMessage(message);
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.showError('Connection error');
        };
        
        this.websocket.onclose = () => {
            console.log('WebSocket closed');
            this.websocket = null;
        };
    }
    
    handleWebSocketMessage(message) {
        const { type, value, text, pcm16_base64, sample_rate } = message;
        
        switch (type) {
            case 'state':
                this.updateState(value);
                break;
                
            case 'stt_partial':
                this.updateTranscript(text, false);
                break;
                
            case 'stt_final':
                this.updateTranscript(text, true);
                break;
                
            case 'llm_token':
                this.appendResponse(text);
                break;
                
            case 'tts_audio':
                this.queueAudioChunk(pcm16_base64, sample_rate);
                break;
                
            case 'tts_end':
                console.log('TTS complete');
                break;
                
            case 'error':
                this.showError(message.message);
                break;
                
            default:
                console.warn('Unknown message type:', type);
        }
    }
    
    updateState(state) {
        console.log('State:', state);
        this.voiceState.textContent = state;
        
        // Update indicator class
        this.voiceIndicator.className = 'voice-indicator';
        if (state === 'listening') {
            this.voiceIndicator.classList.add('listening');
        } else if (state === 'thinking') {
            this.voiceIndicator.classList.add('thinking');
        } else if (state === 'speaking') {
            this.voiceIndicator.classList.add('speaking');
        }
    }
    
    updateTranscript(text, isFinal) {
        if (isFinal) {
            this.voiceTranscript.innerHTML += `<div class="final">You: ${text}</div>`;
        } else {
            // Update or create partial transcript
            let partial = this.voiceTranscript.querySelector('.partial');
            if (!partial) {
                partial = document.createElement('div');
                partial.className = 'partial';
                this.voiceTranscript.appendChild(partial);
            }
            partial.textContent = text;
        }
        
        // Auto-scroll
        this.voiceTranscript.scrollTop = this.voiceTranscript.scrollHeight;
    }
    
    appendResponse(token) {
        // Find or create response div
        let response = this.voiceTranscript.querySelector('.response');
        if (!response) {
            response = document.createElement('div');
            response.className = 'final response';
            response.innerHTML = '<strong>EDISON:</strong> ';
            this.voiceTranscript.appendChild(response);
        }
        
        response.textContent += token;
        this.voiceTranscript.scrollTop = this.voiceTranscript.scrollHeight;
    }
    
    queueAudioChunk(base64Data, sampleRate) {
        // Decode base64 to PCM16
        const binaryString = atob(base64Data);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        
        const pcm16 = new Int16Array(bytes.buffer);
        
        // Convert PCM16 to float32 for Web Audio API
        const float32 = new Float32Array(pcm16.length);
        for (let i = 0; i < pcm16.length; i++) {
            float32[i] = pcm16[i] / 32768.0;
        }
        
        this.audioQueue.push({ data: float32, sampleRate });
        
        // Start playback if not already playing
        if (!this.isPlaying) {
            this.playNextChunk();
        }
    }
    
    async playNextChunk() {
        if (this.audioQueue.length === 0) {
            this.isPlaying = false;
            return;
        }
        
        this.isPlaying = true;
        const { data, sampleRate } = this.audioQueue.shift();
        
        // Create playback context if needed
        if (!this.audioContext || this.audioContext.state === 'closed') {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: sampleRate
            });
        }
        
        // Create buffer
        const audioBuffer = this.audioContext.createBuffer(1, data.length, sampleRate);
        audioBuffer.getChannelData(0).set(data);
        
        // Create source
        const source = this.audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this.audioContext.destination);
        
        // Play
        source.start();
        
        // Play next chunk when this one finishes
        source.onended = () => {
            this.playNextChunk();
        };
    }
    
    stopPlayback() {
        this.audioQueue = [];
        this.isPlaying = false;
        
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
    }
    
    showError(message) {
        this.voiceTranscript.innerHTML += `<div class="final" style="color: #ef4444;">Error: ${message}</div>`;
        this.voiceTranscript.scrollTop = this.voiceTranscript.scrollHeight;
    }
    
    arrayBufferToBase64(buffer) {
        let binary = '';
        const bytes = new Uint8Array(buffer);
        const len = bytes.byteLength;
        for (let i = 0; i < len; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }
}

// Initialize voice mode when app loads
let voiceMode = null;

function initVoiceMode() {
    // Get API endpoint from localStorage, with smart fallback
    let apiEndpoint = localStorage.getItem('apiEndpoint');
    
    // If not set or localhost, try to detect from current page
    if (!apiEndpoint || apiEndpoint.includes('localhost') || apiEndpoint.includes('127.0.0.1')) {
        // Use current host with port 8811
        const currentHost = window.location.hostname;
        apiEndpoint = `http://${currentHost}:8811`;
        console.log('Using detected API endpoint:', apiEndpoint);
    }
    
    voiceMode = new VoiceMode(apiEndpoint);
    console.log('Voice mode initialized with endpoint:', apiEndpoint);
}

// Initialize on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initVoiceMode);
} else {
    initVoiceMode();
}
