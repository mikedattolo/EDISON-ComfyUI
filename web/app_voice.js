/**
 * Voice Mode for Edison - Localhost implementation
 * Works on http://localhost:8080 without SSL
 * Push-to-talk interface with live transcription
 */

class VoiceMode {
    constructor() {
        this.ws = null;
        this.sessionId = null;
        this.audioContext = null;
        this.scriptProcessor = null;
        this.mediaStream = null;
        this.isRecording = false;
        this.isPlaying = false;
        
        // Audio playback queue
        this.audioQueue = [];
        this.isProcessingQueue = false;
        
        this.initUI();
    }
    
    initUI() {
        // Add voice button to input area
        const voiceBtn = document.getElementById('voiceBtn');
        if (!voiceBtn) {
            console.warn('Voice button not found');
            return;
        }
        
        voiceBtn.addEventListener('mousedown', () => this.startRecording());
        voiceBtn.addEventListener('mouseup', () => this.stopRecording());
        voiceBtn.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.startRecording();
        });
        voiceBtn.addEventListener('touchend', (e) => {
            e.preventDefault();
            this.stopRecording();
        });
        
        // Create transcript display
        const transcriptDiv = document.createElement('div');
        transcriptDiv.id = 'voiceTranscript';
        transcriptDiv.style.cssText = `
            position: fixed;
            bottom: 100px;
            left: 50%;
            transform: translateX(-50%);
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            max-width: 600px;
            display: none;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        `;
        document.body.appendChild(transcriptDiv);
        
        // Speaking indicator
        const speakingDiv = document.createElement('div');
        speakingDiv.id = 'voiceSpeaking';
        speakingDiv.style.cssText = `
            position: fixed;
            bottom: 100px;
            left: 50%;
            transform: translateX(-50%);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 20px;
            border-radius: 20px;
            display: none;
            z-index: 1000;
            font-weight: 600;
        `;
        speakingDiv.textContent = '🔊 Edison is speaking...';
        document.body.appendChild(speakingDiv);
    }
    
    async startRecording() {
        if (this.isRecording) return;
        
        try {
            // Stop any current playback (barge-in)
            if (this.isPlaying) {
                this.stopPlayback();
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify({ type: 'barge_in' }));
                }
            }
            
            // Create session if needed
            if (!this.sessionId) {
                await this.createSession();
            }
            
            // Connect WebSocket if needed
            if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
                await this.connectWebSocket();
            }
            
            // Get microphone access
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });
            
            // Setup audio processing
            this.audioContext = new AudioContext({ sampleRate: 16000 });
            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            
            // Create script processor for PCM16 conversion
            this.scriptProcessor = this.audioContext.createScriptProcessor(4096, 1, 1);
            
            this.scriptProcessor.onaudioprocess = (event) => {
                if (!this.isRecording) return;
                
                const inputData = event.inputBuffer.getChannelData(0);
                const pcm16 = new Int16Array(inputData.length);
                
                // Convert float32 to int16
                for (let i = 0; i < inputData.length; i++) {
                    const s = Math.max(-1, Math.min(1, inputData[i]));
                    pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }
                
                // Send to server
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    const arrayBuffer = pcm16.buffer;
                    const base64 = this.arrayBufferToBase64(arrayBuffer);
                    
                    this.ws.send(JSON.stringify({
                        type: 'audio_pcm16',
                        data_b64: base64,
                        sample_rate: 16000
                    }));
                }
            };
            
            source.connect(this.scriptProcessor);
            this.scriptProcessor.connect(this.audioContext.destination);
            
            this.isRecording = true;
            document.getElementById('voiceBtn').classList.add('recording');
            document.getElementById('voiceTranscript').style.display = 'block';
            document.getElementById('voiceTranscript').innerHTML = '<div style="color: var(--text-secondary);">🎤 Listening...</div>';
            
            console.log('Recording started');
            
        } catch (error) {
            console.error('Error starting recording:', error);
            alert('Microphone access failed. Make sure you\'re on localhost.');
        }
    }
    
    stopRecording() {
        if (!this.isRecording) return;
        
        this.isRecording = false;
        
        // Stop audio processing
        if (this.scriptProcessor) {
            this.scriptProcessor.disconnect();
            this.scriptProcessor = null;
        }
        
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }
        
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        
        // Signal end of utterance
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'stop_listening' }));
        }
        
        document.getElementById('voiceBtn').classList.remove('recording');
        document.getElementById('voiceTranscript').innerHTML = '<div style="color: var(--text-secondary);">⏳ Processing...</div>';
        
        console.log('Recording stopped');
    }
    
    async createSession() {
        const response = await fetch('http://localhost:8811/voice/session', {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error('Failed to create voice session');
        }
        
        const data = await response.json();
        this.sessionId = data.session_id;
        console.log('Voice session created:', this.sessionId);
    }
    
    async connectWebSocket() {
        return new Promise((resolve, reject) => {
            this.ws = new WebSocket(`ws://localhost:8811/voice/ws/${this.sessionId}`);
            
            this.ws.onopen = () => {
                console.log('Voice WebSocket connected');
                resolve();
            };
            
            this.ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                this.handleMessage(msg);
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                reject(error);
            };
            
            this.ws.onclose = () => {
                console.log('Voice WebSocket closed');
                this.ws = null;
            };
        });
    }
    
    handleMessage(msg) {
        const transcriptDiv = document.getElementById('voiceTranscript');
        const speakingDiv = document.getElementById('voiceSpeaking');
        
        switch (msg.type) {
            case 'state':
                console.log('State:', msg.value);
                if (msg.value === 'thinking') {
                    transcriptDiv.innerHTML = '<div style="color: var(--text-secondary);">🤔 Thinking...</div>';
                } else if (msg.value === 'speaking') {
                    transcriptDiv.style.display = 'none';
                    speakingDiv.style.display = 'block';
                } else if (msg.value === 'ready') {
                    speakingDiv.style.display = 'none';
                    transcriptDiv.style.display = 'none';
                }
                break;
            
            case 'stt_final':
                transcriptDiv.innerHTML = `<div><strong>You:</strong> ${msg.text}</div>`;
                break;
            
            case 'llm_text':
                transcriptDiv.innerHTML += `<div style="margin-top: 8px;"><strong>Edison:</strong> ${msg.text}</div>`;
                break;
            
            case 'tts_audio':
                this.playAudio(msg.data_b64, msg.sample_rate);
                break;
            
            case 'done':
                setTimeout(() => {
                    transcriptDiv.style.display = 'none';
                    speakingDiv.style.display = 'none';
                }, 1000);
                break;
            
            case 'error':
                console.error('Voice error:', msg.message);
                transcriptDiv.innerHTML = `<div style="color: red;">Error: ${msg.message}</div>`;
                break;
        }
    }
    
    playAudio(base64Data, sampleRate) {
        // Decode base64 to PCM16
        const binaryString = atob(base64Data);
        const len = binaryString.length;
        const bytes = new Uint8Array(len);
        
        for (let i = 0; i < len; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        
        const pcm16 = new Int16Array(bytes.buffer);
        
        // Convert to float32
        const float32 = new Float32Array(pcm16.length);
        for (let i = 0; i < pcm16.length; i++) {
            float32[i] = pcm16[i] / 32768.0;
        }
        
        // Queue for playback
        this.audioQueue.push({ data: float32, sampleRate });
        
        if (!this.isProcessingQueue) {
            this.processAudioQueue();
        }
    }
    
    async processAudioQueue() {
        if (this.isProcessingQueue || this.audioQueue.length === 0) return;
        
        this.isProcessingQueue = true;
        this.isPlaying = true;
        
        // Create audio context for playback
        const audioContext = new AudioContext();
        
        while (this.audioQueue.length > 0 && this.isPlaying) {
            const { data, sampleRate } = this.audioQueue.shift();
            
            // Create buffer
            const buffer = audioContext.createBuffer(1, data.length, sampleRate);
            buffer.getChannelData(0).set(data);
            
            // Play
            const source = audioContext.createBufferSource();
            source.buffer = buffer;
            source.connect(audioContext.destination);
            
            await new Promise((resolve) => {
                source.onended = resolve;
                source.start();
            });
        }
        
        audioContext.close();
        this.isProcessingQueue = false;
        this.isPlaying = false;
    }
    
    stopPlayback() {
        this.isPlaying = false;
        this.audioQueue = [];
        
        document.getElementById('voiceSpeaking').style.display = 'none';
    }
    
    arrayBufferToBase64(buffer) {
        let binary = '';
        const bytes = new Uint8Array(buffer);
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }
}

// Initialize voice mode on localhost
if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    window.voiceMode = new VoiceMode();
    console.log('🎙️ Voice mode initialized (localhost)');
} else {
    console.log('⚠️ Voice mode requires localhost');
    const voiceBtn = document.getElementById('voiceBtn');
    if (voiceBtn) voiceBtn.style.display = 'none';
}
