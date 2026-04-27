/**
 * Video Capture App with WebRTC and WebSocket
 *
 * This file implements WebRTC camera functionality with WebSocket communication.
 */

/**
 * TextStreamer - Modular letter-by-letter text animator
 */
class TextStreamer {
    constructor(containerElement) {
        this.container = containerElement;
        this.queue = []; // Queue of items (text or elements) to display
        this.isStreaming = false;
        this.streamSpeed = 15; // ms per character
    }

    /**
     * Add text to streaming queue
     * @param {string} text - Text to stream
     * @param {string} className - CSS class for styling
     */
    addText(text, className = 'model-text') {
        this.queue.push({ type: 'text', text, className });
        if (!this.isStreaming) {
            this.processQueue();
        }
    }

    /**
     * Add element to queue (will appear in order)
     * @param {HTMLElement} element - DOM element to add
     */
    addElement(element) {
        this.queue.push({ type: 'element', element });
        if (!this.isStreaming) {
            this.processQueue();
        }
    }

    /**
     * Process queue with letter-by-letter animation for text
     */
    async processQueue() {
        if (this.queue.length === 0) {
            this.isStreaming = false;
            return;
        }

        this.isStreaming = true;
        const item = this.queue.shift();

        if (item.type === 'element') {
            // Add element immediately
            this.container.appendChild(item.element);
        } else if (item.type === 'text') {
            // Create container for this text segment
            const textElement = document.createElement('div');
            textElement.className = item.className;
            this.container.appendChild(textElement);

            // Stream letter by letter
            for (let i = 0; i < item.text.length; i++) {
                textElement.textContent += item.text[i];
                await this.sleep(this.streamSpeed);
            }
        }

        // Process next item
        this.processQueue();
    }

    /**
     * Clear all output
     */
    clear() {
        this.queue = [];
        this.isStreaming = false;
        this.container.innerHTML = '';
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

class VideoApp {
    constructor() {
        this.videoElement = document.getElementById('video-preview');
        this.connectionStatus = document.getElementById('connection-status');
        this.toggleButton = document.getElementById('btn-toggle-camera');
        this.permissionInfo = document.getElementById('permission-info');
        this.cameraSelect = document.getElementById('camera-select');
        this.clearOutputBtn = document.getElementById('clear-output-btn');
        this.outputContent = document.getElementById('output-content');
        this.modelsStatus = document.getElementById('models-status');
        this.candidateNameDisplay = document.getElementById('candidate-name-display');
        this.targetRoleInput = document.getElementById('target-role');
        this.interviewRequirementsInput = document.getElementById('interview-requirements');
        this.reqCharCount = document.getElementById('req-char-count');
        this.engagementScore = document.getElementById('engagement-score');
        this.postureStability = document.getElementById('posture-stability');
        this.voicePitch = document.getElementById('voice-pitch');
        this.voiceTempo = document.getElementById('voice-tempo');
        this.sentimentAlert = document.getElementById('sentiment-alert');
        this.localStream = null;
        this.mediaRecorder = null;
        this.socket = null;
        this.socketConnected = false;
        this.mediaConstraints = { video: true, audio: true };
        this.chunkTimer = null;
        this.isRecording = false;
        this.isCameraActive = false;
        this.chunkDurationMs = 30000;  // Default, will be overridden by backend
        this.textStreamer = null;
        this.modelsReady = false;
        this.modelsCheckInterval = null;

        // Session-specific state (initialized when camera starts)
        this.sessionId = null;
        this.chunkIndex = null;

        // Disable camera button until models are ready
        this.toggleButton.disabled = true;
        this.toggleButton.textContent = 'Loading Models...';

        this.initializeWebSocket();
        this.initializeEventListeners();
        this.enumerateCameras();
        this.startModelStatusCheck();

        // Reliably close the session when navigating away mid-interview
        window.addEventListener('beforeunload', () => {
            if (this.isCameraActive && this.sessionId) {
                // Try socket first (works for same-tab navigation)
                this.sendWebSocketMessage('stream_ended', { sessionId: this.sessionId });
                // sendBeacon survives page unload reliably
                const { protocol, hostname } = window.location;
                navigator.sendBeacon(
                    `${protocol}//${hostname}:5555/api/sessions/${encodeURIComponent(this.sessionId)}/complete`
                );
            }
        });
    }

    initializeSession() {
        this.sessionId = `session_${Date.now()}`;
        this.chunkIndex = 1;
    }

    resetSession() {
        this.sessionId = null;
        this.chunkIndex = null;
    }

    initializeWebSocket() {
        if (typeof io === 'undefined') return;
        const { protocol, hostname } = window.location;
        const backendPort = '5555';  // Backend runs on port 5555
        // Long timeouts so connection survives slow model runs (2–3 min per chunk)
        this.socket = io(`${protocol}//${hostname}:${backendPort}`, {
            transports: ['websocket'],
            reconnection: true,
            reconnectionDelay: 1000,
            reconnectionDelayMax: 15000,
            reconnectionAttempts: 20,
            timeout: 120000,
            pingTimeout: 300000,
            pingInterval: 20000
        });
        this.socket.on('connect', () => {
            this.socketConnected = true;
            console.log('WebSocket connected');
            this.socket.emit('request_camera_permission');
        });
        this.socket.on('disconnect', () => {
            this.socketConnected = false;
            console.log('WebSocket disconnected');
        });
        this.socket.on('connect_error', (error) => {
            this.socketConnected = false;
            console.error('WebSocket connection error:', error);
        });
        this.socket.on('stream_acknowledged', (data) => {
            if (data.chunkDurationMs) {
                this.chunkDurationMs = data.chunkDurationMs;
                console.log(`Chunk duration set to ${this.chunkDurationMs}ms`);
            }
        });
        this.socket.on('processing_heartbeat', () => {
            // Keep connection alive during long backend processing
        });
        this.socket.on('chunk_results', (data) => {
            // Only log raw model outputs; UI shows interviewer questions only
            const { chunkIndex, results } = data || {};
            if (chunkIndex !== undefined && results) {
                console.log(
                    `[ChunkResults] Chunk ${chunkIndex} models: ${Object.keys(results).join(', ')}`
                );
                this.updateMetricsFromChunkResults(results);
            } else {
                console.log('[ChunkResults] Received chunk results', data);
            }
        });
        // Generic text streaming handler
        this.socket.on('text_stream', (data) => {
            this.handleTextStream(data);
        });
    }
    
    sendWebSocketMessage(event, data = {}) {
        if (this.socket && this.socketConnected) {
            this.socket.emit(event, data);
        }
    }

    initializeEventListeners() {
        this.toggleButton.addEventListener('click', () => {
            if (this.isCameraActive) {
                this.stopCamera();
            } else {
                this.startCamera();
            }
        });

        this.clearOutputBtn.addEventListener('click', () => {
            if (this.textStreamer) {
                this.textStreamer.clear();
                // Add placeholder back
                const placeholder = document.createElement('div');
                placeholder.className = 'output-placeholder';
                placeholder.textContent = 'Interview questions and AI insights will appear here after you start the camera...';
                this.outputContent.appendChild(placeholder);
            }
        });

        if (this.candidateNameDisplay) {
            this.candidateNameDisplay.addEventListener('input', () => {
                this._updateStartButtonState();
            });
            this.candidateNameDisplay.addEventListener('blur', () => {
                const value = (this.candidateNameDisplay.textContent || '').trim();
                this.candidateNameDisplay.textContent = value;
                this._updateStartButtonState();
            });
        }

        if (this.interviewRequirementsInput && this.reqCharCount) {
            const updateCounter = () => {
                const len = this.interviewRequirementsInput.value.length;
                this.reqCharCount.textContent = len;
                const counter = this.reqCharCount.closest('.req-counter');
                if (counter) counter.classList.toggle('at-limit', len >= 100);
            };
            this.interviewRequirementsInput.addEventListener('input', updateCounter);
        }
    }

    _updateStartButtonState() {
        const name = (this.candidateNameDisplay ? this.candidateNameDisplay.textContent || '' : '').trim();
        const nameEntered = name.length > 0;
        const shouldEnable = this.modelsReady && nameEntered;
        this.toggleButton.disabled = !shouldEnable;
        if (!this.modelsReady) {
            // Text managed by updateModelStatusUI — don't override
        } else if (!nameEntered) {
            this.toggleButton.textContent = 'Enter candidate name to start';
        } else if (!this.isCameraActive) {
            this.toggleButton.textContent = 'Start Camera';
        }
    }

    async enumerateCameras() {
        try {
            if (!navigator.mediaDevices?.enumerateDevices) {
                console.warn('enumerateDevices not supported');
                return;
            }

            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(device => device.kind === 'videoinput');

            this.cameraSelect.innerHTML = '';

            if (videoDevices.length === 0) {
                this.cameraSelect.innerHTML = '<option value="">No cameras found</option>';
                return;
            }

            videoDevices.forEach((device, index) => {
                const option = document.createElement('option');
                option.value = device.deviceId;
                option.text = device.label || `Camera ${index + 1}`;
                this.cameraSelect.appendChild(option);
            });

            console.log(`Found ${videoDevices.length} camera(s)`);
        } catch (error) {
            console.error('Error enumerating cameras:', error);
            this.cameraSelect.innerHTML = '<option value="">Error loading cameras</option>';
        }
    }

    async startCamera() {
        try {
            if (!navigator.mediaDevices?.getUserMedia) {
                throw new Error('getUserMedia is not supported');
            }

            // Get selected camera deviceId
            const selectedDeviceId = this.cameraSelect.value;

            // Build constraints with selected camera
            const constraints = {
                audio: true,
                video: selectedDeviceId ? { deviceId: { exact: selectedDeviceId } } : true
            };

            this.localStream = await navigator.mediaDevices.getUserMedia(constraints);

            this.videoElement.srcObject = this.localStream;
            this.videoElement.play().catch(err => console.error('Video play error:', err));

            // Refresh camera list to get proper labels (available after permission granted)
            await this.enumerateCameras();

            this.updateConnectionStatus(true);
            this.isCameraActive = true;
            this.toggleButton.textContent = 'Stop Camera';
            this.toggleButton.classList.remove('btn-primary');
            this.toggleButton.classList.add('btn-danger');

            // Initialize new session
            this.initializeSession();

            // Start recording after a short delay
            setTimeout(() => {
                this.startRecording();
            }, 1000);

            const tracks = this.localStream.getTracks();
            const candidateName = (this.candidateNameDisplay ? this.candidateNameDisplay.textContent || '' : '').trim();
            const targetRole = (this.targetRoleInput ? this.targetRoleInput.value.trim() : '');
            const interviewRequirements = (this.interviewRequirementsInput ? this.interviewRequirementsInput.value.trim() : '');

            if (this.targetRoleInput) this.targetRoleInput.disabled = true;
            if (this.interviewRequirementsInput) this.interviewRequirementsInput.disabled = true;

            this.sendWebSocketMessage('stream_ready', {
                sessionId: this.sessionId,
                candidateName: candidateName,
                targetRole: targetRole,
                interviewRequirements: interviewRequirements,
                status: 'active',
                video: tracks.some(t => t.kind === 'video'),
                audio: tracks.some(t => t.kind === 'audio')
            });

            console.log('Camera started:', this.sessionId);
        } catch (error) {
            console.error('Camera start error:', error);
            this.handleError(error);
        }
    }
    
    startRecording() {
        if (!this.localStream) {
            console.warn('Stream not available');
            return;
        }

        // Find supported MIME type
        const options = [
            'video/webm;codecs=vp9,opus',
            'video/webm;codecs=vp8,opus',
            'video/webm;codecs=vp9',
            'video/webm;codecs=vp8',
            'video/webm',
            'video/mp4'
        ];

        let mimeType = '';
        for (const option of options) {
            if (MediaRecorder.isTypeSupported(option)) {
                mimeType = option;
                break;
            }
        }

        if (!mimeType) {
            console.error('No supported MIME type found');
            return;
        }

        this.isRecording = true;
        this.startRecorderCycle(mimeType);
    }

    startRecorderCycle(mimeType) {
        if (!this.isRecording || !this.localStream) return;

        const recorderOptions = { mimeType };
        if (mimeType.startsWith('video/webm')) {
            recorderOptions.videoBitsPerSecond = 2500000;
        }

        this.mediaRecorder = new MediaRecorder(this.localStream, recorderOptions);

        this.mediaRecorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) {
                const reader = new FileReader();
                reader.onloadend = () => {
                    // Skip if session has been reset (camera was stopped)
                    if (this.sessionId === null || this.chunkIndex === null) {
                        console.log('Skipping chunk upload - session ended');
                        return;
                    }

                    // Validate that we have actual data
                    if (!reader.result || reader.result.byteLength === 0) {
                        console.error('FileReader produced empty result, skipping chunk');
                        return;
                    }

                    const sizeKb = Math.round(event.data.size / 1024);
                    console.log(`Chunk ${this.chunkIndex} uploaded (${sizeKb} KB)`);
                    this.sendWebSocketMessage('video_chunk', {
                        sessionId: this.sessionId,
                        chunk: reader.result,
                        timestamp: Date.now(),
                        chunkIndex: this.chunkIndex++,
                        mimeType,
                        durationMs: this.chunkDurationMs
                    });
                };
                reader.onerror = (error) => {
                    console.error('FileReader error:', error);
                };
                reader.readAsArrayBuffer(event.data);
            }
        };

        this.mediaRecorder.onerror = (event) => {
            console.error('MediaRecorder error:', event.error);
        };

        this.mediaRecorder.onstop = () => {
            if (this.isRecording) {
                this.startRecorderCycle(mimeType);
            }
        };

        this.mediaRecorder.start();
        this.chunkTimer = setTimeout(() => {
            if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
                this.mediaRecorder.stop();
            }
        }, this.chunkDurationMs);
    }

    stopCamera() {
        this.isRecording = false;

        if (this.chunkTimer) {
            clearTimeout(this.chunkTimer);
            this.chunkTimer = null;
        }

        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
            this.mediaRecorder = null;
        }

        if (this.localStream) {
            this.localStream.getTracks().forEach(track => track.stop());
            this.videoElement.srcObject = null;
            this.localStream = null;
        }

        // Notify backend the session has ended before clearing sessionId
        if (this.sessionId) {
            this.sendWebSocketMessage('stream_ended', { sessionId: this.sessionId });
        }

        this.resetSession();
        this.updateConnectionStatus(false);
        this.isCameraActive = false;
        this.toggleButton.classList.remove('btn-danger');
        this.toggleButton.classList.add('btn-primary');
        this._updateStartButtonState();
        this.resetLiveMetrics();
        if (this.targetRoleInput) this.targetRoleInput.disabled = false;
        if (this.interviewRequirementsInput) this.interviewRequirementsInput.disabled = false;
        console.log('Camera stopped');
    }
    

    updateConnectionStatus(connected) {
        if (connected) {
            this.connectionStatus.classList.add('connected');
            this.connectionStatus.querySelector('.text').textContent = 'Camera Active';
        } else {
            this.connectionStatus.classList.remove('connected');
            this.connectionStatus.querySelector('.text').textContent = 'Camera Off';
        }
    }

    handleError(error) {
        const messages = {
            'NotAllowedError': 'Camera access denied. Please allow access in browser settings.',
            'PermissionDeniedError': 'Camera access denied. Please allow access in browser settings.',
            'NotFoundError': 'No camera or microphone found.',
            'DevicesNotFoundError': 'No camera or microphone found.',
            'NotReadableError': 'Camera is already in use by another application.',
            'TrackStartError': 'Camera is already in use by another application.'
        };
        alert(messages[error.name] || `Error: ${error.message || error.name}`);
        this.updateConnectionStatus(false);
        this.isCameraActive = false;
        this.toggleButton.classList.remove('btn-danger');
        this.toggleButton.classList.add('btn-primary');
        this._updateStartButtonState();
        this.sendWebSocketMessage('camera_error', { error: error.name, message: error.message });
    }

    /**
     * Handle generic text stream from backend
     * @param {Object} data - { text, timestamp, sessionId?, metadata? }
     */
    handleTextStream(data) {
        const { text, timestamp, metadata } = data;

        // Always parse stream text for metric cards, even if this message is not displayed.
        this.updateInsightWidgets(text, metadata);

        // Show only interviewer_ollama questions in the right-hand panel.
        // Other sources (whisper, mediapipe, vocaltone, clifton_fusion, etc.)
        // are kept in logs only.
        if (metadata && metadata.source && metadata.source !== 'interviewer_ollama') {
            console.log('[TextStream] Skipping non-interviewer source:', metadata.source);
            return;
        }

        // Initialize TextStreamer if needed
        if (!this.textStreamer) {
            this.textStreamer = new TextStreamer(this.outputContent);
            const placeholder = this.outputContent.querySelector('.output-placeholder');
            if (placeholder) {
                placeholder.remove();
            }
        }

        // Optional: Add header with metadata
        if (metadata) {
            const header = this.createMetadataHeader(metadata, timestamp);
            this.textStreamer.addElement(header);
        }

        // Stream the text
        this.textStreamer.addText(text + '\n\n');
    }

    updateInsightWidgets(text, metadata) {
        const lowerText = (text || '').toLowerCase();
        this.tryUpdateMetricsFromText(text);

        if (metadata && typeof metadata.engagement === 'number' && this.engagementScore) {
            this.engagementScore.textContent = `${Math.max(0, Math.min(100, Math.round(metadata.engagement)))}%`;
        }
        if (metadata && typeof metadata.posture === 'number' && this.postureStability) {
            this.postureStability.textContent = `${Math.max(0, Math.min(100, Math.round(metadata.posture)))}%`;
        }
        if (metadata && typeof metadata.pitch_hz === 'number' && this.voicePitch) {
            this.voicePitch.textContent = `${Math.round(metadata.pitch_hz)} Hz`;
        }
        if (metadata && typeof metadata.tempo_bpm === 'number' && this.voiceTempo) {
            this.voiceTempo.textContent = `${Math.round(metadata.tempo_bpm)} BPM`;
        }

        if (this.sentimentAlert) {
            this.sentimentAlert.className = 'alert-chip neutral';
            this.sentimentAlert.textContent = 'No sentiment alerts';

            if (lowerText.includes('surprise') || lowerText.includes('stress')) {
                this.sentimentAlert.className = 'alert-chip warning';
                this.sentimentAlert.textContent = 'Emotion shift detected';
            }
            if (lowerText.includes('disgust') || lowerText.includes('anger') || lowerText.includes('frustrat')) {
                this.sentimentAlert.className = 'alert-chip danger';
                this.sentimentAlert.textContent = 'Negative sentiment detected';
            }
        }
    }

    updateMetricsFromChunkResults(results) {
        if (!results || typeof results !== 'object') return;

        const mediapipe = results.mediapipe || {};
        const vocaltone = results.vocaltone || {};

        if (typeof mediapipe.engagement_score === 'number' && this.engagementScore) {
            this.engagementScore.textContent = `${Math.round(Math.max(0, Math.min(1, mediapipe.engagement_score)) * 100)}%`;
        }
        if (typeof mediapipe.posture_score === 'number' && this.postureStability) {
            this.postureStability.textContent = `${Math.round(Math.max(0, Math.min(1, mediapipe.posture_score)) * 100)}%`;
        }
        if (typeof vocaltone.pitch_mean === 'number' && this.voicePitch) {
            this.voicePitch.textContent = `${Math.round(vocaltone.pitch_mean)} Hz`;
        }
        if (typeof vocaltone.tempo === 'number' && this.voiceTempo) {
            this.voiceTempo.textContent = `${Math.round(vocaltone.tempo)} BPM`;
        }

        const emotion = (mediapipe.dominant_emotion || vocaltone.emotion || '').toLowerCase();
        if (this.sentimentAlert && emotion) {
            this.sentimentAlert.className = 'alert-chip neutral';
            this.sentimentAlert.textContent = `Emotion: ${emotion}`;

            if (['surprise', 'fear'].includes(emotion)) {
                this.sentimentAlert.className = 'alert-chip warning';
                this.sentimentAlert.textContent = `Watch emotion trend: ${emotion}`;
            }
            if (['disgust', 'anger', 'sad'].includes(emotion)) {
                this.sentimentAlert.className = 'alert-chip danger';
                this.sentimentAlert.textContent = `Negative emotion detected: ${emotion}`;
            }
        }
    }

    tryUpdateMetricsFromText(text) {
        if (!text || typeof text !== 'string') return;

        const engagementMatch = text.match(/engagement:\s*(\d+)%/i);
        const postureMatch = text.match(/posture:\s*(\d+)%/i);
        const pitchMatch = text.match(/pitch:\s*(\d+(?:\.\d+)?)\s*hz/i);
        const tempoMatch = text.match(/tempo:\s*(\d+(?:\.\d+)?)\s*bpm/i);

        if (engagementMatch && this.engagementScore) {
            this.engagementScore.textContent = `${engagementMatch[1]}%`;
        }
        if (postureMatch && this.postureStability) {
            this.postureStability.textContent = `${postureMatch[1]}%`;
        }
        if (pitchMatch && this.voicePitch) {
            this.voicePitch.textContent = `${Math.round(Number(pitchMatch[1]))} Hz`;
        }
        if (tempoMatch && this.voiceTempo) {
            this.voiceTempo.textContent = `${Math.round(Number(tempoMatch[1]))} BPM`;
        }
    }

    resetLiveMetrics() {
        if (this.engagementScore) this.engagementScore.textContent = '--%';
        if (this.postureStability) this.postureStability.textContent = '--%';
        if (this.voicePitch) this.voicePitch.textContent = '-';
        if (this.voiceTempo) this.voiceTempo.textContent = '-';
        if (this.sentimentAlert) {
            this.sentimentAlert.className = 'alert-chip neutral';
            this.sentimentAlert.textContent = 'No sentiment alerts';
        }
    }

    /**
     * Create a metadata header element for streamed text
     * @param {Object} metadata - Metadata object (chunk, source, model, etc.)
     * @param {string} timestamp - ISO timestamp string
     * @returns {HTMLElement} Header element
     */
    createMetadataHeader(metadata, timestamp) {
        const header = document.createElement('div');
        header.className = 'stream-header';

        let headerText = '';
        if (metadata.chunk !== undefined) {
            headerText += `Chunk ${metadata.chunk}`;
        }
        if (metadata.source) {
            headerText += ` | ${metadata.source}`;
        }
        if (timestamp) {
            const date = new Date(timestamp);
            headerText += ` | ${date.toLocaleTimeString()}`;
        }

        if (headerText) {
            header.textContent = '\n' + headerText + '\n';
        }

        return header;
    }

    /**
     * Handle incoming chunk results from backend
     * Formats results as text and streams using generic handler
     */
    handleChunkResults(data) {
        const { chunkIndex, results, timestamp } = data;

        console.log(`[Output] Chunk ${chunkIndex} received - Models: ${Object.keys(results).join(', ')}`);

        // Format results as text and stream
        let text = '';
        for (const [modelName, modelData] of Object.entries(results)) {
            text += `[${modelName.toUpperCase()}]\n`;
            text += JSON.stringify(modelData, null, 2) + '\n\n';
        }

        this.handleTextStream({
            text,
            timestamp,
            metadata: { chunk: chunkIndex, source: 'chunk_results' }
        });
    }

    /**
     * Start checking model preload status from backend
     */
    startModelStatusCheck() {
        this.checkModelStatus(); // Check immediately

        // Poll every 3 seconds until all models are ready
        this.modelsCheckInterval = setInterval(() => {
            if (!this.modelsReady) {
                this.checkModelStatus();
            } else {
                // Stop polling once all models are ready
                clearInterval(this.modelsCheckInterval);
                this.modelsCheckInterval = null;
            }
        }, 3000);
    }

    /**
     * Check model status from backend API
     */
    async checkModelStatus() {
        try {
            const { protocol, hostname } = window.location;
            const backendPort = '5555';
            const response = await fetch(`${protocol}//${hostname}:${backendPort}/api/models/status`);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.updateModelStatusUI(data);
        } catch (error) {
            console.error('Failed to check model status:', error);
            this.updateModelStatusUI({ all_ready: false, error: true });
        }
    }

    /**
     * Update model status UI based on backend response
     * @param {Object} status - Status object from backend
     */
    updateModelStatusUI(status) {
        if (!this.modelsStatus) return;

        const statusText = this.modelsStatus.querySelector('.status-text');

        // Check if any models are actively loading
        let anyLoading = false;
        let hasError = false;

        for (const [modelName, modelStatus] of Object.entries(status)) {
            if (modelName === 'all_ready') continue;
            if (modelStatus.loading) anyLoading = true;
            if (modelStatus.error) hasError = true;
        }

        if (status.error) {
            this.modelsStatus.className = 'models-status error';
            statusText.textContent = 'Model status unavailable';
            // Allow camera to work even if status check failed
            if (!anyLoading) {
                this.modelsReady = true;
                this.toggleButton.textContent = 'Start Camera (Models may load at runtime)';
                this._updateStartButtonState();
            }
            return;
        }

        if (status.all_ready) {
            this.modelsReady = true;
            this.modelsStatus.className = 'models-status ready';
            statusText.textContent = 'All models ready';
            this._updateStartButtonState();
        } else {
            // Check which models are loading
            const loadingModels = [];
            const readyModels = [];

            for (const [modelName, modelStatus] of Object.entries(status)) {
                if (modelName === 'all_ready') continue;

                if (modelStatus.loading) {
                    loadingModels.push(modelName);
                } else if (modelStatus.ready) {
                    readyModels.push(modelName);
                }
            }

            this.modelsStatus.className = 'models-status';

            // Special handling for Ollama progress
            if (status.ollama && status.ollama.loading && status.ollama.progress !== undefined) {
                const progress = status.ollama.progress;
                const statusTextStr = status.ollama.status_text || 'downloading';
                statusText.textContent = `Downloading Ollama model: ${progress}% (${statusTextStr})`;
                this.toggleButton.textContent = `Loading Models... ${progress}%`;
                this.toggleButton.disabled = true;
            } else if (loadingModels.length > 0) {
                statusText.textContent = `Loading models... (${readyModels.length}/${readyModels.length + loadingModels.length} ready)`;
                this.toggleButton.textContent = `Loading Models... (${readyModels.length}/${readyModels.length + loadingModels.length})`;
                this.toggleButton.disabled = true;
            } else if (hasError) {
                // Models finished loading but some failed — allow camera anyway
                statusText.textContent = 'Some models failed to load';
                this.modelsReady = true;
                this.toggleButton.textContent = 'Start Camera (Some models unavailable)';
                this._updateStartButtonState();
            } else {
                statusText.textContent = 'Initializing models...';
                this.toggleButton.textContent = 'Start Camera';
                this.toggleButton.disabled = true;
            }
        }
    }

}

document.addEventListener('DOMContentLoaded', () => {
    window.app = new VideoApp();
});
