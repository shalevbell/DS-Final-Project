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
        this.stressFaces = document.getElementById('stress-faces');
        this.stressMeterPointer = document.getElementById('stress-meter-pointer');
        this.stressLevelValue = document.getElementById('stress-level-value');
        this.stressLevelDesc = document.getElementById('stress-level-desc');
        this.eyeContactIndicator = document.getElementById('eye-contact-indicator');
        this.eyeContactLabel = document.getElementById('eye-contact-label');
        this.bodyStabilityIndicator = document.getElementById('body-stability-indicator');
        this.bodyStabilityLabel = document.getElementById('body-stability-label');
        this.voiceLoudnessLight = document.getElementById('voice-loudness-light');
        this.voiceLoudnessValue = document.getElementById('voice-loudness-value');
        this.voiceLoudnessDesc = document.getElementById('voice-loudness-desc');
        this.voiceFlowLight = document.getElementById('voice-flow-light');
        this.voiceFlowValue = document.getElementById('voice-flow-value');
        this.voiceFlowDesc = document.getElementById('voice-flow-desc');
        this.voiceConfidenceLight = document.getElementById('voice-confidence-light');
        this.voiceConfidenceValue = document.getElementById('voice-confidence-value');
        this.voiceConfidenceDesc = document.getElementById('voice-confidence-desc');
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

        // Resume upload elements
        this.resumeFileInput = document.getElementById('resume-file-input');
        this.resumeUploadBtn = document.getElementById('resume-upload-btn');
        this.resumeChip = document.getElementById('resume-chip');
        this.resumeFilename = document.getElementById('resume-filename');
        this.resumeRemoveBtn = document.getElementById('resume-remove-btn');

        // Disable camera button until models are ready
        this.toggleButton.disabled = true;
        this.toggleButton.textContent = 'Loading Models...';

        this.initializeWebSocket();
        this.initializeEventListeners();
        this.enumerateCameras();
        this.startModelStatusCheck();
        this._initResumeUpload();

        // Reliably close the session when navigating away mid-interview
        window.addEventListener('beforeunload', () => {
            if (this.isCameraActive && this.sessionId) {
                // Try socket first (works for same-tab navigation)
                this.sendWebSocketMessage('stream_ended', { sessionId: this.sessionId });
                // sendBeacon survives page unload reliably
                const { protocol, hostname, port } = window.location;
                const _origin = port ? `${protocol}//${hostname}:${port}` : `${protocol}//${hostname}`;
                navigator.sendBeacon(
                    `${_origin}/api/sessions/${encodeURIComponent(this.sessionId)}/complete`
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
        const { protocol, hostname, port } = window.location;
        const origin = port ? `${protocol}//${hostname}:${port}` : `${protocol}//${hostname}`;
        // Long timeouts so connection survives slow model runs (2–3 min per chunk)
        this.socket = io(origin, {
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

            let resumeData = null;
            try {
                const raw = localStorage.getItem('helperviewer_resume');
                if (raw) resumeData = JSON.parse(raw).data || null;
            } catch { /* localStorage unavailable or malformed */ }

            this.sendWebSocketMessage('stream_ready', {
                sessionId: this.sessionId,
                candidateName: candidateName,
                targetRole: targetRole,
                interviewRequirements: interviewRequirements,
                status: 'active',
                video: tracks.some(t => t.kind === 'video'),
                audio: tracks.some(t => t.kind === 'audio'),
                resumeData: resumeData
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

        // Show interviewer_ollama and resume_questions in the right-hand panel.
        // Other sources (whisper, mediapipe, vocaltone, clifton_fusion, etc.)
        // are kept in logs only.
        const displayedSources = ['interviewer_ollama', 'resume_questions'];
        if (metadata && metadata.source && !displayedSources.includes(metadata.source)) {
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

        if (metadata && metadata.source === 'mediapipe') {
            this.updateMediapipeInsights(metadata);
        }
        if (metadata && metadata.source === 'vocaltone') {
            this.updateVoiceProfileFromVocaltone(metadata);
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

        this.updateMediapipeInsights(mediapipe);
        this.updateVoiceProfileFromVocaltone(vocaltone);

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

        const stressMatch = text.match(/stress:\s*(\d+)%/i);
        const eyeContactMatch = text.match(/eyecontact:\s*(yes|no)/i);
        const bodyStableMatch = text.match(/bodystable:\s*(yes|no)/i);
        const pitchMatch = text.match(/pitch:\s*(\d+(?:\.\d+)?)\s*hz/i);
        const tempoMatch = text.match(/tempo:\s*(\d+(?:\.\d+)?)\s*bpm/i);

        if (stressMatch || eyeContactMatch || bodyStableMatch) {
            const mediapipeData = {};
            if (stressMatch) mediapipeData.stress_level = Number(stressMatch[1]);
            if (eyeContactMatch) mediapipeData.eye_contact = eyeContactMatch[1].toLowerCase() === 'yes';
            if (bodyStableMatch) mediapipeData.body_stable = bodyStableMatch[1].toLowerCase() === 'yes';
            this.updateMediapipeInsights(mediapipeData);
        }

        if (pitchMatch || tempoMatch) {
            const confidenceMatch = text.match(/\((\d+)%\)/i);
            const vocaltoneData = {};
            if (tempoMatch) vocaltoneData.tempo_bpm = Number(tempoMatch[1]);
            if (confidenceMatch) vocaltoneData.confidence = Number(confidenceMatch[1]) / 100;
            this.updateVoiceProfileFromVocaltone(vocaltoneData);
        }
    }

    updateMediapipeInsights(data) {
        if (!data || typeof data !== 'object') return;

        if (typeof data.stress_level === 'number') {
            const stress = Math.max(0, Math.min(100, Math.round(data.stress_level)));
            let level = 'neutral';
            let desc = 'Neutral';

            if (stress <= 25) {
                level = 'calm';
                desc = 'Calm';
            } else if (stress <= 50) {
                level = 'neutral';
                desc = 'Mild tension';
            } else if (stress <= 75) {
                level = 'tense';
                desc = 'Elevated stress';
            } else {
                level = 'high';
                desc = 'High stress';
            }

            if (this.stressFaces) {
                this.stressFaces.querySelectorAll('.stress-face').forEach((face) => {
                    face.classList.toggle('active', face.dataset.level === level);
                });
            }
            if (this.stressMeterPointer) {
                this.stressMeterPointer.style.left = `${stress}%`;
            }
            if (this.stressLevelValue) {
                this.stressLevelValue.textContent = `${stress}%`;
            }
            if (this.stressLevelDesc) {
                this.stressLevelDesc.textContent = desc;
            }
        }

        if (typeof data.eye_contact === 'boolean') {
            this._setStatusIndicator(
                this.eyeContactIndicator,
                this.eyeContactLabel,
                data.eye_contact,
                'Maintaining eye contact',
                'Limited eye contact'
            );
        }

        if (typeof data.body_stable === 'boolean') {
            this._setStatusIndicator(
                this.bodyStabilityIndicator,
                this.bodyStabilityLabel,
                data.body_stable,
                'Stable posture',
                'Frequent movement'
            );
        }
    }

    _setStatusIndicator(indicatorEl, labelEl, isPositive, yesText, noText) {
        if (indicatorEl) {
            indicatorEl.className = `status-indicator ${isPositive ? 'yes' : 'no'}`;
            indicatorEl.textContent = isPositive ? '✓' : '✕';
        }
        if (labelEl) {
            labelEl.textContent = isPositive ? yesText : noText;
        }
    }

    updateVoiceProfileFromVocaltone(data) {
        if (!data || typeof data !== 'object') return;

        const energy = typeof data.energy_level === 'number' ? data.energy_level : null;
        const tempo = typeof data.tempo_bpm === 'number'
            ? data.tempo_bpm
            : (typeof data.tempo === 'number' ? data.tempo : null);
        const confidence = typeof data.confidence === 'number' ? data.confidence : null;

        if (energy !== null && energy > 0) {
            const volumePct = Math.round(Math.min(energy / 0.12, 1) * 100);
            const state = volumePct < 20 ? 'red' : (volumePct > 75 ? 'yellow' : 'green');
            const desc = state === 'red' ? 'low' : (state === 'yellow' ? 'elevated' : 'balanced');
            this._setVoiceProfileCard(
                this.voiceLoudnessLight,
                this.voiceLoudnessValue,
                this.voiceLoudnessDesc,
                state,
                `${volumePct}%`,
                desc
            );
        }

        if (tempo !== null && tempo > 0) {
            const roundedTempo = Math.round(tempo);
            const state = roundedTempo < 70 ? 'red' : (roundedTempo > 145 ? 'yellow' : 'green');
            const desc = state === 'red' ? 'slow' : (state === 'yellow' ? 'fast' : 'steady');
            this._setVoiceProfileCard(
                this.voiceFlowLight,
                this.voiceFlowValue,
                this.voiceFlowDesc,
                state,
                `${roundedTempo} BPM`,
                desc
            );
        }

        if (confidence !== null && confidence > 0) {
            const confidencePct = Math.round(Math.max(0, Math.min(1, confidence)) * 100);
            const state = confidencePct < 50 ? 'red' : (confidencePct < 75 ? 'yellow' : 'green');
            const desc = state === 'red' ? 'uncertain' : (state === 'yellow' ? 'moderate' : 'strong');
            this._setVoiceProfileCard(
                this.voiceConfidenceLight,
                this.voiceConfidenceValue,
                this.voiceConfidenceDesc,
                state,
                `${confidencePct}%`,
                desc
            );
        }
    }

    _setVoiceProfileCard(lightEl, valueEl, descEl, state, valueText, descText) {
        if (lightEl) lightEl.dataset.state = state;
        if (valueEl) valueEl.textContent = valueText;
        if (descEl) descEl.textContent = descText;
    }

    resetLiveMetrics() {
        if (this.stressFaces) {
            this.stressFaces.querySelectorAll('.stress-face').forEach((face) => face.classList.remove('active'));
        }
        if (this.stressMeterPointer) this.stressMeterPointer.style.left = '0%';
        if (this.stressLevelValue) this.stressLevelValue.textContent = '--';
        if (this.stressLevelDesc) this.stressLevelDesc.textContent = 'Waiting for camera...';
        if (this.eyeContactIndicator) {
            this.eyeContactIndicator.className = 'status-indicator unknown';
            this.eyeContactIndicator.textContent = '-';
        }
        if (this.eyeContactLabel) this.eyeContactLabel.textContent = 'Waiting for camera...';
        if (this.bodyStabilityIndicator) {
            this.bodyStabilityIndicator.className = 'status-indicator unknown';
            this.bodyStabilityIndicator.textContent = '-';
        }
        if (this.bodyStabilityLabel) this.bodyStabilityLabel.textContent = 'Waiting for camera...';
        this._setVoiceProfileCard(this.voiceLoudnessLight, this.voiceLoudnessValue, this.voiceLoudnessDesc, 'none', '-', '-');
        this._setVoiceProfileCard(this.voiceFlowLight, this.voiceFlowValue, this.voiceFlowDesc, 'none', '-', '-');
        this._setVoiceProfileCard(this.voiceConfidenceLight, this.voiceConfidenceValue, this.voiceConfidenceDesc, 'none', '-', '-');
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
            const { protocol, hostname, port } = window.location;
            const origin = port ? `${protocol}//${hostname}:${port}` : `${protocol}//${hostname}`;
            const response = await fetch(`${origin}/api/models/status`);

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

    _initResumeUpload() {
        this._loadResumeFromStorage();

        this.resumeUploadBtn.addEventListener('click', () => {
            this.resumeFileInput.click();
        });

        this.resumeFileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (!file) return;
            this._handleResumeFile(file);
            this.resumeFileInput.value = '';
        });

        this.resumeRemoveBtn.addEventListener('click', () => {
            this._removeResume();
        });
    }

    _handleResumeFile(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                localStorage.setItem('helperviewer_resume', JSON.stringify({
                    name: file.name,
                    data: e.target.result
                }));
            } catch {
                // localStorage quota exceeded — chip still shows for this page session
            }
            this._showResumeChip(file.name);
        };
        reader.readAsDataURL(file);
    }

    _loadResumeFromStorage() {
        try {
            const raw = localStorage.getItem('helperviewer_resume');
            if (raw) {
                const { name } = JSON.parse(raw);
                this._showResumeChip(name);
            }
        } catch {
            localStorage.removeItem('helperviewer_resume');
        }
    }

    _showResumeChip(filename) {
        this.resumeFilename.textContent = filename;
        this.resumeUploadBtn.style.display = 'none';
        this.resumeChip.style.display = 'inline-flex';
    }

    _removeResume() {
        localStorage.removeItem('helperviewer_resume');
        this.resumeChip.style.display = 'none';
        this.resumeUploadBtn.style.display = '';
        this.resumeFilename.textContent = '';
    }

}

document.addEventListener('DOMContentLoaded', () => {
    window.app = new VideoApp();
});
