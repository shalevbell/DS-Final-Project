/**
 * Video Capture App with WebRTC and WebSocket
 *
 * This file implements WebRTC camera functionality with WebSocket communication.
 */

class VideoApp {
    constructor() {
        this.videoElement = document.getElementById('video-preview');
        this.connectionStatus = document.getElementById('connection-status');
        this.toggleButton = document.getElementById('btn-toggle-camera');
        this.permissionInfo = document.getElementById('permission-info');
        this.cameraSelect = document.getElementById('camera-select');
        this.localStream = null;
        this.mediaRecorder = null;
        this.socket = null;
        this.socketConnected = false;
        this.mediaConstraints = { video: true, audio: true };
        this.chunkTimer = null;
        this.isRecording = false;
        this.isCameraActive = false;
        this.chunkDurationMs = 30000;  // Default, will be overridden by backend

        // Session-specific state (initialized when camera starts)
        this.sessionId = null;
        this.chunkIndex = null;

        this.initializeWebSocket();
        this.initializeEventListeners();
        this.enumerateCameras();
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
        this.socket = io(`${protocol}//${hostname}:${backendPort}`, {
            transports: ['websocket'],
            reconnection: true,
            reconnectionDelay: 1000,
            reconnectionAttempts: 5
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
            this.sendWebSocketMessage('stream_ready', {
                sessionId: this.sessionId,
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

        this.resetSession();
        this.updateConnectionStatus(false);
        this.isCameraActive = false;
        this.toggleButton.textContent = 'Start Camera';
        this.toggleButton.classList.remove('btn-danger');
        this.toggleButton.classList.add('btn-primary');
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
        this.toggleButton.textContent = 'Start Camera';
        this.toggleButton.classList.remove('btn-danger');
        this.toggleButton.classList.add('btn-primary');
        this.sendWebSocketMessage('camera_error', { error: error.name, message: error.message });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.app = new VideoApp();
});
