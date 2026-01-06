/**
 * Video Capture App with WebRTC and WebSocket
 *
 * This file implements WebRTC camera functionality with WebSocket communication.
 */

class VideoApp {
    constructor() {
        this.videoElement = document.getElementById('video-preview');
        this.connectionStatus = document.getElementById('connection-status');
        this.startButton = document.getElementById('btn-start-camera');
        this.stopButton = document.getElementById('btn-stop-camera');
        this.permissionInfo = document.getElementById('permission-info');
        this.localStream = null;
        this.mediaRecorder = null;
        this.sessionId = null;
        this.socket = null;
        this.socketConnected = false;
        this.mediaConstraints = { video: true, audio: true };
        this.initializeWebSocket();
        this.initializeEventListeners();
    }

    initializeWebSocket() {
        if (typeof io === 'undefined') return;
        const { protocol, hostname, port } = window.location;
        const backendPort = (hostname === 'localhost' || hostname === '127.0.0.1') 
            ? (port === '3000' ? '5555' : (port || '5000'))
            : (port || '5555');
        this.socket = io(`${protocol}//${hostname}:${backendPort}`, {
            transports: ['websocket', 'polling'],
            reconnection: true,
            reconnectionDelay: 1000,
            reconnectionAttempts: 5
        });
        this.socket.on('connect', () => {
            this.socketConnected = true;
            console.log('✅ WebSocket connected');
            this.socket.emit('request_camera_permission');
        });
        this.socket.on('disconnect', () => {
            this.socketConnected = false;
            console.log('❌ WebSocket disconnected');
        });
        this.socket.on('connect_error', (error) => {
            this.socketConnected = false;
            console.error('❌ WebSocket connection error:', error);
        });
        this.socket.on('redis_test_result', (data) => {
            console.log('📊 Redis test result:', data);
            if (data.status === 'success') {
                console.log('✅ Redis connection is working!');
            } else {
                console.error('❌ Redis connection failed:', data.message);
            }
        });
    }
    
    sendWebSocketMessage(event, data = {}) {
        if (this.socket && this.socketConnected) {
            this.socket.emit(event, data);
        }
    }

    initializeEventListeners() {
        this.startButton.addEventListener('click', () => this.startCamera());
        this.stopButton.addEventListener('click', () => this.stopCamera());
    }

    async startCamera() {
        try {
            if (!navigator.mediaDevices?.getUserMedia) {
                throw new Error('getUserMedia is not supported');
            }
            
            console.log('Requesting camera access...');
            this.localStream = await navigator.mediaDevices.getUserMedia(this.mediaConstraints);
            console.log('Camera access granted');
            
            this.videoElement.srcObject = this.localStream;
            this.videoElement.play().catch(err => console.error('Video play error:', err));
            
            this.updateConnectionStatus(true);
            this.startButton.disabled = true;
            this.stopButton.disabled = false;
            
            // Generate session ID
            this.sessionId = `stream_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
            
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
        } catch (error) {
            console.error('Camera start error:', error);
            this.handleError(error);
        }
    }
    
    startRecording() {
        if (!this.localStream) {
            console.warn('Stream not available for recording');
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
            console.warn('No supported MIME type found for MediaRecorder');
            return;
        }
        
        try {
            const recorderOptions = { mimeType };
            if (mimeType.startsWith('video/webm')) {
                recorderOptions.videoBitsPerSecond = 2500000;
            }
            
            this.mediaRecorder = new MediaRecorder(this.localStream, recorderOptions);
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data && event.data.size > 0) {
                    const reader = new FileReader();
                    reader.onloadend = () => {
                        const base64data = reader.result.split(',')[1];
                        this.sendWebSocketMessage('video_chunk', {
                            sessionId: this.sessionId,
                            chunk: base64data,
                            timestamp: Date.now()
                        });
                    };
                    reader.onerror = () => console.error('FileReader error');
                    reader.readAsDataURL(event.data);
                }
            };
            
            this.mediaRecorder.onerror = (event) => {
                console.error('MediaRecorder error:', event.error);
            };
            
            this.mediaRecorder.onstop = () => {
                console.log('MediaRecorder stopped');
            };
            
            // Start recording and send chunks every 20 seconds
            this.mediaRecorder.start(20000);
            console.log('MediaRecorder started with MIME type:', mimeType);
        } catch (error) {
            console.error('Failed to start MediaRecorder:', error);
            alert('Failed to start video recording: ' + error.message);
        }
    }

    stopCamera() {
        console.log('🛑 Stopping camera...');
        
        // Stop MediaRecorder
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
            this.mediaRecorder = null;
            console.log('✅ MediaRecorder stopped');
        }
        
        if (this.localStream) {
            this.localStream.getTracks().forEach(track => track.stop());
            this.videoElement.srcObject = null;
            this.localStream = null;
            console.log('✅ Stream tracks stopped');
            
            // Test Redis connection via server
            if (this.sessionId) {
                console.log(`📡 Testing Redis connection for session: ${this.sessionId}`);
                this.sendWebSocketMessage('test_redis', {
                    sessionId: this.sessionId
                });
                this.sessionId = null;
            }
            
            this.sendWebSocketMessage('camera_status', { status: 'stopped' });
        }
        this.updateConnectionStatus(false);
        this.startButton.disabled = false;
        this.stopButton.disabled = true;
        console.log('✅ Camera stopped');
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
        this.startButton.disabled = false;
        this.stopButton.disabled = true;
        this.sendWebSocketMessage('camera_error', { error: error.name, message: error.message });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.app = new VideoApp();
});
