/**
 * Video Capture App - Blueprint
 *
 * This file provides the structure for implementing WebRTC camera functionality.
 * Team members should implement the methods below to add camera capture.
 */

class VideoApp {
    constructor() {
        this.localStream = null;
        this.videoElement = document.getElementById('video-preview');
        this.connectionStatus = document.getElementById('connection-status');
        this.startButton = document.getElementById('btn-start-camera');
        this.stopButton = document.getElementById('btn-stop-camera');

        this.initializeEventListeners();
        console.log('Video App initialized - ready for implementation');
    }

    /**
     * Initialize button event listeners
     */
    initializeEventListeners() {
        this.startButton.addEventListener('click', () => {
            this.startCamera();
        });

        this.stopButton.addEventListener('click', () => {
            this.stopCamera();
        });
    }

    /**
     * TODO: Implement camera start functionality
     *
     * Steps to implement:
     * 1. Call navigator.mediaDevices.getUserMedia() with video/audio constraints
     * 2. Assign the stream to this.localStream
     * 3. Set this.videoElement.srcObject to the stream
     * 4. Update UI with this.updateConnectionStatus(true)
     * 5. Enable/disable appropriate buttons
     * 6. Handle errors appropriately
     */
    async startCamera() {
        console.log('TODO: Implement camera start');
        alert('Camera functionality not yet implemented. This is a placeholder for future development.');
    }

    /**
     * TODO: Implement camera stop functionality
     *
     * Steps to implement:
     * 1. Stop all tracks in this.localStream
     * 2. Set this.videoElement.srcObject to null
     * 3. Clear this.localStream
     * 4. Update UI with this.updateConnectionStatus(false)
     * 5. Enable/disable appropriate buttons
     */
    stopCamera() {
        console.log('TODO: Implement camera stop');
    }

    /**
     * Update connection status indicator
     */
    updateConnectionStatus(connected) {
        if (connected) {
            this.connectionStatus.classList.add('connected');
            this.connectionStatus.querySelector('.text').textContent = 'Camera Active';
        } else {
            this.connectionStatus.classList.remove('connected');
            this.connectionStatus.querySelector('.text').textContent = 'Camera Off';
        }
    }

    /**
     * TODO: Implement error handling
     *
     * Handle common errors:
     * - NotAllowedError: Permission denied
     * - NotFoundError: No camera/microphone found
     * - NotReadableError: Device already in use
     * - OverconstrainedError: Constraints not supported
     */
    handleError(error) {
        console.error('Camera error:', error);
        alert('Error: ' + error.message);
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM ready, initializing app...');
    const app = new VideoApp();
    window.app = app; // For debugging
});
