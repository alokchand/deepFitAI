/**
 * Android WebView Camera Fix
 * Handles camera access, video recording, and photo capture for Android WebView
 */

class WebViewCameraManager {
    constructor() {
        this.isWebView = this.detectWebView();
        this.mediaRecorder = null;
        this.recordedChunks = [];
        this.stream = null;
        this.isRecording = false;
        this.photoCanvas = null;
        this.videoElement = null;
        
        console.log('WebView detected:', this.isWebView);
        this.initializeCamera();
    }

    detectWebView() {
        const userAgent = navigator.userAgent;
        const isAndroid = /Android/i.test(userAgent);
        const isWebView = /wv|WebView/i.test(userAgent) || 
                         window.AndroidInterface !== undefined ||
                         /Version\/[\d.]+.*Chrome\/[.\d]+ Mobile/i.test(userAgent);
        
        return isAndroid && isWebView;
    }

    async initializeCamera() {
        try {
            // Enhanced constraints for Android WebView
            const constraints = {
                video: {
                    width: { ideal: 1280, max: 1920 },
                    height: { ideal: 720, max: 1080 },
                    frameRate: { ideal: 30, max: 60 },
                    facingMode: 'user', // Front camera by default
                    aspectRatio: { ideal: 16/9 }
                },
                audio: false // Disable audio for better compatibility
            };

            // For WebView, try different constraint combinations
            if (this.isWebView) {
                console.log('Applying WebView-specific camera constraints');
                
                // Try multiple constraint sets for better compatibility
                const webViewConstraints = [
                    // High quality
                    {
                        video: {
                            width: 1280,
                            height: 720,
                            frameRate: 30,
                            facingMode: 'user'
                        }
                    },
                    // Medium quality fallback
                    {
                        video: {
                            width: 640,
                            height: 480,
                            frameRate: 30,
                            facingMode: 'user'
                        }
                    },
                    // Basic fallback
                    {
                        video: {
                            facingMode: 'user'
                        }
                    },
                    // Minimal fallback
                    {
                        video: true
                    }
                ];

                for (const constraint of webViewConstraints) {
                    try {
                        this.stream = await navigator.mediaDevices.getUserMedia(constraint);
                        console.log('Camera initialized with constraints:', constraint);
                        break;
                    } catch (error) {
                        console.warn('Failed with constraint:', constraint, error);
                        continue;
                    }
                }
            } else {
                this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            }

            if (!this.stream) {
                throw new Error('Failed to initialize camera with any constraints');
            }

            return this.stream;
        } catch (error) {
            console.error('Camera initialization failed:', error);
            this.handleCameraError(error);
            throw error;
        }
    }

    async startCamera(videoElement) {
        try {
            this.videoElement = videoElement;
            
            if (!this.stream) {
                await this.initializeCamera();
            }

            videoElement.srcObject = this.stream;
            
            // WebView-specific video element setup
            if (this.isWebView) {
                videoElement.setAttribute('playsinline', 'true');
                videoElement.setAttribute('webkit-playsinline', 'true');
                videoElement.muted = true;
                videoElement.autoplay = true;
            }

            return new Promise((resolve, reject) => {
                videoElement.onloadedmetadata = () => {
                    videoElement.play()
                        .then(() => {
                            console.log('Camera started successfully');
                            resolve(this.stream);
                        })
                        .catch(reject);
                };
                
                videoElement.onerror = reject;
                
                // Timeout fallback
                setTimeout(() => {
                    if (videoElement.readyState === 0) {
                        reject(new Error('Camera startup timeout'));
                    }
                }, 10000);
            });
        } catch (error) {
            console.error('Failed to start camera:', error);
            throw error;
        }
    }

    async stopCamera() {
        try {
            if (this.isRecording) {
                await this.stopRecording();
            }

            if (this.stream) {
                this.stream.getTracks().forEach(track => {
                    track.stop();
                    console.log('Stopped track:', track.kind);
                });
                this.stream = null;
            }

            if (this.videoElement) {
                this.videoElement.srcObject = null;
                this.videoElement = null;
            }

            console.log('Camera stopped successfully');
        } catch (error) {
            console.error('Error stopping camera:', error);
        }
    }

    async capturePhoto(videoElement, canvas) {
        try {
            if (!videoElement || !this.stream) {
                throw new Error('Camera not initialized');
            }

            // Create canvas if not provided
            if (!canvas) {
                canvas = document.createElement('canvas');
                this.photoCanvas = canvas;
            }

            const context = canvas.getContext('2d');
            
            // Set canvas dimensions to match video
            canvas.width = videoElement.videoWidth || 640;
            canvas.height = videoElement.videoHeight || 480;

            // Draw current video frame to canvas
            context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

            // Convert to blob for better WebView compatibility
            return new Promise((resolve, reject) => {
                canvas.toBlob((blob) => {
                    if (blob) {
                        const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
                        resolve({
                            blob: blob,
                            dataUrl: dataUrl,
                            width: canvas.width,
                            height: canvas.height
                        });
                    } else {
                        reject(new Error('Failed to capture photo'));
                    }
                }, 'image/jpeg', 0.8);
            });
        } catch (error) {
            console.error('Photo capture failed:', error);
            throw error;
        }
    }

    async startRecording() {
        try {
            if (!this.stream) {
                throw new Error('Camera not initialized');
            }

            if (this.isRecording) {
                console.warn('Recording already in progress');
                return;
            }

            this.recordedChunks = [];

            // WebView-compatible MediaRecorder options
            const options = {
                mimeType: this.getSupportedMimeType(),
                videoBitsPerSecond: this.isWebView ? 1000000 : 2500000 // Lower bitrate for WebView
            };

            this.mediaRecorder = new MediaRecorder(this.stream, options);

            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data && event.data.size > 0) {
                    this.recordedChunks.push(event.data);
                }
            };

            this.mediaRecorder.onstart = () => {
                console.log('Recording started');
                this.isRecording = true;
            };

            this.mediaRecorder.onstop = () => {
                console.log('Recording stopped');
                this.isRecording = false;
            };

            this.mediaRecorder.onerror = (error) => {
                console.error('MediaRecorder error:', error);
                this.isRecording = false;
            };

            this.mediaRecorder.start(1000); // Collect data every second
            return true;
        } catch (error) {
            console.error('Failed to start recording:', error);
            throw error;
        }
    }

    async stopRecording() {
        try {
            if (!this.mediaRecorder || !this.isRecording) {
                console.warn('No active recording to stop');
                return null;
            }

            return new Promise((resolve, reject) => {
                this.mediaRecorder.onstop = () => {
                    try {
                        const blob = new Blob(this.recordedChunks, {
                            type: this.getSupportedMimeType()
                        });

                        const videoUrl = URL.createObjectURL(blob);
                        
                        resolve({
                            blob: blob,
                            url: videoUrl,
                            duration: this.getRecordingDuration(),
                            size: blob.size
                        });
                    } catch (error) {
                        reject(error);
                    }
                };

                this.mediaRecorder.stop();
            });
        } catch (error) {
            console.error('Failed to stop recording:', error);
            throw error;
        }
    }

    getSupportedMimeType() {
        const types = [
            'video/webm;codecs=vp9',
            'video/webm;codecs=vp8',
            'video/webm',
            'video/mp4;codecs=h264',
            'video/mp4'
        ];

        for (const type of types) {
            if (MediaRecorder.isTypeSupported(type)) {
                console.log('Using MIME type:', type);
                return type;
            }
        }

        console.warn('No supported MIME type found, using default');
        return 'video/webm';
    }

    getRecordingDuration() {
        // Calculate duration based on chunks (approximate)
        return this.recordedChunks.length; // seconds (approximate)
    }

    async switchCamera() {
        try {
            if (!this.stream) {
                throw new Error('Camera not initialized');
            }

            const videoTrack = this.stream.getVideoTracks()[0];
            const currentFacingMode = videoTrack.getSettings().facingMode;
            const newFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';

            // Stop current stream
            await this.stopCamera();

            // Start with new facing mode
            const constraints = {
                video: {
                    facingMode: newFacingMode,
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            };

            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            
            if (this.videoElement) {
                this.videoElement.srcObject = this.stream;
            }

            return newFacingMode;
        } catch (error) {
            console.error('Failed to switch camera:', error);
            // Fallback to original camera
            await this.initializeCamera();
            throw error;
        }
    }

    handleCameraError(error) {
        let message = 'Camera access failed';
        
        if (error.name === 'NotAllowedError') {
            message = 'Camera permission denied. Please enable camera access in your browser settings.';
        } else if (error.name === 'NotFoundError') {
            message = 'No camera found on this device.';
        } else if (error.name === 'NotSupportedError') {
            message = 'Camera not supported in this browser.';
        } else if (error.name === 'NotReadableError') {
            message = 'Camera is being used by another application.';
        }

        console.error('Camera error:', message, error);
        
        // Show user-friendly error message
        this.showErrorMessage(message);
    }

    showErrorMessage(message) {
        // Create error overlay
        const errorOverlay = document.createElement('div');
        errorOverlay.className = 'camera-error-overlay';
        errorOverlay.innerHTML = `
            <div class="error-content">
                <i class="fas fa-exclamation-triangle"></i>
                <h3>Camera Error</h3>
                <p>${message}</p>
                <button onclick="this.parentElement.parentElement.remove()" class="btn btn-primary">
                    OK
                </button>
            </div>
        `;

        // Add styles
        errorOverlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 10000;
        `;

        document.body.appendChild(errorOverlay);
    }

    // Utility methods for WebView detection and optimization
    isWebViewEnvironment() {
        return this.isWebView;
    }

    getOptimalConstraints() {
        if (this.isWebView) {
            return {
                video: {
                    width: { ideal: 640, max: 1280 },
                    height: { ideal: 480, max: 720 },
                    frameRate: { ideal: 24, max: 30 },
                    facingMode: 'user'
                }
            };
        } else {
            return {
                video: {
                    width: { ideal: 1280, max: 1920 },
                    height: { ideal: 720, max: 1080 },
                    frameRate: { ideal: 30, max: 60 },
                    facingMode: 'user'
                }
            };
        }
    }
}

// Global instance
window.webViewCameraManager = new WebViewCameraManager();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WebViewCameraManager;
}