/**
 * WebView Video Recorder
 * Enhanced video recording functionality for Android WebView
 */

class WebViewVideoRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.recordedChunks = [];
        this.isRecording = false;
        this.startTime = null;
        this.stream = null;
        this.recordingTimer = null;
        this.maxDuration = 300; // 5 minutes max
        this.onDataAvailable = null;
        this.onRecordingComplete = null;
        this.onError = null;
    }

    async initialize(stream) {
        try {
            this.stream = stream;
            
            if (!MediaRecorder.isTypeSupported) {
                throw new Error('MediaRecorder not supported');
            }

            // Find best supported format for WebView
            const mimeType = this.getBestMimeType();
            console.log('Using MIME type for recording:', mimeType);

            const options = {
                mimeType: mimeType,
                videoBitsPerSecond: this.getOptimalBitrate()
            };

            this.mediaRecorder = new MediaRecorder(stream, options);
            this.setupEventHandlers();
            
            return true;
        } catch (error) {
            console.error('Failed to initialize video recorder:', error);
            throw error;
        }
    }

    getBestMimeType() {
        // Priority order for WebView compatibility
        const mimeTypes = [
            'video/webm;codecs=vp9,opus',
            'video/webm;codecs=vp8,opus',
            'video/webm;codecs=vp9',
            'video/webm;codecs=vp8',
            'video/webm',
            'video/mp4;codecs=h264,aac',
            'video/mp4;codecs=h264',
            'video/mp4'
        ];

        for (const mimeType of mimeTypes) {
            if (MediaRecorder.isTypeSupported(mimeType)) {
                return mimeType;
            }
        }

        // Fallback
        return 'video/webm';
    }

    getOptimalBitrate() {
        // Lower bitrate for WebView to ensure compatibility
        const isWebView = /wv|WebView/i.test(navigator.userAgent);
        return isWebView ? 1000000 : 2500000; // 1Mbps for WebView, 2.5Mbps for regular browsers
    }

    setupEventHandlers() {
        this.mediaRecorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) {
                this.recordedChunks.push(event.data);
                
                if (this.onDataAvailable) {
                    this.onDataAvailable(event.data);
                }
            }
        };

        this.mediaRecorder.onstart = () => {
            console.log('Video recording started');
            this.isRecording = true;
            this.startTime = Date.now();
            this.startRecordingTimer();
        };

        this.mediaRecorder.onstop = () => {
            console.log('Video recording stopped');
            this.isRecording = false;
            this.stopRecordingTimer();
            this.processRecording();
        };

        this.mediaRecorder.onerror = (error) => {
            console.error('MediaRecorder error:', error);
            this.isRecording = false;
            this.stopRecordingTimer();
            
            if (this.onError) {
                this.onError(error);
            }
        };

        this.mediaRecorder.onpause = () => {
            console.log('Video recording paused');
        };

        this.mediaRecorder.onresume = () => {
            console.log('Video recording resumed');
        };
    }

    async startRecording(options = {}) {
        try {
            if (!this.mediaRecorder) {
                throw new Error('Recorder not initialized');
            }

            if (this.isRecording) {
                console.warn('Recording already in progress');
                return false;
            }

            // Clear previous recording
            this.recordedChunks = [];
            
            // Set max duration if provided
            if (options.maxDuration) {
                this.maxDuration = options.maxDuration;
            }

            // Start recording with time slice for better WebView compatibility
            const timeSlice = options.timeSlice || 1000; // 1 second chunks
            this.mediaRecorder.start(timeSlice);

            return true;
        } catch (error) {
            console.error('Failed to start recording:', error);
            throw error;
        }
    }

    async stopRecording() {
        try {
            if (!this.isRecording) {
                console.warn('No active recording to stop');
                return null;
            }

            return new Promise((resolve, reject) => {
                // Set up one-time handler for stop event
                const originalOnStop = this.mediaRecorder.onstop;
                this.mediaRecorder.onstop = (event) => {
                    // Call original handler
                    if (originalOnStop) {
                        originalOnStop(event);
                    }
                    
                    // Resolve with recording data
                    resolve(this.getRecordingData());
                };

                this.mediaRecorder.stop();
            });
        } catch (error) {
            console.error('Failed to stop recording:', error);
            throw error;
        }
    }

    pauseRecording() {
        try {
            if (this.isRecording && this.mediaRecorder.state === 'recording') {
                this.mediaRecorder.pause();
                return true;
            }
            return false;
        } catch (error) {
            console.error('Failed to pause recording:', error);
            return false;
        }
    }

    resumeRecording() {
        try {
            if (this.isRecording && this.mediaRecorder.state === 'paused') {
                this.mediaRecorder.resume();
                return true;
            }
            return false;
        } catch (error) {
            console.error('Failed to resume recording:', error);
            return false;
        }
    }

    processRecording() {
        try {
            if (this.recordedChunks.length === 0) {
                console.warn('No recorded data available');
                return null;
            }

            const recordingData = this.getRecordingData();
            
            if (this.onRecordingComplete) {
                this.onRecordingComplete(recordingData);
            }

            return recordingData;
        } catch (error) {
            console.error('Failed to process recording:', error);
            return null;
        }
    }

    getRecordingData() {
        try {
            const blob = new Blob(this.recordedChunks, {
                type: this.mediaRecorder.mimeType || 'video/webm'
            });

            const url = URL.createObjectURL(blob);
            const duration = this.startTime ? (Date.now() - this.startTime) / 1000 : 0;

            return {
                blob: blob,
                url: url,
                duration: duration,
                size: blob.size,
                mimeType: this.mediaRecorder.mimeType,
                chunks: this.recordedChunks.length,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            console.error('Failed to create recording data:', error);
            return null;
        }
    }

    startRecordingTimer() {
        this.recordingTimer = setInterval(() => {
            const elapsed = (Date.now() - this.startTime) / 1000;
            
            // Auto-stop if max duration reached
            if (elapsed >= this.maxDuration) {
                console.log('Max recording duration reached, stopping...');
                this.stopRecording();
            }
        }, 1000);
    }

    stopRecordingTimer() {
        if (this.recordingTimer) {
            clearInterval(this.recordingTimer);
            this.recordingTimer = null;
        }
    }

    getRecordingDuration() {
        if (!this.startTime) return 0;
        return (Date.now() - this.startTime) / 1000;
    }

    getRecordingState() {
        return {
            isRecording: this.isRecording,
            state: this.mediaRecorder ? this.mediaRecorder.state : 'inactive',
            duration: this.getRecordingDuration(),
            chunksCount: this.recordedChunks.length,
            mimeType: this.mediaRecorder ? this.mediaRecorder.mimeType : null
        };
    }

    // Utility method to download recording
    downloadRecording(filename = null) {
        try {
            const recordingData = this.getRecordingData();
            if (!recordingData) {
                throw new Error('No recording data available');
            }

            const link = document.createElement('a');
            link.href = recordingData.url;
            link.download = filename || `recording_${Date.now()}.webm`;
            
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            return true;
        } catch (error) {
            console.error('Failed to download recording:', error);
            return false;
        }
    }

    // Clean up resources
    cleanup() {
        try {
            this.stopRecordingTimer();
            
            if (this.isRecording) {
                this.stopRecording();
            }

            // Clean up blob URLs
            this.recordedChunks.forEach(chunk => {
                if (chunk instanceof Blob) {
                    URL.revokeObjectURL(URL.createObjectURL(chunk));
                }
            });

            this.recordedChunks = [];
            this.mediaRecorder = null;
            this.stream = null;
            
            console.log('Video recorder cleaned up');
        } catch (error) {
            console.error('Error during cleanup:', error);
        }
    }

    // Static method to check MediaRecorder support
    static isSupported() {
        return typeof MediaRecorder !== 'undefined' && 
               typeof MediaRecorder.isTypeSupported === 'function';
    }

    // Static method to get supported MIME types
    static getSupportedMimeTypes() {
        if (!WebViewVideoRecorder.isSupported()) {
            return [];
        }

        const mimeTypes = [
            'video/webm;codecs=vp9,opus',
            'video/webm;codecs=vp8,opus',
            'video/webm;codecs=vp9',
            'video/webm;codecs=vp8',
            'video/webm',
            'video/mp4;codecs=h264,aac',
            'video/mp4;codecs=h264',
            'video/mp4'
        ];

        return mimeTypes.filter(type => MediaRecorder.isTypeSupported(type));
    }
}

// Export for use
window.WebViewVideoRecorder = WebViewVideoRecorder;