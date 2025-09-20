/**
 * WebView Photo Capture
 * Enhanced photo capture functionality for Android WebView
 */

class WebViewPhotoCapture {
    constructor() {
        this.canvas = null;
        this.context = null;
        this.lastCapturedPhoto = null;
        this.compressionQuality = 0.8;
        this.maxWidth = 1920;
        this.maxHeight = 1080;
    }

    initialize() {
        // Create canvas for photo capture
        this.canvas = document.createElement('canvas');
        this.context = this.canvas.getContext('2d');
        console.log('WebView photo capture initialized');
    }

    async captureFromVideo(videoElement, options = {}) {
        try {
            if (!videoElement || videoElement.readyState < 2) {
                throw new Error('Video not ready for capture');
            }

            // Initialize canvas if not done
            if (!this.canvas) {
                this.initialize();
            }

            // Get video dimensions
            const videoWidth = videoElement.videoWidth || videoElement.clientWidth;
            const videoHeight = videoElement.videoHeight || videoElement.clientHeight;

            if (videoWidth === 0 || videoHeight === 0) {
                throw new Error('Invalid video dimensions');
            }

            // Calculate optimal dimensions
            const dimensions = this.calculateOptimalDimensions(videoWidth, videoHeight, options);
            
            // Set canvas size
            this.canvas.width = dimensions.width;
            this.canvas.height = dimensions.height;

            // Clear canvas
            this.context.clearRect(0, 0, dimensions.width, dimensions.height);

            // Draw video frame to canvas
            this.context.drawImage(videoElement, 0, 0, dimensions.width, dimensions.height);

            // Apply filters if specified
            if (options.filters) {
                this.applyFilters(options.filters);
            }

            // Convert to different formats
            const result = await this.processCapture(options);
            
            this.lastCapturedPhoto = result;
            return result;

        } catch (error) {
            console.error('Photo capture failed:', error);
            throw error;
        }
    }

    calculateOptimalDimensions(videoWidth, videoHeight, options = {}) {
        let width = options.width || videoWidth;
        let height = options.height || videoHeight;

        // Maintain aspect ratio
        const aspectRatio = videoWidth / videoHeight;

        // Apply max dimensions
        if (width > this.maxWidth) {
            width = this.maxWidth;
            height = width / aspectRatio;
        }

        if (height > this.maxHeight) {
            height = this.maxHeight;
            width = height * aspectRatio;
        }

        // Ensure even dimensions for better compatibility
        width = Math.floor(width / 2) * 2;
        height = Math.floor(height / 2) * 2;

        return { width, height };
    }

    applyFilters(filters) {
        if (!this.context) return;

        const imageData = this.context.getImageData(0, 0, this.canvas.width, this.canvas.height);
        const data = imageData.data;

        // Apply brightness filter
        if (filters.brightness !== undefined) {
            const brightness = filters.brightness;
            for (let i = 0; i < data.length; i += 4) {
                data[i] = Math.min(255, Math.max(0, data[i] + brightness));     // Red
                data[i + 1] = Math.min(255, Math.max(0, data[i + 1] + brightness)); // Green
                data[i + 2] = Math.min(255, Math.max(0, data[i + 2] + brightness)); // Blue
            }
        }

        // Apply contrast filter
        if (filters.contrast !== undefined) {
            const contrast = filters.contrast;
            const factor = (259 * (contrast + 255)) / (255 * (259 - contrast));
            for (let i = 0; i < data.length; i += 4) {
                data[i] = Math.min(255, Math.max(0, factor * (data[i] - 128) + 128));
                data[i + 1] = Math.min(255, Math.max(0, factor * (data[i + 1] - 128) + 128));
                data[i + 2] = Math.min(255, Math.max(0, factor * (data[i + 2] - 128) + 128));
            }
        }

        // Apply saturation filter
        if (filters.saturation !== undefined) {
            const saturation = filters.saturation;
            for (let i = 0; i < data.length; i += 4) {
                const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
                data[i] = Math.min(255, Math.max(0, gray + saturation * (data[i] - gray)));
                data[i + 1] = Math.min(255, Math.max(0, gray + saturation * (data[i + 1] - gray)));
                data[i + 2] = Math.min(255, Math.max(0, gray + saturation * (data[i + 2] - gray)));
            }
        }

        this.context.putImageData(imageData, 0, 0);
    }

    async processCapture(options = {}) {
        const format = options.format || 'jpeg';
        const quality = options.quality || this.compressionQuality;

        // Create different output formats
        const result = {
            canvas: this.canvas,
            width: this.canvas.width,
            height: this.canvas.height,
            timestamp: new Date().toISOString()
        };

        // Generate data URL
        const mimeType = format === 'png' ? 'image/png' : 'image/jpeg';
        result.dataUrl = this.canvas.toDataURL(mimeType, quality);

        // Generate blob for better WebView compatibility
        result.blob = await this.canvasToBlob(mimeType, quality);

        // Generate base64 string (without data URL prefix)
        result.base64 = result.dataUrl.split(',')[1];

        // Calculate file size estimate
        result.estimatedSize = Math.ceil(result.base64.length * 0.75);

        return result;
    }

    canvasToBlob(mimeType = 'image/jpeg', quality = 0.8) {
        return new Promise((resolve, reject) => {
            try {
                this.canvas.toBlob((blob) => {
                    if (blob) {
                        resolve(blob);
                    } else {
                        reject(new Error('Failed to create blob'));
                    }
                }, mimeType, quality);
            } catch (error) {
                reject(error);
            }
        });
    }

    // Capture multiple photos in sequence
    async captureSequence(videoElement, count = 3, interval = 1000, options = {}) {
        const photos = [];
        
        for (let i = 0; i < count; i++) {
            try {
                const photo = await this.captureFromVideo(videoElement, {
                    ...options,
                    sequenceIndex: i,
                    sequenceTotal: count
                });
                
                photos.push(photo);
                
                // Wait before next capture
                if (i < count - 1) {
                    await this.delay(interval);
                }
            } catch (error) {
                console.error(`Failed to capture photo ${i + 1}:`, error);
            }
        }

        return photos;
    }

    // Capture with countdown
    async captureWithCountdown(videoElement, countdown = 3, options = {}) {
        return new Promise((resolve, reject) => {
            let count = countdown;
            
            const countdownInterval = setInterval(() => {
                if (options.onCountdown) {
                    options.onCountdown(count);
                }
                
                count--;
                
                if (count < 0) {
                    clearInterval(countdownInterval);
                    
                    this.captureFromVideo(videoElement, options)
                        .then(resolve)
                        .catch(reject);
                }
            }, 1000);
        });
    }

    // Compare two photos for similarity (basic implementation)
    comparePhotos(photo1, photo2) {
        if (!photo1 || !photo2) return 0;
        
        try {
            // Simple comparison based on file size and dimensions
            const sizeDiff = Math.abs(photo1.estimatedSize - photo2.estimatedSize);
            const dimensionMatch = (photo1.width === photo2.width && photo1.height === photo2.height);
            
            if (!dimensionMatch) return 0;
            
            // Calculate similarity score (0-1)
            const maxSize = Math.max(photo1.estimatedSize, photo2.estimatedSize);
            const similarity = 1 - (sizeDiff / maxSize);
            
            return Math.max(0, similarity);
        } catch (error) {
            console.error('Photo comparison failed:', error);
            return 0;
        }
    }

    // Download captured photo
    downloadPhoto(photo, filename = null) {
        try {
            if (!photo || !photo.dataUrl) {
                throw new Error('No photo data available');
            }

            const link = document.createElement('a');
            link.href = photo.dataUrl;
            link.download = filename || `photo_${Date.now()}.jpg`;
            
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            return true;
        } catch (error) {
            console.error('Photo download failed:', error);
            return false;
        }
    }

    // Upload photo to server
    async uploadPhoto(photo, endpoint, additionalData = {}) {
        try {
            if (!photo || !photo.blob) {
                throw new Error('No photo blob available');
            }

            const formData = new FormData();
            formData.append('photo', photo.blob, 'captured_photo.jpg');
            
            // Add metadata
            formData.append('metadata', JSON.stringify({
                width: photo.width,
                height: photo.height,
                timestamp: photo.timestamp,
                estimatedSize: photo.estimatedSize,
                ...additionalData
            }));

            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.statusText}`);
            }

            const result = await response.json();
            return result;

        } catch (error) {
            console.error('Photo upload failed:', error);
            throw error;
        }
    }

    // Utility method for delays
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // Get last captured photo
    getLastPhoto() {
        return this.lastCapturedPhoto;
    }

    // Clear captured photos from memory
    cleanup() {
        this.lastCapturedPhoto = null;
        if (this.canvas) {
            this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }
        console.log('Photo capture cleaned up');
    }

    // Static method to check canvas support
    static isSupported() {
        try {
            const canvas = document.createElement('canvas');
            return !!(canvas.getContext && canvas.getContext('2d'));
        } catch (error) {
            return false;
        }
    }

    // Static method to get optimal photo settings for WebView
    static getWebViewOptimalSettings() {
        const isWebView = /wv|WebView/i.test(navigator.userAgent);
        
        return {
            format: 'jpeg',
            quality: isWebView ? 0.7 : 0.8,
            maxWidth: isWebView ? 1280 : 1920,
            maxHeight: isWebView ? 720 : 1080,
            filters: {
                brightness: 0,
                contrast: 0,
                saturation: 1.0
            }
        };
    }
}

// Global instance
window.webViewPhotoCapture = new WebViewPhotoCapture();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WebViewPhotoCapture;
}