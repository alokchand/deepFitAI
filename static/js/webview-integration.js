/**
 * WebView Integration Manager
 * Handles all WebView-specific functionality and Android app integration
 */

class WebViewIntegration {
    constructor() {
        this.isWebView = this.detectWebView();
        this.androidInterface = window.AndroidInterface || null;
        this.callbacks = new Map();
        this.eventListeners = new Map();
        
        this.initializeWebViewFeatures();
        console.log('WebView Integration initialized:', {
            isWebView: this.isWebView,
            hasAndroidInterface: !!this.androidInterface
        });
    }

    detectWebView() {
        const userAgent = navigator.userAgent;
        const isAndroid = /Android/i.test(userAgent);
        const isWebView = /wv|WebView/i.test(userAgent) || 
                         window.AndroidInterface !== undefined ||
                         /Version\/[\d.]+.*Chrome\/[.\d]+ Mobile/i.test(userAgent);
        
        return isAndroid && isWebView;
    }

    initializeWebViewFeatures() {
        if (!this.isWebView) return;

        // Set up WebView-specific CSS
        this.applyWebViewStyles();
        
        // Handle WebView events
        this.setupWebViewEventHandlers();
        
        // Configure viewport for WebView
        this.configureViewport();
        
        // Set up performance optimizations
        this.optimizeForWebView();
    }

    applyWebViewStyles() {
        const style = document.createElement('style');
        style.textContent = `
            /* WebView-specific styles */
            * {
                -webkit-tap-highlight-color: transparent;
                -webkit-touch-callout: none;
                -webkit-user-select: none;
                user-select: none;
            }
            
            input, textarea, select {
                -webkit-user-select: text;
                user-select: text;
            }
            
            video {
                -webkit-playsinline: true;
                playsinline: true;
            }
            
            /* Prevent zoom on input focus */
            input[type="text"], input[type="email"], input[type="password"], 
            input[type="number"], input[type="tel"], textarea, select {
                font-size: 16px !important;
            }
            
            /* Smooth scrolling for WebView */
            html {
                -webkit-overflow-scrolling: touch;
                overflow-scrolling: touch;
            }
            
            /* Fix for WebView button styling */
            button, .btn {
                -webkit-appearance: none;
                appearance: none;
                border-radius: 8px;
            }
        `;
        document.head.appendChild(style);
    }

    setupWebViewEventHandlers() {
        // Handle Android back button
        document.addEventListener('backbutton', (e) => {
            e.preventDefault();
            this.handleBackButton();
        });

        // Handle app pause/resume
        document.addEventListener('pause', () => {
            this.handleAppPause();
        });

        document.addEventListener('resume', () => {
            this.handleAppResume();
        });

        // Handle network changes
        window.addEventListener('online', () => {
            this.handleNetworkChange(true);
        });

        window.addEventListener('offline', () => {
            this.handleNetworkChange(false);
        });

        // Handle orientation changes
        window.addEventListener('orientationchange', () => {
            setTimeout(() => {
                this.handleOrientationChange();
            }, 100);
        });
    }

    configureViewport() {
        let viewport = document.querySelector('meta[name="viewport"]');
        if (!viewport) {
            viewport = document.createElement('meta');
            viewport.name = 'viewport';
            document.head.appendChild(viewport);
        }
        
        viewport.content = 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover';
    }

    optimizeForWebView() {
        // Disable context menu
        document.addEventListener('contextmenu', (e) => {
            e.preventDefault();
        });

        // Optimize touch events
        document.addEventListener('touchstart', (e) => {
            // Prevent default for better touch response
            if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
                e.preventDefault();
            }
        }, { passive: false });

        // Prevent double-tap zoom
        let lastTouchEnd = 0;
        document.addEventListener('touchend', (e) => {
            const now = (new Date()).getTime();
            if (now - lastTouchEnd <= 300) {
                e.preventDefault();
            }
            lastTouchEnd = now;
        }, false);
    }

    // Android Interface Methods
    callAndroidMethod(methodName, params = {}) {
        if (!this.androidInterface) {
            console.warn('Android interface not available');
            return null;
        }

        try {
            if (typeof this.androidInterface[methodName] === 'function') {
                return this.androidInterface[methodName](JSON.stringify(params));
            } else {
                console.warn(`Android method ${methodName} not found`);
                return null;
            }
        } catch (error) {
            console.error(`Error calling Android method ${methodName}:`, error);
            return null;
        }
    }

    // Camera permissions
    async requestCameraPermission() {
        if (this.androidInterface) {
            const result = this.callAndroidMethod('requestCameraPermission');
            return result === 'granted';
        }
        
        // Fallback to web API
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            stream.getTracks().forEach(track => track.stop());
            return true;
        } catch (error) {
            return false;
        }
    }

    // Storage permissions
    async requestStoragePermission() {
        if (this.androidInterface) {
            const result = this.callAndroidMethod('requestStoragePermission');
            return result === 'granted';
        }
        return true; // Web doesn't need explicit storage permission
    }

    // File operations
    async saveFileToDownloads(blob, filename) {
        if (this.androidInterface) {
            try {
                const reader = new FileReader();
                reader.onload = () => {
                    const base64 = reader.result.split(',')[1];
                    this.callAndroidMethod('saveFileToDownloads', {
                        filename: filename,
                        data: base64,
                        mimeType: blob.type
                    });
                };
                reader.readAsDataURL(blob);
                return true;
            } catch (error) {
                console.error('Error saving file:', error);
                return false;
            }
        }
        
        // Fallback to web download
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
        return true;
    }

    // Share functionality
    async shareContent(content) {
        if (this.androidInterface) {
            return this.callAndroidMethod('shareContent', content);
        }
        
        // Fallback to Web Share API
        if (navigator.share) {
            try {
                await navigator.share(content);
                return true;
            } catch (error) {
                console.error('Error sharing:', error);
                return false;
            }
        }
        
        return false;
    }

    // Toast messages
    showToast(message, duration = 3000) {
        if (this.androidInterface) {
            this.callAndroidMethod('showToast', { message, duration });
            return;
        }
        
        // Fallback to web toast
        this.showWebToast(message, duration);
    }

    showWebToast(message, duration = 3000) {
        const toast = document.createElement('div');
        toast.className = 'webview-toast';
        toast.textContent = message;
        toast.style.cssText = `
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 12px 24px;
            border-radius: 24px;
            z-index: 10000;
            font-size: 14px;
            max-width: 80%;
            text-align: center;
            animation: toastSlideIn 0.3s ease-out;
        `;
        
        // Add animation styles
        if (!document.querySelector('#toast-styles')) {
            const style = document.createElement('style');
            style.id = 'toast-styles';
            style.textContent = `
                @keyframes toastSlideIn {
                    from { opacity: 0; transform: translateX(-50%) translateY(20px); }
                    to { opacity: 1; transform: translateX(-50%) translateY(0); }
                }
                @keyframes toastSlideOut {
                    from { opacity: 1; transform: translateX(-50%) translateY(0); }
                    to { opacity: 0; transform: translateX(-50%) translateY(20px); }
                }
            `;
            document.head.appendChild(style);
        }
        
        document.body.appendChild(toast);
        
        setTimeout(() => {
            toast.style.animation = 'toastSlideOut 0.3s ease-out';
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 300);
        }, duration);
    }

    // Vibration
    vibrate(pattern = [100]) {
        if (this.androidInterface) {
            this.callAndroidMethod('vibrate', { pattern });
            return;
        }
        
        // Fallback to web vibration
        if (navigator.vibrate) {
            navigator.vibrate(pattern);
        }
    }

    // Keep screen on
    keepScreenOn(enable = true) {
        if (this.androidInterface) {
            this.callAndroidMethod('keepScreenOn', { enable });
        }
    }

    // Battery optimization
    requestBatteryOptimizationExemption() {
        if (this.androidInterface) {
            return this.callAndroidMethod('requestBatteryOptimizationExemption');
        }
        return false;
    }

    // Event handlers
    handleBackButton() {
        // Check if there are any modals or overlays open
        const modals = document.querySelectorAll('.modal.show, .popup-overlay, .camera-error-overlay');
        if (modals.length > 0) {
            modals[modals.length - 1].remove();
            return;
        }
        
        // Check if we're in a sub-page
        if (window.location.pathname !== '/') {
            window.history.back();
            return;
        }
        
        // Exit app
        if (this.androidInterface) {
            this.callAndroidMethod('exitApp');
        }
    }

    handleAppPause() {
        console.log('App paused');
        
        // Stop camera if active
        if (window.webViewCameraManager) {
            window.webViewCameraManager.stopCamera();
        }
        
        // Pause any ongoing operations
        this.emit('app-pause');
    }

    handleAppResume() {
        console.log('App resumed');
        
        // Resume operations if needed
        this.emit('app-resume');
    }

    handleNetworkChange(isOnline) {
        console.log('Network status changed:', isOnline ? 'online' : 'offline');
        
        this.showToast(
            isOnline ? 'Connection restored' : 'No internet connection',
            2000
        );
        
        this.emit('network-change', { isOnline });
    }

    handleOrientationChange() {
        console.log('Orientation changed');
        
        // Trigger resize events for components
        window.dispatchEvent(new Event('resize'));
        
        this.emit('orientation-change');
    }

    // Event system
    on(event, callback) {
        if (!this.eventListeners.has(event)) {
            this.eventListeners.set(event, []);
        }
        this.eventListeners.get(event).push(callback);
    }

    off(event, callback) {
        if (this.eventListeners.has(event)) {
            const listeners = this.eventListeners.get(event);
            const index = listeners.indexOf(callback);
            if (index > -1) {
                listeners.splice(index, 1);
            }
        }
    }

    emit(event, data = null) {
        if (this.eventListeners.has(event)) {
            this.eventListeners.get(event).forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in event listener for ${event}:`, error);
                }
            });
        }
    }

    // Utility methods
    isWebViewEnvironment() {
        return this.isWebView;
    }

    hasAndroidInterface() {
        return !!this.androidInterface;
    }

    getWebViewInfo() {
        return {
            isWebView: this.isWebView,
            userAgent: navigator.userAgent,
            hasAndroidInterface: !!this.androidInterface,
            screenSize: {
                width: window.screen.width,
                height: window.screen.height
            },
            viewport: {
                width: window.innerWidth,
                height: window.innerHeight
            }
        };
    }
}

// Global instance
window.webViewIntegration = new WebViewIntegration();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WebViewIntegration;
}