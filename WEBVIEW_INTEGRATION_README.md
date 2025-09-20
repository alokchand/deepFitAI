# Android WebView Integration Guide

This document provides comprehensive instructions for integrating the DeepFit application with Android WebView, ensuring all camera, video, and photo functionality works seamlessly in mobile applications.

## 🚀 Overview

The WebView integration includes:
- **Camera Access**: Enhanced camera initialization and management
- **Video Recording**: WebView-compatible video recording with multiple formats
- **Photo Capture**: High-quality photo capture with filters and processing
- **File Upload**: Robust file upload system for videos and photos
- **Android Integration**: Native Android interface for enhanced functionality

## 📁 Files Added/Modified

### New JavaScript Files
- `static/js/webview-camera-fix.js` - Main camera management for WebView
- `static/js/webview-video-recorder.js` - Video recording functionality
- `static/js/webview-photo-capture.js` - Photo capture with advanced features
- `static/js/webview-integration.js` - Android WebView integration manager

### New Python Files
- `app_routes_webview.py` - WebView-specific Flask routes for uploads

### Modified Files
- `templates/signup.html` - Enhanced photo capture for WebView
- `templates/height.html` - WebView-compatible camera integration
- `templates/index_situp.html` - Added WebView scripts
- `static/script.js` - Enhanced situp tracker with WebView support
- `app.py` - Added WebView blueprint registration

## 🔧 WebView Configuration

### 1. Android WebView Settings

```java
// Enable JavaScript
webView.getSettings().setJavaScriptEnabled(true);

// Enable DOM storage
webView.getSettings().setDomStorageEnabled(true);

// Enable camera access
webView.getSettings().setMediaPlaybackRequiresUserGesture(false);

// Allow file access
webView.getSettings().setAllowFileAccess(true);
webView.getSettings().setAllowContentAccess(true);

// Enable hardware acceleration
webView.setLayerType(View.LAYER_TYPE_HARDWARE, null);
```

### 2. Camera Permissions

Add to `AndroidManifest.xml`:
```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
```

### 3. WebView Client Configuration

```java
webView.setWebViewClient(new WebViewClient() {
    @Override
    public void onPermissionRequest(PermissionRequest request) {
        // Grant camera and microphone permissions
        request.grant(request.getResources());
    }
});

webView.setWebChromeClient(new WebChromeClient() {
    @Override
    public void onPermissionRequest(PermissionRequest request) {
        request.grant(request.getResources());
    }
    
    @Override
    public boolean onShowFileChooser(WebView webView, ValueCallback<Uri[]> filePathCallback, 
                                   FileChooserParams fileChooserParams) {
        // Handle file selection for uploads
        return true;
    }
});
```

## 🎥 Camera Integration

### WebView Camera Manager

The `WebViewCameraManager` class provides:

```javascript
// Initialize camera
const cameraManager = window.webViewCameraManager;

// Start camera with WebView-optimized settings
await cameraManager.startCamera(videoElement);

// Capture photo with advanced options
const photo = await cameraManager.capturePhoto(videoElement, canvas);

// Stop camera and cleanup
await cameraManager.stopCamera();
```

### Features:
- **Auto-detection** of WebView environment
- **Fallback constraints** for different Android versions
- **Error handling** with user-friendly messages
- **Camera switching** (front/back)
- **Performance optimization** for mobile devices

## 📹 Video Recording

### WebView Video Recorder

```javascript
// Initialize recorder
const recorder = new WebViewVideoRecorder();
await recorder.initialize(stream);

// Start recording with options
await recorder.startRecording({
    maxDuration: 300, // 5 minutes
    timeSlice: 1000   // 1 second chunks
});

// Stop and get recording
const recordingData = await recorder.stopRecording();

// Upload to server
const formData = new FormData();
formData.append('video', recordingData.blob);
await fetch('/api/webview/upload_video', {
    method: 'POST',
    body: formData
});
```

### Supported Formats:
- **WebM** (VP8/VP9) - Primary format
- **MP4** (H.264) - Fallback format
- **Auto-detection** of best supported format

## 📸 Photo Capture

### WebView Photo Capture

```javascript
// Initialize photo capture
const photoCapture = window.webViewPhotoCapture;

// Capture with options
const photo = await photoCapture.captureFromVideo(videoElement, {
    format: 'jpeg',
    quality: 0.8,
    width: 1280,
    height: 720,
    filters: {
        brightness: 0,
        contrast: 0,
        saturation: 1.0
    }
});

// Upload photo
await photoCapture.uploadPhoto(photo, '/api/webview/upload_photo');
```

### Features:
- **Multiple formats** (JPEG, PNG)
- **Image filters** (brightness, contrast, saturation)
- **Sequence capture** for multiple photos
- **Countdown capture** with timer
- **Automatic optimization** for WebView

## 🔗 Android Interface

### JavaScript to Android Communication

```javascript
// Check if Android interface is available
if (window.webViewIntegration.hasAndroidInterface()) {
    // Call Android methods
    const result = window.webViewIntegration.callAndroidMethod('methodName', {
        param1: 'value1',
        param2: 'value2'
    });
}
```

### Available Android Methods:
- `requestCameraPermission()` - Request camera access
- `requestStoragePermission()` - Request storage access
- `saveFileToDownloads(blob, filename)` - Save files to downloads
- `shareContent(content)` - Share content via Android
- `showToast(message, duration)` - Show native toast
- `vibrate(pattern)` - Device vibration
- `keepScreenOn(enable)` - Prevent screen sleep

## 📱 Mobile Optimizations

### Performance Enhancements:
1. **Reduced bitrates** for WebView video recording
2. **Optimized constraints** for camera initialization
3. **Battery-friendly** video track management
4. **Memory cleanup** on page visibility changes
5. **Touch optimization** for better responsiveness

### UI Adaptations:
1. **WebView-specific CSS** for better rendering
2. **Touch-friendly** button sizes and spacing
3. **Viewport configuration** for proper scaling
4. **Keyboard handling** for input fields
5. **Orientation change** support

## 🛠️ Server-Side Integration

### Flask Routes

New WebView-specific routes added:

```python
# Video upload
POST /api/webview/upload_video

# Photo upload  
POST /api/webview/upload_photo

# Situp video upload
POST /api/webview/upload_situp_video

# Camera status
GET /api/webview/camera_status

# Test upload
POST /api/webview/test_upload
```

### File Handling:
- **Secure filename** generation
- **Metadata storage** with JSON files
- **File size validation** (50MB max)
- **Format validation** for security
- **Error handling** with proper HTTP codes

## 🧪 Testing

### WebView Detection:
```javascript
// Check if running in WebView
const isWebView = window.webViewIntegration.isWebViewEnvironment();

// Get WebView info
const info = window.webViewIntegration.getWebViewInfo();
console.log('WebView Info:', info);
```

### Camera Testing:
```javascript
// Test camera initialization
try {
    await window.webViewCameraManager.initializeCamera();
    console.log('Camera initialized successfully');
} catch (error) {
    console.error('Camera initialization failed:', error);
}
```

### Upload Testing:
```javascript
// Test upload endpoint
const response = await fetch('/api/webview/test_upload', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ test: 'data' })
});
const result = await response.json();
console.log('Upload test result:', result);
```

## 🚨 Troubleshooting

### Common Issues:

1. **Camera Permission Denied**
   - Ensure permissions are granted in Android manifest
   - Check WebView client permission handling
   - Verify user has granted camera access

2. **Video Recording Fails**
   - Check MediaRecorder support
   - Verify MIME type compatibility
   - Ensure sufficient storage space

3. **Photo Capture Issues**
   - Verify canvas support
   - Check video element readiness
   - Ensure proper video dimensions

4. **Upload Failures**
   - Check network connectivity
   - Verify server endpoint availability
   - Check file size limits

### Debug Mode:
Enable debug logging by setting:
```javascript
window.webViewDebug = true;
```

## 📋 Checklist for Integration

- [ ] WebView settings configured correctly
- [ ] Camera permissions added to manifest
- [ ] WebView client and chrome client set up
- [ ] JavaScript files included in templates
- [ ] Flask blueprint registered
- [ ] Upload directories created
- [ ] Error handling implemented
- [ ] Testing completed on target devices

## 🔄 Updates and Maintenance

### Regular Updates:
1. **Browser compatibility** checks
2. **Android version** testing
3. **Performance monitoring**
4. **Security updates** for file handling
5. **User feedback** integration

### Monitoring:
- **Upload success rates**
- **Camera initialization failures**
- **Performance metrics**
- **Error logs analysis**

## 📞 Support

For technical support or questions:
1. Check browser console for errors
2. Verify WebView configuration
3. Test on different Android versions
4. Review server logs for upload issues

---

**Note**: This integration provides comprehensive WebView support while maintaining backward compatibility with standard web browsers. All features gracefully degrade when WebView-specific functionality is not available.