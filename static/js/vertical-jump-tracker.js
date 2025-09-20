class VerticalJumpTracker {
    constructor() {
        this.isRunning = false;
        this.statsInterval = null;
        this.timerInterval = null;
        this.timeRemaining = 180; // 3 minutes
        this.totalTime = 180;
        this.startTime = null;
        this.endTime = null;
        this.timerCompleted = false;
        this.clientVideo = null;
        this.clientStream = null;
        this.initializeElements();
        this.bindEvents();
    }

    initializeElements() {
        this.startBtn = document.getElementById('start-btn');
        this.stopBtn = document.getElementById('stop-btn');
        this.resetBtn = document.getElementById('reset-btn');
        this.submitBtn = document.getElementById('submit-btn');
        this.videoFeed = document.getElementById('video-feed');
        this.placeholder = document.getElementById('placeholder');
        this.totalJumps = document.getElementById('total-jumps');
        this.maxHeight = document.getElementById('max-height');
        this.currentHeight = document.getElementById('current-height');
        this.jumpState = document.getElementById('jump-state');
        this.feedback = document.getElementById('feedback');
        this.timerDisplay = document.getElementById('timer-display');
        this.timerProgress = document.getElementById('timer-progress');
        this.sessionStatus = document.getElementById('session-status');
    }

    bindEvents() {
        this.startBtn.addEventListener('click', () => this.startCamera());
        this.stopBtn.addEventListener('click', () => this.stopCamera());
        this.resetBtn.addEventListener('click', () => this.resetCounter());
        this.submitBtn.addEventListener('click', () => this.submitResults());
    }

    async startCamera() {
        try {
            this.showLoading('Starting camera...');
            
            const response = await fetch('/vertical_jump/start_camera?' + new Date().getTime());
            const data = await response.json();
            
            if (data.status === 'success') {
                this.isRunning = true;
                this.videoFeed.src = '/vertical_jump/video_feed?' + new Date().getTime();
                this.videoFeed.style.display = 'block';
                this.placeholder.style.display = 'none';
                
                this.startBtn.disabled = true;
                this.stopBtn.disabled = false;
                
                this.startTimer();
                this.startStatsUpdate();
                this.showFeedback('Camera started! Begin your vertical jumps.', 'success');
            } else {
                await this.startClientCamera();
            }
        } catch (error) {
            await this.startClientCamera();
        }
    }

    async startClientCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            
            const video = document.createElement('video');
            video.srcObject = stream;
            video.autoplay = true;
            video.playsInline = true;
            video.style.cssText = 'width: 100%; height: 100%; object-fit: cover;';
            
            this.placeholder.style.display = 'none';
            this.videoFeed.style.display = 'none';
            
            const videoContainer = this.videoFeed.parentElement;
            videoContainer.appendChild(video);
            
            this.clientVideo = video;
            this.clientStream = stream;
            this.isRunning = true;
            
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            
            this.startTimer();
            this.startClientStatsUpdate();
            this.showFeedback('Client camera started! Begin your vertical jumps.', 'success');
            
        } catch (error) {
            this.showFeedback('Camera access denied: ' + error.message, 'error');
        }
    }

    async stopCamera() {
        try {
            await fetch('/vertical_jump/stop_camera');
            
            if (this.clientStream) {
                this.clientStream.getTracks().forEach(track => track.stop());
                this.clientStream = null;
            }
            
            if (this.clientVideo) {
                this.clientVideo.remove();
                this.clientVideo = null;
            }
            
            this.isRunning = false;
            this.videoFeed.style.display = 'none';
            this.placeholder.style.display = 'flex';
            this.videoFeed.src = '';
            
            this.startBtn.disabled = false;
            this.stopBtn.disabled = true;
            
            this.stopTimer();
            this.stopStatsUpdate();
            this.showFeedback('Camera stopped.', 'info');
        } catch (error) {
            this.isRunning = false;
            this.videoFeed.style.display = 'none';
            this.placeholder.style.display = 'flex';
            this.startBtn.disabled = false;
            this.stopBtn.disabled = true;
            this.stopStatsUpdate();
        }
    }

    async resetCounter() {
        try {
            if (this.isRunning) {
                await this.stopCamera();
            }
            
            const response = await fetch('/vertical_jump/reset_counter');
            const data = await response.json();
            
            if (data.status === 'success') {
                this.isRunning = false;
                this.videoFeed.style.display = 'none';
                this.placeholder.style.display = 'flex';
                this.videoFeed.src = '';
                
                this.startBtn.disabled = false;
                this.stopBtn.disabled = true;
                
                this.stopStatsUpdate();
                this.resetTimer();
                
                this.updateStats({
                    total_jumps: 0,
                    max_height: 0.0,
                    current_height: 0.0,
                    state: 'GROUND',
                    calibrated: false,
                    feedback: 'System Ready'
                });
                
                this.showFeedback('Counter reset! Ready for new session.', 'success');
            } else {
                this.showFeedback('Error resetting: ' + data.message, 'error');
            }
        } catch (error) {
            this.showFeedback('Error resetting counter: ' + error.message, 'error');
        }
    }

    startStatsUpdate() {
        this.statsInterval = setInterval(async () => {
            if (this.isRunning) {
                try {
                    const response = await fetch('/vertical_jump/get_stats');
                    const stats = await response.json();
                    this.updateStats(stats);
                } catch (error) {
                    console.error('Error fetching stats:', error);
                }
            }
        }, 500);
    }

    stopStatsUpdate() {
        if (this.statsInterval) {
            clearInterval(this.statsInterval);
            this.statsInterval = null;
        }
    }

    startClientStatsUpdate() {
        // Use server-side stats even with client camera
        this.statsInterval = setInterval(async () => {
            if (this.isRunning) {
                try {
                    const response = await fetch('/vertical_jump/get_stats');
                    const stats = await response.json();
                    this.updateStats(stats);
                } catch (error) {
                    console.error('Error fetching stats:', error);
                }
            }
        }, 500);
    }

    updateStats(stats) {
        if (this.totalJumps) this.totalJumps.textContent = stats.total_jumps || 0;
        if (this.maxHeight) this.maxHeight.textContent = (stats.max_height || 0).toFixed(1) + ' cm';
        if (this.currentHeight) this.currentHeight.textContent = (stats.current_height || 0).toFixed(1) + ' cm';
        if (this.jumpState) this.jumpState.textContent = stats.state || 'GROUND';
        if (this.feedback) this.feedback.textContent = stats.feedback || 'System Ready';
        
        // Update state color
        if (this.jumpState) {
            const state = stats.state || 'GROUND';
            if (state === 'JUMPING') {
                this.jumpState.style.color = '#4CAF50';
            } else if (state === 'AIRBORNE') {
                this.jumpState.style.color = '#FF9800';
            } else {
                this.jumpState.style.color = '#2196F3';
            }
        }
    }

    showLoading(message) {
        if (this.feedback) {
            this.feedback.textContent = message;
            this.feedback.style.color = '#2196F3';
        }
    }

    showFeedback(message, type) {
        const colors = {
            success: '#4CAF50',
            error: '#f44336',
            info: '#2196F3',
            warning: '#ff9800'
        };
        
        if (!this.isRunning || type === 'error') {
            if (this.feedback) {
                this.feedback.textContent = message;
                this.feedback.style.color = colors[type] || '#666';
                
                setTimeout(() => {
                    if (!this.isRunning) {
                        this.feedback.textContent = 'System Ready';
                        this.feedback.style.color = '#666';
                    }
                }, 3000);
            }
        }
    }

    startTimer() {
        this.startTime = new Date();
        this.timeRemaining = 180;
        this.timerCompleted = false;
        this.updateTimerDisplay();
        
        this.timerInterval = setInterval(() => {
            this.timeRemaining--;
            this.updateTimerDisplay();
            
            if (this.timeRemaining <= 0) {
                this.onTimerComplete();
            }
        }, 1000);
        
        if (this.sessionStatus) this.sessionStatus.textContent = 'Session in progress';
    }

    stopTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
        this.endTime = new Date();
        if (this.sessionStatus) this.sessionStatus.textContent = 'Session paused';
    }

    resetTimer() {
        this.stopTimer();
        this.timeRemaining = 180;
        this.totalTime = 180;
        this.startTime = null;
        this.endTime = null;
        this.timerCompleted = false;
        this.updateTimerDisplay();
        if (this.sessionStatus) this.sessionStatus.textContent = 'Ready to start';
    }

    updateTimerDisplay() {
        const minutes = Math.floor(this.timeRemaining / 60);
        const seconds = this.timeRemaining % 60;
        if (this.timerDisplay) {
            this.timerDisplay.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        }
        
        const progress = ((this.totalTime - this.timeRemaining) / this.totalTime) * 100;
        if (this.timerProgress) {
            this.timerProgress.style.width = progress + '%';
        }
    }

    onTimerComplete() {
        this.stopTimer();
        this.timerCompleted = true;
        if (this.timerDisplay) this.timerDisplay.textContent = '0:00';
        if (this.timerProgress) this.timerProgress.style.width = '100%';
        if (this.sessionStatus) this.sessionStatus.textContent = 'Time completed!';
        
        this.showTimeCompletedPopup();
    }

    showTimeCompletedPopup() {
        const popup = document.createElement('div');
        popup.className = 'popup-overlay';
        popup.innerHTML = `
            <div class="popup-content">
                <div class="popup-icon">⏰</div>
                <h3>Time Completed!</h3>
                <p>Your 3-minute session is complete. Please click Submit to save your results.</p>
                <button onclick="this.parentElement.parentElement.remove()" class="btn btn-primary">OK</button>
            </div>
        `;
        document.body.appendChild(popup);
    }

    async submitResults() {
        try {
            const response = await fetch('/vertical_jump/get_stats');
            const stats = await response.json();
            
            const totalJumps = stats.total_jumps || 0;
            const maxHeight = stats.max_height || 0;
            
            let timerTime;
            if (this.timerCompleted) {
                timerTime = '3:00';
            } else if (this.startTime && this.endTime) {
                const duration = Math.floor((this.endTime - this.startTime) / 1000);
                const minutes = Math.floor(duration / 60);
                const seconds = duration % 60;
                timerTime = `${minutes}:${seconds.toString().padStart(2, '0')}`;
            } else {
                const elapsed = this.totalTime - this.timeRemaining;
                const minutes = Math.floor(elapsed / 60);
                const seconds = elapsed % 60;
                timerTime = `${minutes}:${seconds.toString().padStart(2, '0')}`;
            }
            
            const submitResponse = await fetch('/api/submit_vertical_jump_result', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    total_jumps: totalJumps,
                    max_height: maxHeight,
                    timer_time: timerTime
                })
            });
            
            const submitData = await submitResponse.json();
            
            if (submitData.success) {
                await this.resetCounter();
                window.location.href = `/displayVerticalJump?jumps=${totalJumps}&height=${maxHeight}&timer=${timerTime}`;
            } else {
                this.showFeedback('Error submitting results: ' + submitData.message, 'error');
            }
        } catch (error) {
            this.showFeedback('Error submitting results: ' + error.message, 'error');
        }
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    new VerticalJumpTracker();
});

// Cleanup
window.addEventListener('beforeunload', async () => {
    try {
        await fetch('/vertical_jump/cleanup', { method: 'POST' });
    } catch (error) {
        console.log('Cleanup error:', error);
    }
});

window.addEventListener('unload', () => {
    navigator.sendBeacon('/vertical_jump/cleanup');
});