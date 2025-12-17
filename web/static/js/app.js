// AI Fruit Sorting System - JavaScript

let socket;
let systemRunning = false;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function () {
    initializeSocketIO();
    loadSystemStatus();
    setupVideoFeed();
});

// Initialize SocketIO connection
function initializeSocketIO() {
    socket = io();

    socket.on('connect', function () {
        console.log('Connected to server');
        updateConnectionStatus(true);
        showToast('Connected to server', 'success');
    });

    socket.on('disconnect', function () {
        console.log('Disconnected from server');
        updateConnectionStatus(false);
        showToast('Disconnected from server', 'error');
    });

    socket.on('stats_update', function (stats) {
        updateStatistics(stats);
    });

    socket.on('connected', function (data) {
        console.log(data.message);
    });
}

// Update connection status
function updateConnectionStatus(connected) {
    const statusDot = document.getElementById('connection-status');
    const statusText = document.getElementById('connection-text');

    if (connected) {
        statusDot.classList.remove('offline');
        statusDot.classList.add('online');
        statusText.textContent = 'Connected';
    } else {
        statusDot.classList.remove('online');
        statusDot.classList.add('offline');
        statusText.textContent = 'Offline';
    }
}

// Load system status
function loadSystemStatus() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            systemRunning = data.running;
            updateUIForSystemStatus(data.running);
            if (data.stats) {
                updateStatistics(data.stats);
            }
        })
        .catch(error => console.error('Error loading status:', error));
}

// Start system
function startSystem() {
    const btn = document.getElementById('btn-start');
    btn.disabled = true;
    btn.textContent = '⏳ Starting...';

    fetch('/api/start', {
        method: 'POST'
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                systemRunning = true;
                updateUIForSystemStatus(true);
                showToast('System started successfully', 'success');
                hideVideoOverlay();
            } else {
                showToast(data.message || 'Failed to start system', 'error');
            }
            btn.disabled = false;
            btn.textContent = '▶️ Start System';
        })
        .catch(error => {
            console.error('Error starting system:', error);
            showToast('Error starting system', 'error');
            btn.disabled = false;
            btn.textContent = '▶️ Start System';
        });
}

// Stop system
function stopSystem() {
    const btn = document.getElementById('btn-stop');
    btn.disabled = true;
    btn.textContent = '⏳ Stopping...';

    fetch('/api/stop', {
        method: 'POST'
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                systemRunning = false;
                updateUIForSystemStatus(false);
                showToast('System stopped', 'info');
            } else {
                showToast(data.message || 'Failed to stop system', 'error');
            }
            btn.disabled = false;
            btn.textContent = '⏹️ Stop System';
        })
        .catch(error => {
            console.error('Error stopping system:', error);
            showToast('Error stopping system', 'error');
            btn.disabled = false;
            btn.textContent = '⏹️ Stop System';
        });
}

// Update UI based on system status
function updateUIForSystemStatus(running) {
    const statusElement = document.getElementById('system-status');
    const startBtn = document.getElementById('btn-start');
    const stopBtn = document.getElementById('btn-stop');

    if (running) {
        statusElement.textContent = 'Running';
        statusElement.style.color = 'var(--success-color)';
        startBtn.disabled = true;
        stopBtn.disabled = false;
    } else {
        statusElement.textContent = 'Stopped';
        statusElement.style.color = 'var(--danger-color)';
        startBtn.disabled = false;
        stopBtn.disabled = true;
    }
}

// Control motor
function controlMotor(action) {
    const speed = document.getElementById('speed-slider').value;

    fetch(`/api/motor/${action}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ speed: parseInt(speed) })
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showToast(`Motor ${action} executed`, 'success');
            } else {
                showToast(data.message || 'Motor control failed', 'error');
            }
        })
        .catch(error => {
            console.error('Error controlling motor:', error);
            showToast('Error controlling motor', 'error');
        });
}

// Control servo
function controlServo(action) {
    fetch(`/api/servo/${action}`, {
        method: 'POST'
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showToast(`Servo moved to ${action}`, 'success');
            } else {
                showToast(data.message || 'Servo control failed', 'error');
            }
        })
        .catch(error => {
            console.error('Error controlling servo:', error);
            showToast('Error controlling servo', 'error');
        });
}

// Update speed display
function updateSpeed(value) {
    document.getElementById('speed-value').textContent = value;
}

// Update statistics
function updateStatistics(stats) {
    document.getElementById('total-detections').textContent = stats.total_detections || 0;
    document.getElementById('fresh-sorted').textContent = stats.fresh_sorted || 0;
    document.getElementById('spoiled-sorted').textContent = stats.spoiled_sorted || 0;
    document.getElementById('errors').textContent = stats.errors || 0;
    document.getElementById('fps-indicator').textContent = `${stats.fps || 0} FPS`;
    document.getElementById('uptime').textContent = formatUptime(stats.uptime || 0);

    // Update last detection with image
    if (stats.last_detection && stats.last_detection_image) {
        const det = stats.last_detection;

        // Show detection container, hide no-detection message
        document.getElementById('detection-container').style.display = 'block';
        document.getElementById('no-detection').style.display = 'none';

        // Set detection image
        const imgElement = document.getElementById('detection-image');
        imgElement.src = 'data:image/jpeg;base64,' + stats.last_detection_image;

        // Color-code based on classification (fresh = green, spoiled = red)
        if (det.classification) {
            const isFresh = det.classification.toLowerCase().includes('fresh');
            imgElement.style.borderColor = isFresh ? '#4CAF50' : '#f44336';
            imgElement.style.boxShadow = isFresh ? '0 0 10px rgba(76, 175, 80, 0.5)' : '0 0 10px rgba(244, 67, 54, 0.5)';
        }

        // Update detection details
        document.getElementById('detection-type').textContent = det.class || '-';

        if (det.classification) {
            const classSpan = document.getElementById('detection-class');
            classSpan.textContent = det.classification;
            const isFresh = det.classification.toLowerCase().includes('fresh');
            classSpan.style.color = isFresh ? '#4CAF50' : '#f44336';
            classSpan.style.fontWeight = 'bold';

            document.getElementById('detection-confidence').textContent =
                `${(det.class_confidence * 100).toFixed(1)}%`;
        } else {
            document.getElementById('detection-class').textContent = 'Pending...';
            document.getElementById('detection-confidence').textContent = '-';
        }

        document.getElementById('detection-time').textContent = det.time || '-';
    } else if (stats.last_detection) {
        // Detection exists but no image yet (shouldn't happen, but fallback)
        document.getElementById('detection-container').style.display = 'none';
        document.getElementById('no-detection').style.display = 'block';
    }
}

// Format uptime
function formatUptime(seconds) {
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    if (minutes < 60) return `${minutes}m ${secs}s`;
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${hours}h ${mins}m`;
}

// Reset statistics
function resetStats() {
    if (!confirm('Reset all statistics?')) return;

    fetch('/api/stats/reset', {
        method: 'POST'
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showToast('Statistics reset', 'success');
                updateStatistics({
                    total_detections: 0,
                    fresh_sorted: 0,
                    spoiled_sorted: 0,
                    errors: 0,
                    fps: 0,
                    uptime: 0
                });
            }
        })
        .catch(error => {
            console.error('Error resetting stats:', error);
            showToast('Error resetting statistics', 'error');
        });
}

// Setup video feed
function setupVideoFeed() {
    const videoFeed = document.getElementById('video-feed');
    const overlay = document.getElementById('video-overlay');

    videoFeed.onload = function () {
        overlay.classList.add('hidden');
    };

    videoFeed.onerror = function () {
        overlay.classList.remove('hidden');
    };
}

// Hide video overlay
function hideVideoOverlay() {
    const overlay = document.getElementById('video-overlay');
    setTimeout(() => {
        overlay.classList.add('hidden');
    }, 1000);
}

// Show toast notification
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️';
    toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;

    container.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Image enlargement modal
function enlargeImage(src) {
    const modal = document.getElementById('image-modal');
    const modalImg = document.getElementById('modal-image');
    modal.style.display = 'block';
    modalImg.src = src;
}

function closeImageModal() {
    document.getElementById('image-modal').style.display = 'none';
}

// Close modal on ESC key
document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape') {
        closeImageModal();
    }
});

// Request stats every 2 seconds
setInterval(() => {
    if (socket && socket.connected) {
        socket.emit('request_stats');
    }
}, 2000);
