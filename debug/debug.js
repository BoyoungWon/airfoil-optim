/**
 * Tailored Airfoil Optimization - Debug Interface
 * WebSocket client and API interactions
 */

// Configuration
const API_BASE = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws';

// Global state
let ws = null;
let reconnectTimer = null;
let pipelineState = {
    status: 'idle',
    current_step: null,
    progress: 0,
    message: '',
    results: {}
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    log('Debug interface initialized', 'info');
    connectWebSocket();
    checkBackendHealth();
});

// =============================================================================
// WebSocket Connection
// =============================================================================

function connectWebSocket() {
    log('Connecting to backend...', 'info');
    
    try {
        ws = new WebSocket(WS_URL);
        
        ws.onopen = () => {
            log('Connected to backend', 'success');
            updateConnectionStatus(true);
            if (reconnectTimer) {
                clearTimeout(reconnectTimer);
                reconnectTimer = null;
            }
        };
        
        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleStateUpdate(data);
            } catch (e) {
                log(`Failed to parse message: ${e.message}`, 'error');
            }
        };
        
        ws.onerror = (error) => {
            log(`WebSocket error: ${error.message || 'Connection failed'}`, 'error');
            updateConnectionStatus(false);
        };
        
        ws.onclose = () => {
            log('Connection closed. Reconnecting...', 'warning');
            updateConnectionStatus(false);
            
            // Attempt reconnection
            reconnectTimer = setTimeout(() => {
                connectWebSocket();
            }, 3000);
        };
    } catch (error) {
        log(`Failed to create WebSocket: ${error.message}`, 'error');
        updateConnectionStatus(false);
    }
}

function updateConnectionStatus(connected) {
    const indicator = document.getElementById('connection-status');
    const text = document.getElementById('connection-text');
    
    if (connected) {
        indicator.className = 'status-indicator connected';
        text.textContent = 'Connected';
    } else {
        indicator.className = 'status-indicator disconnected';
        text.textContent = 'Disconnected';
    }
}

// =============================================================================
// State Management
// =============================================================================

function handleStateUpdate(state) {
    pipelineState = state;
    
    // Update progress bar
    updateProgress(state.progress * 100, state.message);
    
    // Update step status
    if (state.current_step) {
        updateStepStatus(state.current_step, state.status);
    }
    
    // Update results
    if (state.results && Object.keys(state.results).length > 0) {
        displayResults(state.results);
    }
    
    // Log message
    if (state.message) {
        const level = state.status === 'error' ? 'error' : 
                     state.status === 'completed' ? 'success' : 'info';
        log(state.message, level);
    }
}

function updateStepStatus(step, status) {
    const statusElement = document.getElementById(`status-${step}`);
    if (statusElement) {
        const badge = statusElement.querySelector('.status-badge');
        badge.textContent = status.charAt(0).toUpperCase() + status.slice(1);
        badge.className = `status-badge ${status}`;
    }
}

function updateProgress(percentage, message) {
    const progressBar = document.getElementById('progress-bar');
    const progressPercentage = document.getElementById('progress-percentage');
    const progressMessage = document.getElementById('progress-message');
    
    progressBar.style.width = `${percentage}%`;
    progressPercentage.textContent = `${Math.round(percentage)}%`;
    progressMessage.textContent = message || 'Idle';
}

function displayResults(results) {
    const panel = document.getElementById('results-panel');
    panel.innerHTML = '';
    
    for (const [key, value] of Object.entries(results)) {
        const item = document.createElement('div');
        item.className = 'result-item';
        
        const keyElement = document.createElement('strong');
        keyElement.textContent = formatKey(key);
        
        const valueElement = document.createElement('div');
        valueElement.textContent = formatValue(value);
        
        item.appendChild(keyElement);
        item.appendChild(valueElement);
        panel.appendChild(item);
    }
}

function formatKey(key) {
    return key
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

function formatValue(value) {
    if (typeof value === 'number') {
        return value.toFixed(4);
    }
    if (typeof value === 'object') {
        return JSON.stringify(value, null, 2);
    }
    return String(value);
}

// =============================================================================
// API Calls
// =============================================================================

async function apiCall(endpoint, method = 'GET', body = null) {
    try {
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json'
            }
        };
        
        if (body) {
            options.body = JSON.stringify(body);
        }
        
        const response = await fetch(`${API_BASE}${endpoint}`, options);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    } catch (error) {
        log(`API call failed: ${error.message}`, 'error');
        throw error;
    }
}

async function checkBackendHealth() {
    try {
        const health = await apiCall('/health');
        log(`Backend health: ${health.status}`, 'success');
    } catch (error) {
        log('Backend health check failed', 'error');
    }
}

// =============================================================================
// Pipeline Steps
// =============================================================================

async function runStep(stepName) {
    log(`Starting step: ${stepName}`, 'info');
    
    try {
        let response;
        
        switch(stepName) {
            case 'preprocess':
                response = await apiCall('/api/phase1/preprocess', 'POST');
                break;
                
            case 'train_gan':
                const ganEpochs = parseInt(document.getElementById('gan-epochs').value);
                const ganBatch = parseInt(document.getElementById('gan-batch').value);
                response = await apiCall('/api/phase1/train_gan', 'POST', {
                    epochs: ganEpochs,
                    batch_size: ganBatch
                });
                break;
                
            case 'train_validator':
                const valEpochs = parseInt(document.getElementById('val-epochs').value);
                response = await apiCall('/api/phase1/train_validator', 'POST', {
                    epochs: valEpochs
                });
                break;
                
            case 'generate_samples':
                const nSamples = parseInt(document.getElementById('n-samples').value);
                const maxThickness = parseFloat(document.getElementById('max-thickness').value);
                response = await apiCall('/api/phase1/generate_samples', 'POST', {
                    n_samples: nSamples,
                    max_thickness: maxThickness
                });
                break;
                
            case 'extract_modes':
                const nModes = parseInt(document.getElementById('n-modes').value);
                response = await apiCall('/api/phase1/extract_modes', 'POST', {
                    n_modes: nModes
                });
                break;
                
            default:
                throw new Error(`Unknown step: ${stepName}`);
        }
        
        log(`Step ${stepName} completed successfully`, 'success');
        return response;
        
    } catch (error) {
        log(`Step ${stepName} failed: ${error.message}`, 'error');
        throw error;
    }
}

async function runFullPipeline() {
    log('Starting full Phase 1 pipeline...', 'info');
    
    const steps = [
        'preprocess',
        'train_gan',
        'train_validator',
        'generate_samples',
        'extract_modes'
    ];
    
    for (const step of steps) {
        try {
            await runStep(step);
            await new Promise(resolve => setTimeout(resolve, 1000)); // Small delay between steps
        } catch (error) {
            log(`Pipeline stopped at step: ${step}`, 'error');
            return;
        }
    }
    
    log('Full Phase 1 pipeline completed!', 'success');
}

async function resetPipeline() {
    try {
        await apiCall('/api/reset', 'POST');
        log('Pipeline reset', 'info');
        
        // Reset UI
        document.querySelectorAll('.status-badge').forEach(badge => {
            badge.textContent = 'Pending';
            badge.className = 'status-badge pending';
        });
        
        updateProgress(0, 'Idle');
        
        const resultsPanel = document.getElementById('results-panel');
        resultsPanel.innerHTML = '<p class="placeholder">No results yet. Run pipeline steps to see results.</p>';
        
    } catch (error) {
        log(`Reset failed: ${error.message}`, 'error');
    }
}

// =============================================================================
// Logging
// =============================================================================

function log(message, level = 'info') {
    const logPanel = document.getElementById('log-panel');
    const entry = document.createElement('p');
    entry.className = `log-entry ${level}`;
    
    const timestamp = new Date().toLocaleTimeString();
    const levelText = level.toUpperCase().padEnd(7);
    
    entry.innerHTML = `<span class="log-timestamp">[${timestamp}]</span>[${levelText}] ${message}`;
    
    logPanel.appendChild(entry);
    
    // Auto-scroll if enabled
    const autoScroll = document.getElementById('auto-scroll');
    if (autoScroll && autoScroll.checked) {
        logPanel.scrollTop = logPanel.scrollHeight;
    }
}

function clearLog() {
    const logPanel = document.getElementById('log-panel');
    logPanel.innerHTML = '';
    log('Log cleared', 'info');
}

// =============================================================================
// Keyboard Shortcuts
// =============================================================================

document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + R: Reset
    if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
        e.preventDefault();
        resetPipeline();
    }
    
    // Ctrl/Cmd + L: Clear log
    if ((e.ctrlKey || e.metaKey) && e.key === 'l') {
        e.preventDefault();
        clearLog();
    }
});

// =============================================================================
// Helper Functions
// =============================================================================

function downloadFile(filename) {
    window.open(`${API_BASE}/api/download/${filename}`, '_blank');
}

// Export functions for global access
window.runStep = runStep;
window.runFullPipeline = runFullPipeline;
window.resetPipeline = resetPipeline;
window.clearLog = clearLog;
window.log = log;