// ===========================================
// EDISON Enhanced Features
// File Upload, Hardware Monitor, Work Mode, etc.
// Version 7 - VLM support
// ===========================================

console.log('📦 app_features.js v7 loading - VLM enabled...');

// Wait for main app to initialize — derive endpoint from current page or saved settings
let API_ENDPOINT = localStorage.getItem('edisonApiEndpoint') || `${window.location.origin}/api`;

// File Upload Handling - make it globally accessible
window.uploadedFiles = [];

// Global function to trigger file selection
window.triggerFileUpload = function(event) {
    if (event) {
        event.preventDefault();
        event.stopPropagation();
    }
    console.log('📎 triggerFileUpload called');
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        console.log('📎 Clicking file input, current files:', window.uploadedFiles.length);
        fileInput.click();
    } else {
        console.error('❌ File input not found!');
    }
};

function initFileUpload() {
    const attachBtn = document.getElementById('attachBtn');
    const fileInput = document.getElementById('fileInput');
    const attachedFilesDiv = document.getElementById('attachedFiles');

    if (!attachBtn || !fileInput) {
        console.error('File upload elements not found:', { attachBtn: !!attachBtn, fileInput: !!fileInput });
        return;
    }

    // Don't add event listener since we're using onclick attribute
    // attachBtn.addEventListener('click', () => {
    //     console.log('Attach button clicked via event listener');
    //     window.triggerFileUpload();
    // });

    fileInput.addEventListener('change', async (e) => {
        console.log('📎 File input change event triggered');
        const files = Array.from(e.target.files);
        console.log('📎 Files selected:', files.length, files.map(f => f.name));
        
        if (files.length === 0) {
            console.log('⚠️ No files selected');
            return;
        }
        
        for (const file of files) {
            console.log(`📎 Processing file: ${file.name}, size: ${file.size} bytes, type: ${file.type}`);
            
            if (file.size > 10 * 1024 * 1024) { // 10MB limit
                console.log(`❌ File ${file.name} is too large (max 10MB)`);
                alert(`❌ File too large: ${file.name}\n\nMaximum file size is 10MB.`);
                continue;
            }

            const fileData = await readFileContent(file);
            console.log(`✅ File ${file.name} read, content length: ${fileData.length}`);
            window.uploadedFiles.push({
                name: file.name,
                type: file.type,
                content: fileData,
                isImage: file.type.startsWith('image/'),
                isPdf: file.type === 'application/pdf'
            });
        }

        console.log('📎 Total files now:', window.uploadedFiles.length, window.uploadedFiles.map(f => f.name));
        updateAttachedFilesUI();
        
        // Don't reset immediately - causes double-click issue
        // fileInput.value = ''; 
    });
}

async function readFileContent(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = reject;
        
        // Read images and PDFs as base64, text files as text
        if (file.type.startsWith('image/') || file.type === 'application/pdf') {
            reader.readAsDataURL(file);
        } else {
            reader.readAsText(file);
        }
    });
}

function updateAttachedFilesUI() {
    console.log('🔄 updateAttachedFilesUI called, files:', window.uploadedFiles.length);
    const attachedFilesDiv = document.getElementById('attachedFiles');
    
    if (!attachedFilesDiv) {
        console.error('❌ attachedFiles div not found!');
        return;
    }
    
    if (window.uploadedFiles.length === 0) {
        attachedFilesDiv.style.display = 'none';
        console.log('📎 No files to display');
        return;
    }

    attachedFilesDiv.style.display = 'flex';
    attachedFilesDiv.innerHTML = window.uploadedFiles.map((file, index) => `
        <div class="file-chip">
            <span>📄 ${file.name}</span>
            <span class="file-chip-remove" onclick="removeFile(${index})">×</span>
        </div>
    `).join('');
    console.log('✅ Files displayed:', window.uploadedFiles.map(f => f.name));
}

function removeFile(index) {
    console.log('🗑️ Removing file at index:', index, window.uploadedFiles[index]?.name);
    window.uploadedFiles.splice(index, 1);
    console.log('📎 Files remaining:', window.uploadedFiles.length);
    updateAttachedFilesUI();
}

// ── Drag & Drop file support ────────────────────────────────────────────────
function initDragAndDrop() {
    const messagesContainer = document.getElementById('messagesContainer');
    const chatArea = document.querySelector('.chat-area') || messagesContainer?.parentElement;
    if (!chatArea) return;

    let dragCounter = 0;

    chatArea.addEventListener('dragenter', (e) => {
        e.preventDefault();
        dragCounter++;
        chatArea.classList.add('drag-over');
    });
    chatArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dragCounter--;
        if (dragCounter <= 0) {
            dragCounter = 0;
            chatArea.classList.remove('drag-over');
        }
    });
    chatArea.addEventListener('dragover', (e) => { e.preventDefault(); });
    chatArea.addEventListener('drop', async (e) => {
        e.preventDefault();
        dragCounter = 0;
        chatArea.classList.remove('drag-over');
        
        const files = Array.from(e.dataTransfer.files);
        if (files.length === 0) return;

        for (const file of files) {
            if (file.size > 10 * 1024 * 1024) {
                alert(`File too large: ${file.name} (max 10MB)`);
                continue;
            }
            const fileData = await readFileContent(file);
            window.uploadedFiles.push({
                name: file.name,
                type: file.type,
                content: fileData,
                isImage: file.type.startsWith('image/'),
                isPdf: file.type === 'application/pdf'
            });
        }
        updateAttachedFilesUI();
    });
}

// ── Paste image from clipboard ──────────────────────────────────────────────
function initClipboardPaste() {
    const messageInput = document.getElementById('messageInput');
    if (!messageInput) return;
    
    messageInput.addEventListener('paste', async (e) => {
        const items = Array.from(e.clipboardData?.items || []);
        for (const item of items) {
            if (item.type.startsWith('image/')) {
                e.preventDefault();
                const file = item.getAsFile();
                if (!file) continue;
                const fileData = await readFileContent(file);
                window.uploadedFiles.push({
                    name: `pasted-image-${Date.now()}.png`,
                    type: file.type,
                    content: fileData,
                    isImage: true,
                    isPdf: false
                });
                updateAttachedFilesUI();
            }
        }
    });
}

// Hardware Monitoring
let hardwareInterval = null;

// Global function to toggle hardware monitor
window.toggleHardwareMonitor = function(event) {
    if (event) {
        event.preventDefault();
        event.stopPropagation();
    }
    console.log('🖥️ toggleHardwareMonitor called');
    const hardwareMonitor = document.getElementById('hardwareMonitor');
    
    if (!hardwareMonitor) {
        console.error('❌ Hardware monitor element not found!');
        return;
    }
    
    const currentDisplay = window.getComputedStyle(hardwareMonitor).display;
    console.log('Current display:', currentDisplay, 'Element:', hardwareMonitor);
    
    if (currentDisplay === 'none') {
        console.log('✅ Showing hardware monitor');
        hardwareMonitor.style.display = 'block';
        console.log('After setting display to block:', hardwareMonitor.style.display);
        startHardwareMonitoring();
    } else {
        console.log('✅ Hiding hardware monitor');
        hardwareMonitor.style.display = 'none';
        if (hardwareInterval) {
            clearInterval(hardwareInterval);
            hardwareInterval = null;
        }
    }
};

function initHardwareMonitor() {
    const monitorBtn = document.getElementById('monitorBtn');
    const hardwareMonitor = document.getElementById('hardwareMonitor');
    const hwCloseBtn = document.getElementById('hwCloseBtn');

    if (!monitorBtn || !hardwareMonitor) {
        console.error('Hardware monitor elements not found:', { monitorBtn: !!monitorBtn, hardwareMonitor: !!hardwareMonitor });
        return;
    }

    // Don't add event listener since we're using onclick attribute
    // monitorBtn.addEventListener('click', () => {
    //     console.log('Monitor button clicked via event listener');
    //     window.toggleHardwareMonitor();
    // });
    if (hwCloseBtn) {
        hwCloseBtn.addEventListener('click', () => {
            hardwareMonitor.style.display = 'none';
            if (hardwareInterval) {
                clearInterval(hardwareInterval);
                hardwareInterval = null;
            }
        });
    }
}

function startHardwareMonitoring() {
    updateHardwareStats(); // Initial update
    
    if (hardwareInterval) clearInterval(hardwareInterval);
    hardwareInterval = setInterval(updateHardwareStats, 2000); // Update every 2 seconds
}

async function updateHardwareStats() {
    try {
        const response = await fetch(`${API_ENDPOINT}/system/stats`);
        if (!response.ok) {
            console.log('System stats endpoint returned', response.status);
            return;
        }
        
        const data = await response.json();
        updateHardwareUI(data);
    } catch (error) {
        console.log('Hardware monitoring not available - is EDISON server running?');
    }
}

function getBarColor(percent) {
    if (percent >= 90) return '#ef4444';
    if (percent >= 70) return '#f59e0b';
    return '';
}

function getTempColor(tempC) {
    if (tempC >= 85) return '#ef4444';
    if (tempC >= 70) return '#f59e0b';
    return '';
}

function updateHardwareUI(stats) {
    // Hostname
    const hostnameEl = document.getElementById('hwHostname');
    if (hostnameEl && stats.hostname) {
        hostnameEl.textContent = stats.hostname + (stats.os ? ` • ${stats.os}` : '');
    }

    // CPU
    const cpuPercent = stats.cpu?.percent ?? stats.cpu_percent ?? 0;
    const cpuBar = document.getElementById('cpuBar');
    cpuBar.style.width = `${cpuPercent}%`;
    const cpuColor = getBarColor(cpuPercent);
    if (cpuColor) cpuBar.style.background = cpuColor;
    else cpuBar.style.background = '';
    document.getElementById('cpuValue').textContent = `${cpuPercent.toFixed(1)}%`;
    
    const cpuNameEl = document.getElementById('cpuName');
    if (cpuNameEl && stats.cpu) {
        const cores = stats.cpu.cores_physical ? `${stats.cpu.cores_physical}C/${stats.cpu.cores_logical}T` : '';
        const freq = stats.cpu.frequency_ghz ? `${stats.cpu.frequency_ghz} GHz` : '';
        const parts = [stats.cpu.name, cores, freq].filter(Boolean);
        cpuNameEl.textContent = parts.join(' • ');
    }
    
    // CPU Temperature
    const cpuTempC = stats.cpu_temp_c || 0;
    const cpuTempBar = document.getElementById('cpuTempBar');
    const cpuTempValue = document.getElementById('cpuTempValue');
    if (cpuTempC > 0) {
        const tempPercent = Math.min(cpuTempC / 100, 1) * 100;
        cpuTempBar.style.width = `${tempPercent}%`;
        const tempColor = getTempColor(cpuTempC);
        if (tempColor) cpuTempBar.style.background = tempColor;
        else cpuTempBar.style.background = '';
        cpuTempValue.textContent = `${cpuTempC.toFixed(0)}°C`;
    } else {
        cpuTempBar.style.width = '0%';
        cpuTempValue.textContent = 'N/A';
    }
    
    // GPUs (dynamic)
    const gpuContainer = document.getElementById('gpuStatsContainer');
    if (gpuContainer && stats.gpus && stats.gpus.length > 0) {
        gpuContainer.innerHTML = stats.gpus.map((gpu, i) => {
            const memPercent = gpu.memory_total_gb > 0 ? (gpu.memory_used_gb / gpu.memory_total_gb) * 100 : 0;
            const utilColor = getBarColor(gpu.utilization_percent);
            const memColor = getBarColor(memPercent);
            const tempColor = getTempColor(gpu.temperature_c);
            const tempPercent = Math.min(gpu.temperature_c / 100, 1) * 100;
            const powerStr = gpu.power_watts > 0 ? ` • ${gpu.power_watts}W` : '';
            return `
                <div class="hw-section-label">GPU ${gpu.index}</div>
                <div class="hw-detail">${gpu.name}${powerStr}</div>
                <div class="hw-stat">
                    <span class="hw-label">Usage</span>
                    <div class="hw-bar"><div class="hw-fill" style="width:${gpu.utilization_percent}%;${utilColor ? 'background:' + utilColor : ''}"></div></div>
                    <span class="hw-value">${gpu.utilization_percent.toFixed(0)}%</span>
                </div>
                <div class="hw-stat">
                    <span class="hw-label">VRAM</span>
                    <div class="hw-bar"><div class="hw-fill" style="width:${memPercent}%;${memColor ? 'background:' + memColor : ''}"></div></div>
                    <span class="hw-value">${gpu.memory_used_gb.toFixed(1)} / ${gpu.memory_total_gb.toFixed(0)} GB</span>
                </div>
                <div class="hw-stat">
                    <span class="hw-label">Temp</span>
                    <div class="hw-bar hw-bar-temp"><div class="hw-fill hw-fill-temp" style="width:${tempPercent}%;${tempColor ? 'background:' + tempColor : ''}"></div></div>
                    <span class="hw-value">${gpu.temperature_c > 0 ? gpu.temperature_c.toFixed(0) + '°C' : 'N/A'}</span>
                </div>
            `;
        }).join('');
    } else if (gpuContainer) {
        gpuContainer.innerHTML = `
            <div class="hw-section-label">GPU</div>
            <div class="hw-detail" style="color: var(--text-secondary)">No NVIDIA GPU detected</div>
        `;
    }
    
    // RAM
    const ramUsed = stats.ram?.used_gb ?? stats.ram_used_gb ?? 0;
    const ramTotal = stats.ram?.total_gb ?? stats.ram_total_gb ?? 1;
    const ramPercent = stats.ram?.percent ?? ((ramUsed / ramTotal) * 100);
    const ramBar = document.getElementById('ramBar');
    ramBar.style.width = `${ramPercent}%`;
    const ramColor = getBarColor(ramPercent);
    if (ramColor) ramBar.style.background = ramColor;
    else ramBar.style.background = '';
    document.getElementById('ramValue').textContent = `${ramUsed.toFixed(1)} / ${ramTotal.toFixed(0)} GB`;
    
    // Disk
    const diskBar = document.getElementById('diskBar');
    const diskValue = document.getElementById('diskValue');
    if (diskBar && diskValue && stats.disk) {
        diskBar.style.width = `${stats.disk.percent}%`;
        const diskColor = getBarColor(stats.disk.percent);
        if (diskColor) diskBar.style.background = diskColor;
        else diskBar.style.background = '';
        diskValue.textContent = `${stats.disk.used_gb.toFixed(0)} / ${stats.disk.total_gb.toFixed(0)} GB`;
    }
}

// Work Mode Desktop
let workModeActive = false;

function initWorkMode() {
    const workCloseBtn = document.getElementById('workCloseBtn');
    const workDesktop = document.getElementById('workDesktop');

    if (!workCloseBtn || !workDesktop) return;

    workCloseBtn.addEventListener('click', () => {
        workDesktop.classList.remove('visible');
        setTimeout(() => { workDesktop.style.display = 'none'; }, 250);
        workModeActive = false;
        window.workModeActive = false;
    });

    // Listen for work mode selection
    const workModeBtn = document.querySelector('[data-mode="work"]');
    if (workModeBtn) {
        workModeBtn.addEventListener('click', () => {
            workDesktop.style.display = 'block';
            workDesktop.offsetHeight; // trigger reflow
            workDesktop.classList.add('visible');
            workModeActive = true;
            window.workModeActive = true;
            // Reset panels on fresh activation
            resetWorkDesktop();
        });
    }
}

function resetWorkDesktop() {
    const currentTask = document.getElementById('currentTask');
    if (currentTask) currentTask.textContent = 'Waiting for task...';

    const searchResults = document.getElementById('searchResults');
    if (searchResults) searchResults.innerHTML = '<div class="result-item" style="opacity: 0.5;">No search results yet</div>';

    const workDocs = document.getElementById('workDocuments');
    if (workDocs) workDocs.innerHTML = '<div class="doc-item" style="opacity: 0.5;">No documents loaded</div>';

    const thinkingLog = document.getElementById('thinkingLog');
    if (thinkingLog) thinkingLog.innerHTML = '';

    const stepList = document.getElementById('workStepList');
    if (stepList) stepList.innerHTML = '';

    const progressFill = document.getElementById('stepProgressFill');
    if (progressFill) progressFill.style.width = '0%';

    const statusBadge = document.getElementById('workStatusBadge');
    if (statusBadge) {
        statusBadge.textContent = 'Ready';
        statusBadge.className = 'work-status-badge';
    }
}

function updateWorkDesktop(task, searchResults, documents, thinkingLog) {
    const currentTask = document.getElementById('currentTask');
    if (currentTask) {
        if (typeof task === 'string') {
            currentTask.textContent = task || 'Waiting for task...';
        }
    }
    
    // Update search results
    const searchResultsDiv = document.getElementById('searchResults');
    if (searchResultsDiv && searchResults && searchResults.length > 0) {
        searchResultsDiv.innerHTML = searchResults.map(result => `
            <div class="result-item">
                <div class="result-title">${result.title || ''}</div>
                <div class="result-snippet">${(result.body || result.snippet || '').substring(0, 200)}</div>
                ${result.url ? `<a href="${result.url}" target="_blank" rel="noopener" class="result-url">${result.url}</a>` : ''}
            </div>
        `).join('');
    }
    
    // Update documents
    const docsDiv = document.getElementById('workDocuments');
    if (docsDiv && documents && documents.length > 0) {
        docsDiv.innerHTML = documents.map(doc => `
            <div class="doc-item">
                <span class="doc-icon">📄</span>
                <span class="doc-name">${doc.name || doc}</span>
            </div>
        `).join('');
    }
    
    // Update thinking log
    if (thinkingLog) {
        addThinkingLogEntry(thinkingLog);
    }

    // Update status badge
    const statusBadge = document.getElementById('workStatusBadge');
    if (statusBadge) {
        statusBadge.textContent = 'Working...';
        statusBadge.className = 'work-status-badge working';
    }
}

function updateWorkStepProgress(steps) {
    /**Update step progress bar and step list in work desktop*/
    if (!steps || !steps.length) return;

    const completed = steps.filter(s => s.status === 'completed').length;
    const total = steps.length;
    const pct = Math.round((completed / total) * 100);

    const progressFill = document.getElementById('stepProgressFill');
    if (progressFill) progressFill.style.width = `${pct}%`;

    const stepList = document.getElementById('workStepList');
    if (stepList) {
        const kindIcons = {
            'llm': '🧠', 'search': '🔍', 'code': '💻',
            'artifact': '📄', 'vision': '👁️', 'tool': '🔧', 'comfyui': '🎨'
        };
        const statusIcons = {
            'pending': '⏳', 'running': '🔄', 'completed': '✅', 'failed': '❌', 'skipped': '⏭️'
        };

        stepList.innerHTML = steps.map(step => {
            const kindIcon = kindIcons[step.kind] || '⚙️';
            const statusIcon = statusIcons[step.status] || '⏳';
            const timeStr = step.elapsed_ms ? `${(step.elapsed_ms / 1000).toFixed(1)}s` : '';
            return `
                <div class="work-step-desktop-item step-${step.status}">
                    <span class="step-icon">${kindIcon}</span>
                    <span class="step-num">${step.id}.</span>
                    <span class="step-text">${step.title || ''}</span>
                    <span class="step-status-icon">${statusIcon}</span>
                    ${timeStr ? `<span class="step-time-label">${timeStr}</span>` : ''}
                </div>
            `;
        }).join('');
    }

    const statusBadge = document.getElementById('workStatusBadge');
    if (statusBadge) {
        if (completed === total) {
            statusBadge.textContent = 'Complete';
            statusBadge.className = 'work-status-badge complete';
        } else {
            statusBadge.textContent = `${completed}/${total}`;
            statusBadge.className = 'work-status-badge working';
        }
    }
}

function addThinkingLogEntry(entry) {
    const logDiv = document.getElementById('thinkingLog');
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement('div');
    logEntry.className = 'log-entry';
    logEntry.innerHTML = `<span style="opacity: 0.6;">[${timestamp}]</span> ${entry}`;
    logDiv.appendChild(logEntry);
    logDiv.scrollTop = logDiv.scrollHeight;
}

// Make functions globally available
window.updateWorkDesktop = updateWorkDesktop;
window.addThinkingLogEntry = addThinkingLogEntry;
window.updateWorkStepProgress = updateWorkStepProgress;
window.resetWorkDesktop = resetWorkDesktop;

// Chat Search Functionality
function initChatSearch() {
    const chatSearchInput = document.getElementById('chatSearchInput');
    const chatHistory = document.getElementById('chatHistory');

    if (!chatSearchInput || !chatHistory) {
        console.error('Chat search elements not found:', { chatSearchInput: !!chatSearchInput, chatHistory: !!chatHistory });
        return;
    }

    chatSearchInput.addEventListener('input', (e) => {
        console.log('Chat search input:', e.target.value);
        const searchTerm = e.target.value.toLowerCase();
        const historyItems = chatHistory.querySelectorAll('.chat-history-item');

        historyItems.forEach(item => {
            const text = item.textContent.toLowerCase();
            if (text.includes(searchTerm) || searchTerm === '') {
                item.style.display = 'flex';
            } else {
                item.style.display = 'none';
            }
        });
    });
}

// Dynamic Chat History Naming
async function generateChatTitle(messages) {
    if (messages.length === 0) return 'New Chat';
    
    // Extract first user message
    const firstMessage = messages.find(m => m.role === 'user');
    if (!firstMessage) return 'New Chat';
    
    // Use first 50 chars or generate with AI
    const shortTitle = firstMessage.content.substring(0, 50);
    
    // Optionally call AI to generate a better title
    try {
        const response = await fetch(`${API_ENDPOINT}/generate-title`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: firstMessage.content })
        });
        
        if (response.ok) {
            const data = await response.json();
            return data.title || shortTitle;
        }
    } catch (error) {
        console.log('Using default title generation');
    }
    
    // Fallback: create descriptive title from content
    return generateSmartTitle(firstMessage.content);
}

function generateSmartTitle(content) {
    // Remove common prefixes
    let title = content
        .replace(/^(can you|could you|please|help me|i need|how do i|what is|tell me about)/i, '')
        .trim();
    
    // Capitalize first letter
    title = title.charAt(0).toUpperCase() + title.slice(1);
    
    // Truncate to reasonable length
    if (title.length > 40) {
        title = title.substring(0, 40) + '...';
    }
    
    return title || 'New Chat';
}

// Enhanced chat history with dynamic names
function updateChatHistoryWithDynamicNames(chatId, messages) {
    const title = generateChatTitle(messages);
    const historyItem = document.querySelector(`[data-chat-id="${chatId}"]`);
    
    if (historyItem) {
        const titleElement = historyItem.querySelector('.history-title');
        if (titleElement) {
            titleElement.textContent = title;
        }
    }
    
    // Also update in storage
    const chats = JSON.parse(localStorage.getItem('chatHistory') || '{}');
    if (chats[chatId]) {
        chats[chatId].title = title;
        localStorage.setItem('chatHistory', JSON.stringify(chats));
    }
}

// Initialize all features
function initAllFeatures() {
    // Only initialize if not already done
    if (window.EDISON_FEATURES_INITIALIZED) {
        console.log('✓ Enhanced features already initialized');
        return;
    }
    
    // Check if main app is ready
    if (!window.edisonApp) {
        console.log('⏳ Waiting for main app to initialize...');
        setTimeout(initAllFeatures, 200);
        return;
    }
    
    window.EDISON_FEATURES_INITIALIZED = true;
    
    console.log('🚀 Initializing enhanced features...');
    console.log('DOM elements check:', {
        attachBtn: !!document.getElementById('attachBtn'),
        fileInput: !!document.getElementById('fileInput'),
        monitorBtn: !!document.getElementById('monitorBtn'),
        hardwareMonitor: !!document.getElementById('hardwareMonitor'),
        chatSearchInput: !!document.getElementById('chatSearchInput'),
        chatHistory: !!document.getElementById('chatHistory')
    });
    
    initFileUpload();
    console.log('  ✓ File upload ready');
    
    initDragAndDrop();
    console.log('  ✓ Drag & drop ready');
    
    initClipboardPaste();
    console.log('  ✓ Clipboard paste ready');
    
    initHardwareMonitor();
    console.log('  ✓ Hardware monitor ready');
    
    initWorkMode();
    console.log('  ✓ Work mode ready');
    
    initChatSearch();
    console.log('  ✓ Chat search ready');
    
    console.log('✅ All enhanced features initialized');
    
    // Test button clicks manually
    const testBtn = document.getElementById('attachBtn');
    if (testBtn) {
        console.log('Attach button element:', testBtn);
        console.log('Attach button computed style:', window.getComputedStyle(testBtn).pointerEvents);
    }
}

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAllFeatures);
} else {
    initAllFeatures();
}

// Export functions for use in main app
window.EDISON_Features = {
    removeFile,
    updateWorkDesktop,
    updateWorkStepProgress,
    resetWorkDesktop,
    addThinkingLogEntry,
    generateChatTitle,
    updateChatHistoryWithDynamicNames
};
