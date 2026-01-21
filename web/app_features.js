// ===========================================
// EDISON Enhanced Features
// File Upload, Hardware Monitor, Work Mode, etc.
// ===========================================

// Wait for main app to initialize
let API_ENDPOINT = 'http://192.168.1.26:8811';

// File Upload Handling
let uploadedFiles = [];

// Global function to trigger file selection
window.triggerFileUpload = function(event) {
    if (event) {
        event.preventDefault();
        event.stopPropagation();
    }
    console.log('triggerFileUpload called');
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        console.log('Clicking file input');
        fileInput.click();
    } else {
        console.error('File input not found!');
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
        const files = Array.from(e.target.files);
        
        for (const file of files) {
            if (file.size > 10 * 1024 * 1024) { // 10MB limit
                console.log(`File ${file.name} is too large (max 10MB)`);
                continue;
            }

            const fileData = await readFileContent(file);
            uploadedFiles.push({
                name: file.name,
                type: file.type,
                content: fileData
            });
        }

        updateAttachedFilesUI();
        fileInput.value = ''; // Reset input
    });
}

async function readFileContent(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = reject;
        reader.readAsText(file);
    });
}

function updateAttachedFilesUI() {
    const attachedFilesDiv = document.getElementById('attachedFiles');
    
    if (uploadedFiles.length === 0) {
        attachedFilesDiv.style.display = 'none';
        return;
    }

    attachedFilesDiv.style.display = 'flex';
    attachedFilesDiv.innerHTML = uploadedFiles.map((file, index) => `
        <div class="file-chip">
            <span>📄 ${file.name}</span>
            <span class="file-chip-remove" onclick="removeFile(${index})">×</span>
        </div>
    `).join('');
}

function removeFile(index) {
    uploadedFiles.splice(index, 1);
    updateAttachedFilesUI();
}

// Hardware Monitoring
let hardwareInterval = null;

// Global function to toggle hardware monitor
window.toggleHardwareMonitor = function(event) {
    if (event) {
        event.preventDefault();
        event.stopPropagation();
    }
    console.log('toggleHardwareMonitor called');
    const hardwareMonitor = document.getElementById('hardwareMonitor');
    
    if (!hardwareMonitor) {
        console.error('Hardware monitor element not found!');
        return;
    }
    
    const currentDisplay = window.getComputedStyle(hardwareMonitor).display;
    console.log('Current display:', currentDisplay);
    
    if (currentDisplay === 'none') {
        console.log('Showing hardware monitor');
        hardwareMonitor.style.display = 'block';
        startHardwareMonitoring();
    } else {
        console.log('Hiding hardware monitor');
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
            // Fallback to mock data if endpoint doesn't exist yet
            updateHardwareUI({
                cpu_percent: Math.random() * 100,
                gpu_percent: Math.random() * 100,
                ram_used_gb: 12 + Math.random() * 10,
                ram_total_gb: 64,
                temp_c: 45 + Math.random() * 20
            });
            return;
        }
        
        const data = await response.json();
        updateHardwareUI(data);
    } catch (error) {
        console.log('Hardware monitoring not available yet');
    }
}

function updateHardwareUI(stats) {
    document.getElementById('cpuBar').style.width = `${stats.cpu_percent}%`;
    document.getElementById('cpuValue').textContent = `${stats.cpu_percent.toFixed(1)}%`;
    
    document.getElementById('gpuBar').style.width = `${stats.gpu_percent}%`;
    document.getElementById('gpuValue').textContent = `${stats.gpu_percent.toFixed(1)}%`;
    
    const ramPercent = (stats.ram_used_gb / stats.ram_total_gb) * 100;
    document.getElementById('ramBar').style.width = `${ramPercent}%`;
    document.getElementById('ramValue').textContent = `${stats.ram_used_gb.toFixed(1)}GB`;
    
    document.getElementById('tempValue').textContent = `${stats.temp_c.toFixed(0)}°C`;
}

// Work Mode Desktop
let workModeActive = false;

function initWorkMode() {
    const workCloseBtn = document.getElementById('workCloseBtn');
    const workDesktop = document.getElementById('workDesktop');

    if (!workCloseBtn || !workDesktop) return;

    workCloseBtn.addEventListener('click', () => {
        workDesktop.style.display = 'none';
        workModeActive = false;
    });

    // Listen for work mode selection
    const workModeBtn = document.querySelector('[data-mode="work"]');
    if (workModeBtn) {
        workModeBtn.addEventListener('click', () => {
            workDesktop.style.display = 'block';
            workModeActive = true;
        });
    }
}

function updateWorkDesktop(task, searchResults, documents, thinkingLog) {
    document.getElementById('currentTask').textContent = task || 'Waiting for task...';
    
    // Update search results
    const searchResultsDiv = document.getElementById('searchResults');
    if (searchResults && searchResults.length > 0) {
        searchResultsDiv.innerHTML = searchResults.map(result => `
            <div class="result-item">
                <div class="result-title">${result.title}</div>
                <div class="result-snippet">${result.snippet}</div>
            </div>
        `).join('');
    }
    
    // Update documents
    const docsDiv = document.getElementById('workDocuments');
    if (documents && documents.length > 0) {
        docsDiv.innerHTML = documents.map(doc => `
            <div class="doc-item">
                <span class="doc-icon">📄</span>
                <span class="doc-name">${doc.name}</span>
            </div>
        `).join('');
    }
    
    // Update thinking log
    if (thinkingLog) {
        addThinkingLogEntry(thinkingLog);
    }
}

function addThinkingLogEntry(entry) {
    const logDiv = document.getElementById('thinkingLog');
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement('div');
    logEntry.className = 'log-entry';
    logEntry.textContent = `[${timestamp}] ${entry}`;
    logDiv.appendChild(logEntry);
    logDiv.scrollTop = logDiv.scrollHeight;
}

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
    uploadedFiles,
    removeFile,
    updateWorkDesktop,
    addThinkingLogEntry,
    generateChatTitle,
    updateChatHistoryWithDynamicNames
};
