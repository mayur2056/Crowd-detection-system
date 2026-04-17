// ── DOM refs ────────────────────────────────────────────────────────────────
const videoStream = document.getElementById('videoStream');
const noSignal = document.getElementById('noSignal');
const reconnectingOverlay = document.getElementById('reconnectingOverlay');
const peopleCount = document.getElementById('peopleCount');
const statusBadge = document.getElementById('statusBadge');
const statusIndicator = document.getElementById('statusIndicator');
const statusLabel = document.getElementById('statusLabel');
const timestampDisplay = document.getElementById('timestampDisplay');
const latencyCalc = document.getElementById('latencyCalc');
const densityCard = document.getElementById('densityCard');
const alertBox = document.getElementById('alertBox');
const alertIcon = document.getElementById('alertIcon');
const alertTitle = document.getElementById('alertTitle');
const alertMessage = document.getElementById('alertMessage');
const statusText = document.getElementById('statusText');
const statusDot = document.getElementById('statusDot');
const alertSound = document.getElementById('alertSound');
const sessionInstructions = document.getElementById('sessionInstructions');
const connectedSessionCode = document.getElementById('connectedSessionCode');
const currentSessionCode = document.getElementById('currentSessionCode');

// ── State ───────────────────────────────────────────────────────────────────
let ws;
let lastFrameTime = Date.now();
let checkConnectionInterval;
let activeSessionCode = null;     // current 6-char code
let activeServerHost = null;     // current host (no protocol)

let currentStableStatus = null;
let stableStateStartTime = Date.now();
let hasSpokenForState = false;

// FIX: Track the pending preload image so we can cancel it if a newer
// frame arrives before the previous one finishes loading. Without this,
// out-of-order onload callbacks could display an older frame over a newer one.
let pendingImage = null;

const STORAGE_KEY = 'crowdpulse_server_host';
const CODE_STORAGE_KEY = 'crowdpulse_session_code';

// ── Helpers ─────────────────────────────────────────────────────────────────
function getServerHost() {
    const raw = document.getElementById('serverUrlInput').value.trim();
    if (raw) return raw.replace(/^wss?:\/\//, '').replace(/^https?:\/\//, '');
    return localStorage.getItem(STORAGE_KEY) || '';
}

function getHttpBase(host) {
    const isLocal = host.startsWith('localhost') || host.startsWith('127.') || host.startsWith('10.0.');
    return isLocal ? `http://${host}` : `https://${host}`;
}

function getWsBase(host) {
    const isLocal = host.startsWith('localhost') || host.startsWith('127.') || host.startsWith('10.0.');
    return isLocal ? `ws://${host}` : `wss://${host}`;
}

// ── UI Connection ─────────────────────────────────────────────────────────
function connectFromUI() {
    const host = getServerHost();
    const code = document.getElementById('sessionCodeInput').value.trim().toUpperCase();

    if (!host) { alert('Please enter the backend server URL first.'); return; }
    if (code.length < 6) { alert('Please enter a valid 6-char session code.'); return; }

    localStorage.setItem(STORAGE_KEY, host);
    localStorage.setItem(CODE_STORAGE_KEY, code);

    // Show session connection success
    sessionInstructions.classList.remove('hidden');
    connectedSessionCode.textContent = code;
    currentSessionCode.textContent = code;

    connectToDashboard(host, code);
}

// ── WebSocket connection (code-scoped) ────────────────────────────────────────
function connectToDashboard(host, code) {
    activeServerHost = host;
    activeSessionCode = code;

    const wsUrl = `${getWsBase(host)}/ws/dashboard/${code}`;

    statusText.textContent = 'Connecting...';
    statusDot.className = 'w-3 h-3 rounded-full bg-amber-400 animate-pulse';

    if (ws) {
        ws.close();
        ws = null;
    }

    ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
        statusText.textContent = 'Connected';
        statusDot.className = 'w-3 h-3 rounded-full bg-emerald-500 animate-pulse';
        clearInterval(checkConnectionInterval);
        checkConnectionInterval = setInterval(checkFallbackState, 1000);
    };

    ws.onclose = (event) => {
        statusText.textContent = 'Disconnected';
        statusDot.className = 'w-3 h-3 rounded-full bg-red-500';
        clearInterval(checkConnectionInterval);

        // If the backend restarted or the session expired, it sends code 4404
        if (event.code === 4404) {
            console.warn("Session expired or invalid code from server.");
            alert("This session has ended (the server restarted or connection expired). Please click 'New Session' to start a new one.");
            localStorage.removeItem(CODE_STORAGE_KEY);
            activeSessionCode = null;
            return; // Don't auto-retry
        }

        // Auto-retry after 4 seconds with the same code (don't generate a new one)
        setTimeout(() => {
            if (activeSessionCode && activeServerHost) {
                connectToDashboard(activeServerHost, activeSessionCode);
            }
        }, 4000);
    };

    ws.onerror = () => {
        statusText.textContent = 'Error';
        statusDot.className = 'w-3 h-3 rounded-full bg-red-500';
    };

    ws.onmessage = (event) => {
        if (typeof event.data === "string") {
            const data = JSON.parse(event.data);
            const serverToBrowserMs = Date.now() - (data.timestamp * 1000);
            updateDashboard(data);
        } else if (event.data instanceof ArrayBuffer) {
            const view = new Uint8Array(event.data);
            const type = view[0];
            const payload = event.data.slice(1);

            if (type === 0) {
                renderFrame(payload, videoStream, 'previousImageUrl', document.getElementById('noSignal'));
            } else if (type === 1) {
                renderFrame(payload, document.getElementById('heatmapStream'), 'previousHeatmapUrl', document.getElementById('noSignalHm'));
            }
        }
    };
}

// ── Frame rendering ───────────────────────────────────────────────────────────
let currentFrameId = 0;

function renderFrame(arrayBuffer, targetImgEl, keyName, fallbackEl) {
    const blob = new Blob([arrayBuffer], { type: "image/jpeg" });
    const newUrl = URL.createObjectURL(blob);
    
    currentFrameId++;
    const thisFrameId = currentFrameId;

    const img = new Image();

    img.onload = () => {
        if (window[keyName]) {
            URL.revokeObjectURL(window[keyName]);
        }

        targetImgEl.src = newUrl;
        window[keyName] = newUrl;

        targetImgEl.classList.remove('hidden');
        if (fallbackEl) fallbackEl.classList.add('hidden');
        reconnectingOverlay.classList.replace('opacity-100', 'opacity-0');
        reconnectingOverlay.classList.add('pointer-events-none');

        lastFrameTime = Date.now();
    };

    img.onerror = () => {
        URL.revokeObjectURL(newUrl);
    };

    img.src = newUrl;
}


// ── Dashboard update ─────────────────────────────────────────────────────────
function updateDashboard(data) {
    peopleCount.textContent = data.count;

    const d = new Date();
    timestampDisplay.textContent = d.toLocaleTimeString();

    applyStatusColor(data.status);

    if (data.status !== currentStableStatus) {
        currentStableStatus = data.status;
        stableStateStartTime = Date.now();
        hasSpokenForState = false;
    } else {
        const duration = (Date.now() - stableStateStartTime) / 1000;
        if (duration > 3.0 && !hasSpokenForState) {
            hasSpokenForState = true;
            triggerSpeechAlert(data.count, data.status);
            
            // Visual alert box trigger
            if (data.status === "RED") triggerAlert();
            else clearAlert();
        }
    }
}

function applyStatusColor(status) {
    statusLabel.textContent = status;
    statusBadge.className = "inline-flex w-full justify-center items-center gap-2 px-4 py-3 rounded-xl font-bold transition-colors duration-300";
    densityCard.className = "glass-panel rounded-2xl p-6 transition-all duration-500 border-2";

    if (status === "GREEN") {
        statusIndicator.className = "w-2.5 h-2.5 rounded-full bg-emerald-500";
        statusBadge.classList.add('bg-emerald-100', 'text-emerald-700');
        densityCard.classList.add('border-emerald-200');
    } else if (status === "BLUE") {
        statusIndicator.className = "w-2.5 h-2.5 rounded-full bg-blue-500 animate-pulse";
        statusBadge.classList.add('bg-blue-100', 'text-blue-700');
        densityCard.classList.add('border-blue-200');
    } else if (status === "RED") {
        statusIndicator.className = "w-2.5 h-2.5 rounded-full bg-red-600 animate-ping";
        statusBadge.classList.add('bg-red-100', 'text-red-700');
        densityCard.classList.add('border-red-300', 'shadow-lg', 'shadow-red-500/20');
    }
}

async function triggerSpeechAlert(count, status) {
    try {
        const hostUrl = getHttpBase(activeServerHost);
        const res = await fetch(`${hostUrl}/api/generate_speech`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({count, status})
        });
        const data = await res.json();
        
        const utterance = new SpeechSynthesisUtterance(data.speech);
        // Try to pick a natural Microsoft voice if available
        const voices = speechSynthesis.getVoices();
        const msVoice = voices.find(v => v.name.includes("Microsoft") && v.name.includes("Natural"));
        if (msVoice) utterance.voice = msVoice;
        
        speechSynthesis.speak(utterance);
    } catch (e) {
        console.error("Speech AI error:", e);
    }
}

function triggerAlert() {
    isAlertActive = true;
    alertBox.className = "glass-panel rounded-2xl p-6 border-l-4 border-red-500 opacity-100 transition-all duration-300 pulse-red bg-red-50";
    alertIcon.className = "p-2 rounded-lg bg-red-100 text-red-600 animate-bounce";
    alertIcon.innerHTML = `<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>`;
    alertTitle.textContent = "CRITICAL SERVER ALERT";
    alertTitle.classList.add('text-red-700');
    alertMessage.textContent = "High density threshold exceeded. Dispatching security.";
    alertMessage.classList.replace('text-slate-500', 'text-red-600');
}

function clearAlert() {
    isAlertActive = false;
    alertBox.className = "glass-panel rounded-2xl p-6 border-l-4 border-slate-300 opacity-50 transition-all duration-300";
    alertIcon.className = "p-2 rounded-lg bg-slate-100 text-slate-400";
    alertIcon.innerHTML = `<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>`;
    alertTitle.textContent = "System Monitoring";
    alertTitle.classList.remove('text-red-700');
    alertMessage.textContent = "Density stabilized.";
    alertMessage.classList.replace('text-red-600', 'text-slate-500');
}

function checkFallbackState() {
    // If the video stream is still hidden, we haven't received the first frame yet.
    // Let the default "Waiting for mobile stream..." UI handle it.
    if (videoStream.classList.contains('hidden')) {
        return;
    }

    const timeSinceLastFrame = Date.now() - lastFrameTime;
    latencyCalc.textContent = `${Math.floor(timeSinceLastFrame / 1000)}s ago`;

    if (timeSinceLastFrame > 35000) {
        reconnectingOverlay.classList.replace('opacity-0', 'opacity-100');
        reconnectingOverlay.classList.remove('pointer-events-none');
    }
}

// ── On load: restore saved session ───────────────────────────────────────────
const savedHost = localStorage.getItem(STORAGE_KEY);
const savedCode = localStorage.getItem(CODE_STORAGE_KEY);

if (savedHost) {
    document.getElementById('serverUrlInput').value = savedHost;
}
if (savedCode) {
    document.getElementById('sessionCodeInput').value = savedCode;
}

if (savedHost && savedCode) {
    connectFromUI();
}