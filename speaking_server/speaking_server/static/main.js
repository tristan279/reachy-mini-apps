console.log("[MAIN.JS] Script loaded");

let antennasEnabled = true;

async function updateAntennasState(enabled) {
    try {
        const resp = await fetch("/antennas", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ enabled }),
        });
        const data = await resp.json();
        antennasEnabled = data.antennas_enabled;
        updateUI();
    } catch (e) {
        document.getElementById("status").textContent = "Backend error";
    }
}

async function playSound() {
    try {
        await fetch("/play_sound", { method: "POST" });
    } catch (e) {
        console.error("Error triggering sound:", e);
    }
}

function updateUI() {
    const checkbox = document.getElementById("antenna-checkbox");
    const status = document.getElementById("status");

    checkbox.checked = antennasEnabled;

    if (antennasEnabled) {
        status.textContent = "Antennas status: running";
    } else {
        status.textContent = "Antennas status: stopped";
    }
}

document.getElementById("antenna-checkbox").addEventListener("change", (e) => {
    updateAntennasState(e.target.checked);
});

document.getElementById("conversation-checkbox").addEventListener("change", async (e) => {
    if (e.target.checked) {
        try {
            const resp = await fetch("/api/conversation", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
            });
            const data = await resp.json();
            console.log("[CONVERSATION] Response:", data);
            
            // Display the response text if needed
            if (data.text) {
                console.log("[CONVERSATION] Text:", data.text);
            }
            
            // Handle spoken flag if needed
            if (data.spoken !== undefined) {
                console.log("[CONVERSATION] Spoken:", data.spoken);
            }
        } catch (e) {
            console.error("[CONVERSATION] Error:", e);
            document.getElementById("status").textContent = "Conversation error: " + e.message;
        }
    }
});

document.getElementById("sound-btn").addEventListener("click", () => {
    playSound();
});

// Recording functionality
let isRecording = false;

// Make functions globally accessible
window.toggleRecording = async function toggleRecording() {
    console.log("[RECORDING] Button clicked, isRecording:", isRecording);
    const btn = document.getElementById("record-btn");
    const statusDiv = document.getElementById("recording-status");
    const infoDiv = document.getElementById("recording-info");
    
    if (!isRecording) {
        // Start recording
        console.log("[RECORDING] Starting recording...");
        btn.disabled = true; // Disable button while processing
        statusDiv.textContent = "⏳ Starting...";
        statusDiv.style.color = "#666";
        
        try {
            console.log("[RECORDING] Sending POST /recording/start");
            const resp = await fetch("/recording/start", { method: "POST" });
            console.log("[RECORDING] Response status:", resp.status);
            
            if (!resp.ok) {
                const errorText = await resp.text();
                throw new Error(`Server error: ${resp.status} - ${errorText}`);
            }
            
            const data = await resp.json();
            console.log("[RECORDING] Response data:", data);
            
            isRecording = true;
            btn.textContent = "Stop Recording";
            btn.style.background = "#f44336";
            btn.disabled = false;
            statusDiv.textContent = "🔴 Recording...";
            statusDiv.style.color = "#f44336";
            infoDiv.innerHTML = "<em>Recording in progress...</em>";
        } catch (e) {
            console.error("[RECORDING] Error:", e);
            statusDiv.textContent = "Error: " + e.message;
            statusDiv.style.color = "#f44336";
            btn.disabled = false;
            btn.textContent = "Start Recording";
            btn.style.background = "#4CAF50";
        }
    } else {
        // Stop recording
        console.log("[RECORDING] Stopping recording...");
        btn.disabled = true;
        statusDiv.textContent = "⏳ Stopping...";
        statusDiv.style.color = "#666";
        
        try {
            console.log("[RECORDING] Sending POST /recording/stop");
            const resp = await fetch("/recording/stop", { method: "POST" });
            console.log("[RECORDING] Response status:", resp.status);
            
            if (!resp.ok) {
                const errorText = await resp.text();
                throw new Error(`Server error: ${resp.status} - ${errorText}`);
            }
            
            const data = await resp.json();
            console.log("[RECORDING] Response data:", data);
            
            isRecording = false;
            btn.textContent = "Start Recording";
            btn.style.background = "#4CAF50";
            btn.disabled = false;
            statusDiv.textContent = "⏹️ Stopped";
            statusDiv.style.color = "#666";
            
            // Display recording info
            const replayBtn = document.getElementById("replay-btn");
            
            if (data.frames_recorded > 0) {
                infoDiv.innerHTML = `
                    <strong>Recording Summary:</strong><br>
                    Frames: ${data.frames_recorded}<br>
                    Audio Duration: ${data.audio_duration}s<br>
                    Recording Duration: ${data.recording_duration}s<br>
                    Total Samples: ${data.total_samples}<br>
                    Sample Rate: ${data.sample_rate}Hz
                `;
                replayBtn.disabled = false;
            } else {
                infoDiv.innerHTML = "<em>No audio recorded</em>";
                replayBtn.disabled = true;
            }
            
            // Update transcription display with final results
            if (data.transcription && data.transcription.enabled) {
                updateTranscriptionDisplay(data.transcription);
            }
        } catch (e) {
            console.error("[RECORDING] Error:", e);
            statusDiv.textContent = "Error: " + e.message;
            statusDiv.style.color = "#f44336";
            btn.disabled = false;
        }
    }
}

// Poll for status updates while recording
setInterval(async () => {
    if (isRecording) {
        try {
            const resp = await fetch("/recording/status");
            const data = await resp.json();
            const statusDiv = document.getElementById("recording-status");
            statusDiv.textContent = `🔴 Recording... (${data.frames_recorded} frames, ${data.recording_duration}s)`;
        } catch (e) {
            // Ignore errors
        }
    }
}, 1000);

// Poll for transcription updates while recording
setInterval(async () => {
    if (isRecording) {
        try {
            const resp = await fetch("/recording/transcription");
            const data = await resp.json();
            updateTranscriptionDisplay(data);
        } catch (e) {
            // Ignore errors
        }
    }
}, 500); // Poll more frequently for transcription (every 500ms)

function updateTranscriptionDisplay(data) {
    const partialDiv = document.getElementById("transcription-partial");
    const finalDiv = document.getElementById("transcription-final");
    
    if (!partialDiv || !finalDiv) {
        return;
    }
    
    // Update partial transcript
    if (data.partial && data.partial.trim()) {
        partialDiv.textContent = data.partial;
        partialDiv.style.color = "#2196F3";
        partialDiv.style.fontStyle = "normal";
    } else {
        partialDiv.innerHTML = "<em>Listening...</em>";
        partialDiv.style.color = "#666";
        partialDiv.style.fontStyle = "italic";
    }
    
    // Update final transcripts
    if (data.full_text && data.full_text.trim()) {
        finalDiv.textContent = data.full_text;
        finalDiv.style.color = "#2e7d32";
    } else if (data.final_segments && data.final_segments.length > 0) {
        finalDiv.textContent = data.final_segments.join(" ");
        finalDiv.style.color = "#2e7d32";
    } else {
        finalDiv.innerHTML = "<em>No final transcripts yet</em>";
        finalDiv.style.color = "#666";
    }
}

window.replayRecording = async function replayRecording() {
    console.log("[RECORDING] Replay button clicked");
    const replayBtn = document.getElementById("replay-btn");
    const statusDiv = document.getElementById("recording-status");
    
    replayBtn.disabled = true;
    statusDiv.textContent = "⏳ Starting replay...";
    statusDiv.style.color = "#666";
    
    try {
        console.log("[RECORDING] Sending POST /recording/replay");
        const resp = await fetch("/recording/replay", { method: "POST" });
        console.log("[RECORDING] Response status:", resp.status);
        
        if (!resp.ok) {
            const errorText = await resp.text();
            throw new Error(`Server error: ${resp.status} - ${errorText}`);
        }
        
        const data = await resp.json();
        console.log("[RECORDING] Response data:", data);
        
        statusDiv.textContent = "▶️ Replaying...";
        statusDiv.style.color = "#2196F3";
        setTimeout(() => {
            statusDiv.textContent = "";
            replayBtn.disabled = false;
        }, 2000);
    } catch (e) {
        console.error("[RECORDING] Error:", e);
        statusDiv.textContent = "Error: " + e.message;
        statusDiv.style.color = "#f44336";
        replayBtn.disabled = false;
    }
}

// Wait for DOM to be ready before attaching event listeners
function initRecordingButtons() {
    console.log("[RECORDING] Initializing recording buttons...");
    
    const recordBtn = document.getElementById("record-btn");
    const replayBtn = document.getElementById("replay-btn");
    
    if (!recordBtn) {
        console.error("[RECORDING] record-btn element not found!");
        return;
    }
    
    if (!replayBtn) {
        console.error("[RECORDING] replay-btn element not found!");
        return;
    }
    
    console.log("[RECORDING] Found buttons, attaching event listeners");
    recordBtn.addEventListener("click", toggleRecording);
    replayBtn.addEventListener("click", replayRecording);
    console.log("[RECORDING] Event listeners attached successfully");
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        initRecordingButtons();
        updateUI();
    });
} else {
    // DOM is already ready
    initRecordingButtons();
    updateUI();
}