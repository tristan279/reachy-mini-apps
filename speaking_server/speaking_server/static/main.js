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

document.getElementById("sound-btn").addEventListener("click", () => {
    playSound();
});

// Recording functionality
let isRecording = false;

async function toggleRecording() {
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

async function replayRecording() {
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

document.getElementById("record-btn").addEventListener("click", toggleRecording);
document.getElementById("replay-btn").addEventListener("click", replayRecording);

updateUI();