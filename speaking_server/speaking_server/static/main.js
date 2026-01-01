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
    const btn = document.getElementById("record-btn");
    const statusDiv = document.getElementById("recording-status");
    
    if (!isRecording) {
        // Start recording
        try {
            const resp = await fetch("/recording/start", { method: "POST" });
            const data = await resp.json();
            isRecording = true;
            btn.textContent = "Stop Recording";
            btn.style.background = "#f44336";
            statusDiv.textContent = "🔴 Recording...";
            statusDiv.style.color = "#f44336";
        } catch (e) {
            statusDiv.textContent = "Error: " + e.message;
            statusDiv.style.color = "#f44336";
        }
    } else {
        // Stop recording
        try {
            const resp = await fetch("/recording/stop", { method: "POST" });
            const data = await resp.json();
            isRecording = false;
            btn.textContent = "Start Recording";
            btn.style.background = "#4CAF50";
            statusDiv.textContent = "⏹️ Stopped";
            statusDiv.style.color = "#666";
            
            // Display recording info
            const infoDiv = document.getElementById("recording-info");
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
            statusDiv.textContent = "Error: " + e.message;
            statusDiv.style.color = "#f44336";
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
    const replayBtn = document.getElementById("replay-btn");
    const statusDiv = document.getElementById("recording-status");
    
    try {
        replayBtn.disabled = true;
        statusDiv.textContent = "▶️ Replaying...";
        const resp = await fetch("/recording/replay", { method: "POST" });
        const data = await resp.json();
        statusDiv.textContent = "▶️ Replay started";
        setTimeout(() => {
            statusDiv.textContent = "";
            replayBtn.disabled = false;
        }, 2000);
    } catch (e) {
        statusDiv.textContent = "Error: " + e.message;
        replayBtn.disabled = false;
    }
}

document.getElementById("record-btn").addEventListener("click", toggleRecording);
document.getElementById("replay-btn").addEventListener("click", replayRecording);

updateUI();