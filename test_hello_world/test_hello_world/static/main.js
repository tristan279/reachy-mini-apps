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
        document.getElementById("status").textContent = "Error: " + e.message;
    }
}

async function playSound() {
    try {
        await fetch("/play_sound", { method: "POST" });
        document.getElementById("status").textContent = "Sound played!";
        setTimeout(updateUI, 2000);
    } catch (e) {
        document.getElementById("status").textContent = "Error: " + e.message;
    }
}

async function speak() {
    const textInput = document.getElementById("speak-text");
    const text = textInput.value.trim();
    
    if (!text) {
        document.getElementById("status").textContent = "Please enter text to speak";
        return;
    }
    
    try {
        const resp = await fetch("/speak", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });
        const data = await resp.json();
        document.getElementById("status").textContent = `Speaking: "${text}"`;
        setTimeout(updateUI, 3000);
    } catch (e) {
        document.getElementById("status").textContent = "Error: " + e.message;
    }
}

function updateUI() {
    const checkbox = document.getElementById("antenna-checkbox");
    const status = document.getElementById("status");

    checkbox.checked = antennasEnabled;

    if (antennasEnabled) {
        status.textContent = "Antennas: running";
    } else {
        status.textContent = "Antennas: stopped";
    }
}

document.getElementById("antenna-checkbox").addEventListener("change", (e) => {
    updateAntennasState(e.target.checked);
});

document.getElementById("sound-btn").addEventListener("click", () => {
    playSound();
});

document.getElementById("speak-btn").addEventListener("click", () => {
    speak();
});

// Allow Enter key to trigger speak
document.getElementById("speak-text").addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
        speak();
    }
});

updateUI();