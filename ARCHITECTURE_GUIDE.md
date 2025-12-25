# Reachy Mini Wireless Architecture & Connection Guide

## Overview

**Reachy Mini Wireless** runs on a **Raspberry Pi 5** onboard the robot. It connects to your laptop via **Zenoh** protocol over Wi-Fi.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Reachy Mini Wireless (Raspberry Pi 5)                  │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Reachy Mini Daemon (runs on robot)              │  │
│  │  - Controls motors, sensors, camera              │  │
│  │  - Exposes Zenoh endpoints                       │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Your App (can run here OR on laptop)            │  │
│  │  - Connects via ReachyMini() client               │  │
│  │  - Uses robot.media.speak() for audio            │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                    ↕ Wi-Fi (Zenoh)
┌─────────────────────────────────────────────────────────┐
│  Your Laptop                                             │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Your Server (HTTP/WebSocket/etc.)                │  │
│  │  - Can be accessed by robot app                  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Key Concepts

### 1. **App Execution Location**
- **Option A**: App runs **on the robot** (Raspberry Pi)
  - Installed via `pip install` on the robot
  - Runs automatically when started from robot dashboard
  - Can make HTTP requests to your laptop server
  
- **Option B**: App runs **on your laptop**
  - Connects to robot via `ReachyMini()` (uses Zenoh)
  - Robot must be on same Wi-Fi network
  - Robot's IP/hostname must be discoverable

### 2. **Speech/Audio**
- **❌ DON'T USE**: `pyttsx3` - runs locally, won't use robot's speaker
- **✅ USE**: `robot.media.speak()` or `robot.media.play_sound()` - uses robot's 5W speaker
- The robot has built-in TTS capabilities via the media API

### 3. **Connecting to Laptop Server**

When your app runs **on the robot**, it can connect to your laptop server:

```python
import requests

# Get your laptop's IP (e.g., 192.168.1.100)
LAPTOP_IP = "192.168.1.100"  # Replace with your laptop's IP
SERVER_URL = f"http://{LAPTOP_IP}:8000"

# Make requests from robot to laptop
response = requests.get(f"{SERVER_URL}/api/status")
```

**Important**: 
- Both devices must be on the **same Wi-Fi network**
- Your laptop's firewall must allow incoming connections
- Use your laptop's **local IP address** (not localhost/127.0.0.1)

## Connection Setup

### Finding Your Laptop's IP

**Mac/Linux:**
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```

**Windows:**
```bash
ipconfig
```

Look for IPv4 address (e.g., `192.168.1.100`)

### Testing Connection from Robot

SSH into your robot and test:
```bash
# On robot
curl http://YOUR_LAPTOP_IP:8000/api/test
```

## Recommended Setup for Your Use Case

Since you want the robot to connect to a server on your laptop:

1. **Run the app on the robot** (installed via pip)
2. **Use `robot.media.speak()`** for speech (not pyttsx3)
3. **Make HTTP requests** from the robot app to your laptop server
4. **Run your server on laptop** with a public IP binding (0.0.0.0, not localhost)

Example:
```python
# In your app running on robot
import requests
from reachy_mini import ReachyMini

robot = ReachyMini()

# Connect to laptop server
laptop_url = "http://192.168.1.100:8000"
response = requests.get(f"{laptop_url}/api/command")
command = response.json()

# Execute command and speak
robot.media.speak(command["text"])
```

## Troubleshooting

### Robot Not Speaking
- ✅ Use `robot.media.speak()` instead of `pyttsx3`
- ✅ Check robot's media backend is configured
- ✅ Ensure app is running on the robot (not just laptop)

### Can't Connect to Laptop Server
- ✅ Check both devices on same Wi-Fi
- ✅ Verify laptop firewall allows connections
- ✅ Use laptop's local IP, not localhost
- ✅ Test with `curl` from robot first

### App Not Starting
- ✅ Ensure app is installed on robot: `pip install your-app`
- ✅ Check app entry point in `pyproject.toml`
- ✅ Verify robot daemon is running

