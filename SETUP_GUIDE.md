# Quick Setup Guide: Connect Reachy Mini to Laptop Server

## Step 1: Find Your Laptop's IP Address

**Mac/Linux:**
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```

**Windows:**
```bash
ipconfig
```

Look for something like `192.168.1.100` or `192.168.0.50`

## Step 2: Start Your Laptop Server

On your laptop, run:
```bash
# Install FastAPI if needed
pip install fastapi uvicorn

# Run the example server
python example_laptop_server.py
```

The server will start on `http://0.0.0.0:8000`

**Important**: Make sure your laptop's firewall allows incoming connections on port 8000.

## Step 3: Install App on Robot

SSH into your Reachy Mini robot and install the app:

```bash
# SSH into robot (replace with your robot's IP or hostname)
ssh reachy@reachy-mini.local

# Install the app
pip install git+https://github.com/YOUR_USERNAME/reachy-mini-apps.git#subdirectory=test_hello_world
```

## Step 4: Configure Laptop Server URL

On the robot, set the environment variable:

```bash
# Replace 192.168.1.100 with YOUR laptop's IP
export LAPTOP_SERVER_URL=http://192.168.1.100:8000
```

Or add it to your robot's environment permanently.

## Step 5: Start the App

The app should now:
1. ✅ Speak using the robot's built-in speaker (`robot.media.speak()`)
2. ✅ Connect to your laptop server on startup
3. ✅ Be able to send/receive data from your laptop

## Testing the Connection

### From Robot to Laptop

The app will automatically try to connect on startup. Check the logs to see if it connected.

You can also test manually from the robot:
```bash
# On robot
curl http://YOUR_LAPTOP_IP:8000/api/status
```

### From Laptop to Robot

The robot app exposes endpoints at `http://0.0.0.0:8042` (or check the robot's IP):
- `GET /laptop_server_status` - Check if robot can reach laptop server
- `POST /send_to_laptop` - Send data from robot to laptop

## Troubleshooting

### Robot Not Speaking
- ✅ The code now uses `robot.media.speak()` instead of `pyttsx3`
- ✅ This uses the robot's 5W speaker
- ✅ Make sure the app is running on the robot (not just on your laptop)

### Can't Connect to Laptop Server
1. **Check Wi-Fi**: Both devices must be on the same network
2. **Check Firewall**: Laptop firewall must allow port 8000
3. **Check IP**: Use laptop's local IP, not `localhost` or `127.0.0.1`
4. **Test with curl**: From robot, test: `curl http://LAPTOP_IP:8000/api/status`

### App Not Starting on Robot
1. Check app is installed: `pip list | grep test-hello-world`
2. Check entry point in `pyproject.toml`
3. Check robot daemon is running
4. Check logs on robot

## Architecture Summary

```
Robot (Raspberry Pi)          Your Laptop
┌─────────────────┐          ┌─────────────────┐
│  Your App       │          │  Your Server    │
│  (runs here)    │  HTTP    │  (runs here)    │
│                 │ ────────>│                 │
│  robot.media    │          │  Port 8000      │
│  .speak()       │          │                 │
└─────────────────┘          └─────────────────┘
     ↕ Zenoh (for robot control)
```

The app runs **on the robot** and makes HTTP requests **to your laptop**.

