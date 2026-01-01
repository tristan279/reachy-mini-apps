"""
Example laptop server that the Reachy Mini robot can connect to.

Run this on your laptop:
    python example_laptop_server.py

Make sure:
1. Your laptop and robot are on the same Wi-Fi network
2. Your laptop's firewall allows connections on port 8000
3. Find your laptop's IP: ifconfig (Mac/Linux) or ipconfig (Windows)
4. Set LAPTOP_SERVER_URL environment variable on robot: 
   export LAPTOP_SERVER_URL=http://YOUR_LAPTOP_IP:8000
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(title="Reachy Mini Laptop Server")


class RobotCommand(BaseModel):
    text: str
    action: Optional[str] = None


@app.get("/")
def root():
    return {
        "message": "Reachy Mini Laptop Server",
        "status": "running",
        "endpoints": {
            "/api/status": "Get server status",
            "/api/command": "Get command for robot",
            "/api/data": "Receive data from robot"
        }
    }


@app.get("/api/status")
def get_status():
    """Return status that robot can use."""
    return {
        "status": "ok",
        "message": "Hello from laptop server!",
        "server_time": "2024-01-01T12:00:00"  # You can add actual timestamp
    }


@app.get("/api/command")
def get_command():
    """Get a command for the robot to execute."""
    return {
        "text": "I received a command from the laptop server!",
        "action": "speak"
    }


@app.post("/api/data")
def receive_data(data: dict):
    """Receive data from the robot."""
    print(f"Received from robot: {data}")
    return {
        "status": "received",
        "data": data
    }


@app.post("/api/command")
def send_command(command: RobotCommand):
    """Send a command to the robot (robot will poll this)."""
    # In a real implementation, you might store this in a queue
    # For now, just return it
    return {
        "text": command.text,
        "action": command.action or "speak"
    }


if __name__ == "__main__":
    # Run on 0.0.0.0 to accept connections from other devices on the network
    # Not just localhost (127.0.0.1)
    print("\n" + "="*50)
    print("Starting Reachy Mini Laptop Server")
    print("="*50)
    print("\nServer will be accessible at:")
    print("  - http://0.0.0.0:8000")
    print("  - http://localhost:8000")
    print("\nTo connect from robot, use your laptop's IP address:")
    print("  export LAPTOP_SERVER_URL=http://YOUR_LAPTOP_IP:8000")
    print("\nFind your laptop IP with:")
    print("  Mac/Linux: ifconfig | grep 'inet '")
    print("  Windows: ipconfig")
    print("\n" + "="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)







