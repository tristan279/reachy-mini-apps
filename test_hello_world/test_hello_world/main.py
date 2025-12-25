"""Entrypoint for the Test Hello World app."""

import logging
import os
import sys
import threading
import time
from typing import Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini.utils import create_head_pose

# Set up logging to both file and console
LOG_FILE = "/tmp/test_hello_world.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def log_print(message: str, level: str = "INFO"):
    """Print to both console and log file."""
    print(message)
    if level == "DEBUG":
        logger.debug(message)
    elif level == "WARNING":
        logger.warning(message)
    elif level == "ERROR":
        logger.error(message)
    else:
        logger.info(message)



class AntennaState(BaseModel):
    enabled: bool


class SpeakRequest(BaseModel):
    text: str


class ControlLoop:
    """Manages the robot control loop for head and antenna movements."""
    
    def __init__(self, robot: ReachyMini, stop_event: Optional[threading.Event] = None):
        self.robot = robot
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self.thread: Optional[threading.Thread] = None
        
        self.antennas_enabled = True
        self.sound_play_requested = False
    
    def speak(self, text: str) -> None:
        """Make the robot speak using the robot's built-in media API."""
        log_print(f"Attempting to speak: '{text}'")
        
        # Check if media object exists
        if not hasattr(self.robot, 'media'):
            log_print("ERROR: robot.media attribute does not exist!", "ERROR")
            return
        
        if self.robot.media is None:
            log_print("ERROR: robot.media is None - media backend may not be initialized", "ERROR")
            return
        
        log_print(f"Media object available: {type(self.robot.media)}")
        log_print(f"Media methods: {[m for m in dir(self.robot.media) if not m.startswith('_')]}")
        
        try:
            # Use the robot's media.speak() method - this uses the robot's 5W speaker
            if hasattr(self.robot.media, 'speak'):
                log_print("Calling robot.media.speak()...")
                self.robot.media.speak(text)
                log_print(f"✓ Robot said: {text}")
            else:
                log_print("ERROR: robot.media.speak() method does not exist", "ERROR")
                log_print(f"Available methods: {[m for m in dir(self.robot.media) if not m.startswith('_')]}", "ERROR")
        except AttributeError:
            # Fallback if media.speak() doesn't exist, try play_sound with a generated file
            log_print("Warning: robot.media.speak() not available, trying alternative method", "WARNING")
            try:
                # Some versions might use different methods
                if hasattr(self.robot.media, 'play_sound'):
                    # This would require a pre-generated audio file
                    log_print("Warning: Need to generate audio file for play_sound()", "WARNING")
                else:
                    log_print("Error: No speech method available on robot.media", "ERROR")
            except Exception as e:
                log_print(f"Error with fallback speech method: {e}", "ERROR")
        except Exception as e:
            log_print(f"ERROR speaking: {e}", "ERROR")
            import traceback
            log_print(traceback.format_exc(), "ERROR")
    
    def _run_loop(self) -> None:
        """Main control loop running in a separate thread."""
        t0 = time.time()
        loop_count = 0
        
        log_print("Entering main control loop...")
        
        # Make the robot speak when it starts
        log_print("Attempting to speak...")
        self.speak("Hello! I am Reachy Mini. Ready to interact!")
        log_print("Speech attempt completed.")
        
        while not self.stop_event.is_set():
            t = time.time() - t0

            yaw_deg = 30.0 * np.sin(2.0 * np.pi * 0.2 * t)
            head_pose = create_head_pose(yaw=yaw_deg, degrees=True)

            if self.antennas_enabled:
                amp_deg = 25.0
                a = amp_deg * np.sin(2.0 * np.pi * 0.5 * t)
                antennas_deg = np.array([a, -a])
            else:
                antennas_deg = np.array([0.0, 0.0])

            if self.sound_play_requested:
                print("Playing sound...")
                self.speak("Playing sound effect!")
                try:
                    self.robot.media.play_sound("wake_up.wav")
                except Exception as e:
                    print(f"Error playing sound file: {e}")
                self.sound_play_requested = False

            antennas_rad = np.deg2rad(antennas_deg)

            # Debug output every 50 iterations (roughly once per second)
            if loop_count % 50 == 0:
                print(f"Loop {loop_count}: t={t:.2f}s, yaw={yaw_deg:.1f}°, antennas={antennas_deg}")
            
            try:
                self.robot.set_target(
                    head=head_pose,
                    antennas=antennas_rad,
                )
            except Exception as e:
                print(f"ERROR in set_target: {e}")
                import traceback
                print(traceback.format_exc())
                # Continue running despite errors
                time.sleep(0.1)
                continue

            loop_count += 1
            time.sleep(0.02)
        
        print("Control loop ended.")
    
    def start(self) -> None:
        """Start the control loop in a separate thread."""
        if self.thread is not None and self.thread.is_alive():
            print("Warning: Control loop already running")
            return
        
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        print("Control loop started")
    
    def stop(self) -> None:
        """Stop the control loop."""
        print("Stopping control loop...")
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        print("Control loop stopped")


class TestHelloWorld(ReachyMiniApp):
    """Test Hello World app for Reachy Mini."""
    
    # Optional: URL to a custom configuration page for the app
    custom_app_url: str | None = "http://0.0.0.0:8042"
    # Optional: specify a media backend ("gstreamer", "default", etc.)
    request_media_backend: str | None = "gstreamer"
    
    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        """Run the app with the provided ReachyMini instance."""
        log_print("=" * 50)
        log_print("TestHelloWorld app starting...")
        log_print(f"Logs are being written to: {LOG_FILE}")
        log_print("=" * 50)
        
        # Check if robot needs to be enabled/turned on
        if hasattr(reachy_mini, 'turn_on'):
            print("Turning on robot...")
            try:
                reachy_mini.turn_on()
                print("Robot turned on successfully")
            except Exception as e:
                print(f"Warning: Error turning on robot: {e}")
        elif hasattr(reachy_mini, 'enable'):
            print("Enabling robot...")
            try:
                reachy_mini.enable()
                print("Robot enabled successfully")
            except Exception as e:
                print(f"Warning: Error enabling robot: {e}")
        else:
            print("No turn_on() or enable() method found - robot may already be enabled")
        
        # Check if running in simulation mode
        try:
            status = reachy_mini.client.get_status()
            if status.get("simulation_enabled", False):
                log_print("Running in simulation mode")
        except Exception as e:
            log_print(f"Could not check simulation status: {e}", "WARNING")
        
        # Check media backend availability
        log_print("\n" + "=" * 50)
        log_print("Checking media backend...")
        log_print("=" * 50)
        if hasattr(reachy_mini, 'media'):
            log_print(f"✓ robot.media exists: {type(reachy_mini.media)}")
            if reachy_mini.media is not None:
                log_print(f"✓ robot.media is not None")
                methods = [m for m in dir(reachy_mini.media) if not m.startswith('_')]
                log_print(f"✓ Available media methods: {methods}")
                if 'speak' in methods:
                    log_print("✓ robot.media.speak() is available")
                else:
                    log_print("✗ robot.media.speak() is NOT available", "ERROR")
            else:
                log_print("✗ robot.media is None - media backend may not be initialized", "ERROR")
        else:
            log_print("✗ robot.media attribute does not exist", "ERROR")
        log_print("=" * 50 + "\n")
        
        # Create control loop
        control_loop = ControlLoop(reachy_mini, stop_event)
        
        # Set up settings endpoints
        @self.settings_app.post("/antennas")
        def update_antennas_state(state: AntennaState):
            control_loop.antennas_enabled = state.enabled
            print(f"Antennas enabled: {control_loop.antennas_enabled}")
            return {"antennas_enabled": control_loop.antennas_enabled}
        
        @self.settings_app.post("/play_sound")
        def request_sound_play():
            control_loop.sound_play_requested = True
            print("Sound play requested")
            return {"status": "ok"}
        
        @self.settings_app.post("/speak")
        def request_speak(request: SpeakRequest):
            """Make the robot speak text."""
            control_loop.speak(request.text)
            print(f"Speak requested: {request.text}")
            return {"status": "ok", "text": request.text}
        
        log_print(f"Web interface available at: {self.custom_app_url}")
        log_print("Starting control loop...")
        
        # Start the control loop
        control_loop.start()
        
        try:
            # Wait for stop event
            while not stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Keyboard interruption...")
        finally:
            # Stop the control loop
            control_loop.stop()
            print("Shutdown complete.")


def main() -> None:
    """Entrypoint for the Test Hello World app."""
    print("=" * 50)
    print("TestHelloWorld app starting...")
    print("=" * 50)
    
    robot = ReachyMini()
    
    # Try to enable/turn on the robot if such a method exists
    if hasattr(robot, 'turn_on'):
        print("Turning on robot...")
        try:
            robot.turn_on()
            print("Robot turned on successfully")
        except Exception as e:
            print(f"Warning: Error turning on robot: {e}")
    elif hasattr(robot, 'enable'):
        print("Enabling robot...")
        try:
            robot.enable()
            print("Robot enabled successfully")
        except Exception as e:
            print(f"Warning: Error enabling robot: {e}")
    else:
        print("No turn_on() or enable() method found - robot may already be enabled")
    
    # Check if running in simulation mode
    try:
        status = robot.client.get_status()
        if status.get("simulation_enabled", False):
            print("Running in simulation mode")
    except Exception as e:
        print(f"Could not check simulation status: {e}")
    
    # Create control loop
    control_loop = ControlLoop(robot)
    
    # Start the control loop
    control_loop.start()
    
    try:
        print("App running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Keyboard interruption in main thread... closing.")
    finally:
        # Stop the control loop
        control_loop.stop()
        
        # Disconnect from robot
        try:
            robot.client.disconnect()
        except Exception as e:
            print(f"Warning: Error disconnecting from robot: {e}")
        
        print("Shutdown complete.")


if __name__ == "__main__":
    print("Starting TestHelloWorld app...")
    app = TestHelloWorld()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
