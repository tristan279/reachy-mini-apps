"""Entrypoint for the Test Hello World app."""

import os
import sys
import threading
import time
import logging
from typing import Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini.utils import create_head_pose

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


class AntennaState(BaseModel):
    enabled: bool


class ControlLoop:
    """Manages the robot control loop for head and antenna movements."""
    
    def __init__(self, robot: ReachyMini, logger: logging.Logger, stop_event: Optional[threading.Event] = None):
        self.robot = robot
        self.logger = logger
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self.thread: Optional[threading.Thread] = None
        
        self.antennas_enabled = True
        self.sound_play_requested = False
        self.tts_engine = None
        
        # Initialize TTS engine if available
        if TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 0.9)
                self.logger.info("TTS engine initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize TTS: {e}")
                self.tts_engine = None
    
    def speak(self, text: str) -> None:
        """Make the bot speak using TTS or WAV file."""
        if self.tts_engine is not None:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                self.logger.info(f"Bot said: {text}")
            except Exception as e:
                self.logger.error(f"Error speaking: {e}")
        else:
            # Fallback: try to play a WAV file if TTS is not available
            wav_file = "speech.wav"
            if os.path.exists(wav_file):
                try:
                    self.robot.media.play_sound(wav_file)
                    self.logger.info(f"Playing WAV file: {wav_file}")
                except Exception as e:
                    self.logger.error(f"Error playing WAV file: {e}")
            else:
                self.logger.warning(f"TTS not available and WAV file '{wav_file}' not found")
    
    def _run_loop(self) -> None:
        """Main control loop running in a separate thread."""
        t0 = time.time()
        loop_count = 0
        
        self.logger.info("Entering main control loop...")
        
        # Make the bot speak when it starts
        self.logger.info("Attempting to speak...")
        self.speak("Hello! I am Reachy Mini. Ready to interact!")
        self.logger.info("Speech attempt completed.")
        
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
                self.logger.info("Playing sound...")
                self.speak("Playing sound effect!")
                try:
                    self.robot.media.play_sound("wake_up.wav")
                except Exception as e:
                    self.logger.error(f"Error playing sound file: {e}")
                self.sound_play_requested = False

            antennas_rad = np.deg2rad(antennas_deg)

            # Debug output every 50 iterations (roughly once per second)
            if loop_count % 50 == 0:
                self.logger.debug(f"Loop {loop_count}: t={t:.2f}s, yaw={yaw_deg:.1f}°, antennas={antennas_deg}")
            
            try:
                self.robot.set_target(
                    head=head_pose,
                    antennas=antennas_rad,
                )
            except Exception as e:
                self.logger.error(f"ERROR in set_target: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                # Continue running despite errors
                time.sleep(0.1)
                continue

            loop_count += 1
            time.sleep(0.02)
        
        self.logger.info("Control loop ended.")
    
    def start(self) -> None:
        """Start the control loop in a separate thread."""
        if self.thread is not None and self.thread.is_alive():
            self.logger.warning("Control loop already running")
            return
        
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        self.logger.info("Control loop started")
    
    def stop(self) -> None:
        """Stop the control loop."""
        self.logger.info("Stopping control loop...")
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self.logger.info("Control loop stopped")


def setup_logger(debug: bool = False) -> logging.Logger:
    """Set up logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


class TestHelloWorld(ReachyMiniApp):
    """Test Hello World app for Reachy Mini."""
    
    # Optional: URL to a custom configuration page for the app
    custom_app_url: str | None = "http://0.0.0.0:8042"
    # Optional: specify a media backend ("gstreamer", "default", etc.)
    request_media_backend: str | None = None
    
    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        """Run the app with the provided ReachyMini instance."""
        logger = setup_logger()
        logger.info("=" * 50)
        logger.info("TestHelloWorld app starting...")
        logger.info("=" * 50)
        
        # Check if robot needs to be enabled/turned on
        logger.debug(f"ReachyMini object: {reachy_mini}")
        logger.debug(f"ReachyMini type: {type(reachy_mini)}")
        logger.debug(f"ReachyMini methods: {[m for m in dir(reachy_mini) if not m.startswith('_')]}")
        
        # Try to enable/turn on the robot if such a method exists
        if hasattr(reachy_mini, 'turn_on'):
            logger.info("Turning on robot...")
            try:
                reachy_mini.turn_on()
                logger.info("Robot turned on successfully")
            except Exception as e:
                logger.warning(f"Error turning on robot: {e}")
        elif hasattr(reachy_mini, 'enable'):
            logger.info("Enabling robot...")
            try:
                reachy_mini.enable()
                logger.info("Robot enabled successfully")
            except Exception as e:
                logger.warning(f"Error enabling robot: {e}")
        else:
            logger.info("No turn_on() or enable() method found - robot may already be enabled")
        
        # Check if running in simulation mode
        try:
            status = reachy_mini.client.get_status()
            if status.get("simulation_enabled", False):
                logger.info("Running in simulation mode")
        except Exception as e:
            logger.debug(f"Could not check simulation status: {e}")
        
        # Create control loop
        control_loop = ControlLoop(reachy_mini, logger, stop_event)
        
        # Set up settings endpoints
        @self.settings_app.post("/antennas")
        def update_antennas_state(state: AntennaState):
            control_loop.antennas_enabled = state.enabled
            logger.info(f"Antennas enabled: {control_loop.antennas_enabled}")
            return {"antennas_enabled": control_loop.antennas_enabled}
        
        @self.settings_app.post("/play_sound")
        def request_sound_play():
            control_loop.sound_play_requested = True
            logger.info("Sound play requested")
            return {"status": "ok"}
        
        # Start the control loop
        control_loop.start()
        
        try:
            # Wait for stop event
            while not stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Keyboard interruption...")
        finally:
            # Stop the control loop
            control_loop.stop()
            logger.info("Shutdown complete.")


def main() -> None:
    """Entrypoint for the Test Hello World app."""
    logger = setup_logger()
    logger.info("=" * 50)
    logger.info("TestHelloWorld app starting...")
    logger.info("=" * 50)
    
    robot = ReachyMini()
    
    # Check if robot needs to be enabled/turned on
    logger.debug(f"ReachyMini object: {robot}")
    logger.debug(f"ReachyMini type: {type(robot)}")
    logger.debug(f"ReachyMini methods: {[m for m in dir(robot) if not m.startswith('_')]}")
    
    # Try to enable/turn on the robot if such a method exists
    if hasattr(robot, 'turn_on'):
        logger.info("Turning on robot...")
        try:
            robot.turn_on()
            logger.info("Robot turned on successfully")
        except Exception as e:
            logger.warning(f"Error turning on robot: {e}")
    elif hasattr(robot, 'enable'):
        logger.info("Enabling robot...")
        try:
            robot.enable()
            logger.info("Robot enabled successfully")
        except Exception as e:
            logger.warning(f"Error enabling robot: {e}")
    else:
        logger.info("No turn_on() or enable() method found - robot may already be enabled")
    
    # Check if running in simulation mode
    try:
        status = robot.client.get_status()
        if status.get("simulation_enabled", False):
            logger.info("Running in simulation mode")
    except Exception as e:
        logger.debug(f"Could not check simulation status: {e}")
    
    # Create control loop
    control_loop = ControlLoop(robot, logger)
    
    # Set up FastAPI app for settings
    app = FastAPI()
    
    @app.post("/antennas")
    def update_antennas_state(state: AntennaState):
        control_loop.antennas_enabled = state.enabled
        logger.info(f"Antennas enabled: {control_loop.antennas_enabled}")
        return {"antennas_enabled": control_loop.antennas_enabled}
    
    @app.post("/play_sound")
    def request_sound_play():
        control_loop.sound_play_requested = True
        logger.info("Sound play requested")
        return {"status": "ok"}
    
    # Start the control loop
    control_loop.start()
    
    try:
        # Keep the main thread alive
        # In a real app, you might want to run the FastAPI app here
        # For now, we'll just wait
        logger.info("App running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interruption in main thread... closing.")
    finally:
        # Stop the control loop
        control_loop.stop()
        
        # Disconnect from robot
        try:
            robot.client.disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting from robot: {e}")
        
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    app = TestHelloWorld()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
