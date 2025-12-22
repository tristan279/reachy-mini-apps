import threading
from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini.utils import create_head_pose
import numpy as np
import time
from pydantic import BaseModel
import os
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


class TestHelloWorld(ReachyMiniApp):
    # Optional: URL to a custom configuration page for the app
    # eg. "http://localhost:8042"
    custom_app_url: str | None = "http://0.0.0.0:8042"
    # Optional: specify a media backend ("gstreamer", "default", etc.)
    request_media_backend: str | None = None

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        print("=" * 50)
        print("TestHelloWorld app starting...")
        print("=" * 50)
        
        # Check if robot needs to be enabled/turned on
        print(f"ReachyMini object: {reachy_mini}")
        print(f"ReachyMini type: {type(reachy_mini)}")
        print(f"ReachyMini methods: {[m for m in dir(reachy_mini) if not m.startswith('_')]}")
        
        # Try to enable/turn on the robot if such a method exists
        if hasattr(reachy_mini, 'turn_on'):
            print("Turning on robot...")
            try:
                reachy_mini.turn_on()
                print("Robot turned on successfully")
            except Exception as e:
                print(f"Error turning on robot: {e}")
        elif hasattr(reachy_mini, 'enable'):
            print("Enabling robot...")
            try:
                reachy_mini.enable()
                print("Robot enabled successfully")
            except Exception as e:
                print(f"Error enabling robot: {e}")
        else:
            print("No turn_on() or enable() method found - robot may already be enabled")
        
        t0 = time.time()

        antennas_enabled = True
        sound_play_requested = False
        tts_engine = None
        
        # Initialize TTS engine if available
        if TTS_AVAILABLE:
            try:
                tts_engine = pyttsx3.init()
                tts_engine.setProperty('rate', 150)  # Speed of speech
                tts_engine.setProperty('volume', 0.9)  # Volume level
                print("TTS engine initialized")
            except Exception as e:
                print(f"Failed to initialize TTS: {e}")
                tts_engine = None
        
        def speak(text: str):
            """Make the bot speak using TTS or WAV file"""
            if tts_engine is not None:
                try:
                    tts_engine.say(text)
                    tts_engine.runAndWait()
                    print(f"Bot said: {text}")
                except Exception as e:
                    print(f"Error speaking: {e}")
            else:
                # Fallback: try to play a WAV file if TTS is not available
                wav_file = "speech.wav"
                if os.path.exists(wav_file):
                    try:
                        reachy_mini.media.play_sound(wav_file)
                        print(f"Playing WAV file: {wav_file}")
                    except Exception as e:
                        print(f"Error playing WAV file: {e}")
                else:
                    print(f"TTS not available and WAV file '{wav_file}' not found")
        
        # Make the bot speak when it starts
        print("Attempting to speak...")
        speak("Hello! I am Reachy Mini. Ready to interact!")
        print("Speech attempt completed.")

        # You can ignore this part if you don't want to add settings to your app. If you set custom_app_url to None, you have to remove this part as well.
        # === vvv ===
        class AntennaState(BaseModel):
            enabled: bool

        @self.settings_app.post("/antennas")
        def update_antennas_state(state: AntennaState):
            nonlocal antennas_enabled
            antennas_enabled = state.enabled
            return {"antennas_enabled": antennas_enabled}

        @self.settings_app.post("/play_sound")
        def request_sound_play():
            nonlocal sound_play_requested
            sound_play_requested = True
            
        # === ^^^ ===

        print("Entering main control loop...")
        loop_count = 0
        
        # Main control loop
        while not stop_event.is_set():
            t = time.time() - t0

            yaw_deg = 30.0 * np.sin(2.0 * np.pi * 0.2 * t)
            head_pose = create_head_pose(yaw=yaw_deg, degrees=True)

            if antennas_enabled:
                amp_deg = 25.0
                a = amp_deg * np.sin(2.0 * np.pi * 0.5 * t)
                antennas_deg = np.array([a, -a])
            else:
                antennas_deg = np.array([0.0, 0.0])

            if sound_play_requested:
                print("Playing sound...")
                speak("Playing sound effect!")
                try:
                    reachy_mini.media.play_sound("wake_up.wav")
                except Exception as e:
                    print(f"Error playing sound file: {e}")
                sound_play_requested = False

            antennas_rad = np.deg2rad(antennas_deg)

            # Debug output every 50 iterations (roughly once per second)
            if loop_count % 50 == 0:
                print(f"Loop {loop_count}: t={t:.2f}s, yaw={yaw_deg:.1f}°, antennas={antennas_deg}")
            
            try:
                reachy_mini.set_target(
                    head=head_pose,
                    antennas=antennas_rad,
                )
            except Exception as e:
                print(f"ERROR in set_target: {e}")
                import traceback
                traceback.print_exc()
                # Continue running despite errors
                time.sleep(0.1)
                continue

            loop_count += 1
            time.sleep(0.02)
        
        print("Control loop ended.")


if __name__ == "__main__":
    app = TestHelloWorld()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()