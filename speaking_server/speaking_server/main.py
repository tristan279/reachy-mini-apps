"""Speaking Server - HTTP server for text-to-speech using AWS Polly."""

import logging
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pydub import AudioSegment
import uvicorn

# Log immediately when module is imported
print("=" * 50)
print("speaking_server.main module is being imported...")
print("=" * 50)

print("FastAPI and uvicorn imported successfully")

from reachy_mini import ReachyMini, ReachyMiniApp

print("ReachyMini and ReachyMiniApp imported successfully")

# Try to import from installed package first, then fall back to local path
try:
    from test_hello_world.utils import speak as aws_speak
    print("Successfully imported aws_speak from test_hello_world.utils")
except ImportError as e:
    print(f"Failed to import from installed package: {e}")
    # If not installed, add local path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "test_hello_world"))
    try:
        from test_hello_world.utils import speak as aws_speak
        print("Successfully imported aws_speak from local path")
    except ImportError as e2:
        print(f"Failed to import from local path: {e2}")
        raise

# Set up logging to both file and console
LOG_FILE = "/tmp/speaking_server.log"

# Write immediately to file to verify it works
with open(LOG_FILE, "a") as f:
    f.write(f"\n{'='*50}\n")
    f.write(f"Module imported at: {__file__}\n")
    f.write(f"{'='*50}\n")

# Remove any existing handlers to avoid duplicates
logging.getLogger().handlers = []

# Create formatter
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

# File handler
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

# Root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Log that logging is set up
print(f"Logging initialized. Log file: {LOG_FILE}")
logger.info(f"Logging initialized. Log file: {LOG_FILE}")


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


class SpeakRequest(BaseModel):
    text: str
    voice_id: str = "Joanna"
    engine: str = "standard"  # Use cheapest standard engine
    output_format: str = "mp3"


class SpeakResponse(BaseModel):
    status: str
    audio_file: str
    text: str


class SpeakingServer(ReachyMiniApp):
    """Speaking Server app for Reachy Mini - HTTP server for text-to-speech using AWS Polly."""
    
    # Optional: URL to a custom configuration page for the app
    custom_app_url: str | None = "http://0.0.0.0:8001"
    
    def __init__(self, *args, **kwargs):
        """Initialize the SpeakingServer app."""
        print("SpeakingServer.__init__() called")
        logger.info("SpeakingServer.__init__() called")
        try:
            super().__init__(*args, **kwargs)
            self.robot: Optional[ReachyMini] = None
            print("SpeakingServer.__init__() completed")
            logger.info("SpeakingServer.__init__() completed")
        except Exception as e:
            print(f"ERROR in SpeakingServer.__init__(): {e}")
            logger.error(f"ERROR in SpeakingServer.__init__(): {e}", exc_info=True)
            raise
    
    def _play_audio_stream(self, audio_path: str) -> None:
        """Play audio by streaming samples using start_playing and push_audio_sample."""
        if self.robot is None:
            log_print("ERROR: Robot instance not available!", "ERROR")
            return
            
        try:
            # Check if ffprobe is available (required for pydub to decode MP3)
            if not shutil.which('ffprobe'):
                log_print("ERROR: ffprobe (from ffmpeg) is required to decode MP3 files.", "ERROR")
                log_print("Please install ffmpeg:", "ERROR")
                log_print("  Ubuntu/Debian: sudo apt install ffmpeg", "ERROR")
                log_print("  macOS: brew install ffmpeg", "ERROR")
                log_print("  Or download from: https://ffmpeg.org/download.html", "ERROR")
                raise FileNotFoundError("ffprobe not found. Please install ffmpeg.")
            
            # Load and decode audio file
            log_print("Loading audio file...")
            audio = AudioSegment.from_file(audio_path)
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Get sample rate and convert to numpy array
            sample_rate = audio.frame_rate
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            # Normalize to [-1, 1] range
            if samples.dtype == np.int16:
                samples = samples.astype(np.float32) / 32768.0
            elif samples.dtype == np.int32:
                samples = samples.astype(np.float32) / 2147483648.0
            
            # Get output sample rate from robot
            output_sample_rate = self.robot.media.get_output_audio_samplerate()
            output_channels = self.robot.media.get_output_channels()
            
            log_print(f"Audio: {sample_rate}Hz, {audio.channels}ch -> Robot: {output_sample_rate}Hz, {output_channels}ch")
            
            # Resample if needed
            if sample_rate != output_sample_rate:
                log_print(f"Resampling from {sample_rate}Hz to {output_sample_rate}Hz...")
                from scipy import signal
                num_samples = int(len(samples) * output_sample_rate / sample_rate)
                samples = signal.resample(samples, num_samples)
            
            # Convert to correct channel format
            if output_channels == 2 and samples.ndim == 1:
                # Convert mono to stereo
                samples = np.column_stack([samples, samples])
            elif output_channels == 1 and samples.ndim > 1:
                # Convert stereo to mono (take first channel)
                samples = samples[:, 0] if samples.ndim == 2 else samples[0]
            
            # Start playing
            log_print("Starting audio playback stream...")
            self.robot.media.start_playing()
            
            # Stream audio in chunks (100ms chunks)
            chunk_size_samples = int(output_sample_rate * 0.1)
            total_samples = len(samples) if samples.ndim == 1 else samples.shape[0]
            sent_samples = 0
            
            while sent_samples < total_samples:
                end_idx = min(sent_samples + chunk_size_samples, total_samples)
                if samples.ndim == 1:
                    chunk = samples[sent_samples:end_idx]
                else:
                    chunk = samples[sent_samples:end_idx, :]
                
                self.robot.media.push_audio_sample(chunk)
                sent_samples = end_idx
                
                # Small delay to avoid overwhelming the buffer
                time.sleep(0.05)
            
            # Stop playing
            log_print("Stopping audio playback...")
            self.robot.media.stop_playing()
            log_print("✓ Audio playback completed")
            
        except FileNotFoundError as e:
            log_print(f"ERROR: {e}", "ERROR")
            raise
        except Exception as e:
            log_print(f"ERROR playing audio stream: {e}", "ERROR")
            import traceback
            log_print(traceback.format_exc(), "ERROR")
            # Try to stop playing if we started
            try:
                self.robot.media.stop_playing()
            except:
                pass
            raise
    
    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        """Run the speaking server with the provided ReachyMini instance."""
        # Store robot instance for use in endpoints
        self.robot = reachy_mini
        
        # Write immediately to verify run() is called
        with open(LOG_FILE, "a") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"SpeakingServer.run() CALLED!\n")
            f.write(f"ReachyMini: {reachy_mini}\n")
            f.write(f"Stop event: {stop_event}\n")
            f.write(f"{'='*50}\n")
        
        print("=" * 50)
        print("SpeakingServer.run() CALLED!")
        print("=" * 50)
        log_print("=" * 50, "DEBUG")
        log_print("SpeakingServer.run() called", "DEBUG")
        log_print(f"ReachyMini instance: {reachy_mini}", "DEBUG")
        log_print(f"Stop event: {stop_event}", "DEBUG")
        log_print("=" * 50)
        log_print("Speaking Server app starting...")
        log_print(f"Logs are being written to: {LOG_FILE}")
        log_print("=" * 50, "DEBUG")
        
        # Use self.settings_app (provided by ReachyMiniApp) instead of creating new FastAPI app
        log_print("Setting up endpoints on settings_app...", "DEBUG")
        log_print(f"settings_app type: {type(self.settings_app)}", "DEBUG")
        
        @self.settings_app.get("/")
        def root():
            """Root endpoint with server information."""
            log_print("GET / endpoint called", "DEBUG")
            response = {
                "service": "Speaking Server",
                "description": "Text-to-speech server using AWS Polly",
                "endpoints": {
                    "POST /speak": "Convert text to speech",
                    "GET /health": "Health check"
                }
            }
            log_print(f"GET / response: {response}", "DEBUG")
            return response
        
        @self.settings_app.get("/health")
        def health():
            """Health check endpoint."""
            log_print("GET /health endpoint called", "DEBUG")
            response = {"status": "healthy"}
            log_print(f"GET /health response: {response}", "DEBUG")
            return response
        
        @self.settings_app.post("/speak", response_model=SpeakResponse)
        def speak_text(request: SpeakRequest):
            """Convert text to speech using AWS Polly and play it on the robot."""
            log_print("=" * 50, "DEBUG")
            log_print("POST /speak endpoint called", "DEBUG")
            log_print(f"Request received: {request}", "DEBUG")
            log_print(f"Text: '{request.text}'", "DEBUG")
            log_print(f"Voice ID: {request.voice_id}", "DEBUG")
            log_print(f"Engine: {request.engine}", "DEBUG")
            log_print(f"Output format: {request.output_format}", "DEBUG")
            
            temp_audio_path = None
            try:
                log_print(f"Received speak request: '{request.text[:50]}...'")
                log_print("Calling aws_speak()...", "DEBUG")
                
                # Generate speech using AWS Polly
                audio_file = aws_speak(
                    text=request.text,
                    voice_id=request.voice_id,
                    engine=request.engine,
                    output_format=request.output_format
                )
                temp_audio_path = audio_file
                
                log_print(f"aws_speak() returned: {audio_file}", "DEBUG")
                log_print(f"Audio file exists: {Path(audio_file).exists() if audio_file else False}", "DEBUG")
                log_print(f"Speech generated: {audio_file}")
                
                # Check if robot.media is available
                if not hasattr(self.robot, 'media') or self.robot.media is None:
                    log_print("WARNING: robot.media is not available, cannot play audio", "WARNING")
                else:
                    # Check if we're using WebRTC backend
                    is_webrtc = False
                    if hasattr(self.robot.media, 'backend'):
                        backend_str = str(self.robot.media.backend).lower()
                        backend_type = type(self.robot.media.backend).__name__.lower()
                        is_webrtc = 'webrtc' in backend_str or 'webrtc' in backend_type
                        if is_webrtc:
                            log_print("WebRTC backend detected - will use streaming method")
                    
                    # Try play_sound() first ONLY if NOT WebRTC
                    if not is_webrtc and hasattr(self.robot.media, 'play_sound'):
                        try:
                            log_print("Attempting to play audio using play_sound()...")
                            self.robot.media.play_sound(temp_audio_path)
                            log_print(f"✓ Robot said: {request.text}")
                            # Cleanup after a delay
                            def cleanup():
                                time.sleep(3)  # Wait for playback to complete
                                try:
                                    if os.path.exists(temp_audio_path):
                                        os.unlink(temp_audio_path)
                                        log_print(f"Cleaned up temporary audio file: {temp_audio_path}")
                                except Exception as e:
                                    log_print(f"Warning: Could not delete temp file {temp_audio_path}: {e}", "WARNING")
                            threading.Thread(target=cleanup, daemon=True).start()
                            temp_audio_path = None  # Don't delete in finally block
                        except Exception as e:
                            log_print(f"play_sound() failed: {e}, trying streaming method...", "WARNING")
                    
                    # Use streaming method (required for WebRTC backend, or fallback)
                    if temp_audio_path and hasattr(self.robot.media, 'start_playing') and hasattr(self.robot.media, 'push_audio_sample'):
                        if is_webrtc:
                            log_print("Using audio streaming method (WebRTC backend)...")
                        else:
                            log_print("Using audio streaming method...")
                        self._play_audio_stream(temp_audio_path)
                        log_print(f"✓ Robot said: {request.text}")
                        # Cleanup
                        if os.path.exists(temp_audio_path):
                            os.unlink(temp_audio_path)
                        temp_audio_path = None
                    elif temp_audio_path:
                        log_print("WARNING: No audio playback methods available", "WARNING")
                        log_print(f"Available methods: {[m for m in dir(self.robot.media) if not m.startswith('_')]}", "WARNING")
                
                response = SpeakResponse(
                    status="ok",
                    audio_file=audio_file,
                    text=request.text
                )
                log_print(f"Response prepared: {response}", "DEBUG")
                return response
                
            except ValueError as e:
                # Credentials or configuration error
                log_print(f"ValueError in speak_text: {e}", "ERROR")
                import traceback
                log_print(traceback.format_exc(), "ERROR")
                log_print(f"Configuration error: {e}", "ERROR")
                if temp_audio_path and os.path.exists(temp_audio_path):
                    try:
                        os.unlink(temp_audio_path)
                    except:
                        pass
                raise HTTPException(
                    status_code=500,
                    detail=f"AWS configuration error: {str(e)}"
                )
            except Exception as e:
                # Other errors
                log_print(f"Exception in speak_text: {e}", "ERROR")
                import traceback
                log_print(traceback.format_exc(), "ERROR")
                log_print(f"Error generating speech: {e}", "ERROR")
                if temp_audio_path and os.path.exists(temp_audio_path):
                    try:
                        os.unlink(temp_audio_path)
                    except:
                        pass
                raise HTTPException(
                    status_code=500,
                    detail=f"Error generating speech: {str(e)}"
                )
        
        @self.settings_app.get("/speak/{file_path:path}")
        def get_audio_file(file_path: str):
            """Serve the generated audio file."""
            log_print(f"GET /speak/{file_path} endpoint called", "DEBUG")
            full_path = Path(file_path)
            
            if not full_path.exists():
                log_print(f"Audio file not found: {file_path}", "ERROR")
                raise HTTPException(status_code=404, detail="Audio file not found")
            
            if not full_path.is_file():
                log_print(f"Path is not a file: {file_path}", "ERROR")
                raise HTTPException(status_code=400, detail="Path is not a file")
            
            log_print(f"Serving audio file: {file_path}", "DEBUG")
            return FileResponse(
                path=full_path,
                media_type="audio/mpeg",
                filename=full_path.name
            )
        
        log_print("Endpoints registered on settings_app", "DEBUG")
        log_print(f"Web interface available at: {self.custom_app_url}")
        log_print("Speaking Server started")
        log_print("=" * 50, "DEBUG")
        log_print("Entering main event loop...", "DEBUG")
        
        with open(LOG_FILE, "a") as f:
            f.write("Endpoints registered, entering main event loop...\n")
        
        try:
            # Wait for stop event
            loop_count = 0
            while not stop_event.is_set():
                stop_event.wait(0.1)
                loop_count += 1
                if loop_count % 50 == 0:  # Log every 5 seconds
                    log_print(f"Main loop running... (iteration {loop_count})", "DEBUG")
        except KeyboardInterrupt:
            log_print("KeyboardInterrupt received", "WARNING")
            log_print("Keyboard interruption...")
        except Exception as e:
            log_print(f"Exception in main loop: {e}", "ERROR")
            import traceback
            log_print(traceback.format_exc(), "ERROR")
            raise
        finally:
            log_print("Exiting main loop, shutting down...", "DEBUG")
            log_print("Shutdown complete.")
            log_print("Shutdown complete.", "DEBUG")


def main() -> None:
    """Entrypoint for the Speaking Server app."""
    log_print("=" * 50, "DEBUG")
    log_print("main() function called", "DEBUG")
    print("=" * 50)
    print("Speaking Server app starting...")
    print("=" * 50)
    log_print("Creating SpeakingServer instance...", "DEBUG")
    
    app = SpeakingServer()
    log_print(f"SpeakingServer instance created: {app}", "DEBUG")
    log_print(f"custom_app_url: {app.custom_app_url}", "DEBUG")
    
    try:
        log_print("Calling app.wrapped_run()...", "DEBUG")
        app.wrapped_run()
    except KeyboardInterrupt:
        log_print("KeyboardInterrupt in main()", "WARNING")
        app.stop()
    except Exception as e:
        log_print(f"Exception in main(): {e}", "ERROR")
        import traceback
        log_print(traceback.format_exc(), "ERROR")
        raise


if __name__ == "__main__":
    main()
