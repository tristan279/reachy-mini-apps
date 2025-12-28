"""Speaking Server - HTTP server for text-to-speech using AWS Polly."""

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Optional

# Log immediately when module is imported
print("=" * 50)
print("speaking_server.main module is being imported...")
print("=" * 50)

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

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
            print("SpeakingServer.__init__() completed")
            logger.info("SpeakingServer.__init__() completed")
        except Exception as e:
            print(f"ERROR in SpeakingServer.__init__(): {e}")
            logger.error(f"ERROR in SpeakingServer.__init__(): {e}", exc_info=True)
            raise
    
    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        """Run the speaking server with the provided ReachyMini instance."""
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
            """Convert text to speech using AWS Polly."""
            log_print("=" * 50, "DEBUG")
            log_print("POST /speak endpoint called", "DEBUG")
            log_print(f"Request received: {request}", "DEBUG")
            log_print(f"Text: '{request.text}'", "DEBUG")
            log_print(f"Voice ID: {request.voice_id}", "DEBUG")
            log_print(f"Engine: {request.engine}", "DEBUG")
            log_print(f"Output format: {request.output_format}", "DEBUG")
            
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
                
                log_print(f"aws_speak() returned: {audio_file}", "DEBUG")
                log_print(f"Audio file exists: {Path(audio_file).exists() if audio_file else False}", "DEBUG")
                log_print(f"Speech generated: {audio_file}")
                
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
