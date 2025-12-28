"""Speaking Server - HTTP server for text-to-speech using AWS Polly."""

import logging
import os
import sys
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from reachy_mini import ReachyMini, ReachyMiniApp

# Try to import from installed package first, then fall back to local path
try:
    from test_hello_world.utils import speak as aws_speak
except ImportError:
    # If not installed, add local path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "test_hello_world"))
    from test_hello_world.utils import speak as aws_speak

# Set up logging to both file and console
LOG_FILE = "/tmp/speaking_server.log"

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
log_print(f"Logging initialized. Log file: {LOG_FILE}", "DEBUG")


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
    
    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        """Run the speaking server with the provided ReachyMini instance."""
        log_print("=" * 50, "DEBUG")
        log_print("SpeakingServer.run() called", "DEBUG")
        log_print(f"ReachyMini instance: {reachy_mini}", "DEBUG")
        log_print(f"Stop event: {stop_event}", "DEBUG")
        log_print("=" * 50)
        log_print("Speaking Server app starting...")
        log_print(f"Logs are being written to: {LOG_FILE}")
        log_print("=" * 50, "DEBUG")
        
        # Create FastAPI app
        log_print("Creating FastAPI application...", "DEBUG")
        app = FastAPI(title="Speaking Server", description="Text-to-speech server using AWS Polly")
        log_print("FastAPI app created successfully", "DEBUG")
        
        @app.get("/")
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
        
        @app.get("/health")
        def health():
            """Health check endpoint."""
            log_print("GET /health endpoint called", "DEBUG")
            response = {"status": "healthy"}
            log_print(f"GET /health response: {response}", "DEBUG")
            return response
        
        @app.post("/speak", response_model=SpeakResponse)
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
        
        @app.get("/speak/{file_path:path}")
        def get_audio_file(file_path: str):
            """Serve the generated audio file."""
            full_path = Path(file_path)
            
            if not full_path.exists():
                raise HTTPException(status_code=404, detail="Audio file not found")
            
            if not full_path.is_file():
                raise HTTPException(status_code=400, detail="Path is not a file")
            
            return FileResponse(
                path=full_path,
                media_type="audio/mpeg",
                filename=full_path.name
            )
        
        # Start the server in a separate thread
        def run_server():
            log_print("=" * 50, "DEBUG")
            log_print("run_server() function called", "DEBUG")
            log_print(f"custom_app_url: {self.custom_app_url}", "DEBUG")
            log_print(f"Starting HTTP server on {self.custom_app_url}")
            log_print(f"API documentation: {self.custom_app_url}/docs")
            
            host = "0.0.0.0"
            port = 8001  # Default port
            
            # Parse port from custom_app_url if provided
            if self.custom_app_url:
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(self.custom_app_url)
                    log_print(f"Parsed URL: {parsed}", "DEBUG")
                    if parsed.port:
                        port = parsed.port
                        log_print(f"Using port from URL: {port}", "DEBUG")
                    else:
                        log_print(f"No port in URL, using default: {port}", "DEBUG")
                except Exception as e:
                    log_print(f"Error parsing URL: {e}, using default port {port}", "WARNING")
            
            log_print(f"Starting uvicorn server on {host}:{port}", "DEBUG")
            log_print(f"FastAPI app: {app}", "DEBUG")
            log_print("=" * 50, "DEBUG")
            
            try:
                uvicorn.run(
                    app,
                    host=host,
                    port=port,
                    log_level="debug",  # Changed to debug for more verbose logging
                    access_log=True
                )
            except Exception as e:
                log_print(f"Error starting uvicorn server: {e}", "ERROR")
                import traceback
                log_print(traceback.format_exc(), "ERROR")
                raise
        
        log_print("Creating server thread...", "DEBUG")
        server_thread = threading.Thread(target=run_server, daemon=True)
        log_print(f"Server thread created: {server_thread}", "DEBUG")
        log_print("Starting server thread...", "DEBUG")
        server_thread.start()
        log_print(f"Server thread started. Is alive: {server_thread.is_alive()}", "DEBUG")
        
        log_print("Speaking Server started")
        log_print("=" * 50, "DEBUG")
        log_print("Entering main event loop...", "DEBUG")
        
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
