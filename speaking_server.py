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
    sys.path.insert(0, str(Path(__file__).parent / "test_hello_world"))
    from test_hello_world.utils import speak as aws_speak

# Set up logging to both file and console
LOG_FILE = "/tmp/speaking_server.log"
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


class SpeakRequest(BaseModel):
    text: str
    voice_id: str = "Joanna"
    engine: str = "standard"  # Use cheapest standard engine
    output_format: str = "mp3"


class SpeakResponse(BaseModel):
    status: str
    audio_file: str
    text: str


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
        log_print("=" * 50)
        log_print("Speaking Server app starting...")
        log_print(f"Logs are being written to: {LOG_FILE}")
        log_print("=" * 50)
        
        # Create FastAPI app
        app = FastAPI(title="Speaking Server", description="Text-to-speech server using AWS Polly")
        
        @app.get("/")
        def root():
            """Root endpoint with server information."""
            return {
                "service": "Speaking Server",
                "description": "Text-to-speech server using AWS Polly",
                "endpoints": {
                    "POST /speak": "Convert text to speech",
                    "GET /health": "Health check"
                }
            }
        
        @app.get("/health")
        def health():
            """Health check endpoint."""
            return {"status": "healthy"}
        
        @app.post("/speak", response_model=SpeakResponse)
        def speak_text(request: SpeakRequest):
            """Convert text to speech using AWS Polly."""
            try:
                log_print(f"Received speak request: '{request.text[:50]}...'")
                
                # Generate speech using AWS Polly
                audio_file = aws_speak(
                    text=request.text,
                    voice_id=request.voice_id,
                    engine=request.engine,
                    output_format=request.output_format
                )
                
                log_print(f"Speech generated: {audio_file}")
                
                return SpeakResponse(
                    status="ok",
                    audio_file=audio_file,
                    text=request.text
                )
                
            except ValueError as e:
                # Credentials or configuration error
                log_print(f"Configuration error: {e}", "ERROR")
                raise HTTPException(
                    status_code=500,
                    detail=f"AWS configuration error: {str(e)}"
                )
            except Exception as e:
                # Other errors
                log_print(f"Error generating speech: {e}", "ERROR")
                import traceback
                log_print(traceback.format_exc(), "ERROR")
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
            log_print(f"Starting HTTP server on {self.custom_app_url}")
            log_print(f"API documentation: {self.custom_app_url}/docs")
            
            host = "0.0.0.0"
            port = 8001  # Default port
            
            # Parse port from custom_app_url if provided
            if self.custom_app_url:
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(self.custom_app_url)
                    if parsed.port:
                        port = parsed.port
                except Exception:
                    pass
            
            uvicorn.run(
                app,
                host=host,
                port=port,
                log_level="info",
                access_log=True
            )
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        log_print("Speaking Server started")
        log_print("=" * 50)
        
        try:
            # Wait for stop event
            while not stop_event.is_set():
                stop_event.wait(0.1)
        except KeyboardInterrupt:
            log_print("Keyboard interruption...")
        finally:
            log_print("Shutdown complete.")


def main() -> None:
    """Entrypoint for the Speaking Server app."""
    print("=" * 50)
    print("Speaking Server app starting...")
    print("=" * 50)
    
    app = SpeakingServer()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()


if __name__ == "__main__":
    main()

