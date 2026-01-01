"""Speaking Server - HTTP server for text-to-speech using AWS Polly.
Also supports real-time conversation mode with AWS Transcribe + AWS Polly."""

import logging
import os
import sys
import threading
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from reachy_mini import ReachyMini, ReachyMiniApp

# Import conversation components
from .aws_realtime import AwsRealtimeHandler
from .console import LocalStream

# Try to import from installed package first, then fall back to local path
try:
    from test_hello_world.utils import speak as aws_speak
except ImportError:
    # If not installed, add local path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "test_hello_world"))
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


class SpeakingServer(ReachyMiniApp):
    """Speaking Server app for Reachy Mini - HTTP server for text-to-speech using AWS Polly.
    
    Supports two modes:
    1. HTTP Server mode (default): Provides REST API for TTS
    2. Conversation mode: Real-time voice conversation with AWS Transcribe + Polly
    """
    
    # Optional: URL to a custom configuration page for the app
    custom_app_url: str | None = "http://0.0.0.0:8001"
    
    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        """Run the speaking server with the provided ReachyMini instance."""
        log_print("=" * 50)
        log_print("Speaking Server app starting...")
        log_print(f"Logs are being written to: {LOG_FILE}")
        log_print("=" * 50)
        
        # Check if conversation mode is enabled (default: true)
        conversation_mode = os.getenv("CONVERSATION_MODE", "true").lower() == "true"
        
        if conversation_mode:
            log_print("Starting in CONVERSATION MODE")
            self._run_conversation_mode(reachy_mini, stop_event)
        else:
            log_print("Starting in HTTP SERVER MODE")
            self._run_http_server_mode(reachy_mini, stop_event)
    
    def _run_conversation_mode(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        """Run in real-time conversation mode with AWS Transcribe + Polly."""
        # Get configuration from environment variables
        aws_region = os.getenv("AWS_REGION", "us-east-1")
        language_code = os.getenv("TRANSCRIBE_LANGUAGE", "en-US")
        voice_id = os.getenv("POLLY_VOICE", "Joanna")
        engine = os.getenv("POLLY_ENGINE", "neural")
        llm_endpoint = os.getenv("LLM_ENDPOINT", None)
        llm_api_key = os.getenv("LLM_API_KEY", None)
        
        # VAD configuration (for cost optimization)
        use_vad = os.getenv("USE_VAD", "true").lower() == "true"
        vad_type = os.getenv("VAD_TYPE", "simple")  # "simple" or "webrtc"
        
        log_print(f"AWS Region: {aws_region}")
        log_print(f"Transcribe Language: {language_code}")
        log_print(f"Polly Voice: {voice_id}, Engine: {engine}")
        if llm_endpoint:
            log_print(f"LLM Endpoint: {llm_endpoint}")
        else:
            log_print("LLM Endpoint: None (using echo mode)")
        
        # Create AWS handler
        handler = AwsRealtimeHandler(
            region_name=aws_region,
            language_code=language_code,
            voice_id=voice_id,
            engine=engine,
            llm_endpoint=llm_endpoint,
            llm_api_key=llm_api_key,
            log_print_func=log_print,
            use_vad=use_vad,
            vad_type=vad_type,
        )
        
        # Create LocalStream
        local_stream = LocalStream(
            robot=reachy_mini,
            handler=handler,
            stop_event=stop_event,
            log_print_func=log_print,
        )
        
        # Run async event loop in a separate thread
        def run_async_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def main_async():
                # Start handler
                await handler.start()
                
                # Start record and play loops
                record_task = asyncio.create_task(local_stream.record_loop())
                play_task = asyncio.create_task(local_stream.play_loop())
                
                # Wait for stop event
                while not stop_event.is_set():
                    await asyncio.sleep(0.1)
                
                # Cleanup
                await handler.stop()
                record_task.cancel()
                play_task.cancel()
                
                try:
                    await record_task
                    await play_task
                except asyncio.CancelledError:
                    pass
            
            try:
                loop.run_until_complete(main_async())
            except Exception as e:
                log_print(f"Error in async loop: {e}", "ERROR")
                import traceback
                log_print(traceback.format_exc(), "ERROR")
            finally:
                loop.close()
        
        async_thread = threading.Thread(target=run_async_loop, daemon=True)
        async_thread.start()
        
        log_print("Conversation mode started")
        log_print("Listening for audio input...")
        log_print("=" * 50)
        
        try:
            # Wait for stop event
            while not stop_event.is_set():
                stop_event.wait(0.1)
        except KeyboardInterrupt:
            log_print("Keyboard interruption...")
        finally:
            log_print("Shutdown complete.")
    
    def _run_http_server_mode(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        """Run in HTTP server mode (original functionality)."""
        
        # Mount endpoints to existing settings_app (no new server needed)
        log_print("Mounting endpoints to existing server...")
        
        @self.settings_app.get("/")
        def root():
            """Root endpoint with server information."""
            return {
                "service": "Speaking Server",
                "description": "Text-to-speech server using AWS Polly",
                "endpoints": {
                    "POST /speak": "Convert text to speech",
                    "GET /health": "Health check",
                    "GET /speak/{file_path}": "Get audio file"
                }
            }
        
        @self.settings_app.get("/health")
        def health():
            """Health check endpoint."""
            return {"status": "healthy"}
        
        @self.settings_app.post("/speak", response_model=SpeakResponse)
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
        
        @self.settings_app.get("/speak/{file_path:path}")
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
        
        log_print("Endpoints mounted successfully")
        log_print(f"API available at: {self.custom_app_url}")
        log_print(f"API documentation: {self.custom_app_url}/docs")
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
