"""Speaking Server - HTTP server for text-to-speech using AWS Polly.
Also supports real-time conversation mode with AWS Transcribe + AWS Polly."""

import logging
import os
import sys
import threading
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from reachy_mini import ReachyMini, ReachyMiniApp

# Import conversation components
from .aws_realtime import AwsRealtimeHandler
from .console import LocalStream
from .simple_recorder import SimpleRecorder

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
        
        # Check if conversation mode is enabled (default: false - use simple recording)
        conversation_mode = os.getenv("CONVERSATION_MODE", "false").lower() == "true"
        
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
        aws_profile = os.getenv("AWS_PROFILE", "reachy-mini")  # Default to reachy-mini profile
        language_code = os.getenv("TRANSCRIBE_LANGUAGE", "en-US")
        voice_id = os.getenv("POLLY_VOICE", "Joanna")
        engine = os.getenv("POLLY_ENGINE", "neural")
        llm_endpoint = os.getenv("LLM_ENDPOINT", None)
        llm_api_key = os.getenv("LLM_API_KEY", None)
        
        # VAD configuration (for cost optimization)
        use_vad = os.getenv("USE_VAD", "true").lower() == "true"
        vad_type = os.getenv("VAD_TYPE", "simple")  # "simple" or "webrtc"
        
        log_print(f"AWS Region: {aws_region}")
        log_print(f"AWS Profile: {aws_profile}")
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
            aws_profile=aws_profile,
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
        
        # Create a new FastAPI app that will bind to 0.0.0.0 for external access
        app = FastAPI(title="Speaking Server")
        
        # Get port from environment variable or use default
        server_port = int(os.getenv("SPEAKING_SERVER_PORT", "8000"))
        server_host = os.getenv("SPEAKING_SERVER_HOST", "0.0.0.0")
        
        log_print(f"Starting HTTP server on {server_host}:{server_port}...")
        
        @app.get("/")
        def root():
            """Root endpoint with server information."""
            return {
                "service": "Speaking Server",
                "description": "Text-to-speech server using AWS Polly",
                "endpoints": {
                    "POST /speak": "Convert text to speech",
                    "GET /health": "Health check",
                    "GET /speak/{file_path}": "Get audio file",
                    "POST /api/conversation": "Conversation endpoint"
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
        
        # Simple recording endpoints (no AWS transcription yet)
        simple_recorder = None
        
        def init_simple_recording():
            """Initialize simple recording components."""
            nonlocal simple_recorder
            
            if simple_recorder is None:
                # Create recorder (no AWS handler needed yet)
                simple_recorder = SimpleRecorder(
                    robot=reachy_mini,
                    log_print_func=log_print,
                )
                
                log_print("[SIMPLE RECORDER] Initialized (no transcription)")
            
            return simple_recorder
        
        @app.post("/recording/start")
        async def start_recording():
            """Start recording audio."""
            try:
                recorder = init_simple_recording()
                await recorder.start_recording()
                log_print("[RECORDING] Started")
                return {"status": "recording", "message": "Recording started"}
            except Exception as e:
                log_print(f"[RECORDING] Error starting: {e}", "ERROR")
                import traceback
                log_print(traceback.format_exc(), "ERROR")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/recording/stop")
        async def stop_recording():
            """Stop recording."""
            try:
                if simple_recorder is None:
                    raise HTTPException(status_code=400, detail="Recording not initialized")
                
                await simple_recorder.stop_recording()
                
                status = simple_recorder.get_status()
                log_print(f"[RECORDING] Stopped. Recorded {status['frames_recorded']} frames ({status['audio_duration']:.2f}s)")
                
                return {
                    "status": "stopped",
                    "frames_recorded": status["frames_recorded"],
                    "audio_duration": status["audio_duration"],
                    "recording_duration": status["recording_duration"],
                    "total_samples": status["total_samples"],
                    "sample_rate": status["sample_rate"]
                }
            except Exception as e:
                log_print(f"[RECORDING] Error stopping: {e}", "ERROR")
                import traceback
                log_print(traceback.format_exc(), "ERROR")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/recording/replay")
        async def replay_recording():
            """Replay the recorded audio."""
            try:
                if simple_recorder is None:
                    raise HTTPException(status_code=400, detail="Recording not initialized")
                
                # Run replay in background so it doesn't block the response
                asyncio.create_task(simple_recorder.replay_recording())
                
                log_print("[RECORDING] Replay started")
                return {"status": "replaying", "message": "Replaying recorded audio"}
            except Exception as e:
                log_print(f"[RECORDING] Error replaying: {e}", "ERROR")
                import traceback
                log_print(traceback.format_exc(), "ERROR")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/recording/status")
        def get_recording_status():
            """Get current recording status."""
            if simple_recorder is None:
                return {"status": "not_initialized"}
            return simple_recorder.get_status()
        
        @app.get("/recording/transcription")
        def get_transcription():
            """Get current transcription results."""
            if simple_recorder is None:
                return {"error": "Recording not initialized"}
            return simple_recorder.get_transcription()
        
        @app.post("/api/conversation")
        def conversation():
            """Conversation endpoint that returns text and spoken flag."""
            return {
                "text": "I'm an AI server API. Here's how the server is set up:\n\nMake the robot have a conversation by sending text to this endpoint.\n\nI can have conversations, but there are some things I can't do yet:\n- Full voice interaction\n- Complex multi-turn conversations\n- Integration with these APIs\n\nThis endpoint will help guide you through the conversation features.",
                "spoken": False
            }
        
        # Run uvicorn server in a separate thread
        server_instance = None
        
        def run_server():
            """Run the uvicorn server in a separate thread."""
            nonlocal server_instance
            config = uvicorn.Config(
                app,
                host=server_host,
                port=server_port,
                log_level="info",
                access_log=False,  # Reduce noise in logs
            )
            server_instance = uvicorn.Server(config)
            
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                loop.run_until_complete(server_instance.serve())
            except Exception as e:
                log_print(f"Server error: {e}", "ERROR")
                import traceback
                log_print(traceback.format_exc(), "ERROR")
            finally:
                loop.close()
        
        # Start server in background thread
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Give server a moment to start
        import time
        time.sleep(0.5)
        
        log_print("Endpoints mounted successfully")
        log_print(f"API available at: http://{server_host}:{server_port}")
        log_print(f"API documentation: http://{server_host}:{server_port}/docs")
        log_print(f"Access from other devices: http://<robot-ip>:{server_port}")
        log_print("=" * 50)
        
        try:
            # Wait for stop event
            while not stop_event.is_set():
                stop_event.wait(0.1)
        except KeyboardInterrupt:
            log_print("Keyboard interruption...")
        finally:
            # Shutdown server gracefully
            if server_instance is not None:
                log_print("Shutting down HTTP server...")
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(server_instance.shutdown())
                    loop.close()
                except Exception as e:
                    log_print(f"Error shutting down server: {e}", "WARNING")
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
