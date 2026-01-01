"""AWS Realtime Handler for Speech-to-Text (Transcribe) and Text-to-Speech (Polly)."""

import asyncio
import base64
import json
import os
import time
from typing import Optional, Tuple, Dict, Any
import numpy as np
from scipy import signal
import boto3
from botocore.exceptions import ClientError
import logging

# Try to import amazon-transcribe for better async support
try:
    from amazon_transcribe.client import TranscribeStreamingClient
    from amazon_transcribe.handlers import TranscriptResultStreamHandler
    from amazon_transcribe.model import TranscriptEvent
    HAS_AMAZON_TRANSCRIBE = True
except ImportError:
    HAS_AMAZON_TRANSCRIBE = False

# Import VAD for cost optimization
try:
    from .vad import SimpleVAD, WebRTCVAD, HAS_WEBRTC_VAD
    HAS_VAD = True
except ImportError:
    HAS_VAD = False
    HAS_WEBRTC_VAD = False

logger = logging.getLogger(__name__)


class AwsRealtimeHandler:
    """Handler for AWS Transcribe Streaming + AWS Polly TTS."""
    
    def __init__(
        self,
        region_name: str = "us-east-1",
        language_code: str = "en-US",
        voice_id: str = "Joanna",
        engine: str = "standard",
        llm_endpoint: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        output_sample_rate: int = 24000,
        log_print_func=None,
        use_vad: bool = True,  # Enable VAD to save costs
        vad_type: str = "simple",  # "simple" or "webrtc"
        aws_profile: Optional[str] = "reachy-mini",  # AWS profile name
    ):
        self.region_name = region_name
        self.language_code = language_code
        self.voice_id = voice_id
        self.engine = engine
        self.llm_endpoint = llm_endpoint
        self.llm_api_key = llm_api_key
        self.output_sample_rate = output_sample_rate
        self.log_print = log_print_func or print
        self.use_vad = use_vad and HAS_VAD
        self.aws_profile = aws_profile
        
        # Audio processing - set sample rates before VAD initialization
        self.input_sample_rate = 16000  # AWS Transcribe requirement
        self.output_queue = asyncio.Queue()
        
        # Initialize VAD if enabled
        if self.use_vad:
            if vad_type == "webrtc" and HAS_WEBRTC_VAD:
                self.vad = WebRTCVAD(sample_rate=self.input_sample_rate)
                self.log_print("Using WebRTC VAD for speech detection")
            else:
                self.vad = SimpleVAD(sample_rate=self.input_sample_rate)
                self.log_print("Using simple VAD for speech detection")
        else:
            self.vad = None
            if use_vad:
                self.log_print("VAD requested but not available. Install webrtcvad for better accuracy.")
        
        # AWS clients - use profile if specified
        try:
            # Create boto3 session with profile
            if self.aws_profile:
                session = boto3.Session(profile_name=self.aws_profile)
                self.log_print(f"Using AWS profile: {self.aws_profile}")
            else:
                session = boto3.Session()
                self.log_print("Using default AWS credentials")
            
            if HAS_AMAZON_TRANSCRIBE:
                # amazon-transcribe library uses boto3 credential chain
                # Set AWS_PROFILE in environment so it picks up the profile
                if self.aws_profile:
                    # Store original if exists, then set our profile
                    original_profile = os.environ.get('AWS_PROFILE')
                    os.environ['AWS_PROFILE'] = self.aws_profile
                    try:
                        self.transcribe_client = TranscribeStreamingClient(region=region_name)
                    except Exception as e:
                        # Restore original on error
                        if original_profile:
                            os.environ['AWS_PROFILE'] = original_profile
                        elif 'AWS_PROFILE' in os.environ:
                            del os.environ['AWS_PROFILE']
                        raise
                    # Keep the profile set (don't restore) so it's available for the session
                else:
                    self.transcribe_client = TranscribeStreamingClient(region=region_name)
                self.log_print("Using amazon-transcribe library for streaming")
            else:
                self.transcribe_client = session.client(
                    'transcribe-streaming',
                    region_name=region_name
                )
                self.log_print("Using boto3 for transcribe streaming (consider installing amazon-transcribe for better async support)")
            
            self.polly_client = session.client('polly', region_name=region_name)
            self.log_print("AWS clients initialized successfully")
        except Exception as e:
            self.log_print(f"Error initializing AWS clients: {e}", "ERROR")
            raise
        
        # State management
        self.transcribe_stream = None
        self.input_stream = None
        self.output_stream = None
        self.is_connected = False
        self.last_activity_time = time.time()
        self.current_transcript = ""
        self.partial_transcript = ""
        
        # Task for running transcription session
        self.transcribe_task = None
        self._stop_event = None
        
        # Audio buffer for batching
        self._audio_buffer = b''
        
        # VAD state
        self._vad_speech_buffer = []  # Buffer audio during speech
        self._streaming_active = False  # Whether we're currently streaming to AWS
        
    async def start(self):
        """Start the transcription session."""
        if self.transcribe_task is None or self.transcribe_task.done():
            self.transcribe_task = asyncio.create_task(self._run_transcribe_session())
            self.log_print("Started AWS Transcribe streaming session")
    
    async def stop(self):
        """Stop the transcription session."""
        self.is_connected = False
        if self.transcribe_stream:
            try:
                self.transcribe_stream.close()
            except Exception as e:
                self.log_print(f"Error closing transcribe stream: {e}")
        if self.transcribe_task and not self.transcribe_task.done():
            self.transcribe_task.cancel()
            try:
                await self.transcribe_task
            except asyncio.CancelledError:
                pass
    
    async def receive(self, audio_frame: Tuple[int, np.ndarray]):
        """Receive audio frame from LocalStream and send to AWS Transcribe."""
        if not self.is_connected:
            return
        
        sample_rate, audio_data = audio_frame
        
        # Convert stereo to mono if needed
        if audio_data.ndim == 2:
            if audio_data.shape[1] > 0:
                audio_data = audio_data[:, 0]
            else:
                audio_data = audio_data.flatten()
        
        # Resample to 16kHz (AWS Transcribe requirement)
        if sample_rate != self.input_sample_rate:
            num_samples = int(len(audio_data) * self.input_sample_rate / sample_rate)
            if num_samples > 0:
                audio_data = signal.resample(audio_data, num_samples)
            else:
                return
        
        # Convert to int16
        if audio_data.dtype != np.int16:
            # Normalize to [-1, 1] range if float
            if audio_data.dtype in [np.float32, np.float64]:
                audio_data = np.clip(audio_data, -1.0, 1.0)
                audio_int16 = (audio_data * 32767).astype(np.int16)
            else:
                audio_int16 = audio_data.astype(np.int16)
        else:
            audio_int16 = audio_data
        
        # VAD-based streaming (cost optimization)
        # Note: AWS Transcribe charges per minute of audio sent, not connection time
        # So we only send audio when speech is detected to save costs
        if self.use_vad and self.vad:
            is_speaking, speech_started = self.vad.process_frame(audio_int16)
            
            if speech_started:
                # Speech just started
                if not self._streaming_active:
                    self._streaming_active = True
                    self.log_print("Speech detected - starting audio transmission")
                # Send buffered audio if any (to capture speech start)
                if self._vad_speech_buffer:
                    for buffered_frame in self._vad_speech_buffer:
                        await self._send_audio_chunk(buffered_frame)
                    self._vad_speech_buffer = []
            
            if self._streaming_active:
                if is_speaking:
                    # Still speaking - send audio
                    await self._send_audio_chunk(audio_int16)
                else:
                    # Speech ended - stop sending (but keep connection alive)
                    self._streaming_active = False
                    self.log_print("Speech ended - pausing audio transmission (saving costs)")
            else:
                # Not currently streaming - buffer audio in case speech starts
                # This helps capture the beginning of speech
                if is_speaking or len(self._vad_speech_buffer) > 0:
                    self._vad_speech_buffer.append(audio_int16)
                    # Limit buffer size (keep last ~500ms)
                    max_buffer_frames = max(1, int(self.input_sample_rate * 0.5 / max(len(audio_int16), 1)))
                    if len(self._vad_speech_buffer) > max_buffer_frames:
                        self._vad_speech_buffer.pop(0)
        else:
            # No VAD - stream continuously (original behavior)
            # This charges for all audio including silence
            await self._send_audio_chunk(audio_int16)
    
    async def _send_audio_chunk(self, audio_int16: np.ndarray):
        """Send audio chunk to AWS Transcribe."""
        if not self.is_connected or not self.input_stream:
            return
        
        try:
            # Convert to bytes
            audio_bytes = audio_int16.tobytes()
            
            # Accumulate in buffer
            self._audio_buffer += audio_bytes
            
            # Send chunks when buffer is large enough (3200 bytes = 100ms at 16kHz, int16)
            chunk_size = 3200
            while len(self._audio_buffer) >= chunk_size:
                chunk = self._audio_buffer[:chunk_size]
                self._audio_buffer = self._audio_buffer[chunk_size:]
                
                # Send chunk through the input stream
                if HAS_AMAZON_TRANSCRIBE:
                    await self.input_stream.send_audio_event(audio_chunk=chunk)
        except Exception as e:
            self.log_print(f"Error sending audio to Transcribe: {e}", "ERROR")
    
    async def _run_transcribe_session(self):
        """Run AWS Transcribe Streaming session."""
        try:
            # Start streaming transcription
            self.log_print("Starting AWS Transcribe streaming session...")
            
            if HAS_AMAZON_TRANSCRIBE:
                # Use amazon-transcribe library
                stream = await self.transcribe_client.start_stream_transcription(
                    language_code=self.language_code,
                    media_sample_rate_hz=self.input_sample_rate,
                    media_encoding="pcm",
                )
                
                self.input_stream = stream.input_stream
                self.output_stream = stream.output_stream
                self.transcribe_stream = stream
                self.is_connected = True
                self.log_print("AWS Transcribe stream connected (amazon-transcribe)")
                
                # Create event handler - must pass the output stream
                event_handler = TranscriptEventHandler(
                    transcript_result_stream=self.output_stream,
                    handler=self,
                    log_print_func=self.log_print
                )
                
                # Process events from the stream
                # Note: When using VAD, we may need to restart the stream for each speech segment
                # For now, we'll keep the stream open and let AWS handle it
                await event_handler.handle_events()
            else:
                # Fallback: Use boto3 (simpler but less async-friendly)
                self.log_print("Warning: amazon-transcribe not installed. Using basic boto3 mode.")
                self.log_print("For better performance, install: pip install amazon-transcribe")
                # For now, mark as connected but actual streaming would need more setup
                self.is_connected = True
                # In a real implementation, you'd set up boto3 streaming here
                # This is a placeholder
                await asyncio.sleep(1)  # Prevent immediate exit
                
        except Exception as e:
            self.log_print(f"Transcription session error: {e}", "ERROR")
            import traceback
            self.log_print(traceback.format_exc(), "ERROR")
        finally:
            self.is_connected = False
            self.log_print("Transcription session ended")
    
    async def _process_transcribe_events(self):
        """Process events from the transcription stream."""
        if not HAS_AMAZON_TRANSCRIBE or not self.output_stream:
            return
            
        async for event in self.output_stream:
            if isinstance(event, TranscriptEvent):
                results = event.transcript.results
                for result in results:
                    if result.alternatives:
                        transcript_text = result.alternatives[0].transcript
                        if result.is_partial:
                            self.partial_transcript = transcript_text
                            self.log_print(f"Partial: {transcript_text}")
                        else:
                            self.current_transcript = transcript_text
                            self.partial_transcript = ""
                            self.log_print(f"Final: {transcript_text}")
                            await self._process_transcript(transcript_text)
    
    async def _process_transcript(self, transcript: str):
        """Process transcript through LLM and generate TTS."""
        if not transcript.strip():
            return
        
        self.last_activity_time = time.time()
        
        # Get LLM response
        llm_response = await self._get_llm_response(transcript)
        
        if llm_response:
            # Generate TTS audio
            await self._generate_tts_audio(llm_response)
    
    async def _get_llm_response(self, user_text: str) -> str:
        """Get response from LLM (AWS Bedrock or external)."""
        if self.llm_endpoint:
            # Call external LLM endpoint
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    headers = {"Content-Type": "application/json"}
                    if self.llm_api_key:
                        headers["Authorization"] = f"Bearer {self.llm_api_key}"
                    
                    async with session.post(
                        self.llm_endpoint,
                        json={"text": user_text, "message": user_text},
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data.get("response") or data.get("text") or data.get("message", "I didn't understand that.")
                        else:
                            self.log_print(f"LLM endpoint returned status {response.status}", "ERROR")
                            return "Sorry, I'm having trouble processing that."
            except asyncio.TimeoutError:
                self.log_print("LLM request timed out", "ERROR")
                return "Sorry, I'm having trouble processing that."
            except Exception as e:
                self.log_print(f"LLM error: {e}", "ERROR")
                return "Sorry, I'm having trouble processing that."
        else:
            # Simple echo for testing
            return f"You said: {user_text}"
    
    async def _generate_tts_audio(self, text: str):
        """Generate TTS audio using AWS Polly."""
        if not text.strip():
            return
        
        try:
            # Split text into sentences for chunking (better latency)
            sentences = self._split_into_sentences(text)
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                
                # Call AWS Polly
                response = self.polly_client.synthesize_speech(
                    Text=sentence,
                    OutputFormat='pcm',  # PCM format for direct playback
                    VoiceId=self.voice_id,
                    Engine=self.engine,
                    SampleRate=str(self.output_sample_rate),
                )
                
                # Read audio stream
                audio_stream = response['AudioStream'].read()
                
                # Convert bytes to numpy array
                # PCM format: int16, mono, at specified sample rate
                audio_data = np.frombuffer(audio_stream, dtype=np.int16)
                
                # Convert to float32 for robot (range [-1, 1])
                audio_float = audio_data.astype(np.float32) / 32767.0
                
                # Chunk audio into frames for streaming (100ms chunks)
                chunk_size = int(self.output_sample_rate * 0.1)
                for i in range(0, len(audio_float), chunk_size):
                    chunk = audio_float[i:i + chunk_size]
                    if len(chunk) > 0:
                        # Reshape for play_loop (expects 2D: [channels, samples] or 1D)
                        if chunk.ndim == 1:
                            chunk = chunk.reshape(1, -1)
                        await self.output_queue.put((self.output_sample_rate, chunk))
                        
        except ClientError as e:
            self.log_print(f"AWS Polly error: {e}", "ERROR")
        except Exception as e:
            self.log_print(f"TTS generation error: {e}", "ERROR")
            import traceback
            self.log_print(traceback.format_exc(), "ERROR")
    
    def _split_into_sentences(self, text: str) -> list:
        """Split text into sentences for chunked TTS."""
        import re
        # Split on sentence endings
        sentences = re.split(r'([.!?]+)', text)
        result = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                sentence = sentences[i] + sentences[i + 1]
                if sentence.strip():
                    result.append(sentence.strip())
        if not result:
            # Fallback: split by commas or just use whole text
            result = [s.strip() for s in text.split(',') if s.strip()]
            if not result:
                result = [text]
        return result
    
    async def emit(self):
        """Emit audio frames for playback."""
        # Check for idle state
        idle_duration = time.time() - self.last_activity_time
        if idle_duration > 15.0:
            # Could send idle signal here
            pass
        
        # Return next item from output queue (blocking)
        try:
            return await asyncio.wait_for(self.output_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            # Return None if no audio available (non-blocking)
            return None


if HAS_AMAZON_TRANSCRIBE:
    class TranscriptEventHandler(TranscriptResultStreamHandler):
        """Event handler for AWS Transcribe streaming results."""
        
        def __init__(self, transcript_result_stream, handler, log_print_func):
            super().__init__(transcript_result_stream)
            self.handler = handler
            self.log_print = log_print_func
        
        async def handle_transcript_event(self, transcript_event: TranscriptEvent):
            """Handle transcript events from AWS Transcribe."""
            results = transcript_event.transcript.results
            for result in results:
                if result.alternatives:
                    transcript_text = result.alternatives[0].transcript
                    if result.is_partial:
                        self.handler.partial_transcript = transcript_text
                        self.log_print(f"Partial: {transcript_text}")
                    else:
                        self.handler.current_transcript = transcript_text
                        self.handler.partial_transcript = ""
                        self.log_print(f"Final: {transcript_text}")
                        await self.handler._process_transcript(transcript_text)
else:
    # Dummy class if amazon-transcribe is not available
    class TranscriptEventHandler:
        def __init__(self, handler, log_print_func):
            self.handler = handler
            self.log_print = log_print_func
        
        async def handle_events(self):
            pass
