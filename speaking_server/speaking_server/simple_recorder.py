"""Simple audio recorder for testing - records audio and replays it.
Also supports real-time transcription using AWS Transcribe."""

import asyncio
import time
import os
import numpy as np
from typing import Optional, List, Tuple
import logging
from scipy import signal
import boto3
from botocore.exceptions import ClientError
import aiohttp

# Try to import amazon-transcribe for better async support
try:
    from amazon_transcribe.client import TranscribeStreamingClient
    from amazon_transcribe.handlers import TranscriptResultStreamHandler
    from amazon_transcribe.model import TranscriptEvent
    HAS_AMAZON_TRANSCRIBE = True
except ImportError:
    HAS_AMAZON_TRANSCRIBE = False

logger = logging.getLogger(__name__)


class SimpleRecorder:
    """Simple recorder that captures audio and replays it directly.
    Also supports real-time transcription using AWS Transcribe."""
    
    def __init__(self, robot, log_print_func=None):
        self._robot = robot
        self.log_print = log_print_func or print
        
        self.is_recording = False
        self.recorded_audio: List[Tuple[int, np.ndarray]] = []  # List of (sample_rate, audio_data)
        self.recording_start_time = None
        
        # Conversation mode
        self.conversation_mode = False
        self.conversation_api_url = "https://reachy.tristy.dev/api/conversation"
        
        # Transcription state
        self.transcription_enabled = os.getenv("ENABLE_TRANSCRIPTION", "true").lower() == "true"
        self.transcribe_client = None
        self.transcribe_stream = None
        self.input_stream = None
        self.output_stream = None
        self.is_transcribing = False
        self.transcribe_task = None
        self.transcription_handler = None
        
        # Transcription results
        self.partial_transcript = ""
        self.final_transcripts: List[str] = []  # List of final transcript segments
        self.all_transcripts: List[dict] = []  # List of {type: "partial"/"final", text: str, timestamp: float}
        
        # AWS configuration
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.aws_profile = os.getenv("AWS_PROFILE", "reachy-mini")
        self.language_code = os.getenv("TRANSCRIBE_LANGUAGE", "en-US")
        self.transcribe_sample_rate = 16000  # AWS Transcribe requirement
        
        # Polly configuration
        self.polly_voice_id = os.getenv("POLLY_VOICE", "Joanna")
        self.polly_engine = os.getenv("POLLY_ENGINE", "standard")
        self.polly_sample_rate = 24000  # Default output sample rate for Polly
        
        # AWS clients
        self.polly_client = None
        
        # Initialize AWS clients if enabled
        if self.transcription_enabled:
            self._init_aws_clients()
    
    def set_conversation_mode(self, enabled: bool):
        """Set conversation mode on/off."""
        self.conversation_mode = enabled
        self.log_print(f"[RECORDER] Conversation mode: {enabled}")
        
    async def start_recording(self):
        """Start recording audio."""
        if self.is_recording:
            self.log_print("[RECORDER] Already recording")
            return
        
        # CRITICAL: Start the microphone before trying to get audio samples
        try:
            self._robot.media.start_recording()
            self.log_print("[RECORDER] Microphone started")
        except Exception as e:
            self.log_print(f"[RECORDER] Error starting microphone: {e}", "ERROR")
            raise
        
        self.is_recording = True
        self.recorded_audio = []
        self.recording_start_time = time.time()
        self._stopping_recording = False  # Flag to indicate we're in grace period
        
        # Reset transcription
        self.partial_transcript = ""
        self.final_transcripts = []
        self.all_transcripts = []
        
        self.log_print("[RECORDER] Started recording")
        
        # Start transcription if enabled
        if self.transcription_enabled and self.transcribe_client:
            await self._start_transcription()
        
        # Start recording loop
        asyncio.create_task(self._record_loop())
    
    async def stop_recording(self):
        """Stop recording."""
        if not self.is_recording:
            self.log_print("[RECORDER] Not currently recording")
            return
        
        # Mark as stopping but allow loop to continue briefly to drain buffer
        self._stopping_recording = True
        self.log_print("[RECORDER] Stopping recording, draining buffer...")
        
        # Give the recording loop a short grace period to capture any buffered audio
        # This ensures we don't lose audio that's still in the system buffer
        grace_period = 0.5  # 500ms grace period
        await asyncio.sleep(grace_period)
        
        # Now actually stop recording
        self.is_recording = False
        
        # Stop transcription
        if self.is_transcribing:
            await self._stop_transcription()
        
        # CRITICAL: Stop the microphone after grace period
        try:
            self._robot.media.stop_recording()
            self.log_print("[RECORDER] Microphone stopped")
        except Exception as e:
            self.log_print(f"[RECORDER] Error stopping microphone: {e}", "ERROR")
        
        duration = time.time() - self.recording_start_time if self.recording_start_time else 0
        self.log_print(f"[RECORDER] Stopped recording. Captured {len(self.recorded_audio)} audio frames over {duration:.2f} seconds")
        
        # Log summary
        if self.recorded_audio:
            total_samples = sum(len(audio) for _, audio in self.recorded_audio)
            sample_rate = self.recorded_audio[0][0] if self.recorded_audio else 0
            duration_seconds = total_samples / sample_rate if sample_rate > 0 else 0
            self.log_print(f"[RECORDER] Total audio: {total_samples} samples @ {sample_rate}Hz = {duration_seconds:.2f} seconds")
        
        # Log transcription summary
        if self.final_transcripts:
            full_text = " ".join(self.final_transcripts)
            self.log_print(f"[TRANSCRIPTION] Final transcript: {full_text}")
            
            # Send to API and speak response
            if full_text.strip():
                asyncio.create_task(self._send_to_api_and_speak(full_text))
        elif self.partial_transcript:
            self.log_print(f"[TRANSCRIPTION] Partial transcript: {self.partial_transcript}")
            # Use partial if no final transcripts
            if self.partial_transcript.strip():
                asyncio.create_task(self._send_to_api_and_speak(self.partial_transcript))
    
    async def _record_loop(self):
        """Record audio frames while recording is active."""
        frame_count = 0
        none_count = 0
        loop_count = 0
        
        # Try to get input sample rate for debugging
        try:
            input_rate = self._robot.media.get_input_audio_samplerate()
            self.log_print(f"[RECORDER] Input audio sample rate: {input_rate} Hz")
        except Exception as e:
            self.log_print(f"[RECORDER] Could not get input sample rate: {e}")
        
        # Check if robot is in simulation mode
        try:
            if hasattr(self._robot, 'client'):
                status = self._robot.client.get_status()
                if status.get("simulation_enabled", False):
                    self.log_print("[RECORDER] WARNING: Robot is in simulation mode - microphone may not work")
        except Exception as e:
            self.log_print(f"[RECORDER] Could not check simulation status: {e}")
        
        self.log_print("[RECORDER] Starting recording loop...")
        
        # Track consecutive None returns during grace period
        consecutive_nones = 0
        max_consecutive_nones = 10  # Exit early if we get 10 None returns in a row during grace period
        
        while self.is_recording or self._stopping_recording:
            loop_count += 1
            try:
                audio_frame = self._robot.media.get_audio_sample()
                if audio_frame is not None:
                    consecutive_nones = 0  # Reset counter when we get audio
                    # Handle different return formats
                    # After start_recording(), it might return just audio data, not a tuple
                    if isinstance(audio_frame, tuple):
                        # Format: (sample_rate, audio_data)
                        sample_rate, audio_data = audio_frame
                    else:
                        # Format: just audio_data (numpy array)
                        # Get sample rate separately
                        audio_data = audio_frame
                        sample_rate = self._robot.media.get_input_audio_samplerate()
                    
                    self.recorded_audio.append((sample_rate, audio_data))
                    frame_count += 1
                    if frame_count == 1:
                        self.log_print(f"[RECORDER] First audio frame captured! ({len(audio_data)} samples @ {sample_rate}Hz)")
                    elif frame_count % 50 == 0:  # Log every 50th frame to reduce spam
                        self.log_print(f"[RECORDER] Captured {frame_count} frames ({len(audio_data)} samples @ {sample_rate}Hz each)")
                    
                    # Send to transcription if enabled
                    if self.is_transcribing and self.input_stream:
                        await self._send_audio_to_transcribe(sample_rate, audio_data)
                else:
                    none_count += 1
                    consecutive_nones += 1
                    
                    # During grace period, exit early if buffer is empty
                    if self._stopping_recording and consecutive_nones >= max_consecutive_nones:
                        self.log_print(f"[RECORDER] Buffer appears empty ({consecutive_nones} consecutive None returns), exiting early")
                        break
                    
                    if none_count == 1:
                        self.log_print("[RECORDER] get_audio_sample() returned None - waiting for audio input...")
                    elif none_count % 100 == 0:  # Log every 100th None to reduce spam
                        self.log_print(f"[RECORDER] Still waiting for audio... ({none_count} None returns, {loop_count} total loops)")
            except Exception as e:
                self.log_print(f"[RECORDER] Error capturing audio: {e}", "ERROR")
                # Log what we actually got for debugging
                try:
                    audio_frame = self._robot.media.get_audio_sample()
                    if audio_frame is not None:
                        self.log_print(f"[RECORDER] Debug: audio_frame type={type(audio_frame)}, value={audio_frame}")
                except:
                    pass
                import traceback
                self.log_print(traceback.format_exc(), "ERROR")
            
            # Yield to event loop (like console.py does)
            await asyncio.sleep(0)
        
        # Reset stopping flag
        self._stopping_recording = False
        self.log_print(f"[RECORDER] Loop ended: {frame_count} frames captured, {none_count} None returns, {loop_count} total loops")
    
    async def replay_recording(self):
        """Replay the recorded audio directly."""
        if not self.recorded_audio:
            self.log_print("[RECORDER] No audio to replay")
            return
        
        # Get input and output sample rates for resampling
        # CRITICAL: Always get the actual output sample rate from hardware
        try:
            input_sample_rate = self._robot.media.get_input_audio_samplerate()
            output_sample_rate = self._robot.media.get_output_audio_samplerate()
            self.log_print(f"[RECORDER] Hardware rates - Input: {input_sample_rate} Hz, Output: {output_sample_rate} Hz")
        except Exception as e:
            self.log_print(f"[RECORDER] Could not get sample rates from hardware: {e}")
            # Try to get at least output rate
            try:
                output_sample_rate = self._robot.media.get_output_audio_samplerate()
                self.log_print(f"[RECORDER] Got output rate: {output_sample_rate} Hz")
            except:
                # Last resort: use recorded rate, but warn
                if self.recorded_audio:
                    output_sample_rate = self.recorded_audio[0][0]
                    self.log_print(f"[RECORDER] WARNING: Using recorded rate as output: {output_sample_rate} Hz (may cause playback issues)")
                else:
                    self.log_print("[RECORDER] No recorded audio and cannot get sample rates")
                    return
        
        # CRITICAL: Start the speaker before playing
        try:
            self._robot.media.start_playing()
            self.log_print("[RECORDER] Speaker started")
        except Exception as e:
            self.log_print(f"[RECORDER] Error starting speaker: {e}", "ERROR")
            return
        
        self.log_print(f"[RECORDER] Replaying {len(self.recorded_audio)} audio frames...")
        
        try:
            from scipy import signal
            
            # Step 1: Validate all frames have same sample rate
            sample_rates = {sr for sr, _ in self.recorded_audio}
            if len(sample_rates) > 1:
                self.log_print(f"[RECORDER] WARNING: Mixed sample rates detected: {sample_rates}")
            recorded_sample_rate = sample_rates.pop() if sample_rates else input_sample_rate
            
            # Validate sample rates
            if recorded_sample_rate <= 0 or output_sample_rate <= 0:
                self.log_print(f"[RECORDER] Invalid sample rates: input={recorded_sample_rate}, output={output_sample_rate}")
                return
            
            # Step 2: Concatenate all audio frames
            self.log_print("[RECORDER] Concatenating audio frames...")
            all_audio_chunks = []
            
            for i, (sample_rate, audio_data) in enumerate(self.recorded_audio):
                # Convert to float32 if needed
                if audio_data.dtype != np.float32:
                    if audio_data.dtype == np.int16:
                        # int16 range is [-32768, 32767], normalize to [-1.0, ~1.0]
                        audio_float = audio_data.astype(np.float32) / 32768.0
                    else:
                        audio_float = audio_data.astype(np.float32)
                else:
                    audio_float = audio_data
                
                # Ensure mono (1D array) - handle both (samples, channels) and (channels, samples)
                if audio_float.ndim == 2:
                    # Handle both (samples, channels) and (channels, samples)
                    if audio_float.shape[0] < audio_float.shape[1]:
                        audio_float = audio_float.T
                    # Average channels if multiple, otherwise take first
                    if audio_float.shape[1] > 1:
                        audio_float = np.mean(audio_float, axis=1)
                    else:
                        audio_float = audio_float[:, 0]
                
                all_audio_chunks.append(audio_float)
            
            # Concatenate all chunks
            if not all_audio_chunks:
                self.log_print("[RECORDER] No audio data to replay")
                return
            
            concatenated_audio = np.concatenate(all_audio_chunks)
            
            if len(concatenated_audio) == 0:
                self.log_print("[RECORDER] Concatenated audio is empty")
                return
            
            self.log_print(f"[RECORDER] Concatenated {len(self.recorded_audio)} frames into {len(concatenated_audio)} total samples @ {recorded_sample_rate}Hz")
            self.log_print(f"[RECORDER] Target output sample rate: {output_sample_rate}Hz")
            
            # Step 3: ALWAYS resample to output sample rate
            # CRITICAL: Force resampling ALWAYS to fix slow/low audio
            # Even if rates appear to match, hardware may expect different rate
            self.log_print(f"[RECORDER] Sample rate check: recorded={recorded_sample_rate}Hz, output={output_sample_rate}Hz")
            self.log_print(f"[RECORDER] FORCING resampling to output rate (always resample)")
            
            # ALWAYS resample to output rate to ensure correct playback speed
            num_samples = round(output_sample_rate * len(concatenated_audio) / recorded_sample_rate)
            
            if num_samples <= 0:
                self.log_print("[RECORDER] Invalid resampling target size")
                return
            
            # Validate minimum array size for resampling
            if len(concatenated_audio) < 10:
                self.log_print(f"[RECORDER] Audio too short for resampling ({len(concatenated_audio)} samples), skipping")
                return
            
            original_duration = len(concatenated_audio) / recorded_sample_rate
            expected_duration = num_samples / output_sample_rate
            self.log_print(f"[RECORDER] Resampling from {recorded_sample_rate}Hz to {output_sample_rate}Hz")
            self.log_print(f"[RECORDER]   Original: {len(concatenated_audio)} samples = {original_duration:.3f}s")
            self.log_print(f"[RECORDER]   Target: {num_samples} samples = {expected_duration:.3f}s")
            
            # Use async resampling to avoid blocking the event loop
            concatenated_audio = await asyncio.to_thread(
                signal.resample, concatenated_audio, num_samples
            )
            
            # Normalize audio to prevent clipping (capping)
            # Clip values to [-1.0, 1.0] range to prevent distortion
            max_val = np.abs(concatenated_audio).max()
            if max_val > 1.0:
                self.log_print(f"[RECORDER] WARNING: Audio clipping detected (max={max_val:.3f}), normalizing...")
                concatenated_audio = np.clip(concatenated_audio, -1.0, 1.0)
            elif max_val > 0:
                # Optional: normalize to use full dynamic range (comment out if not needed)
                # concatenated_audio = concatenated_audio / max_val
                pass
            
            actual_duration = len(concatenated_audio) / output_sample_rate
            self.log_print(f"[RECORDER] Resampling complete: {len(concatenated_audio)} samples (duration: {actual_duration:.3f}s)")
            
            # Step 4: Push audio in chunks with proper timing
            # Push chunks and sleep to match real-time playback rate
            chunk_size = round(output_sample_rate * 0.01)  # 10ms chunks
            
            if chunk_size <= 0:
                self.log_print("[RECORDER] Invalid chunk size, using default")
                chunk_size = round(output_sample_rate * 0.01)  # 10ms chunks
            
            chunk_duration = chunk_size / output_sample_rate  # Duration of each chunk
            self.log_print(f"[RECORDER] Pushing audio in chunks of ~{chunk_size} samples ({chunk_duration*1000:.1f}ms each)...")
            
            total_pushed = 0
            start_time = time.time()
            last_chunk_time = start_time
            
            for i in range(0, len(concatenated_audio), chunk_size):
                chunk = concatenated_audio[i:i+chunk_size]
                if len(chunk) > 0:
                    chunk_start = time.time()
                    self._robot.media.push_audio_sample(chunk)
                    total_pushed += len(chunk)
                    
                    # Calculate how long to sleep to maintain real-time playback
                    # Account for time spent pushing
                    push_time = time.time() - chunk_start
                    sleep_time = max(0, chunk_duration - push_time)
                    
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
                    
                    if (i // chunk_size) % 100 == 0:
                        self.log_print(f"[RECORDER] Pushed {total_pushed}/{len(concatenated_audio)} samples ({100*total_pushed/len(concatenated_audio):.1f}%)")
            
            elapsed = time.time() - start_time
            expected_duration = len(concatenated_audio) / output_sample_rate
            self.log_print(f"[RECORDER] Finished pushing all {total_pushed} samples (elapsed: {elapsed:.3f}s, expected: {expected_duration:.3f}s)")
            
            # Wait for remaining audio to finish playing
            # The audio system buffers chunks, so we need to wait for the buffer to drain
            remaining_time = max(0, expected_duration - elapsed)
            if remaining_time > 0:
                self.log_print(f"[RECORDER] Waiting {remaining_time:.3f}s for audio buffer to drain...")
                await asyncio.sleep(remaining_time)
            
            # Additional small buffer to ensure all audio is played
            # Audio systems often have internal buffering
            await asyncio.sleep(0.1)
            self.log_print(f"[RECORDER] Audio playback complete")
                    
        except Exception as e:
            self.log_print(f"[RECORDER] Error during replay: {e}", "ERROR")
            import traceback
            self.log_print(traceback.format_exc(), "ERROR")
        finally:
            # CRITICAL: Stop the speaker after playing
            # Only stop after ensuring all audio has been played
            try:
                self._robot.media.stop_playing()
                self.log_print("[RECORDER] Speaker stopped")
            except Exception as e:
                self.log_print(f"[RECORDER] Error stopping speaker: {e}", "ERROR")
        
        self.log_print("[RECORDER] Replay complete")
    
    def get_status(self):
        """Get current recording status."""
        duration = 0
        if self.recording_start_time and self.is_recording:
            duration = time.time() - self.recording_start_time
        
        total_samples = sum(len(audio) for _, audio in self.recorded_audio)
        sample_rate = self.recorded_audio[0][0] if self.recorded_audio else 0
        audio_duration = total_samples / sample_rate if sample_rate > 0 else 0
        
        result = {
            "is_recording": self.is_recording,
            "frames_recorded": len(self.recorded_audio),
            "recording_duration": round(duration, 2),
            "audio_duration": round(audio_duration, 2),
            "total_samples": total_samples,
            "sample_rate": sample_rate
        }
        
        # Add transcription info
        if self.transcription_enabled:
            result["transcription"] = {
                "enabled": True,
                "partial": self.partial_transcript,
                "final_segments": self.final_transcripts,
                "full_text": " ".join(self.final_transcripts) if self.final_transcripts else "",
                "all_segments": self.all_transcripts[-10:]  # Last 10 segments for display
            }
        else:
            result["transcription"] = {"enabled": False}
        
        return result
    
    def _init_aws_clients(self):
        """Initialize AWS Transcribe and Polly clients."""
        try:
            if self.aws_profile:
                session = boto3.Session(profile_name=self.aws_profile)
                self.log_print(f"[AWS] Using AWS profile: {self.aws_profile}")
            else:
                session = boto3.Session()
                self.log_print("[AWS] Using default AWS credentials")
            
            # Initialize Polly client
            self.polly_client = session.client('polly', region_name=self.aws_region)
            self.log_print("[AWS] Initialized Polly client")
            
            # Initialize Transcribe client
            if HAS_AMAZON_TRANSCRIBE:
                if self.aws_profile:
                    original_profile = os.environ.get('AWS_PROFILE')
                    os.environ['AWS_PROFILE'] = self.aws_profile
                    try:
                        self.transcribe_client = TranscribeStreamingClient(region=self.aws_region)
                    except Exception as e:
                        if original_profile:
                            os.environ['AWS_PROFILE'] = original_profile
                        elif 'AWS_PROFILE' in os.environ:
                            del os.environ['AWS_PROFILE']
                        raise
                else:
                    self.transcribe_client = TranscribeStreamingClient(region=self.aws_region)
                self.log_print("[AWS] Initialized Transcribe with amazon-transcribe library")
            else:
                self.transcribe_client = session.client(
                    'transcribe-streaming',
                    region_name=self.aws_region
                )
                self.log_print("[AWS] Initialized Transcribe with boto3 (consider installing amazon-transcribe)")
        except Exception as e:
            self.log_print(f"[AWS] Error initializing: {e}", "ERROR")
            self.transcription_enabled = False
            self.transcribe_client = None
            self.polly_client = None
    
    async def _start_transcription(self):
        """Start AWS Transcribe streaming session."""
        if not self.transcribe_client or self.is_transcribing:
            return
        
        try:
            if HAS_AMAZON_TRANSCRIBE:
                stream = await self.transcribe_client.start_stream_transcription(
                    language_code=self.language_code,
                    media_sample_rate_hz=self.transcribe_sample_rate,
                    media_encoding="pcm",
                )
                
                self.input_stream = stream.input_stream
                self.output_stream = stream.output_stream
                self.transcribe_stream = stream
                self.is_transcribing = True
                self.log_print("[TRANSCRIBE] Started streaming transcription")
                
                # Create event handler
                self.transcription_handler = TranscriptEventHandler(
                    transcript_result_stream=self.output_stream,
                    recorder=self,
                    log_print_func=self.log_print
                )
                
                # Start processing events
                self.transcribe_task = asyncio.create_task(
                    self.transcription_handler.handle_events()
                )
            else:
                self.log_print("[TRANSCRIBE] amazon-transcribe not available, transcription disabled", "WARNING")
                self.transcription_enabled = False
        except Exception as e:
            self.log_print(f"[TRANSCRIBE] Error starting transcription: {e}", "ERROR")
            self.is_transcribing = False
    
    async def _stop_transcription(self):
        """Stop AWS Transcribe streaming session."""
        if not self.is_transcribing:
            return
        
        self.is_transcribing = False
        
        try:
            if self.input_stream:
                await self.input_stream.end_stream()
        except Exception as e:
            self.log_print(f"[TRANSCRIBE] Error ending stream: {e}")
        
        if self.transcribe_task and not self.transcribe_task.done():
            self.transcribe_task.cancel()
            try:
                await self.transcribe_task
            except asyncio.CancelledError:
                pass
        
        self.log_print("[TRANSCRIBE] Stopped transcription")
    
    async def _send_audio_to_transcribe(self, sample_rate: int, audio_data: np.ndarray):
        """Send audio frame to AWS Transcribe."""
        if not self.is_transcribing or not self.input_stream:
            return
        
        try:
            # Convert to mono if needed
            if audio_data.ndim == 2:
                if audio_data.shape[1] > 0:
                    audio_data = audio_data[:, 0]
                else:
                    audio_data = audio_data.flatten()
            
            # Resample to 16kHz (AWS Transcribe requirement)
            if sample_rate != self.transcribe_sample_rate:
                num_samples = int(len(audio_data) * self.transcribe_sample_rate / sample_rate)
                if num_samples > 0:
                    audio_data = signal.resample(audio_data, num_samples)
                else:
                    return
            
            # Convert to int16
            if audio_data.dtype != np.int16:
                if audio_data.dtype in [np.float32, np.float64]:
                    audio_data = np.clip(audio_data, -1.0, 1.0)
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                else:
                    audio_int16 = audio_data.astype(np.int16)
            else:
                audio_int16 = audio_data
            
            # Convert to bytes and send
            audio_bytes = audio_int16.tobytes()
            await self.input_stream.send_audio_event(audio_chunk=audio_bytes)
        except Exception as e:
            # Don't spam errors if transcription fails
            pass
    
    def _add_transcript(self, text: str, is_partial: bool):
        """Add transcript result (partial or final)."""
        timestamp = time.time()
        
        if is_partial:
            self.partial_transcript = text
        else:
            if text.strip():
                self.final_transcripts.append(text)
                self.partial_transcript = ""
        
        # Store in all_transcripts for history
        self.all_transcripts.append({
            "type": "partial" if is_partial else "final",
            "text": text,
            "timestamp": timestamp
        })
        
        # Limit history size
        if len(self.all_transcripts) > 100:
            self.all_transcripts = self.all_transcripts[-100:]
    
    def get_transcription(self):
        """Get current transcription results."""
        return {
            "partial": self.partial_transcript,
            "final_segments": self.final_transcripts,
            "full_text": " ".join(self.final_transcripts) if self.final_transcripts else "",
            "all_segments": self.all_transcripts[-20:]  # Last 20 segments
        }
    
    async def _send_to_api_and_speak(self, text: str):
        """Send transcription to API endpoint and speak the response."""
        if not text.strip():
            return
        
        # Choose API endpoint based on conversation mode
        if self.conversation_mode:
            api_url = self.conversation_api_url
        else:
            api_url = "https://reachy.tristy.dev/api/reachy/input"
        
        response_text = None
        should_speak = True  # Default to speaking the response
        
        # Try to send to API
        try:
            self.log_print(f"[API] Sending to {api_url}: {text}")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    api_url,
                    json={"text": text},
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Handle conversation API response format: {"text": "...", "spoken": false}
                        if self.conversation_mode:
                            response_text = data.get("text", "")
                            should_speak = data.get("spoken", True)  # Default to True if not specified
                            if response_text:
                                self.log_print(f"[API] Received response: {response_text} (spoken: {should_speak})")
                            else:
                                self.log_print("[API] Response missing 'text' key or empty")
                                response_text = None
                        else:
                            # Handle original API response format: {"response": "..."}
                            response_text = data.get("response", "")
                            if response_text:
                                self.log_print(f"[API] Received response: {response_text}")
                            else:
                                self.log_print("[API] Response missing 'response' key or empty")
                                response_text = None
                    else:
                        self.log_print(f"[API] Error: HTTP {response.status}", "ERROR")
                        response_text = None
        except asyncio.TimeoutError:
            self.log_print(f"[API] Request timed out", "ERROR")
            response_text = None
        except Exception as e:
            self.log_print(f"[API] Error sending request: {e}", "ERROR")
            response_text = None
        
        # Speak the response if we have one and should_speak is True
        if response_text and should_speak:
            await self._speak_text(response_text)
        elif response_text and not should_speak:
            self.log_print(f"[API] Response received but not speaking (spoken=false): {response_text}")
        else:
            # Speak error message
            error_text = f"Failed to send {text}"
            await self._speak_text(error_text)
    
    async def _speak_text(self, text: str):
        """Speak text using AWS Polly."""
        if not text.strip() or not self.polly_client:
            return
        
        self.log_print(f"[TTS] Speaking: {text}")
        
        try:
            # Get output sample rate from robot
            try:
                output_sample_rate = self._robot.media.get_output_audio_samplerate()
            except:
                output_sample_rate = self.polly_sample_rate
                self.log_print(f"[TTS] Could not get output sample rate, using {output_sample_rate}Hz")
            
            # Generate TTS using AWS Polly
            response = self.polly_client.synthesize_speech(
                Text=text,
                OutputFormat='pcm',  # PCM format for direct playback
                VoiceId=self.polly_voice_id,
                Engine=self.polly_engine,
                SampleRate=str(output_sample_rate),
            )
            
            # Read audio stream
            audio_stream = response['AudioStream'].read()
            
            # Convert bytes to numpy array
            # PCM format: int16, mono, at specified sample rate
            audio_data = np.frombuffer(audio_stream, dtype=np.int16)
            
            # Convert to float32 for robot (range [-1, 1])
            audio_float = audio_data.astype(np.float32) / 32767.0
            
            # Start the speaker
            try:
                self._robot.media.start_playing()
                self.log_print("[TTS] Speaker started")
            except Exception as e:
                self.log_print(f"[TTS] Error starting speaker: {e}", "ERROR")
                return
            
            # Push audio in chunks with proper timing
            chunk_size = round(output_sample_rate * 0.01)  # 10ms chunks
            chunk_duration = chunk_size / output_sample_rate
            
            total_pushed = 0
            start_time = time.time()
            
            for i in range(0, len(audio_float), chunk_size):
                chunk = audio_float[i:i+chunk_size]
                if len(chunk) > 0:
                    chunk_start = time.time()
                    self._robot.media.push_audio_sample(chunk)
                    total_pushed += len(chunk)
                    
                    # Calculate sleep time to maintain real-time playback
                    push_time = time.time() - chunk_start
                    sleep_time = max(0, chunk_duration - push_time)
                    
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
            
            # Wait for audio to finish playing
            elapsed = time.time() - start_time
            expected_duration = len(audio_float) / output_sample_rate
            remaining_time = max(0, expected_duration - elapsed)
            if remaining_time > 0:
                await asyncio.sleep(remaining_time)
            
            # Additional buffer
            await asyncio.sleep(0.1)
            
            # Stop the speaker
            try:
                self._robot.media.stop_playing()
                self.log_print("[TTS] Speaker stopped")
            except Exception as e:
                self.log_print(f"[TTS] Error stopping speaker: {e}", "ERROR")
            
            self.log_print(f"[TTS] Finished speaking: {text}")
            
        except ClientError as e:
            self.log_print(f"[TTS] AWS Polly error: {e}", "ERROR")
        except Exception as e:
            self.log_print(f"[TTS] Error generating/playing speech: {e}", "ERROR")
            import traceback
            self.log_print(traceback.format_exc(), "ERROR")


if HAS_AMAZON_TRANSCRIBE:
    class TranscriptEventHandler(TranscriptResultStreamHandler):
        """Event handler for AWS Transcribe streaming results."""
        
        def __init__(self, transcript_result_stream, recorder, log_print_func):
            super().__init__(transcript_result_stream)
            self.recorder = recorder
            self.log_print = log_print_func
        
        async def handle_transcript_event(self, transcript_event: TranscriptEvent):
            """Handle transcript events from AWS Transcribe."""
            results = transcript_event.transcript.results
            for result in results:
                if result.alternatives:
                    transcript_text = result.alternatives[0].transcript
                    if result.is_partial:
                        self.recorder._add_transcript(transcript_text, is_partial=True)
                        self.log_print(f"[TRANSCRIBE - PARTIAL] {transcript_text}")
                    else:
                        self.recorder._add_transcript(transcript_text, is_partial=False)
                        self.log_print(f"[TRANSCRIBE - FINAL] {transcript_text}")
else:
    # Dummy class if amazon-transcribe is not available
    class TranscriptEventHandler:
        def __init__(self, transcript_result_stream, recorder, log_print_func):
            self.recorder = recorder
            self.log_print = log_print_func
        
        async def handle_events(self):
            pass