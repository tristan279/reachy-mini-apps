"""Simple audio recorder for testing - records audio and replays it."""

import asyncio
import time
import numpy as np
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class SimpleRecorder:
    """Simple recorder that captures audio and replays it directly."""
    
    def __init__(self, robot, log_print_func=None):
        self._robot = robot
        self.log_print = log_print_func or print
        
        self.is_recording = False
        self.recorded_audio: List[Tuple[int, np.ndarray]] = []  # List of (sample_rate, audio_data)
        self.recording_start_time = None
        
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
        self.log_print("[RECORDER] Started recording")
        
        # Start recording loop
        asyncio.create_task(self._record_loop())
    
    async def stop_recording(self):
        """Stop recording."""
        if not self.is_recording:
            self.log_print("[RECORDER] Not currently recording")
            return
        
        self.is_recording = False
        
        # CRITICAL: Stop the microphone
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
        
        while self.is_recording:
            loop_count += 1
            try:
                audio_frame = self._robot.media.get_audio_sample()
                if audio_frame is not None:
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
                else:
                    none_count += 1
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
            
            # Step 3: ALWAYS resample to match output sample rate (even if rates appear to match)
            # This ensures correct playback speed regardless of any sample rate mismatches
            if recorded_sample_rate != output_sample_rate:
                # Use round() instead of int() to avoid precision loss
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
                self.log_print(f"[RECORDER] Resampling complete: {len(concatenated_audio)} samples")
            else:
                self.log_print(f"[RECORDER] No resampling needed (both rates are {recorded_sample_rate}Hz)")
                # Double-check: if rates match but audio sounds wrong, force resampling anyway
                # This handles cases where the reported rates don't match actual hardware
                if len(concatenated_audio) > 0:
                    actual_duration = len(concatenated_audio) / recorded_sample_rate
                    self.log_print(f"[RECORDER] Audio duration: {actual_duration:.3f}s at {recorded_sample_rate}Hz")
            
            # Step 4: Push audio in chunks quickly, then sleep for total duration
            # The audio system handles buffering, so we push quickly and sleep once
            chunk_size = round(output_sample_rate * 0.01)  # 10ms chunks for smooth pushing
            
            if chunk_size <= 0:
                self.log_print("[RECORDER] Invalid chunk size, using default")
                chunk_size = round(output_sample_rate * 0.01)  # 10ms chunks
            
            self.log_print(f"[RECORDER] Pushing audio in chunks of ~{chunk_size} samples...")
            
            total_pushed = 0
            start_time = time.time()
            
            # Push all chunks quickly (tiny delay just to yield to event loop)
            for i in range(0, len(concatenated_audio), chunk_size):
                chunk = concatenated_audio[i:i+chunk_size]
                if len(chunk) > 0:
                    self._robot.media.push_audio_sample(chunk)
                    total_pushed += len(chunk)
                    
                    # Tiny yield to event loop, not a timing delay
                    await asyncio.sleep(0)
                    
                    if (i // chunk_size) % 100 == 0:
                        self.log_print(f"[RECORDER] Pushed {total_pushed}/{len(concatenated_audio)} samples ({100*total_pushed/len(concatenated_audio):.1f}%)")
            
            # Calculate total audio duration and sleep for that duration
            # This matches the example: time.sleep(len(samples) / output_sample_rate)
            total_duration = len(concatenated_audio) / output_sample_rate
            self.log_print(f"[RECORDER] Pushed all {total_pushed} samples, sleeping for {total_duration:.3f}s (total audio duration)")
            
            # Sleep for the total duration of the audio at the output sample rate
            await asyncio.sleep(total_duration)
            
            elapsed = time.time() - start_time
            self.log_print(f"[RECORDER] Finished (elapsed: {elapsed:.3f}s, expected: {total_duration:.3f}s)")
                    
        except Exception as e:
            self.log_print(f"[RECORDER] Error during replay: {e}", "ERROR")
            import traceback
            self.log_print(traceback.format_exc(), "ERROR")
        finally:
            # CRITICAL: Stop the speaker after playing
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
        
        return {
            "is_recording": self.is_recording,
            "frames_recorded": len(self.recorded_audio),
            "recording_duration": round(duration, 2),
            "audio_duration": round(audio_duration, 2),
            "total_samples": total_samples,
            "sample_rate": sample_rate
        }
