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
        
        while self.is_recording:
            loop_count += 1
            try:
                audio_frame = self._robot.media.get_audio_sample()
                if audio_frame is not None:
                    sample_rate, audio_data = audio_frame
                    self.recorded_audio.append((sample_rate, audio_data))
                    frame_count += 1
                    if frame_count % 50 == 0:  # Log every 50th frame to reduce spam
                        self.log_print(f"[RECORDER] Captured {frame_count} frames ({len(audio_data)} samples @ {sample_rate}Hz each)")
                else:
                    none_count += 1
                    if none_count % 100 == 0:  # Log every 100th None to reduce spam
                        self.log_print(f"[RECORDER] get_audio_sample() returned None ({none_count} times, {loop_count} total loops)")
            except Exception as e:
                self.log_print(f"[RECORDER] Error capturing audio: {e}", "ERROR")
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
        
        self.log_print(f"[RECORDER] Replaying {len(self.recorded_audio)} audio frames...")
        
        for i, (sample_rate, audio_data) in enumerate(self.recorded_audio):
            try:
                # Convert to float32 if needed
                if audio_data.dtype != np.float32:
                    if audio_data.dtype == np.int16:
                        audio_float = audio_data.astype(np.float32) / 32767.0
                    else:
                        audio_float = audio_data.astype(np.float32)
                else:
                    audio_float = audio_data
                
                # Ensure mono
                if audio_float.ndim == 2:
                    audio_float = audio_float[:, 0] if audio_float.shape[1] > 0 else audio_float.flatten()
                
                # Push to robot speaker
                self._robot.media.push_audio_sample(audio_float)
                
                if i % 50 == 0:  # Log every 50th frame
                    self.log_print(f"[RECORDER] Replayed {i+1}/{len(self.recorded_audio)} frames")
                
                # Small delay to match original timing
                await asyncio.sleep(0.01)
                
            except Exception as e:
                self.log_print(f"[RECORDER] Error replaying frame {i}: {e}", "ERROR")
        
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
