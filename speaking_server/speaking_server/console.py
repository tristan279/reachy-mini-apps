"""LocalStream for audio capture and playback with Reachy Mini."""

import asyncio
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def audio_to_float32(audio: np.ndarray) -> np.ndarray:
    """Convert audio to float32 format in range [-1, 1]."""
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32767.0
    elif audio.dtype == np.int32:
        return audio.astype(np.float32) / 2147483647.0
    elif audio.dtype in [np.float32, np.float64]:
        return audio.astype(np.float32)
    else:
        return audio.astype(np.float32)


def resample(audio: np.ndarray, num_samples: int) -> np.ndarray:
    """Simple resampling using linear interpolation."""
    from scipy import signal
    if len(audio) == num_samples:
        return audio
    return signal.resample(audio, num_samples)


class LocalStream:
    """Manages audio streaming between robot and handler."""
    
    def __init__(
        self,
        robot,
        handler,
        stop_event,
        log_print_func=None,
    ):
        self._robot = robot
        self.handler = handler
        self._stop_event = stop_event
        self.log_print = log_print_func or print
        
        # Audio sample rates
        self._input_sample_rate = None
        self._output_sample_rate = None
        
    async def record_loop(self):
        """Continuously poll robot for audio and forward to handler."""
        try:
            # Get input sample rate
            try:
                self._input_sample_rate = self._robot.media.get_input_audio_samplerate()
                self.log_print(f"Input audio sample rate: {self._input_sample_rate} Hz")
            except Exception as e:
                self.log_print(f"Could not get input sample rate: {e}")
                self._input_sample_rate = 16000  # Default fallback
            
            while not self._stop_event.is_set():
                try:
                    audio_frame = self._robot.media.get_audio_sample()
                    if audio_frame is not None:
                        sample_rate, audio_data = audio_frame
                        # Forward to handler
                        await self.handler.receive((sample_rate, audio_data))
                except Exception as e:
                    self.log_print(f"Error in record_loop: {e}", "ERROR")
                
                # Yield to event loop
                await asyncio.sleep(0)
                
        except Exception as e:
            self.log_print(f"Record loop error: {e}", "ERROR")
            import traceback
            self.log_print(traceback.format_exc(), "ERROR")
    
    async def play_loop(self):
        """Continuously poll handler for audio and play on robot."""
        try:
            # Get output sample rate
            try:
                self._output_sample_rate = self._robot.media.get_output_audio_samplerate()
                self.log_print(f"Output audio sample rate: {self._output_sample_rate} Hz")
            except Exception as e:
                self.log_print(f"Could not get output sample rate: {e}")
                self._output_sample_rate = 24000  # Default fallback
            
            while not self._stop_event.is_set():
                try:
                    # Get output from handler
                    handler_output = await self.handler.emit()
                    
                    if handler_output is None:
                        # No audio available, yield and continue
                        await asyncio.sleep(0.01)
                        continue
                    
                    if isinstance(handler_output, tuple):
                        # Audio frame: (sample_rate, audio_data)
                        input_sample_rate, audio_data = handler_output
                        
                        # Convert multi-channel to mono if needed
                        if audio_data.ndim == 2:
                            if audio_data.shape[0] == 1:
                                audio_data = audio_data[0, :]
                            elif audio_data.shape[1] == 1:
                                audio_data = audio_data[:, 0]
                            else:
                                # Take first channel
                                audio_data = audio_data[0, :]
                        
                        # Convert to float32
                        audio_frame = audio_to_float32(audio_data)
                        
                        # Resample if needed
                        if input_sample_rate != self._output_sample_rate:
                            num_samples = int(
                                len(audio_frame) * self._output_sample_rate / input_sample_rate
                            )
                            if num_samples > 0:
                                audio_frame = resample(audio_frame, num_samples)
                            else:
                                continue
                        
                        # Push to robot speaker
                        try:
                            self._robot.media.push_audio_sample(audio_frame)
                        except Exception as e:
                            self.log_print(f"Error pushing audio to robot: {e}", "ERROR")
                    elif isinstance(handler_output, str):
                        # Text message (for logging)
                        self.log_print(f"Handler message: {handler_output}")
                    
                except Exception as e:
                    self.log_print(f"Error in play_loop: {e}", "ERROR")
                
                # Small sleep to prevent tight loop
                await asyncio.sleep(0.001)
                
        except Exception as e:
            self.log_print(f"Play loop error: {e}", "ERROR")
            import traceback
            self.log_print(traceback.format_exc(), "ERROR")
