"""Voice Activity Detection (VAD) for cost optimization."""

import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class SimpleVAD:
    """Simple Voice Activity Detection using energy threshold."""
    
    def __init__(
        self,
        energy_threshold: float = 0.01,
        frame_duration_ms: int = 30,
        min_speech_frames: int = 3,  # ~90ms of speech to trigger
        min_silence_frames: int = 10,  # ~300ms of silence to stop
        sample_rate: int = 16000,
    ):
        self.energy_threshold = energy_threshold
        self.frame_duration_ms = frame_duration_ms
        self.min_speech_frames = min_speech_frames
        self.min_silence_frames = min_silence_frames
        self.sample_rate = sample_rate
        
        # State
        self.speech_frame_count = 0
        self.silence_frame_count = 0
        self.is_speaking = False
        self.speech_started = False
        
    def process_frame(self, audio_frame: np.ndarray) -> Tuple[bool, bool]:
        """
        Process audio frame and detect speech.
        
        Returns:
            (is_speaking, speech_just_started)
            - is_speaking: Current speech state
            - speech_just_started: True if speech just started (for triggering transcription)
        """
        # Calculate RMS energy
        if audio_frame.ndim > 1:
            audio_frame = audio_frame.flatten()
        
        # Convert to float if needed
        if audio_frame.dtype != np.float32:
            if audio_frame.dtype == np.int16:
                audio_frame = audio_frame.astype(np.float32) / 32767.0
            else:
                audio_frame = audio_frame.astype(np.float32)
        
        # Calculate RMS energy
        rms_energy = np.sqrt(np.mean(audio_frame ** 2))
        
        speech_just_started = False
        
        if rms_energy > self.energy_threshold:
            # Potential speech detected
            self.speech_frame_count += 1
            self.silence_frame_count = 0
            
            # Check if we've detected enough consecutive speech frames
            if not self.is_speaking and self.speech_frame_count >= self.min_speech_frames:
                self.is_speaking = True
                self.speech_started = True
                speech_just_started = True
        else:
            # Silence detected
            self.silence_frame_count += 1
            self.speech_frame_count = 0
            
            # Check if we've detected enough consecutive silence frames
            if self.is_speaking and self.silence_frame_count >= self.min_silence_frames:
                self.is_speaking = False
                self.speech_started = False
        
        return self.is_speaking, speech_just_started
    
    def reset(self):
        """Reset VAD state."""
        self.speech_frame_count = 0
        self.silence_frame_count = 0
        self.is_speaking = False
        self.speech_started = False


try:
    import webrtcvad
    HAS_WEBRTC_VAD = True
except ImportError:
    HAS_WEBRTC_VAD = False


class WebRTCVAD:
    """WebRTC VAD for better accuracy (requires webrtcvad package)."""
    
    def __init__(
        self,
        aggressiveness: int = 2,  # 0-3, higher = more aggressive
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
    ):
        if not HAS_WEBRTC_VAD:
            raise ImportError("webrtcvad not installed. Install with: pip install webrtcvad")
        
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        
        # State for debouncing
        self.speech_frame_count = 0
        self.silence_frame_count = 0
        self.is_speaking = False
        self.min_speech_frames = 3
        self.min_silence_frames = 10
        
    def process_frame(self, audio_frame: np.ndarray) -> Tuple[bool, bool]:
        """
        Process audio frame using WebRTC VAD.
        
        Returns:
            (is_speaking, speech_just_started)
        """
        # Convert to int16 if needed
        if audio_frame.dtype != np.int16:
            if audio_frame.dtype in [np.float32, np.float64]:
                audio_frame = (audio_frame * 32767).astype(np.int16)
            else:
                audio_frame = audio_frame.astype(np.int16)
        
        # Ensure correct frame size
        if len(audio_frame) != self.frame_size:
            # Resample or pad/trim
            if len(audio_frame) > self.frame_size:
                audio_frame = audio_frame[:self.frame_size]
            else:
                # Pad with zeros
                padded = np.zeros(self.frame_size, dtype=np.int16)
                padded[:len(audio_frame)] = audio_frame
                audio_frame = padded
        
        # Convert to bytes
        audio_bytes = audio_frame.tobytes()
        
        # Check if speech
        is_speech = self.vad.is_speech(audio_bytes, self.sample_rate)
        
        speech_just_started = False
        
        if is_speech:
            self.speech_frame_count += 1
            self.silence_frame_count = 0
            
            if not self.is_speaking and self.speech_frame_count >= self.min_speech_frames:
                self.is_speaking = True
                speech_just_started = True
        else:
            self.silence_frame_count += 1
            self.speech_frame_count = 0
            
            if self.is_speaking and self.silence_frame_count >= self.min_silence_frames:
                self.is_speaking = False
        
        return self.is_speaking, speech_just_started
    
    def reset(self):
        """Reset VAD state."""
        self.speech_frame_count = 0
        self.silence_frame_count = 0
        self.is_speaking = False
