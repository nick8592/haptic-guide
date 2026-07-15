# =============================================================================
# HapticGuide — Real-Time Spatial Audio Engine
# PipeWire/PulseAudio + sounddevice for low-latency audio output
# =============================================================================

from __future__ import annotations

import math
import time
from enum import Enum
from typing import Callable

import numpy as np
import sounddevice as sd
from loguru import logger

from .feedback_engine import FeedbackSignal, FeedbackMode


# ---------------------------------------------------------------------------
# Earcon Definitions
# ---------------------------------------------------------------------------

class EarconType(str, Enum):
    ASCENDING_TRIAD = "ascending_triad"
    DESCENDING_TRIAD = "descending_triad"
    DOUBLE_BEEP = "double_beep"
    SWEEP = "sweep"


def _generate_earcon(earcon_type: EarconType, sample_rate: int = 44100) -> np.ndarray:
    """Generate a short distinctive earcon sound."""
    duration_s = 0.3
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), dtype=np.float32)

    if earcon_type == EarconType.ASCENDING_TRIAD:
        # Three ascending tones: C5, E5, G5
        freqs = [523.25, 659.25, 783.99]
        segment_len = len(t) // 3
        signal = np.zeros_like(t)
        for i, freq in enumerate(freqs):
            start = i * segment_len
            end = start + segment_len
            segment_t = t[start:end] - t[start]
            signal[start:end] = np.sin(2 * np.pi * freq * segment_t) * 0.6
        # Envelope
        envelope = np.ones_like(t)
        fade = int(0.01 * sample_rate)
        envelope[:fade] = np.linspace(0, 1, fade)
        envelope[-fade:] = np.linspace(1, 0, fade)
        return (signal * envelope).astype(np.float32)

    elif earcon_type == EarconType.DOUBLE_BEEP:
        freq = 880
        signal = np.zeros_like(t)
        beep_len = int(0.05 * sample_rate)
        gap = int(0.03 * sample_rate)
        for start in [0, beep_len + gap]:
            end = start + beep_len
            if end > len(signal):
                break
            seg_t = np.arange(beep_len) / sample_rate
            signal[start:end] = np.sin(2 * np.pi * freq * seg_t) * 0.7
        return signal.astype(np.float32)

    else:
        # Default: simple beep
        return (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)


# ---------------------------------------------------------------------------
# Audio Engine
# ---------------------------------------------------------------------------

class AudioEngine:
    """
    Real-time spatial audio feedback engine.

    Converts FeedbackSignal into audible output:
    - Proximity tone: pitch + beat rate encode distance
    - Stereo pan: encodes direction (left/right)
    - Earcon: distinctive sound when target is locked

    Uses sounddevice (PortAudio) for cross-platform low-latency audio.
    On Linux, PortAudio routes through PipeWire/PulseAudio.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        buffer_size: int = 512,
        device: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.device = device

        # Current signal state (written by feedback thread, read by audio thread)
        self._current_signal: FeedbackSignal = FeedbackSignal()
        self._signal_lock = False

        # Phase accumulator for tone generation
        self._phase = 0.0
        self._beat_phase = 0.0

        # Earcon playback
        self._earcon_queue: list[np.ndarray] = []
        self._earcon_pos: int = 0
        self._playing_earcon: bool = False
        self._prev_locked: bool = False

        # Stream
        self._stream: sd.OutputStream | None = None

        logger.info(
            f"AudioEngine initialized: rate={sample_rate}, "
            f"buffer={buffer_size}, device={device}"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start audio output stream. Gracefully degrades to silent mode if no device."""
        if self._stream is not None:
            logger.warning("Audio stream already running")
            return

        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                blocksize=self.buffer_size,
                channels=2,
                dtype="float32",
                device=self.device,
                callback=self._audio_callback,
                latency="low",
            )
            self._stream.start()
            logger.info("Audio stream started")
        except (sd.PortAudioError, OSError) as e:
            logger.warning(f"No audio device available — running in silent mode ({e})")
            self._stream = None

    def stop(self) -> None:
        """Stop audio output stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Audio stream stopped")

    # ------------------------------------------------------------------
    # Signal Update (called from feedback thread)
    # ------------------------------------------------------------------

    def update_signal(self, signal: FeedbackSignal) -> None:
        """
        Update the current feedback signal.

        Called by the main loop at detection rate (15-30 Hz).
        Audio callback reads this continuously at audio rate (44100 Hz).
        """
        # Detect lock-on transition → trigger earcon
        if signal.audio_locked and not self._prev_locked:
            earcon = _generate_earcon(EarconType.ASCENDING_TRIAD, self.sample_rate)
            self._earcon_queue.append(earcon)
            self._earcon_pos = 0
            self._playing_earcon = True

        self._prev_locked = signal.audio_locked
        self._current_signal = signal

    # ------------------------------------------------------------------
    # Audio Callback (real-time, called by PortAudio)
    # ------------------------------------------------------------------

    def _audio_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: sd.CallbackInfo,
        status: sd.CallbackFlags,
    ) -> None:
        """
        Real-time audio callback. Generates stereo output.

        outdata shape: (frames, 2) — stereo float32
        """
        signal = self._current_signal

        if signal.mode == FeedbackMode.IDLE:
            outdata[:] = 0
            return

        # If earcon is playing, mix it in
        if self._playing_earcon and self._earcon_queue:
            earcon = self._earcon_queue[0]
            remaining = len(earcon) - self._earcon_pos

            if remaining <= 0:
                # Earcon finished
                self._earcon_queue.pop(0)
                self._playing_earcon = False
            else:
                # Mix earcon into output
                n = min(frames, remaining)
                earcon_slice = earcon[self._earcon_pos:self._earcon_pos + n]

                # Earcon is mono → make stereo centered
                outdata[:n, 0] = earcon_slice * 0.8
                outdata[:n, 1] = earcon_slice * 0.8

                if n < frames:
                    outdata[n:, :] = 0

                self._earcon_pos += n
                return

        # Generate proximity tone
        t = np.arange(frames) / self.sample_rate

        pitch_hz = signal.audio_pitch_hz
        beat_bpm = signal.audio_beat_bpm
        pan = signal.audio_pan  # -1 to +1

        if pitch_hz <= 0 or beat_bpm <= 0:
            outdata[:] = 0
            return

        # Base tone (sine wave at proximity pitch)
        self._phase += pitch_hz * frames / self.sample_rate
        tone = np.sin(
            2 * np.pi * pitch_hz * t + self._phase * 2 * np.pi
        )

        # Beat envelope (square-wave amplitude modulation at beat rate)
        beat_freq_hz = beat_bpm / 60.0
        self._beat_phase += beat_freq_hz * frames / self.sample_rate
        beat_env = 0.5 * (1 + np.sign(np.sin(
            2 * np.pi * beat_freq_hz * t + self._beat_phase * 2 * np.pi
        )))

        # Combine: tone × beat envelope
        amplitude = tone * beat_env * 0.5  # Master volume

        # For locked-on state: continuous tone (no beat gaps)
        if signal.mode == FeedbackMode.LOCKED:
            amplitude = tone * 0.5

        # Stereo pan: pan = 0 → both, pan = -1 → left only, pan = +1 → right only
        # Constant-power pan law
        left_gain = math.cos((pan + 1) * math.pi / 4)
        right_gain = math.sin((pan + 1) * math.pi / 4)

        outdata[:, 0] = (amplitude * left_gain).astype(np.float32)
        outdata[:, 1] = (amplitude * right_gain).astype(np.float32)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def list_devices() -> None:
        """Print all available audio devices."""
        print(sd.query_devices())
