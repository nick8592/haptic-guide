# =============================================================================
# HapticGuide — Main Application Entry Point
# =============================================================================

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import yaml
from loguru import logger

from .camera import CameraCapture, CameraConfig
from .detector import DetectorConfig, InferenceBackend, YOLO26Detector
from .feedback_engine import FeedbackConfig, SpatialFeedbackEngine
from .audio_engine import AudioEngine
from .visualizer import DetectionVisualizer


# ---------------------------------------------------------------------------
# Configuration Loader
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    if not config_path.exists():
        logger.warning(f"Config not found: {config_path}, using defaults")
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def build_configs(raw: dict) -> tuple[CameraConfig, DetectorConfig, FeedbackConfig, dict]:
    """Parse raw config dict into typed config objects."""
    camera_cfg = CameraConfig(**raw.get("camera", {}))

    model_raw = raw.get("model", {})
    # Convert backend string to enum
    if "backend" in model_raw:
        model_raw["backend"] = InferenceBackend(model_raw["backend"])
    detector_cfg = DetectorConfig(**model_raw)

    feedback_cfg = FeedbackConfig(**raw.get("feedback", {}))

    # Target + voice configs kept as dicts (flexible structure)
    target_cfg = raw.get("target", {})
    voice_cfg = raw.get("voice", {})

    return camera_cfg, detector_cfg, feedback_cfg, target_cfg


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class HapticGuideApp:
    """
    Main application: ties together camera → detector → feedback → audio.

    Architecture:
      Camera Thread ──→ Main Loop ──→ Audio Thread
                         │
                    Detector (GPU/CPU)
                         │
                    Feedback Engine
                         │
                    Audio Output
    """

    def __init__(
        self,
        camera_config: CameraConfig,
        detector_config: DetectorConfig,
        feedback_config: FeedbackConfig,
        target_config: dict,
        voice_config: dict,
        display: bool = False,
        no_audio: bool = False,
    ) -> None:
        self.camera_config = camera_config
        self.detector_config = detector_config
        self.feedback_config = feedback_config
        self.target_config = target_config
        self.voice_config = voice_config
        self.display = display
        self.no_audio = no_audio

        self.camera: CameraCapture | None = None
        self.detector: YOLO26Detector | None = None
        self.feedback_engine: SpatialFeedbackEngine | None = None
        self.audio_engine: AudioEngine | None = None
        self.visualizer: DetectionVisualizer | None = None

        self._running = False
        self._target_class = target_config.get("default", "cell phone")
        self._prev_mode = None
        self._fps_display = 0.0

    def start(self) -> None:
        logger.info("=" * 60)
        logger.info("HapticGuide — Real-Time Spatial Finder")
        logger.info("=" * 60)

        logger.info("[1/5] Initializing camera...")
        self.camera = CameraCapture(self.camera_config)
        self.camera.open()
        self.camera.start_capture()

        logger.info("[2/5] Loading YOLO26 model...")
        self.detector = YOLO26Detector(self.detector_config)
        self.detector.load()

        logger.info("[3/5] Initializing feedback engine...")
        self.feedback_engine = SpatialFeedbackEngine(self.feedback_config)

        logger.info("[4/5] Starting audio engine...")
        if self.no_audio:
            logger.info("Audio: DISABLED (--no-audio flag)")
        else:
            audio_cfg = self.voice_config.get("audio", {})
            self.audio_engine = AudioEngine(
                sample_rate=audio_cfg.get("sample_rate", 44100),
                buffer_size=audio_cfg.get("buffer_size", 512),
            )
            self.audio_engine.start()

        if self.display:
            logger.info("[5/5] Enabling visual display...")
            self.visualizer = DetectionVisualizer(self._target_class)
        else:
            logger.info("[5/5] Visual display: OFF (use --display to enable)")

        logger.info(f"Target: '{self._target_class}'")
        logger.info("System ready — press 'q' or ESC to quit")
        if self.display:
            logger.info("Visual mode: ON — detection overlay window active")

        self._running = True
        self._run_loop()

    def stop(self) -> None:
        logger.info("Shutting down...")
        self._running = False

        if self.visualizer:
            self.visualizer.close()
        if self.audio_engine:
            self.audio_engine.stop()
        if self.camera:
            self.camera.close()
        if self.detector:
            self.detector.unload()

        logger.info("HapticGuide stopped")

    # ------------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        frame_count = 0
        fps_timer = time.perf_counter()
        fps_count = 0
        last_status_time = 0.0

        try:
            while self._running:
                frame = self.camera.get_latest_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue

                result = self.detector.detect(frame)
                signal = self.feedback_engine.compute(result, self._target_class)
                if self.audio_engine:
                    self.audio_engine.update_signal(signal)

                fps_count += 1
                elapsed = time.perf_counter() - fps_timer
                if elapsed >= 1.0:
                    self._fps_display = fps_count / elapsed
                    fps_count = 0
                    fps_timer = time.perf_counter()

                if self.visualizer is not None:
                    vis_frame = self.visualizer.render(
                        frame, result, signal, self._fps_display,
                    )
                    if not self.visualizer.show(vis_frame):
                        logger.info("Display window closed by user")
                        break

                # Periodic status logging (every 2 seconds)
                now = time.time()
                if now - last_status_time >= 2.0:
                    mode_str = signal.mode.value
                    prox_str = f"{signal.proximity:.2f}" if signal.proximity > 0 else "—"
                    dir_str = signal.direction.value if signal.proximity > 0 else "—"
                    inf_str = f"{result.inference_ms:.1f}ms"

                    logger.info(
                        f"[{self._fps_display:.1f} FPS] mode={mode_str} "
                        f"proximity={prox_str} direction={dir_str} "
                        f"inference={inf_str} target='{self._target_class}'"
                    )
                    last_status_time = now

                frame_count += 1

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()

    # ------------------------------------------------------------------
    # Target Selection
    # ------------------------------------------------------------------

    def set_target(self, target_class: str) -> None:
        """Change the target object class."""
        self._target_class = target_class
        logger.info(f"Target changed to: '{target_class}'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HapticGuide — Real-Time Spatial Finder for the Visually Impaired"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/default.yaml"),
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--model-variant", type=str, default=None,
        choices=["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"],
        help="YOLO26 model variant (overrides config)",
    )
    parser.add_argument(
        "--backend", type=str, default=None,
        choices=["pytorch", "onnx", "tensorrt"],
        help="Inference backend (overrides config)",
    )
    parser.add_argument(
        "--camera", type=int, default=None,
        help="Camera device index (overrides config)",
    )
    parser.add_argument(
        "--target", type=str, default=None,
        help="Target object class to search for (overrides config)",
    )
    parser.add_argument(
        "--list-cameras", action="store_true",
        help="List available cameras and exit",
    )
    parser.add_argument(
        "--list-audio", action="store_true",
        help="List available audio devices and exit",
    )
    parser.add_argument(
        "--display", action="store_true",
        help="Show real-time detection overlay window (requires X11/display)",
    )
    parser.add_argument(
        "--no-audio", action="store_true",
        help="Disable audio feedback (silent mode for visual-only testing)",
    )

    args = parser.parse_args()

    # Utility modes
    if args.list_cameras:
        devices = CameraCapture.list_devices()
        for d in devices:
            print(f"  /dev/video{d['index']}: {d['resolution']} @ {d['fps']} FPS")
        sys.exit(0)

    if args.list_audio:
        AudioEngine.list_devices()
        sys.exit(0)

    # Load config
    raw = load_config(args.config)
    camera_cfg, detector_cfg, feedback_cfg, target_cfg = build_configs(raw)

    # CLI overrides
    if args.model_variant:
        detector_cfg.variant = args.model_variant
    if args.backend:
        detector_cfg.backend = InferenceBackend(args.backend)
    if args.camera is not None:
        camera_cfg.device_index = args.camera

    target_override = args.target or target_cfg.get("default", "cell phone")
    voice_cfg = raw.get("voice", {})

    # Configure logging
    log_level = raw.get("logging", {}).get("level", "INFO")
    logger.remove()
    logger.add(sys.stderr, level=log_level, format="{time:HH:mm:ss} | {level:<7} | {message}")

    # Create and start app
    app = HapticGuideApp(
        camera_config=camera_cfg,
        detector_config=detector_cfg,
        feedback_config=feedback_cfg,
        target_config=target_cfg,
        voice_config=voice_cfg,
        display=args.display,
        no_audio=args.no_audio,
    )

    # Handle signals
    def signal_handler(sig, frame):
        app.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    app.start()


if __name__ == "__main__":
    main()
