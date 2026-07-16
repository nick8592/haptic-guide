"""
HapticGuide — Gradio Interactive Web Interface
===============================================

Provides a browser-based UI for:
  - Uploading images / using webcam for YOLO26 detection
  - Selecting target object class and model variant
  - Visualizing detection results with overlay
  - Computing spatial feedback signals
  - Listing available cameras and audio devices
  - Running benchmarks

Usage:
    python -m src.gradio_app
    # or
    python src/gradio_app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import yaml
from loguru import logger

from .camera import CameraCapture, CameraConfig
from .detector import (
    Detection,
    DetectorConfig,
    DetectionResult,
    InferenceBackend,
    YOLO26Detector,
)
from .feedback_engine import (
    Direction,
    FeedbackConfig,
    FeedbackMode,
    FeedbackSignal,
    SpatialFeedbackEngine,
)
from .visualizer import DetectionVisualizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG_PATH = Path("configs/default.yaml")

# COCO classes commonly used as targets
PRESET_TARGETS = [
    "cell phone", "person", "chair", "bottle",
    "laptop", "cup", "book", "remote",
    "keyboard", "mouse", "backpack", "umbrella",
    "handbag", "wallet", "keys",
]

MODEL_VARIANTS = ["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"]
INFERENCE_BACKENDS = ["onnx", "pytorch", "tensorrt"]

# Module-level state
_detector: YOLO26Detector | None = None
_feedback_engine: SpatialFeedbackEngine | None = None
_visualizer: DetectionVisualizer | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_default_config() -> dict:
    """Load the default YAML configuration."""
    if not CONFIG_PATH.exists():
        logger.warning(f"Config not found: {CONFIG_PATH}, using defaults")
        return {}
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


def build_detector_config(
    variant: str = "yolo26n",
    backend: str = "onnx",
    confidence: float = 0.5,
    iou: float = 0.45,
    max_detections: int = 10,
) -> DetectorConfig:
    """Build a DetectorConfig from UI parameters."""
    return DetectorConfig(
        variant=variant,
        backend=InferenceBackend(backend),
        confidence_threshold=confidence,
        iou_threshold=iou,
        max_detections=max_detections,
    )


def build_feedback_config(raw: dict | None = None) -> FeedbackConfig:
    """Build a FeedbackConfig from raw YAML or defaults."""
    if raw and "feedback" in raw:
        return FeedbackConfig(**raw["feedback"])
    return FeedbackConfig()


def ensure_detector(config: DetectorConfig) -> YOLO26Detector:
    """Lazy-load the YOLO26 detector (reuse if same variant+backend)."""
    global _detector
    if _detector is None:
        _detector = YOLO26Detector(config)
        _detector.load()
    elif (
        _detector.config.variant != config.variant
        or _detector.config.backend != config.backend
    ):
        _detector.unload()
        _detector = YOLO26Detector(config)
        _detector.load()
    return _detector


def draw_detection_overlay(
    frame: np.ndarray,
    result: DetectionResult,
    signal: FeedbackSignal,
    target_class: str,
    fps: float = 0.0,
) -> np.ndarray:
    """Draw detection boxes, proximity bar, and mode on the frame.

    Returns an RGB image suitable for Gradio display.
    """
    vis = DetectionVisualizer(target_class)
    overlay = vis.render(frame, result, signal, fps)
    # Convert BGR → RGB for Gradio
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay_rgb


def format_signal(signal: FeedbackSignal) -> str:
    """Format a FeedbackSignal as a human-readable status string."""
    if signal.mode == FeedbackMode.IDLE:
        return (
            f"Mode: **IDLE** — No target visible\n\n"
            f"Vibration: Off | Audio: Off"
        )
    lines = [
        f"Mode: **{signal.mode.value.upper()}**",
        f"Proximity: `{signal.proximity:.2f}` | Direction: **{signal.direction.value}**",
        f"Vibration: `{signal.vibration_freq_hz:.0f} Hz` at `{signal.vibration_intensity:.0%}`",
        f"Audio Pitch: `{signal.audio_pitch_hz:.0f} Hz` | Beat: `{signal.audio_beat_bpm:.0f} BPM`",
        f"Stereo Pan: `{signal.audio_pan:+.2f}` ({'left' if signal.audio_pan < -0.1 else 'right' if signal.audio_pan > 0.1 else 'center'})",
    ]
    if signal.audio_locked:
        lines.append("**EARCON** — Target locked!")
    return "\n".join(lines)


def format_detections(result: DetectionResult, target_class: str) -> str:
    """Format DetectionResult as a readable list."""
    if not result.detections:
        return "No objects detected."

    lines = [f"**{len(result.detections)}** objects detected ({result.inference_ms:.1f} ms):"]
    for i, det in enumerate(result.detections):
        is_target = det.class_name.lower() == target_class.lower()
        marker = "TARGET" if is_target else ""
        lines.append(
            f"  {i+1}. `{det.class_name}` — conf: `{det.confidence:.2f}` "
            f"bbox: `[{det.bbox[0]:.0f}, {det.bbox[1]:.0f}, {det.bbox[2]:.0f}, {det.bbox[3]:.0f}]` "
            f"{marker}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core Processing Functions
# ---------------------------------------------------------------------------

def detect_on_image(
    image: np.ndarray,
    target_class: str,
    model_variant: str,
    backend: str,
    confidence: float,
    iou_threshold: float,
    max_detections: int,
) -> tuple[np.ndarray | None, str, str]:
    """Run YOLO26 detection on a single uploaded image.

    Returns:
        (annotated_image_rgb, detections_text, signal_text)
    """
    if image is None:
        return None, "No image provided.", ""

    config = build_detector_config(
        variant=model_variant,
        backend=backend,
        confidence=confidence,
        iou=iou_threshold,
        max_detections=max_detections,
    )

    try:
        detector = ensure_detector(config)
    except Exception as e:
        logger.error(f"Failed to load detector: {e}")
        return None, f"Error loading model: {e}", ""

    global _feedback_engine
    if _feedback_engine is None:
        raw = load_default_config()
        _feedback_engine = SpatialFeedbackEngine(build_feedback_config(raw))

    # Convert RGB → BGR for OpenCV processing
    frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        result = detector.detect(frame_bgr)
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return None, f"Detection error: {e}", ""

    signal = _feedback_engine.compute(result, target_class)

    # Draw overlay
    overlay_rgb = draw_detection_overlay(frame_bgr, result, signal, target_class)

    det_text = format_detections(result, target_class)
    sig_text = format_signal(signal)

    return overlay_rgb, det_text, sig_text


def detect_on_webcam(
    frame: np.ndarray,
    target_class: str,
    model_variant: str,
    backend: str,
    confidence: float,
) -> np.ndarray | None:
    """Process a single webcam frame for real-time detection.

    Used as a Gradio streaming callback.
    """
    if frame is None:
        return None

    config = build_detector_config(
        variant=model_variant,
        backend=backend,
        confidence=confidence,
    )

    try:
        detector = ensure_detector(config)
    except Exception as e:
        logger.error(f"Detector error: {e}")
        return frame

    global _feedback_engine
    if _feedback_engine is None:
        raw = load_default_config()
        _feedback_engine = SpatialFeedbackEngine(build_feedback_config(raw))

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    try:
        result = detector.detect(frame_bgr)
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        return frame

    signal = _feedback_engine.compute(result, target_class)
    overlay_rgb = draw_detection_overlay(frame_bgr, result, signal, target_class)

    return overlay_rgb


def list_cameras() -> str:
    """List available camera devices."""
    try:
        devices = CameraCapture.list_devices()
        if not devices:
            return "No cameras found."
        lines = []
        for d in devices:
            lines.append(
                f"/dev/video{d['index']}: {d['resolution']} @ {d['fps']} FPS"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing cameras: {e}"


def list_audio_devices() -> str:
    """List available audio devices."""
    try:
        # Capture stdout from AudioEngine.list_devices()
        import io
        from contextlib import redirect_stdout

        from .audio_engine import AudioEngine

        buf = io.StringIO()
        with redirect_stdout(buf):
            AudioEngine.list_devices()
        output = buf.getvalue()
        return output.strip() if output.strip() else "No audio devices found."
    except Exception as e:
        return f"Error listing audio devices: {e}"


def run_benchmark(
    model_variant: str,
    backend: str,
    iterations: int,
) -> str:
    """Run a quick inference benchmark."""
    config = build_detector_config(variant=model_variant, backend=backend)

    try:
        detector = YOLO26Detector(config)
        detector.load()
    except Exception as e:
        return f"Error loading model: {e}"

    # Synthetic 640x640 frame
    synthetic = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Warmup
    for _ in range(3):
        detector.detect(synthetic)

    # Benchmark
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        detector.detect(synthetic)
        times.append((time.perf_counter() - t0) * 1000)

    detector.unload()

    times_arr = np.array(times)
    lines = [
        f"**Benchmark: {model_variant} ({backend})** — {iterations} iterations",
        f"Mean: **{times_arr.mean():.2f} ms** | Median: **{np.median(times_arr):.2f} ms**",
        f"P95: **{np.percentile(times_arr, 95):.2f} ms** | P99: **{np.percentile(times_arr, 99):.2f} ms**",
        f"Min: {times_arr.min():.2f} ms | Max: {times_arr.max():.2f} ms",
        f"FPS (inference-only): **{1000.0 / times_arr.mean():.0f}**",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    """Build the Gradio Blocks interface."""

    with gr.Blocks(
        title="HapticGuide — Real-Time Spatial Finder",
    ) as demo:

        gr.Markdown(
            """
            # HapticGuide — Real-Time Spatial Finder
            Find objects in real-time using a camera, local AI, and haptic/audio feedback.
            **Zero cloud dependency.** All inference runs locally on GPU/CPU.
            """
        )

        # ----------------------------------------------------------------
        # Tab 1: Image Detection
        # ----------------------------------------------------------------
        with gr.Tab("Image Detection"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(
                        label="Upload Image",
                        type="numpy",
                        sources=["upload", "clipboard"],
                    )
                    with gr.Row():
                        target_input = gr.Dropdown(
                            choices=PRESET_TARGETS,
                            value="cell phone",
                            label="Target Object",
                            allow_custom_value=True,
                            scale=3,
                        )
                        variant_input = gr.Dropdown(
                            choices=MODEL_VARIANTS,
                            value="yolo26n",
                            label="Model Variant",
                            scale=1,
                        )
                    with gr.Row():
                        backend_input = gr.Dropdown(
                            choices=INFERENCE_BACKENDS,
                            value="onnx",
                            label="Inference Backend",
                            scale=1,
                        )
                        confidence_input = gr.Slider(
                            minimum=0.1,
                            maximum=0.99,
                            value=0.5,
                            step=0.05,
                            label="Confidence Threshold",
                            scale=2,
                        )
                    with gr.Row():
                        iou_input = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.45,
                            step=0.05,
                            label="IoU Threshold",
                            scale=1,
                        )
                        max_det_input = gr.Slider(
                            minimum=1,
                            maximum=50,
                            value=10,
                            step=1,
                            label="Max Detections",
                            scale=1,
                        )
                    detect_btn = gr.Button("Detect", variant="primary")

                with gr.Column(scale=1):
                    output_image = gr.Image(label="Detection Result", type="numpy")
                    detection_text = gr.Markdown(label="Detections")
                    signal_text = gr.Markdown(label="Feedback Signal")

            detect_btn.click(
                fn=detect_on_image,
                inputs=[
                    input_image,
                    target_input,
                    variant_input,
                    backend_input,
                    confidence_input,
                    iou_input,
                    max_det_input,
                ],
                outputs=[output_image, detection_text, signal_text],
            )

        # ----------------------------------------------------------------
        # Tab 2: Webcam (Streaming)
        # ----------------------------------------------------------------
        with gr.Tab("Live Webcam"):
            with gr.Row():
                with gr.Column(scale=1):
                    webcam_input = gr.Image(
                        sources=["webcam"],
                        label="Webcam Feed",
                        type="numpy",
                        streaming=True,
                    )
                    with gr.Row():
                        wc_target = gr.Dropdown(
                            choices=PRESET_TARGETS,
                            value="cell phone",
                            label="Target Object",
                            allow_custom_value=True,
                            scale=3,
                        )
                        wc_variant = gr.Dropdown(
                            choices=MODEL_VARIANTS,
                            value="yolo26n",
                            label="Model Variant",
                            scale=1,
                        )
                    with gr.Row():
                        wc_backend = gr.Dropdown(
                            choices=INFERENCE_BACKENDS,
                            value="onnx",
                            label="Inference Backend",
                            scale=1,
                        )
                        wc_confidence = gr.Slider(
                            minimum=0.1,
                            maximum=0.99,
                            value=0.5,
                            step=0.05,
                            label="Confidence Threshold",
                            scale=2,
                        )

                with gr.Column(scale=1):
                    webcam_output = gr.Image(label="Detection Overlay", type="numpy")

            webcam_input.stream(
                fn=detect_on_webcam,
                inputs=[webcam_input, wc_target, wc_variant, wc_backend, wc_confidence],
                outputs=webcam_output,
            )

        # ----------------------------------------------------------------
        # Tab 3: Devices
        # ----------------------------------------------------------------
        with gr.Tab("Devices"):
            with gr.Row():
                with gr.Column():
                    cam_btn = gr.Button("List Cameras", variant="secondary")
                    cam_output = gr.Textbox(label="Camera Devices", lines=8)
                    cam_btn.click(fn=list_cameras, outputs=cam_output)

                with gr.Column():
                    audio_btn = gr.Button("List Audio Devices", variant="secondary")
                    audio_output = gr.Textbox(label="Audio Devices", lines=8)
                    audio_btn.click(fn=list_audio_devices, outputs=audio_output)

        # ----------------------------------------------------------------
        # Tab 4: Benchmark
        # ----------------------------------------------------------------
        with gr.Tab("Benchmark"):
            with gr.Row():
                with gr.Column(scale=1):
                    bench_variant = gr.Dropdown(
                        choices=MODEL_VARIANTS,
                        value="yolo26n",
                        label="Model Variant",
                    )
                    bench_backend = gr.Dropdown(
                        choices=INFERENCE_BACKENDS,
                        value="onnx",
                        label="Inference Backend",
                    )
                    bench_iterations = gr.Slider(
                        minimum=10,
                        maximum=500,
                        value=100,
                        step=10,
                        label="Iterations",
                    )
                    bench_btn = gr.Button("Run Benchmark", variant="primary")

                with gr.Column(scale=1):
                    bench_output = gr.Markdown(label="Benchmark Results")

            bench_btn.click(
                fn=run_benchmark,
                inputs=[bench_variant, bench_backend, bench_iterations],
                outputs=bench_output,
            )

        # ----------------------------------------------------------------
        # Tab 5: Config
        # ----------------------------------------------------------------
        with gr.Tab("Configuration"):
            raw = load_default_config()
            config_yaml = yaml.dump(raw, default_flow_style=False, sort_keys=False) if raw else "No config file found."
            gr.Code(
                value=config_yaml,
                language="yaml",
                label="configs/default.yaml",
                interactive=False,
            )
            gr.Markdown(
                """
                Edit `configs/default.yaml` directly to change runtime settings.
                CLI flags and environment variables override these values at startup.
                """
            )

    return demo


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch the Gradio interface."""
    logger.info("Starting HapticGuide Gradio interface...")
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(
            primary_hue="emerald",
            secondary_hue="amber",
            neutral_hue="stone",
        ),
    )


if __name__ == "__main__":
    main()
