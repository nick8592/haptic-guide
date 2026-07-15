# =============================================================================
# HapticGuide — YOLO26 Object Detection Pipeline
# Supports: PyTorch, ONNX Runtime (GPU/CPU), TensorRT
# =============================================================================

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Data Types
# ---------------------------------------------------------------------------

class InferenceBackend(str, Enum):
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"


class Detection(BaseModel):
    """Single detected object."""
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float] = Field(description="[x1, y1, x2, y2] in pixel coords")
    center: tuple[float, float] = Field(description="(cx, cy) in pixel coords")

    model_config = {"arbitrary_types_allowed": True}


class DetectionResult(BaseModel):
    """Frame-level detection result."""
    detections: list[Detection]
    frame_shape: tuple[int, int] = Field(description="(height, width)")
    inference_ms: float = Field(description="Inference time in milliseconds")
    timestamp: float = Field(description="Unix timestamp")

    model_config = {"arbitrary_types_allowed": True}

    @property
    def frame_center(self) -> tuple[float, float]:
        h, w = self.frame_shape
        return (w / 2, h / 2)


# ---------------------------------------------------------------------------
# Detector Configuration
# ---------------------------------------------------------------------------

class DetectorConfig(BaseModel):
    variant: str = "yolo26n"
    backend: InferenceBackend = InferenceBackend.ONNX
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 10
    input_size: int = 640
    half_precision: bool = True
    end2end: bool = True  # YOLO26 one-to-one head (NMS-free, default)


class YOLOE26Config(BaseModel):
    """Configuration for YOLOE-26 open-vocabulary detector ('find anything' mode)."""
    variant: str = "yoloe-26l-seg"
    backend: InferenceBackend = InferenceBackend.PYTORCH
    confidence_threshold: float = 0.3
    max_detections: int = 5


# ---------------------------------------------------------------------------
# YOLO26 Detector
# ---------------------------------------------------------------------------

class YOLO26Detector:
    """
    YOLO26-based object detector with multi-backend support.

    Pipeline:  Frame → Preprocess → Inference → NMS → DetectionResult

    Backends:
      - pytorch:  Ultralytics native (slowest, most flexible)
      - onnx:     ONNX Runtime with CUDA EP (fast, portable)
      - tensorrt: TensorRT engine (fastest, Jetson/ NVIDIA only)
    """

    COCO_CLASSES: list[str] = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",
    ]

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self._model: Any = None
        self._class_names: list[str] = self.COCO_CLASSES
        self._warm = False
        # YOLO26 head type: determines post-processing path
        # end2end=True  → one-to-one head: output (N, 300, 6), no NMS needed
        # end2end=False → one-to-many head: output (N, 4+nc, 8400), needs NMS

    # ------------------------------------------------------------------
    # Model Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load model based on configured backend."""
        logger.info(
            f"Loading YOLO26 model: variant={self.config.variant}, "
            f"backend={self.config.backend.value}"
        )

        if self.config.backend == InferenceBackend.PYTORCH:
            self._load_pytorch()
        elif self.config.backend == InferenceBackend.ONNX:
            self._load_onnx()
        elif self.config.backend == InferenceBackend.TENSORRT:
            self._load_tensorrt()
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")

        self._warmup()

    def _load_pytorch(self) -> None:
        """Load via Ultralytics Python API."""
        from ultralytics import YOLO
        model_file = f"{self.config.variant}.pt"
        self._model = YOLO(model_file)
        self._class_names = self._model.names

    def _load_onnx(self) -> None:
        """Load ONNX model with GPU/CPU execution provider."""
        import onnxruntime as ort

        model_path = self._resolve_onnx_path()
        if not model_path.exists():
            logger.info(f"ONNX model not found, exporting from PyTorch...")
            self._export_onnx()
            model_path = self._resolve_onnx_path()

        providers = self._get_onnx_providers()
        logger.info(f"ONNX Runtime providers: {providers}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._model = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers,
        )
        logger.info(f"ONNX model loaded: {model_path}")

    def _load_tensorrt(self) -> None:
        """Load TensorRT engine (pre-built .engine file)."""
        import onnxruntime as ort

        engine_path = self._resolve_engine_path()
        if not engine_path.exists():
            logger.info(f"TensorRT engine not found, building from ONNX...")
            self._build_tensorrt_engine()
            engine_path = self._resolve_engine_path()

        providers = [
            (
                "TensorrtExecutionProvider",
                {
                    "trt_engine_path": str(engine_path),
                    "trt_fp16_enable": self.config.half_precision,
                },
            ),
            "CUDAExecutionProvider",
        ]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._model = ort.InferenceSession(
            str(engine_path) if engine_path.suffix == ".onnx" else "",
            sess_options=sess_options,
            providers=providers,
        )

    # ------------------------------------------------------------------
    # Export Helpers
    # ------------------------------------------------------------------

    def _export_onnx(self) -> None:
        import shutil
        from ultralytics import YOLO
        model = YOLO(f"{self.config.variant}.pt")
        export_path = model.export(
            format="onnx",
            imgsz=self.config.input_size,
            quantize="fp16" if self.config.half_precision else "fp32",
            simplify=True,
            opset=17,
            end2end=self.config.end2end,
        )
        target = Path("/app/models") / f"{self.config.variant}.onnx"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(export_path), str(target))
        logger.info(f"Exported ONNX model to: {target} (end2end={self.config.end2end})")

    def _build_tensorrt_engine(self) -> None:
        from ultralytics import YOLO
        model = YOLO(f"{self.config.variant}.pt")
        export_path = model.export(
            format="engine",
            imgsz=self.config.input_size,
            quantize="fp16" if self.config.half_precision else "fp32",
            end2end=self.config.end2end,
            device=0,
        )
        logger.info(f"Exported TensorRT engine to: {export_path} (end2end={self.config.end2end})")

    def _resolve_onnx_path(self) -> Path:
        model_dir = Path("/app/models")
        name = f"{self.config.variant}.onnx"
        candidates = [
            model_dir / name,
            Path(name).resolve(),
        ]
        for p in candidates:
            if p.exists():
                return p
        return model_dir / name  # Will trigger export

    def _resolve_engine_path(self) -> Path:
        model_dir = Path("/app/models")
        name = f"{self.config.variant}.engine"
        candidates = [
            model_dir / name,
            Path(name).resolve(),
        ]
        for p in candidates:
            if p.exists():
                return p
        return model_dir / name

    def _get_onnx_providers(self) -> list[str | tuple[str, dict]]:
        """Select best available ONNX Runtime execution provider."""
        import onnxruntime as ort
        available = ort.get_available_providers()
        logger.debug(f"Available ONNX providers: {available}")

        if "CUDAExecutionProvider" in available:
            return [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kSameAsRequested",
                        "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                    },
                ),
                "CPUExecutionProvider",
            ]
        return ["CPUExecutionProvider"]

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def _warmup(self) -> None:
        logger.info("Warming up model...")

        if self.config.backend == InferenceBackend.PYTORCH:
            dummy = np.zeros(
                (self.config.input_size, self.config.input_size, 3),
                dtype=np.uint8,
            )
            self._model.predict(
                dummy,
                verbose=False,
                imgsz=self.config.input_size,
                end2end=self.config.end2end,
            )
        elif self.config.backend in (InferenceBackend.ONNX, InferenceBackend.TENSORRT):
            input_type = self._model.get_inputs()[0].type
            dtype = np.float16 if input_type == "tensor(float16)" else np.float32
            dummy = np.zeros(
                (1, 3, self.config.input_size, self.config.input_size),
                dtype=dtype,
            )
            input_name = self._model.get_inputs()[0].name
            for _ in range(3):
                self._model.run(None, {input_name: dummy})

        self._warm = True
        logger.info("Model warmup complete")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run object detection on a single BGR frame.

        Args:
            frame: numpy array of shape (H, W, 3) in BGR format

        Returns:
            DetectionResult with all detected objects
        """
        t_start = time.perf_counter()

        if self.config.backend == InferenceBackend.PYTORCH:
            raw_detections = self._infer_pytorch(frame)
        else:
            raw_detections = self._infer_onnx(frame)

        detections = self._postprocess(raw_detections, frame.shape[:2])

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000

        return DetectionResult(
            detections=detections,
            frame_shape=frame.shape[:2],
            inference_ms=t_elapsed_ms,
            timestamp=time.time(),
        )

    def _infer_pytorch(self, frame: np.ndarray) -> Any:
        results = self._model.predict(
            frame,
            imgsz=self.config.input_size,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            max_det=self.config.max_detections,
            quantize="fp16" if self.config.half_precision else "fp32",
            end2end=self.config.end2end,
            verbose=False,
        )
        return results[0]

    def _infer_onnx(self, frame: np.ndarray) -> list[np.ndarray]:
        """ONNX Runtime inference."""
        blob = self._preprocess(frame)
        input_name = self._model.get_inputs()[0].name
        output_names = [o.name for o in self._model.get_outputs()]

        outputs = self._model.run(output_names, {input_name: blob})
        return outputs

    # ------------------------------------------------------------------
    # Pre/Post Processing
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess BGR frame for ONNX inference:
        Letterbox resize → HWC→CHW → BGR→RGB → normalize → NCHW
        """
        img = frame.copy()
        img = cv2.resize(img, (self.config.input_size, self.config.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC → CHW
        img = np.expand_dims(img, axis=0)  # Add batch dim

        if self.config.backend in (InferenceBackend.ONNX, InferenceBackend.TENSORRT):
            input_type = self._model.get_inputs()[0].type
            if input_type == "tensor(float16)":
                img = img.astype(np.float16)

        return img

    def _postprocess(
        self, raw: Any, frame_shape: tuple[int, int]
    ) -> list[Detection]:
        detections: list[Detection] = []

        if self.config.backend == InferenceBackend.PYTORCH:
            boxes = raw.boxes
            if boxes is None:
                return detections

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                xyxy = boxes.xyxy[i].cpu().numpy()
                cx = float((xyxy[0] + xyxy[2]) / 2)
                cy = float((xyxy[1] + xyxy[3]) / 2)
                class_name = self._class_names.get(cls_id, str(cls_id))

                detections.append(Detection(
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=conf,
                    bbox=xyxy.tolist(),
                    center=(cx, cy),
                ))
        else:
            if isinstance(raw, list) and len(raw) > 0:
                pred = raw[0]
                if self.config.end2end:
                    detections = self._parse_yolo26_e2e_onnx(pred, frame_shape)
                else:
                    detections = self._parse_yolo_onnx_output(pred, frame_shape)

        return detections

    def _parse_yolo26_e2e_onnx(
        self, pred: np.ndarray, frame_shape: tuple[int, int]
    ) -> list[Detection]:
        """
        Parse YOLO26 end-to-end (one-to-one) ONNX output.
        Shape: (1, 300, 6) — each row: [x1, y1, x2, y2, conf, class_id]
        No NMS needed — model already filtered.
        """
        detections: list[Detection] = []

        if len(pred.shape) == 3:
            pred = pred[0]  # (300, 6)

        scale_x = frame_shape[1] / self.config.input_size
        scale_y = frame_shape[0] / self.config.input_size

        for row in pred:
            x1, y1, x2, y2, conf, cls_id = row

            if conf < self.config.confidence_threshold:
                continue

            cls_id = int(cls_id)
            class_name = self._class_names[cls_id] if cls_id < len(self._class_names) else str(cls_id)

            x1_s = float(x1 * scale_x)
            y1_s = float(y1 * scale_y)
            x2_s = float(x2 * scale_x)
            y2_s = float(y2 * scale_y)

            detections.append(Detection(
                class_id=cls_id,
                class_name=class_name,
                confidence=float(conf),
                bbox=[x1_s, y1_s, x2_s, y2_s],
                center=((x1_s + x2_s) / 2, (y1_s + y2_s) / 2),
            ))

            if len(detections) >= self.config.max_detections:
                break

        return detections

    def _parse_yolo_onnx_output(
        self, pred: np.ndarray, frame_shape: tuple[int, int]
    ) -> list[Detection]:
        """
        Parse YOLO ONNX output tensor to Detection list.
        Handles YOLO26 output format: (1, 4+num_classes, num_detections)
        """
        detections: list[Detection] = []

        # YOLO output: (1, 4+nc, detections) → transpose to (1, detections, 4+nc)
        if len(pred.shape) == 3:
            pred = pred[0].T  # (detections, 4+nc)

        # Split into boxes and class scores
        boxes = pred[:, :4]  # cx, cy, w, h (YOLO format)
        class_scores = pred[:, 4:]

        # Get class with max score
        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        # Filter by confidence
        mask = confidences >= self.config.confidence_threshold
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        # Convert YOLO cx,cy,w,h → x1,y1,x2,y2
        scale_x = frame_shape[1] / self.config.input_size
        scale_y = frame_shape[0] / self.config.input_size

        for i in range(len(boxes)):
            cx, cy, w, h = boxes[i]
            x1 = (cx - w / 2) * scale_x
            y1 = (cy - h / 2) * scale_y
            x2 = (cx + w / 2) * scale_x
            y2 = (cy + h / 2) * scale_y

            cls_id = int(class_ids[i])
            class_name = self._class_names[cls_id] if cls_id < len(self._class_names) else str(cls_id)

            detections.append(Detection(
                class_id=cls_id,
                class_name=class_name,
                confidence=float(confidences[i]),
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                center=(float(cx * scale_x), float(cy * scale_y)),
            ))

        # NMS
        detections = self._nms(detections)

        return detections[: self.config.max_detections]

    def _nms(self, detections: list[Detection]) -> list[Detection]:
        """Non-Maximum Suppression on detection list."""
        if not detections:
            return detections

        boxes = np.array([d.bbox for d in detections])
        scores = np.array([d.confidence for d in detections])

        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes.tolist(),
            scores=scores.tolist(),
            score_threshold=self.config.confidence_threshold,
            nms_threshold=self.config.iou_threshold,
        )

        if isinstance(indices, np.ndarray):
            indices = indices.flatten()

        return [detections[i] for i in indices]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("Model unloaded")


# ---------------------------------------------------------------------------
# YOLOE-26 Open-Vocabulary Detector ("Find Anything" mode)
# ---------------------------------------------------------------------------

class YOLOE26Detector:
    """
    YOLOE-26 open-vocabulary detector for "find anything" scan mode.

    Unlike YOLO26 which detects only COCO classes, YOLOE-26 accepts
    text prompts like "red wallet" or "empty seat" and detects them
    without retraining. This enables the dual-mode strategy:

      SCAN MODE  → YOLOE-26 (text prompt "Find my wallet") ~100ms
      TRACK MODE → YOLO26 (fast tracking of detected object) ~8ms

    Usage:
        detector = YOLOE26Detector(YOLOE26Config())
        detector.load()
        detector.set_classes(["wallet", "keys"])
        result = detector.detect(frame)
    """

    def __init__(self, config: YOLOE26Config) -> None:
        self.config = config
        self._model: Any = None
        self._class_names: list[str] = []

    def load(self) -> None:
        from ultralytics import YOLO
        logger.info(f"Loading YOLOE-26 model: {self.config.variant}")
        self._model = YOLO(f"{self.config.variant}.pt")

    def set_classes(self, class_names: list[str]) -> None:
        """
        Set target classes via text prompt (open-vocabulary).
        Only needs to be called once after loading.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        self._class_names = class_names
        self._model.set_classes(class_names)
        logger.info(f"YOLOE-26 target classes set: {class_names}")

    def detect(self, frame: np.ndarray) -> DetectionResult:
        t_start = time.perf_counter()

        results = self._model.predict(
            frame,
            conf=self.config.confidence_threshold,
            max_det=self.config.max_detections,
            verbose=False,
        )

        detections: list[Detection] = []
        boxes = results[0].boxes
        if boxes is not None:
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                xyxy = boxes.xyxy[i].cpu().numpy()
                cx = float((xyxy[0] + xyxy[2]) / 2)
                cy = float((xyxy[1] + xyxy[3]) / 2)
                class_name = self._class_names[cls_id] if cls_id < len(self._class_names) else str(cls_id)

                detections.append(Detection(
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=conf,
                    bbox=xyxy.tolist(),
                    center=(cx, cy),
                ))

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000

        return DetectionResult(
            detections=detections,
            frame_shape=frame.shape[:2],
            inference_ms=t_elapsed_ms,
            timestamp=time.time(),
        )

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("YOLOE-26 model unloaded")
