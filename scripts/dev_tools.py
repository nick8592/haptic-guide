#!/usr/bin/env python3
# =============================================================================
# HapticGuide — Dev Utilities
# =============================================================================

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="HapticGuide dev utilities")
    sub = parser.add_subparsers(dest="command")

    # --- download-model ---
    dl = sub.add_parser("download-model", help="Download YOLO26 model")
    dl.add_argument("variant", choices=["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"])
    dl.add_argument("--format", default="pt", choices=["pt", "onnx", "engine"])

    # --- export-onnx ---
    ex = sub.add_parser("export-onnx", help="Export model to ONNX")
    ex.add_argument("variant", choices=["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"])
    ex.add_argument("--output", default=None, help="Output directory")
    ex.add_argument("--half", action="store_true", help="FP16 (half precision)")

    # --- export-tensorrt ---
    et = sub.add_parser("export-tensorrt", help="Export model to TensorRT engine")
    et.add_argument("variant", choices=["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"])
    et.add_argument("--half", action="store_true", default=True, help="FP16 (default: True)")

    # --- benchmark ---
    bm = sub.add_parser("benchmark", help="Run inference benchmark")
    bm.add_argument("variant", choices=["yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"])
    bm.add_argument("--backend", default="pytorch", choices=["pytorch", "onnx", "tensorrt"])
    bm.add_argument("--iterations", type=int, default=100)
    bm.add_argument("--half", action="store_true")

    # --- list-devices ---
    sub.add_parser("list-devices", help="List cameras and audio devices")

    args = parser.parse_args()

    if args.command == "download-model":
        _download_model(args.variant, args.format)
    elif args.command == "export-onnx":
        _export_onnx(args.variant, args.output, args.half)
    elif args.command == "export-tensorrt":
        _export_tensorrt(args.variant, args.half)
    elif args.command == "benchmark":
        _benchmark(args.variant, args.backend, args.iterations, args.half)
    elif args.command == "list-devices":
        _list_devices()
    else:
        parser.print_help()


def _download_model(variant: str, fmt: str) -> None:
    """Download a YOLO26 model."""
    from ultralytics import YOLO
    model = YOLO(f"{variant}.pt")
    print(f"Downloaded: {variant}.pt")

    if fmt == "onnx":
        path = model.export(format="onnx", simplify=True)
        print(f"Exported ONNX: {path}")
    elif fmt == "engine":
        path = model.export(format="engine", device=0)
        print(f"Exported TensorRT: {path}")


def _export_onnx(variant: str, output: str | None, half: bool) -> None:
    """Export model to ONNX format."""
    from ultralytics import YOLO
    model = YOLO(f"{variant}.pt")
    path = model.export(
        format="onnx",
        simplify=True,
        half=half,
        opset=17,
    )
    print(f"Exported ONNX: {path}")

    if output:
        import shutil
        dest = Path(output) / Path(path).name
        shutil.copy2(path, dest)
        print(f"Copied to: {dest}")


def _export_tensorrt(variant: str, half: bool) -> None:
    """Export model to TensorRT engine."""
    from ultralytics import YOLO
    model = YOLO(f"{variant}.pt")
    path = model.export(format="engine", half=half, device=0)
    print(f"Exported TensorRT engine: {path}")


def _benchmark(variant: str, backend: str, iterations: int, half: bool) -> None:
    """Run inference latency benchmark."""
    import numpy as np
    import time

    from ultralytics import YOLO

    model = YOLO(f"{variant}.pt")

    # Warmup
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(10):
        model.predict(dummy, verbose=False, half=half)

    # Benchmark
    latencies = []
    for i in range(iterations):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=False, half=half)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    print(f"\n{'='*50}")
    print(f"YOLO26 Benchmark: {variant} | backend={backend} | half={half}")
    print(f"{'='*50}")
    print(f"  Iterations: {iterations}")
    print(f"  Mean:       {latencies.mean():.2f} ms")
    print(f"  Median:     {np.median(latencies):.2f} ms")
    print(f"  P95:        {np.percentile(latencies, 95):.2f} ms")
    print(f"  P99:        {np.percentile(latencies, 99):.2f} ms")
    print(f"  Min:        {latencies.min():.2f} ms")
    print(f"  Max:        {latencies.max():.2f} ms")
    print(f"  FPS:        {1000/latencies.mean():.1f}")
    print(f"{'='*50}")


def _list_devices() -> None:
    """List available cameras and audio devices."""
    import cv2
    import sounddevice as sd

    print("\nCameras:")
    print("-" * 40)
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"  /dev/video{i}: {w}x{h} @ {fps} FPS")
            cap.release()

    print("\nAudio Devices:")
    print("-" * 40)
    print(sd.query_devices())


if __name__ == "__main__":
    main()
