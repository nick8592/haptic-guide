# HapticGuide

Real-Time Spatial Finder for the Visually Impaired — Linux Edition

## What It Does

HapticGuide helps visually impaired people find objects in real-time using a camera, local AI, and haptic/audio feedback — the **"metal detector" metaphor**:

- Pan your camera across the room
- The app vibrates faster and raises pitch as you get closer to the target
- Stereo panning tells you left vs right
- A distinctive earcon sounds when the target is locked
- **Visual display mode** (`--display`) shows real-time detection overlay for sighted debugging/demo

**Zero cloud dependency.** All inference runs locally on GPU/CPU.

## Quick Start

### Prerequisites

- **Docker** + Docker Compose
- **NVIDIA GPU** + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- **USB camera** (`/dev/video0`)
- **X11 display** (only needed for `--display` mode — Wayland users need XWayland)

### 1. Build

```bash
./scripts/haptic-guide.sh build
```

First build takes ~5–10 minutes (downloads CUDA base + Python deps + YOLO26n model).

### 2. Run

```bash
# Default: yolo26n, ONNX GPU, camera 0, target "cell phone"
./scripts/haptic-guide.sh run

# Search for a person
TARGET="person" ./scripts/haptic-guide.sh run

# With visual display window (detection overlay)
DISPLAY_MODE=on ./scripts/haptic-guide.sh run

# More accurate model variant
MODEL_VARIANT=yolo26s ./scripts/haptic-guide.sh run

# Combine all options
MODEL_VARIANT=yolo26s TARGET="person" DISPLAY_MODE=on ./scripts/haptic-guide.sh run
```

### 3. Docker Compose

```bash
# Headless (audio-only)
docker compose run --rm haptic-guide

# With display
docker compose run --rm -e DISPLAY=$DISPLAY haptic-guide --display

# With custom target
docker compose run --rm haptic-guide --target "person" --display
```

Docker Compose handles GPU, camera, audio, and X11 passthrough automatically.

### Display Window

When `--display` is active, the window shows:

- **Green boxes** on target objects with confidence %
- **Gray boxes** on other detections
- **Proximity bar** (left side) — fills as you get closer
- **Mode indicator** (top-right) — SCANNING / TRACKING / LOCKED
- **Crosshair** at frame center
- **FPS counter** (bottom-left)

Press **q** or **ESC** in the window to quit.

> **Note:** First run with display takes ~15s extra to install X11 dependencies. Subsequent runs skip this (libs are cached in the image after rebuild).

## CLI Reference

| Flag | Description |
|------|-------------|
| `--display` | Show real-time detection overlay window (requires X11) |
| `--no-audio` | Disable audio feedback (silent/visual-only mode) |
| `--target "person"` | Set target object class |
| `--model-variant yolo26s` | Choose model variant (n/s/m/l/x) |
| `--backend onnx` | Set inference backend (pytorch / onnx / tensorrt) |
| `--camera 1` | Set camera device index |
| `--config configs/custom.yaml` | Load custom config file |
| `--list-cameras` | List available cameras and exit |
| `--list-audio` | List audio devices and exit |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_VARIANT` | `yolo26n` | Model variant (n/s/m/l/x) |
| `INFERENCE_BACKEND` | `onnx` | pytorch / onnx / tensorrt |
| `CAMERA_DEVICE` | `0` | Camera index |
| `TARGET` | `cell phone` | Object class to find |
| `DISPLAY_MODE` | `off` | `on` = enable X11 overlay window |

## Dev Tools

```bash
./scripts/haptic-guide.sh build       # Build Docker image
./scripts/haptic-guide.sh run         # Run app
./scripts/haptic-guide.sh shell       # Shell into container
./scripts/haptic-guide.sh benchmark   # Benchmark inference latency
./scripts/haptic-guide.sh download    # Download model variant
./scripts/haptic-guide.sh devices     # List cameras & audio devices
```

## Benchmarks (RTX 4060 Laptop 8GB)

**Inference-only:** 100 iterations, synthetic 640×640 frames, ONNX FP32 CUDA.
**E2E:** 200 iterations, MJPG 640×480 real camera, full pipeline (camera → inference → feedback).

### All YOLO26 Variants — ONNX FP32 CUDA

| Variant | Params | Infer (ms) | P95 (ms) | FPS | E2E (ms) | E2E FPS |
|---------|--------|-----------|----------|-----|----------|---------|
| **yolo26n** | 2.6M | **4.68** | 5.24 | **213** | 32.99 | ~30 |
| **yolo26s** | 9.5M | **6.13** | 6.61 | **163** | 32.95 | ~30 |
| **yolo26m** | 20.4M | **12.00** | 13.04 | **83** | 32.97 | ~30 |
| yolo26l | 24.8M | 15.09 | 16.40 | 66 | — | — |
| yolo26x | 55.7M | 29.14 | 29.46 | 34 | — | — |

> **Camera bottleneck:** All variants saturate at the same E2E FPS (~30) because MJPG decode/capture dominates (~27ms). Switching to a faster capture path (V4L2 DmaBuf or lower resolution) would reveal per-model E2E differences. Inference-only is the true GPU throughput metric.

### YOLO26n — Backend Comparison

| Backend | Inference Only | E2E | E2E FPS |
|---------|---------------|-----|---------|
| **ONNX FP32 CUDA** | 4.68ms | 32.99ms | ~30 |
| PyTorch FP16 | ~12ms | ~14ms | ~71 |
| ONNX FP32 CPU | ~43ms | ~48ms | ~21 |

> ONNX GPU is **~2.5× faster** than PyTorch FP16 for inference on this hardware.

With `--display` window active (YOLO26n): **55–60 FPS** (display rendering adds ~5ms overhead).

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│   Camera     │────→│  YOLO26      │────→│  Spatial       │────→│   Audio      │
│   (V4L2)     │     │  Detector    │     │  Feedback      │     │   Engine     │
│              │     │  (ONNX/TRT)  │     │  Engine        │     │   (PipeWire) │
│  ~27ms MJPG  │     │  4-29ms inf  │     │  1ms compute   │     │  3ms output │
└─────────────┘     └──────────────┘     └────────────────┘     └──────────────┘
                           │                     │
                     ┌─────┴─────┐         ┌─────┴──────┐
                     │  Tracker  │         │  Visualizer │
                     │  (IoU)    │         │  (OpenCV)   │
                     │  <1ms     │         │  --display  │
                     └───────────┘         └────────────┘
```

## YOLO26 Model Variants

| Variant | Params | GFLOPs | Use Case |
|---------|--------|--------|----------|
| yolo26n | 2.6M   | 5.4    | Real-time (default, recommended) |
| yolo26s | 9.5M   | 20.7   | Better accuracy, real-time GPU |
| yolo26m | 20.4M  | 68.2   | GPU-only real-time |
| yolo26l | 24.8M  | 86.4   | High accuracy (GPU) |
| yolo26x | 55.7M  | 193.9  | Research / offline |

**YOLO26 key feature**: Native end-to-end (NMS-free) inference via one-to-one head.
Default output shape: `(1, 300, 6)` — no post-processing NMS needed.

### Dual-Mode Strategy (YOLOE-26 + YOLO26)

| Mode | Model | Latency | Purpose |
|------|-------|---------|---------|
| SCAN | YOLOE-26 (open-vocab) | ~100ms GPU | "Find my wallet" via text prompt |
| TRACK | YOLO26n (e2e) | ~6ms GPU | Continuous real-time tracking |

## Feedback Mapping

| Target Proximity | Vibration | Audio Pitch | Audio Beat | Stereo |
|-----------------|----------|-------------|------------|--------|
| Not visible | None | None | None | — |
| Far from center | 10Hz, 20% | 200Hz | 60 BPM | Panned |
| Getting closer | 40Hz, 60% | 500Hz | 180 BPM | Narrowing |
| Near center | 70Hz, 80% | 600Hz | 300 BPM | Centered |
| Locked on | 80Hz, 100% | Earcon | Sustained | Center |

## Testing

```bash
# Via docker with volume mounts (no rebuild needed):
docker run --rm --gpus all \
  -e LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/local/cuda/lib64 \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/tests:/app/tests \
  --entrypoint python3 haptic-guide:latest -m pytest tests/ -v
```

All 12 tests pass (11 unit + 1 integration with GPU inference).

## Docker Run (Manual)

Prefer `docker compose run` or `./scripts/haptic-guide.sh run` — they handle all passthrough flags automatically.

For manual `docker run`, the full command is:

```bash
# With audio + display
docker run --rm --gpus all \
  --device /dev/video0 \
  --device /dev/snd \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e XAUTHORITY=/root/.Xauthority \
  -v ${XAUTHORITY:-/dev/null}:/root/.Xauthority \
  -e QT_X11_NO_MITSHM=1 \
  -e PULSE_SERVER=unix:/run/user/1000/pulse/native \
  -v /run/user/1000/pulse:/run/user/1000/pulse \
  -e LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/local/cuda/lib64 \
  -v $(pwd)/models:/app/models \
  haptic-guide:latest --display --target "person"

# Without audio (visual-only / no sound device available)
docker run --rm --gpus all \
  --device /dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e XAUTHORITY=/root/.Xauthority \
  -v ${XAUTHORITY:-/dev/null}:/root/.Xauthority \
  -e QT_X11_NO_MITSHM=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/local/cuda/lib64 \
  -v $(pwd)/models:/app/models \
  haptic-guide:latest --display --no-audio --target "person"
```

## Project Structure

```
haptic-guide/
├── Dockerfile              # Ubuntu 24.04 + CUDA 12.8 + Qt5 XCB
├── docker-compose.yml      # GPU + camera + audio + X11 passthrough
├── requirements.txt        # Python dependencies
├── configs/
│   └── default.yaml        # Runtime configuration
├── models/                 # Downloaded/exported models (persisted via volume)
├── scripts/
│   ├── entrypoint.sh       # Container entrypoint (auto-installs X11 deps)
│   ├── haptic-guide.sh     # Build, run, shell, benchmark CLI
│   └── dev_tools.py        # Download, export, benchmark tools
├── src/
│   ├── __init__.py
│   ├── main.py             # App entry point + CLI (--display, --no-audio)
│   ├── detector.py         # YOLO26 inference (PyTorch/ONNX/TRT)
│   ├── feedback_engine.py  # Spatial → haptic/audio mapping
│   ├── audio_engine.py     # Real-time spatial audio output
│   ├── camera.py            # Low-latency V4L2 camera capture
│   ├── tracker.py           # IoU centroid object tracker
│   └── visualizer.py        # OpenCV detection overlay (--display mode)
└── tests/
    └── test_core.py         # Unit + integration tests
```

## Key Runtime Details

| Topic | Detail |
|-------|--------|
| **ONNX GPU** | `LD_LIBRARY_PATH` must include `nvidia/cudnn/lib` and `nvidia/cu13/lib` (pip-installed CUDA/cuDNN libs). Docker Compose sets this automatically. |
| **Model persistence** | ONNX export saves to `./models/yolo26n.onnx` (volume-mounted). First run exports (~15s), subsequent runs load instantly. |
| **Display deps** | The `entrypoint.sh` auto-installs Qt5 XCB libs when `DISPLAY` is set. After image rebuild, these are pre-installed and no runtime install is needed. |
| **half_precision** | Default `false` in `configs/default.yaml`. ONNX FP32 is faster on GPU than FP16. Set `true` only for PyTorch backend. |
| **Volume mounts** | `src/`, `configs/`, `scripts/`, `tests/`, and `models/` are all volume-mounted — code changes take effect immediately without rebuild. |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `libcudnn.so.9: not found` | Add to LD_LIBRARY_PATH: `/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib` |
| `libcudart.so.13: not found` | Add to LD_LIBRARY_PATH: `/opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib` |
| `Qt platform plugin "xcb" could not load` | Run with `DISPLAY` set, or `apt install` the Qt5 XCB libs (see `scripts/entrypoint.sh`) |
| `QFontDatabase: Cannot find font directory` | `apt install fonts-dejavu-core` |
| `Cannot open camera` | Check `ls /dev/video*` and pass `--device /dev/videoN` |
| X11 display not working | Run `xhost +local:docker` on host, ensure `DISPLAY` env is set |
| Container crash-looping | `docker-compose.yml` uses `restart: "no"` — check logs with `docker compose logs` |
| ONNX model re-exports every run | Ensure `./models/` volume is mounted so `yolo26n.onnx` persists |
| `unrecognized arguments: --display` | Rebuild image, or ensure `./src/` is volume-mounted (docker-compose does this by default) |
| `PortAudioError: Error querying device` | No audio device in container. Add `--device /dev/snd` + PulseAudio socket mount, or use `--no-audio` to run silently. Audio engine now auto-degrades to silent mode if no device is found. |

## License

Proprietary — Pegatron Internal Use
