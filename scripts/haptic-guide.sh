#!/bin/bash
# =============================================================================
# HapticGuide — Docker Build & Run Script
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Defaults
VARIANT="${MODEL_VARIANT:-yolo26n}"
BACKEND="${INFERENCE_BACKEND:-onnx}"
CAMERA="${CAMERA_DEVICE:-0}"
TARGET="${TARGET:-cell phone}"
DISPLAY_MODE="${DISPLAY_MODE:-off}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

build() {
    info "Building HapticGuide Docker image..."
    docker build \
        --build-arg NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
        -t haptic-guide:latest \
        -f "$PROJECT_DIR/Dockerfile" \
        "$PROJECT_DIR"
    info "Build complete: haptic-guide:latest"
}

run() {
    # Check camera device
    if [ ! -e "/dev/video${CAMERA}" ]; then
        warn "Camera /dev/video${CAMERA} not found. Listing available:"
        ls -la /dev/video* 2>/dev/null || warn "No video devices found"
    fi

    # Check NVIDIA runtime
    if ! docker info 2>/dev/null | grep -q "nvidia"; then
        warn "NVIDIA Container Toolkit not detected. GPU will not be available."
        warn "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
    fi

    info "Starting HapticGuide..."
    info "  Model:    ${VARIANT}"
    info "  Backend:  ${BACKEND}"
    info "  Camera:   /dev/video${CAMERA}"
    info "  Target:   '${TARGET}'"
    info "  Display:  ${DISPLAY_MODE}"

    DISPLAY_FLAGS=""
    EXTRA_ARGS=""
    if [ "${DISPLAY_MODE}" = "on" ]; then
        xhost +local:docker 2>/dev/null || true
        DISPLAY_FLAGS="-e DISPLAY=${DISPLAY:-:0}"
        EXTRA_ARGS="--display"
    fi

    docker compose -f "$PROJECT_DIR/docker-compose.yml" run --rm \
        -e MODEL_VARIANT="$VARIANT" \
        -e INFERENCE_BACKEND="$BACKEND" \
        -e CAMERA_DEVICE="$CAMERA" \
        ${DISPLAY_FLAGS} \
        haptic-guide \
        --target "$TARGET" ${EXTRA_ARGS}
}

shell() {
    info "Opening shell in HapticGuide container..."
    docker compose -f "$PROJECT_DIR/docker-compose.yml" run --rm \
        --entrypoint /bin/bash \
        haptic-guide
}

benchmark() {
    local variant="${1:-yolo26n}"
    local iters="${2:-100}"
    info "Benchmarking ${variant} (${iters} iterations)..."
    docker compose -f "$PROJECT_DIR/docker-compose.yml" run --rm \
        --entrypoint python3 \
        haptic-guide \
        /app/scripts/dev_tools.py benchmark "$variant" --iterations "$iters"
}

download_model() {
    local variant="${1:-yolo26n}"
    info "Downloading ${variant}..."
    docker compose -f "$PROJECT_DIR/docker-compose.yml" run --rm \
        --entrypoint python3 \
        haptic-guide \
        /app/scripts/dev_tools.py download-model "$variant"
}

gradio() {
    info "Starting HapticGuide Gradio web UI..."
    info "  Model:    ${VARIANT}"
    info "  Backend:  ${BACKEND}"
    info "  Camera:   /dev/video${CAMERA}"
    info "  Port:     7860"
    info "  Open:     http://localhost:7860"

    docker compose -f "$PROJECT_DIR/docker-compose.yml" run --rm \
        -e MODEL_VARIANT="$VARIANT" \
        -e INFERENCE_BACKEND="$BACKEND" \
        -e CAMERA_DEVICE="$CAMERA" \
        gradio
}

list_devices() {
    info "Listing available devices..."
    docker compose -f "$PROJECT_DIR/docker-compose.yml" run --rm \
        --entrypoint python3 \
        haptic-guide \
        /app/scripts/dev_tools.py list-devices
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

usage() {
    cat <<EOF
HapticGuide — Real-Time Spatial Finder for the Visually Impaired

Usage: $(basename "$0") <command> [options]

Commands:
  build             Build Docker image
  run               Run HapticGuide (default: yolo26n, camera 0)
  gradio            Launch Gradio web UI on http://localhost:7860
  shell             Open shell in container
  benchmark [VAR]   Benchmark model variant (default: yolo26n)
  download [VAR]    Download model variant (default: yolo26n)
  devices           List cameras and audio devices

Environment:
  MODEL_VARIANT      Model variant (yolo26n/s/m/l/x)  [default: yolo26n]
  INFERENCE_BACKEND  pytorch | onnx | tensorrt         [default: onnx]
  CAMERA_DEVICE      Camera index (0, 1, ...)         [default: 0]
  TARGET             Object class to search for        [default: cell phone]
  DISPLAY_MODE       on | off                          [default: off]

Examples:
  $(basename "$0") build
  $(basename "$0") run
  $(basename "$0") gradio
  MODEL_VARIANT=yolo26s $(basename "$0") run
  DISPLAY_MODE=on $(basename "$0") run
  $(basename "$0") benchmark yolo26n 200
  CAMERA_DEVICE=1 TARGET=person DISPLAY_MODE=on $(basename "$0") run
EOF
}

case "${1:-}" in
    build)    build ;;
    run)      run ;;
    gradio)   gradio ;;
    shell)    shell ;;
    benchmark) benchmark "${2:-yolo26n}" "${3:-100}" ;;
    download) download_model "${2:-yolo26n}" ;;
    devices)  list_devices ;;
    -h|--help|help) usage ;;
    *)        usage; exit 1 ;;
esac
