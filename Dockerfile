# =============================================================================
# HapticGuide — Real-Time Spatial Finder for the Visually Impaired
# Dockerfile: Ubuntu 24.04 + NVIDIA GPU + YOLO26 + ONNX Runtime
# =============================================================================

FROM nvidia/cuda:12.8.2-runtime-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ---------------------------------------------------------------------------
# System packages
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    v4l-utils \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libsndfile1 \
    libportaudio2 \
    libgpiod2 \
    build-essential \
    cmake \
    pkg-config \
    wget \
    curl \
    git \
    espeak-ng \
    libespeak-ng-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libqt5x11extras5 \
    libqt5core5t64 \
    libqt5gui5t64 \
    libqt5widgets5t64 \
    libqt5dbus5t64 \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-cursor0 \
    libsm6 \
    libice6 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python virtual environment
# ---------------------------------------------------------------------------
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
ENV PIP_NO_CACHE_DIR=1

# Install pip into the venv
RUN pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# Python dependencies (installed in venv)
# ---------------------------------------------------------------------------
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# ---------------------------------------------------------------------------
# Pre-download YOLO26 nano model
# Runs at container build so inference is ready on first start.
# ---------------------------------------------------------------------------
RUN python3 -c "from ultralytics import YOLO; model = YOLO('yolo26n.pt'); print(f'YOLO26n loaded: {model.model.info()}')" \
    || echo "WARN: Model download failed — will download on first run"

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
WORKDIR /app

COPY src/ /app/src/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
COPY tests/ /app/tests/
COPY pytest.ini /app/pytest.ini

RUN chmod +x /app/scripts/*.sh 2>/dev/null || true

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
ENV CAMERA_DEVICE=0
ENV MODEL_VARIANT=yolo26n
ENV INFERENCE_BACKEND=onnx
ENV LOG_LEVEL=INFO
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import ultralytics; print('OK')" || exit 1

ENTRYPOINT ["python3", "-m", "src.main"]
CMD ["--config", "/app/configs/default.yaml"]
