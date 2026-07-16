#!/bin/bash
# HapticGuide — Container entrypoint wrapper
# Installs X11 display dependencies when DISPLAY is set, then runs the app.
# If GRADIO_MODE=on, launches the Gradio web UI instead of the CLI app.

set -e

if [ -n "${DISPLAY:-}" ]; then
    if ! dpkg -s libqt5x11extras5 &>/dev/null; then
        echo "[entrypoint] Installing X11 display dependencies..."
        apt-get update -qq 2>/dev/null
        apt-get install -y --no-install-recommends -qq \
            libqt5x11extras5 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 \
            libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 \
            libxcb-xfixes0 libxcb-xinerama0 libxcb-cursor0 libsm6 libice6 \
            fonts-dejavu-core 2>/dev/null
        echo "[entrypoint] X11 dependencies installed"
    fi
fi

if [ "${GRADIO_MODE:-off}" = "on" ]; then
    echo "[entrypoint] Starting Gradio web UI on port ${GRADIO_PORT:-7860}..."
    exec python3 -m src.gradio_app "$@"
else
    exec python3 -m src.main "$@"
fi
