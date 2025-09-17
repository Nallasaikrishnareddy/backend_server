#!/bin/bash
# render_start.sh - Start script for Render deployment

# Set environment variables for CPU-only TensorFlow
export PYTHONUNBUFFERED=1
export PORT=${PORT:-8000}
export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=1

# Create necessary directories
mkdir -p /tmp/face_recognition

# Start with higher timeout for initial model loading
exec gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 300 --preload