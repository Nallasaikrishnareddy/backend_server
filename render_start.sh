#!/bin/bash
# render_start.sh - Optimized for Render Free Tier

# Set strict memory and CPU limits
export PYTHONUNBUFFERED=1
export PORT=${PORT:-8000}
export CUDA_VISIBLE_DEVICES=-1
export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Reduce memory usage
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_VLOG_LEVEL=3


# Start with minimal resources
exec gunicorn main:app \
  -w 1 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:$PORT \
  --timeout 600 \
  --max-requests 100 \
  --max-requests-jitter 10 \
  --preload \
  --worker-tmp-dir /dev/shm