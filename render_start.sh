#!/bin/bash
# render_start.sh - Start script for Render deployment

# Set environment variables
export PYTHONUNBUFFERED=1
export PORT=${PORT:-8000}

# Create necessary directories
mkdir -p /tmp/face_recognition

# Start the FastAPI application with Gunicorn
exec gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120