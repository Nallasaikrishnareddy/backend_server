#!/bin/bash

# Install missing system libraries for OpenCV / face recognition
apt-get update
apt-get install -y libgl1 libglib2.0-0

# Optional: install other libraries if needed
# apt-get install -y ffmpeg

# Start the FastAPI app
# Replace main:app with your module and app name
uvicorn main:app --host 0.0.0.0 --port 8000
