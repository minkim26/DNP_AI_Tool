#!/bin/bash
# run_classifier.sh - Simple script for users to run the application

echo "Starting Solder Joint Classifier..."

# Allow X11 forwarding
xhost +local:docker

# Create necessary directories
mkdir -p input output

# Run the container with GUI support
docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    --gpus=all \
    --ipc=host \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $(pwd)/input:/app/input \
    -v $(pwd)/output:/app/output \
    -e TZ="America/Los_Angeles" \
    -e HTTP_PROXY="http://proxy-dmz.intel.com:912" \
    -e HTTPS_PROXY="http://proxy-dmz.intel.com:912" \
    --name solder-classifier \
    solder-joint-classifier:latest

echo "Application closed."

# Disable X11 forwarding for security
xhost -local:docker