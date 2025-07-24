#!/bin/bash
# build.sh - Build the Docker image

echo "Building Solder Joint Classifier Docker image..."

# Build the image
docker build \
    --build-arg HTTP_PROXY="http://proxy-dmz.intel.com:912" \
    --build-arg HTTPS_PROXY="http://proxy-dmz.intel.com:912" \
    --network=host \
    -t solder-joint-classifier:latest .

echo "Build complete!"
echo "To run the application, use: ./run_classifier.sh"
echo "Or use docker-compose: docker-compose up"