#!/bin/bash
# build.sh - Build the Multi-Stage Docker image

echo "Building Solder Joint Classifier Docker image with multi-stage build..."

# Build the image with multi-stage optimization
docker build \
    --build-arg HTTP_PROXY="http://proxy-dmz.intel.com:912" \
    --build-arg HTTPS_PROXY="http://proxy-dmz.intel.com:912" \
    --network=host \
    --target=runtime \
    -t solder-joint-classifier:latest \
    .

if [ $? -eq 0 ]; then
    echo "✅ Build complete!"
    
    # Show image size
    echo ""
    echo "Image size:"
    docker images solder-joint-classifier:latest --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
    
    echo ""
    echo "To run the application:"
    echo "   ./run_classifier.sh"
else
    echo "❌ Build failed!"
    exit 1
fi