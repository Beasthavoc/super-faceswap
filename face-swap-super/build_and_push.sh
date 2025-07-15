#!/bin/bash

# Build and push script for Face Swap Super
# Usage: ./build_and_push.sh [tag]

set -e

# Configuration
IMAGE_NAME="tjmance/face-swap-super"
TAG=${1:-"latest"}
FULL_TAG="${IMAGE_NAME}:${TAG}"

echo "🚀 Building Face Swap Super Docker image..."
echo "   Image: ${FULL_TAG}"

# Build the image
echo "📦 Building Docker image..."
docker build -t ${FULL_TAG} .

# Tag as latest if not already latest
if [ "$TAG" != "latest" ]; then
    docker tag ${FULL_TAG} ${IMAGE_NAME}:latest
fi

# Push to Docker Hub
echo "🚀 Pushing to Docker Hub..."
docker push ${FULL_TAG}

if [ "$TAG" != "latest" ]; then
    docker push ${IMAGE_NAME}:latest
fi

echo "✅ Successfully built and pushed ${FULL_TAG}"

# Optional: Run the image to test
echo "🧪 Testing the image..."
docker run --rm -p 8000:8000 --name face-swap-super-test ${FULL_TAG} &

# Wait a bit and test health endpoint
sleep 10
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Health check passed!"
else
    echo "❌ Health check failed!"
fi

# Stop the test container
docker stop face-swap-super-test || true

echo "🎉 Build and push completed successfully!"
echo "📝 To run the image:"
echo "   docker run -p 8000:8000 --gpus all ${FULL_TAG}"
echo "   docker-compose up"
