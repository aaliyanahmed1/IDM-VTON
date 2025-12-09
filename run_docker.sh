#!/bin/bash
# Quick script to build and run IDM-VTON API in Docker on Lightning AI

set -e

echo "=== IDM-VTON Docker Setup ==="
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Check GPU support
echo "Checking GPU support..."
if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "✅ GPU support available"
else
    echo "⚠️  GPU support may not be available, but continuing..."
fi

echo ""
echo "Building Docker image (this may take 10-15 minutes)..."
docker build -t idm-vton-api:latest .

echo ""
echo "Stopping old container if exists..."
docker stop idm-vton-api 2>/dev/null || true
docker rm idm-vton-api 2>/dev/null || true

echo ""
echo "Starting new container with GPU..."
docker run -d \
  --name idm-vton-api \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/ckpt:/app/ckpt \
  -v $(pwd)/results:/app/results \
  --restart unless-stopped \
  idm-vton-api:latest

echo ""
echo "Waiting for server to initialize (30 seconds)..."
sleep 30

echo ""
echo "=== Container Status ==="
docker ps | grep idm-vton-api || echo "Container not running!"

echo ""
echo "=== Recent Logs ==="
docker logs --tail 30 idm-vton-api

echo ""
echo "=== Testing Health Endpoint ==="
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Server is responding!"
    curl -s http://localhost:8000/health | head -3
else
    echo "⚠️  Server may still be starting. Check logs with: docker logs -f idm-vton-api"
fi

echo ""
echo "=== Access URLs ==="
echo "1. Lightning AI Proxy: https://lightning.ai/f2022376124/vision-model/studios/scratch-studio/web-ui?port=8000"
echo "2. Local: http://localhost:8000"
echo "3. API Docs: http://localhost:8000/docs"
echo ""
echo "To view logs: docker logs -f idm-vton-api"
echo "To stop: docker stop idm-vton-api"
echo "To restart: docker start idm-vton-api"

