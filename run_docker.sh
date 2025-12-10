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

# Stop any existing processes on port 8000
echo ""
echo "Checking for processes on port 8000..."
if lsof -ti:8000 > /dev/null 2>&1; then
    echo "⚠️  Port 8000 is in use. Stopping existing processes..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Stop any existing Python API processes
echo "Stopping any existing API processes..."
pkill -f "api.main" 2>/dev/null || true
pkill -f "python.*main.py" 2>/dev/null || true
sleep 2

# Stop old Docker container if exists
echo ""
echo "Stopping old Docker container if exists..."
docker stop idm-vton-api 2>/dev/null || true
docker rm idm-vton-api 2>/dev/null || true

# Check if we need to build
if docker images | grep -q "idm-vton-api.*latest"; then
    read -p "Docker image exists. Rebuild? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Building Docker image (this may take 10-15 minutes)..."
        docker build -t idm-vton-api:latest .
    else
        echo "Using existing image..."
    fi
else
    echo ""
    echo "Building Docker image (this may take 10-15 minutes)..."
    docker build -t idm-vton-api:latest .
fi

# Create cache directory if it doesn't exist
mkdir -p hf_cache

echo ""
echo "Starting new container with GPU..."
docker run -d \
  --name idm-vton-api \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/ckpt:/app/ckpt \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/hf_cache:/root/.cache/huggingface \
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
    echo "Wait for: 'INFO:     Application startup complete.'"
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
