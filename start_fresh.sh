#!/bin/bash
# Complete script to start Docker build from scratch

set -e

echo "=== Starting Fresh Docker Build ==="
echo ""

# 1. Stop and remove all containers
echo "1. Stopping and removing all containers..."
docker stop $(docker ps -aq) 2>/dev/null || true
docker rm $(docker ps -aq) 2>/dev/null || true

# 2. Remove all images
echo ""
echo "2. Removing all Docker images..."
docker rmi $(docker images -q) 2>/dev/null || true

# 3. Clean up everything
echo ""
echo "3. Cleaning up Docker system..."
docker system prune -a -f --volumes

# 4. Check disk space
echo ""
echo "4. Checking disk space..."
df -h | head -5

# 5. Pull latest code
echo ""
echo "5. Pulling latest code..."
cd ~/IDM-VTON
git pull origin feature/mobile-api-integration

# 6. Build fresh image
echo ""
echo "6. Building fresh Docker image (this will take 15-25 minutes)..."
docker build -t idm-vton-api:latest .

# 7. Start container
echo ""
echo "7. Starting container..."
docker run -d \
  --name idm-vton-api \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/ckpt:/app/ckpt \
  -v $(pwd)/results:/app/results \
  --restart unless-stopped \
  idm-vton-api:latest

# 8. Wait for startup
echo ""
echo "8. Waiting for server to start (30 seconds)..."
sleep 30

# 9. Check status
echo ""
echo "=== Container Status ==="
docker ps | grep idm-vton-api || echo "Container not running!"

# 10. Show logs
echo ""
echo "=== Recent Logs ==="
docker logs --tail 30 idm-vton-api

# 11. Test health
echo ""
echo "=== Testing Health Endpoint ==="
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Server is responding!"
    curl -s http://localhost:8000/health | head -3
else
    echo "⚠️  Server may still be starting. Check logs: docker logs -f idm-vton-api"
fi

echo ""
echo "=== Access URLs ==="
echo "1. Lightning AI: https://lightning.ai/f2022376124/vision-model/studios/scratch-studio/web-ui?port=8000"
echo "2. Local: http://localhost:8000"
echo "3. API Docs: http://localhost:8000/docs"
echo ""
echo "To view logs: docker logs -f idm-vton-api"
echo "To stop: docker stop idm-vton-api"

