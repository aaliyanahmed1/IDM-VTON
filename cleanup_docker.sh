#!/bin/bash
# Script to clean up Docker to free disk space

echo "=== Docker Cleanup Script ==="
echo ""

echo "1. Stopping all containers..."
docker stop $(docker ps -aq) 2>/dev/null || true

echo ""
echo "2. Removing all stopped containers..."
docker rm $(docker ps -aq) 2>/dev/null || true

echo ""
echo "3. Removing unused images..."
docker image prune -a -f

echo ""
echo "4. Removing unused volumes..."
docker volume prune -f

echo ""
echo "5. Removing build cache..."
docker builder prune -a -f

echo ""
echo "6. Checking disk space..."
df -h

echo ""
echo "=== Cleanup Complete ==="
echo "You can now rebuild the Docker image"

