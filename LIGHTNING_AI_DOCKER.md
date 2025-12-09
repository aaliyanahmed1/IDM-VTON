# Running Docker on Lightning AI - Complete Guide

## Prerequisites

1. **Docker installed on Lightning AI** (usually pre-installed)
2. **NVIDIA Container Toolkit** (for GPU access)
3. **Your code pushed to repository**

## Step-by-Step Instructions

### Step 1: Check Docker and GPU Support

```bash
# Check if Docker is installed
docker --version

# Check if NVIDIA Container Toolkit is available
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

If `nvidia-smi` works, you're ready!

### Step 2: Build Docker Image

```bash
# Navigate to your project directory
cd ~/IDM-VTON

# Pull latest code
git pull origin feature/mobile-api-integration

# Build the Docker image (this will take 10-15 minutes)
docker build -t idm-vton-api:latest .

# Check if image was created
docker images | grep idm-vton-api
```

### Step 3: Run Docker Container with GPU

```bash
# Run the container with GPU access
docker run -d \
  --name idm-vton-api \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/ckpt:/app/ckpt \
  -v $(pwd)/results:/app/results \
  --restart unless-stopped \
  idm-vton-api:latest
```

### Step 4: Check Container Status

```bash
# Check if container is running
docker ps

# Check container logs
docker logs idm-vton-api

# Follow logs in real-time
docker logs -f idm-vton-api
```

**Wait for these lines in logs:**
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 5: Test the API

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test root endpoint (UI)
curl http://localhost:8000/ | head -5
```

### Step 6: Access the UI

Once the server is running, access via Lightning AI:

1. **Lightning AI Proxy URL:**
   ```
   https://lightning.ai/f2022376124/vision-model/studios/scratch-studio/web-ui?port=8000
   ```

2. **Or check your Lightning AI dashboard** for the public URL

## Common Commands

### Stop Container
```bash
docker stop idm-vton-api
```

### Start Container
```bash
docker start idm-vton-api
```

### Restart Container
```bash
docker restart idm-vton-api
```

### Remove Container
```bash
docker stop idm-vton-api
docker rm idm-vton-api
```

### Rebuild and Restart
```bash
# Stop and remove old container
docker stop idm-vton-api
docker rm idm-vton-api

# Rebuild image
docker build -t idm-vton-api:latest .

# Run new container
docker run -d \
  --name idm-vton-api \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/ckpt:/app/ckpt \
  -v $(pwd)/results:/app/results \
  --restart unless-stopped \
  idm-vton-api:latest
```

### View Logs
```bash
# Last 50 lines
docker logs --tail 50 idm-vton-api

# Follow logs
docker logs -f idm-vton-api

# Logs with timestamps
docker logs -t idm-vton-api
```

### Execute Commands in Container
```bash
# Open shell in running container
docker exec -it idm-vton-api /bin/bash

# Run Python command
docker exec idm-vton-api python3 -c "import torch; print(torch.cuda.is_available())"
```

## Troubleshooting

### Issue: "docker: Error response from daemon: could not select device driver"
**Solution:** NVIDIA Container Toolkit not installed
```bash
# Check if nvidia-docker is available
which nvidia-docker
```

### Issue: "No space left on device"
**Solution:** Clean up Docker
```bash
# Remove unused images
docker image prune -a

# Remove unused containers
docker container prune

# Check disk space
df -h
```

### Issue: Container exits immediately
**Solution:** Check logs
```bash
docker logs idm-vton-api
```

### Issue: GPU not detected
**Solution:** Verify GPU access
```bash
# Test GPU in container
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Issue: Port already in use
**Solution:** Stop existing container or use different port
```bash
# Find what's using port 8000
lsof -i :8000

# Or use different port
docker run -d --name idm-vton-api --gpus all -p 8001:8000 idm-vton-api:latest
```

## Quick Start Script

Create a file `run_docker.sh`:

```bash
#!/bin/bash
set -e

echo "Building Docker image..."
docker build -t idm-vton-api:latest .

echo "Stopping old container if exists..."
docker stop idm-vton-api 2>/dev/null || true
docker rm idm-vton-api 2>/dev/null || true

echo "Starting new container..."
docker run -d \
  --name idm-vton-api \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/ckpt:/app/ckpt \
  -v $(pwd)/results:/app/results \
  --restart unless-stopped \
  idm-vton-api:latest

echo "Waiting for server to start..."
sleep 10

echo "Checking logs..."
docker logs --tail 20 idm-vton-api

echo "Testing health endpoint..."
curl -s http://localhost:8000/health | head -3

echo "Done! Access UI at: https://lightning.ai/f2022376124/vision-model/studios/scratch-studio/web-ui?port=8000"
```

Make it executable and run:
```bash
chmod +x run_docker.sh
./run_docker.sh
```

## Using Docker Compose (Alternative)

```bash
# Build and start
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Restart
docker-compose restart
```

## Performance Tips

1. **Use volume mounts** for checkpoints (already configured)
2. **Keep container running** to avoid model reload time
3. **Monitor GPU memory**: `docker stats idm-vton-api`
4. **Check logs regularly** for errors

## Next Steps

Once Docker is running:
1. Test the UI at Lightning AI URL
2. Upload test images
3. Verify try-on generation works
4. Check response times (should be 15-25s)

