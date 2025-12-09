# Deployment Guide for Tesla T4 / Cloud Platform

## Overview

This API is ready for deployment on cloud platforms with NVIDIA Tesla T4 GPUs (or similar).

## Requirements

- **GPU**: NVIDIA Tesla T4 or similar (16GB VRAM recommended)
- **CUDA**: 11.8 compatible
- **Python**: 3.8+
- **Storage**: ~20GB for models

## Quick Deployment

### 1. Install Dependencies

```bash
# Create conda environment (recommended)
conda env create -f environment.yaml
conda activate idm

# OR install with pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.36.2
pip install diffusers==0.25.0
pip install huggingface_hub==0.19.4
pip install accelerate==0.25.0
pip install opencv-python pillow numpy pandas
pip install fastapi uvicorn python-multipart pydantic

# Install detectron2 (for DensePose)
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html

# Install API dependencies
pip install -r api/requirements.txt
```

### 2. Download Models

Models will be automatically downloaded from HuggingFace on first run:
- Base model: `yisol/IDM-VTON`
- Checkpoints are already in `ckpt/` folder

### 3. Run the API

```bash
# From repository root
python -m api.main

# OR using uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Docker Deployment (Recommended for Cloud)

### Dockerfile

```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY api/requirements.txt /app/api/
COPY environment.yaml /app/

# Install Python dependencies
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir transformers==4.36.2 diffusers==0.25.0 huggingface_hub==0.19.4 && \
    pip install --no-cache-dir accelerate==0.25.0 opencv-python pillow numpy pandas && \
    pip install --no-cache-dir fastapi uvicorn python-multipart pydantic && \
    pip install --no-cache-dir detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html && \
    pip install --no-cache-dir -r api/requirements.txt

# Copy application
COPY . /app/

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "-m", "api.main"]
```

### Docker Commands

```bash
# Build
docker build -t idm-vton-api .

# Run with GPU
docker run --gpus all -p 8000:8000 idm-vton-api

# Run with specific GPU
docker run --gpus '"device=0"' -p 8000:8000 idm-vton-api
```

## Cloud Platform Deployment

### AWS EC2 / GCP / Azure

1. **Launch GPU instance** (Tesla T4, V100, or similar)
2. **Install dependencies** (see Quick Deployment above)
3. **Clone repository**
4. **Run API** with proper firewall rules (port 8000)
5. **Use load balancer** if needed for multiple instances

### Environment Variables

```bash
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU
export PORT=8000
export HOST=0.0.0.0
```

## Performance Expectations (Tesla T4)

- **Processing Time**: 15-25 seconds per request
- **Memory Usage**: 8-12GB VRAM
- **Concurrent Requests**: 1-2 max (GPU limited)
- **Throughput**: ~3-4 requests/minute

## Health Check

```bash
curl http://your-server:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_device": "Tesla T4",
  "service_ready": true
}
```

## API Endpoint

```bash
POST http://your-server:8000/api/v1/tryon
Content-Type: multipart/form-data

Parameters:
- person_image: file
- garment_image: file
- denoise_steps: 20 (default, for speed)
```

## Monitoring

- Check GPU usage: `nvidia-smi`
- Check API logs: Server logs to console
- Monitor memory: Watch for OOM errors (507 status code)

## Troubleshooting

### GPU Not Detected
```bash
# Check CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory
- Reduce `denoise_steps` to 15-20
- Process smaller images
- Enable all memory optimizations (already in code)

### Slow Processing
- Ensure GPU is being used (check `/health` endpoint)
- Verify CUDA is properly installed
- Check GPU utilization with `nvidia-smi`

## Security Notes

1. **CORS**: Configure `allow_origins` in production
2. **Rate Limiting**: Add rate limiting middleware
3. **Authentication**: Add API key authentication
4. **HTTPS**: Use reverse proxy (nginx) with SSL

## Next Steps

1. Push to your repository
2. Deploy on cloud platform
3. Test with `/health` endpoint
4. Test with actual try-on requests
5. Monitor performance and adjust as needed

