# Production Dockerfile for IDM-VTON API with GPU support
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV DEBIAN_FRONTEND=noninteractive

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support first
RUN pip3 install --no-cache-dir \
    torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install API dependencies
COPY api/requirements.txt /app/api/requirements.txt
RUN pip3 install --no-cache-dir -r api/requirements.txt

# Install additional ML dependencies
RUN pip3 install --no-cache-dir \
    transformers==4.36.2 \
    diffusers==0.25.0 \
    accelerate==0.25.0 \
    einops==0.7.0 \
    scipy==1.11.1 \
    opencv-python-headless \
    fvcore \
    cloudpickle \
    omegaconf \
    pycocotools \
    basicsr \
    av \
    onnxruntime==1.16.2 \
    huggingface-hub==0.19.4 \
    safetensors

# Install detectron2 dependencies
RUN pip3 install --no-cache-dir \
    'git+https://github.com/facebookresearch/detectron2.git' || echo "detectron2 install may need manual setup"

# Copy entire application code
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/ckpt /app/results

# Set CUDA environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API server
CMD ["python3", "-m", "api.main"]

