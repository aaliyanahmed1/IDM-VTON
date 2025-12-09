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
    gcc \
    g++ \
    ninja-build \
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

# Install PyTorch with CUDA support first (specify exact version to avoid large downloads)
RUN pip3 install --no-cache-dir \
    torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install API dependencies
COPY api/requirements.txt /app/api/requirements.txt
RUN pip3 install --no-cache-dir -r api/requirements.txt

# Install NumPy < 2.0 FIRST and pin it (required for PyTorch 2.0.1 and onnxruntime compatibility)
# Use specific version to prevent upgrades
RUN pip3 install --no-cache-dir "numpy==1.24.3" && \
    pip3 install --upgrade --force-reinstall "numpy<2.0" --no-deps || true

# Install additional ML dependencies (split into smaller chunks to save space)
# Constrain numpy in all installs to prevent upgrades
RUN pip3 install --no-cache-dir \
    "numpy<2.0" \
    transformers==4.36.2 \
    diffusers==0.25.0 \
    accelerate==0.25.0 \
    einops==0.7.0 \
    scipy==1.11.1 \
    opencv-python-headless \
    fvcore \
    cloudpickle \
    omegaconf \
    av \
    onnxruntime==1.16.2 \
    huggingface-hub==0.19.4 \
    safetensors \
    matplotlib \
    torchmetrics==1.2.1 \
    tqdm==4.66.1 \
    && pip3 cache purge

# Verify numpy version is < 2.0
RUN python3 -c "import numpy; assert numpy.__version__.startswith('1.'), f'NumPy version {numpy.__version__} is >= 2.0!'; print(f'NumPy version OK: {numpy.__version__}')"

# Install basicsr separately (can be large, constrain numpy)
RUN pip3 install --no-cache-dir "numpy<2.0" basicsr || echo "basicsr install failed, continuing..."

# Install bitsandbytes separately (optional, can fail, constrain numpy)
RUN pip3 install --no-cache-dir "numpy<2.0" bitsandbytes==0.39.0 || echo "bitsandbytes install failed, continuing..."

# Final numpy version check before detectron2
RUN python3 -c "import numpy; print(f'NumPy version before detectron2: {numpy.__version__}'); assert numpy.__version__.startswith('1.'), 'NumPy must be < 2.0'"

# Install detectron2 dependencies (numpy already installed above)
RUN pip3 install --no-cache-dir \
    cython \
    && pip3 cache purge

# Install pycocotools (requires numpy, which is already installed)
RUN pip3 install --no-cache-dir \
    'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' \
    && pip3 cache purge

# Install detectron2 from Facebook Research repo (builds from source, takes 5-10 minutes)
# Using specific tag v0.6 for stability with PyTorch 2.0.1
RUN pip3 install --no-cache-dir \
    'git+https://github.com/facebookresearch/detectron2.git@v0.6' \
    && pip3 cache purge

# Verify detectron2 installation (fail build if not installed correctly)
RUN python3 -c "import detectron2; print('Detectron2 installed successfully'); print(f'Version: {detectron2.__version__}')" || \
    (echo "ERROR: Detectron2 installation failed!" && exit 1)

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

