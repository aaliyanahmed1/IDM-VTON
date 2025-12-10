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

# Install NumPy < 2.0 FIRST (before PyTorch to prevent it from pulling NumPy 2.x)
RUN pip3 install --no-cache-dir "numpy==1.24.3"

# Install PyTorch with CUDA support (constrain numpy to prevent upgrade)
RUN pip3 install --no-cache-dir \
    "numpy<2.0" \
    torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Verify numpy stayed < 2.0 after PyTorch install
RUN python3 -c "import numpy; assert numpy.__version__.startswith('1.'), f'NumPy upgraded to {numpy.__version__} after PyTorch!'; print(f'NumPy OK after PyTorch: {numpy.__version__}')"

# Install Pillow 9.5.0 FIRST (before other deps, detectron2 v0.6 needs it)
# Pillow 10.0+ removed Image.LINEAR which detectron2 v0.6 uses
RUN pip3 install --no-cache-dir "pillow==9.5.0"

# Install API dependencies (constrain numpy and ensure Pillow stays pinned)
COPY api/requirements.txt /app/api/requirements.txt
RUN pip3 install --no-cache-dir "numpy<2.0" -r api/requirements.txt && \
    pip3 install --force-reinstall --no-deps "pillow==9.5.0" && \
    python3 -c "import PIL; print(f'Pillow version: {PIL.__version__}')"

# Install additional ML dependencies (split into smaller chunks to save space)
# Constrain numpy and ensure Pillow stays pinned
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
    && pip3 install --force-reinstall --no-deps "pillow==9.5.0" && \
    pip3 cache purge

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

# Install pycocotools (use --no-build-isolation so numpy is available in build env)
RUN pip3 install --no-cache-dir --no-build-isolation \
    'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' \
    && pip3 cache purge

# Install detectron2 from Facebook Research repo (builds from source, takes 5-10 minutes)
# Using specific tag v0.6 for stability with PyTorch 2.0.1
# Use --no-build-isolation so torch and numpy are available in build environment
RUN pip3 install --no-cache-dir --no-build-isolation \
    'git+https://github.com/facebookresearch/detectron2.git@v0.6' \
    && pip3 cache purge

# Ensure numpy stays < 2.0 after detectron2 install (it might upgrade it)
RUN python3 -c "import numpy; print(f'NumPy after detectron2: {numpy.__version__}')" && \
    (python3 -c "import numpy; assert numpy.__version__.startswith('1.')" || \
     (pip3 install --force-reinstall --no-deps "numpy==1.24.3" && \
      python3 -c "import numpy; print(f'NumPy reinstalled: {numpy.__version__}')"))

# Verify detectron2 installation (fail build if not installed correctly)
RUN python3 -c "import detectron2; print('Detectron2 installed successfully'); print(f'Version: {detectron2.__version__}')" || \
    (echo "ERROR: Detectron2 installation failed!" && exit 1)

# Patch detectron2 to fix Image.LINEAR compatibility issue
# Replace Image.LINEAR with Image.BILINEAR in detectron2's transform.py
RUN python3 -c "import detectron2; import os; transform_file = os.path.join(os.path.dirname(detectron2.__file__), 'data/transforms/transform.py'); \
    with open(transform_file, 'r') as f: content = f.read(); \
    content = content.replace('Image.LINEAR', 'Image.BILINEAR'); \
    with open(transform_file, 'w') as f: f.write(content); \
    print('Patched detectron2: Image.LINEAR -> Image.BILINEAR')" || \
    echo "Warning: Could not patch detectron2, will try Pillow 9.5.0 compatibility"

# Ensure Pillow is still 9.5.0 after all installs
RUN pip3 install --force-reinstall --no-deps "pillow==9.5.0" && \
    python3 -c "import PIL; assert PIL.__version__ == '9.5.0', f'Pillow version is {PIL.__version__}, expected 9.5.0'; print(f'✓ Pillow pinned to: {PIL.__version__}')"

# Final verification: Check both numpy and detectron2 versions
# Force numpy < 2.0 if it got upgraded
RUN python3 -c "import numpy; print(f'NumPy before final check: {numpy.__version__}')" && \
    (python3 -c "import numpy; assert numpy.__version__.startswith('1.')" || \
     (pip3 install --force-reinstall --no-deps "numpy==1.24.3" && \
      python3 -c "import numpy; print(f'NumPy reinstalled: {numpy.__version__}')")) && \
    python3 -c "import numpy; import detectron2; assert numpy.__version__.startswith('1.'), f'NumPy {numpy.__version__} is incompatible!'; print(f'✓ NumPy: {numpy.__version__}'); print(f'✓ Detectron2: {detectron2.__version__}')"

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

