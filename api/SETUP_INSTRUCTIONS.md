# Setup Instructions

## Quick Setup Guide

### Option 1: Using Conda (Recommended)

```bash
# From repository root (IDM-VTON-fork/)
conda env create -f environment.yaml
conda activate idm

# Install API dependencies
pip install -r api/requirements.txt
```

### Option 2: Using pip only (if conda not available)

```bash
# From repository root (IDM-VTON-fork/)
# Install main dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers diffusers accelerate
pip install opencv-python pillow numpy pandas
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html

# Install API dependencies
pip install -r api/requirements.txt
```

### Option 3: Install API dependencies only (if main environment already set up)

```bash
# From repository root (IDM-VTON-fork/)
pip install -r api/requirements.txt
```

## Running the API

### From Repository Root (Recommended)
```bash
# From IDM-VTON-fork/ directory
python -m api.main

# OR using uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### From API Directory
```bash
# From IDM-VTON-fork/api/ directory
python main.py

# OR
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Troubleshooting

### Issue: "conda: command not found"
- **Solution**: Use Option 2 (pip only) or install Anaconda/Miniconda

### Issue: "No module named 'api'"
- **Solution**: Run from repository root using `python -m api.main`

### Issue: "No module named 'diffusers'"
- **Solution**: Install dependencies first using one of the options above

### Issue: "CUDA not available"
- **Solution**: Install CPU version of PyTorch or ensure CUDA is properly installed

## Verify Installation

```bash
# Test structure
cd api
python test_api_structure.py

# Should show:
# ✓ File Structure
# ✓ Python Imports
# ✓ API Imports
# ✓ PyTorch Optimizations
```

