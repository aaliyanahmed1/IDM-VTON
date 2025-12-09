# Fix Dependency Issues

## Current Issue: huggingface_hub version incompatibility

The error `cannot import name 'cached_download' from 'huggingface_hub'` means you need a compatible version.

## Fix Commands

```powershell
# Install compatible huggingface_hub version
pip install "huggingface_hub<0.20.0"

# OR install specific version that works with diffusers 0.25.0
pip install huggingface_hub==0.19.4
```

## Complete Dependency Fix

```powershell
# Uninstall incompatible versions
pip uninstall huggingface_hub -y

# Install compatible versions
pip install huggingface_hub==0.19.4
pip install diffusers==0.25.0
```

## Verify

After installing, test again:
```powershell
cd api
python test_api_structure.py
```

