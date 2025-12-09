# Quick API Testing Guide

## Pre-Flight Checks

Before running the full API, test the structure:

```bash
cd api
python test_api_structure.py
```

## Installation Check

1. **Install API dependencies:**
```bash
pip install -r requirements.txt
```

2. **Ensure main dependencies are installed:**
```bash
# From repository root
conda env create -f environment.yaml
conda activate idm
```

## Testing the API

### 1. Start the Server

```bash
cd api
python main.py
```

The server should start and show:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Initializing IDM-VTON service...
INFO:     Service initialized successfully
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 2. Test Health Endpoint

```bash
# Using curl
curl http://localhost:8000/health

# Or open in browser
# http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_device": "NVIDIA GeForce RTX 3090",
  "service_ready": true
}
```

### 3. Test Try-On Endpoint

```bash
# Using curl with multipart form data
curl -X POST "http://localhost:8000/api/v1/tryon" \
  -F "person_image=@person.jpg" \
  -F "garment_image=@garment.jpg" \
  -F "denoise_steps=20" \
  -o response.json
```

### 4. Test with Python

```python
import requests

files = {
    'person_image': open('person.jpg', 'rb'),
    'garment_image': open('garment.jpg', 'rb')
}
data = {'denoise_steps': 20}

response = requests.post('http://localhost:8000/api/v1/tryon', files=files, data=data)
result = response.json()

if result['success']:
    print("✓ Try-on successful!")
    # Decode base64 image
    import base64
    from PIL import Image
    import io
    
    image_data = base64.b64decode(result['result_image'])
    image = Image.open(io.BytesIO(image_data))
    image.save('result.png')
    print("Result saved to result.png")
else:
    print(f"Error: {result.get('message', 'Unknown error')}")
```

### 5. Test API Documentation

Open in browser:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Common Issues

### Issue: "Service not initialized"
- **Cause**: Models failed to load
- **Fix**: Check GPU availability, ensure models are downloaded from HuggingFace

### Issue: "GPU out of memory"
- **Cause**: Not enough VRAM
- **Fix**: Reduce image sizes, use smaller denoise_steps, or enable more memory optimizations

### Issue: "Module not found"
- **Cause**: Dependencies not installed
- **Fix**: Run `pip install -r requirements.txt` and ensure conda environment is activated

### Issue: "torch.compile failed"
- **Cause**: PyTorch version or compatibility issue
- **Fix**: Service will continue without compilation (slightly slower but functional)

## Performance Testing

Test processing time:
```python
import time
import requests

start = time.time()
response = requests.post('http://localhost:8000/api/v1/tryon', files=files, data=data)
elapsed = time.time() - start

print(f"Processing time: {elapsed:.2f} seconds")
print(f"Status: {response.status_code}")
```

Expected: 15-25 seconds with optimizations enabled.

## Load Testing

For basic load testing:
```bash
# Install Apache Bench (ab) or use Python
pip install locust

# Create locustfile.py
# Run: locust -f locustfile.py
```

## Next Steps

Once basic tests pass:
1. Test with actual mobile app integration
2. Monitor GPU memory usage
3. Test error handling with invalid inputs
4. Test concurrent requests (limited to 1-2 for GPU)

