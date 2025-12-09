# IDM-VTON REST API

Production-ready FastAPI server for integration with **any platform or device**.

## Features

- ✅ **Universal Compatibility** - Works with any HTTP client (Web, Mobile, Desktop, Server-side, IoT)
- ✅ **Base64 Image Encoding** - Returns base64-encoded images for easy display on any platform
- ✅ **CORS Support** - Cross-origin requests enabled for web applications
- ✅ **Performance** - FP16 precision, model compilation, memory-efficient (8-12GB VRAM), fast response (15-25s)
- ✅ **Production-Ready** - Error handling, validation, health checks, logging
- ✅ **Easy Integration** - Simple multipart form-data upload, standard REST API

## Installation

```bash
# Install API dependencies
pip install -r api/requirements.txt

# Ensure main dependencies are installed (from root)
conda env create -f environment.yaml
conda activate idm
```

## Running the Server

```bash
# From the repository root
cd api
python main.py

# Or using uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The API will be available at:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### POST `/api/v1/tryon`

Generate virtual try-on image.

**Request:**
- `person_image` (file): Person image (JPEG/PNG)
- `garment_image` (file): Garment image (JPEG/PNG)
- `garment_description` (optional string): Text description
- `denoise_steps` (int, 10-50, default: 30): Diffusion steps
- `seed` (optional int): Random seed
- `auto_mask` (bool, default: true): Auto-generate mask
- `crop_image` (bool, default: false): Crop to optimal ratio

**Response:**
```json
{
  "success": true,
  "result_image": "base64_encoded_png",
  "mask_image": "base64_encoded_png",
  "message": "Try-on completed successfully"
}
```

### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_device": "NVIDIA GeForce RTX 3090",
  "service_ready": true
}
```

## Platform Integration

The API uses standard HTTP REST endpoints, making it compatible with any platform that supports HTTP requests. Simply upload images via multipart form-data and receive base64-encoded results.

### Web Applications (JavaScript/TypeScript)
```javascript
// Using Fetch API
const formData = new FormData();
formData.append('person_image', personImageFile);
formData.append('garment_image', garmentImageFile);
formData.append('denoise_steps', '20');

const response = await fetch('http://your-server:8000/api/v1/tryon', {
    method: 'POST',
    body: formData
});

const result = await response.json();
const resultImage = `data:image/png;base64,${result.result_image}`;
// Display: <img src={resultImage} />
```

### Python Applications
```python
import requests

files = {
    'person_image': open('person.jpg', 'rb'),
    'garment_image': open('garment.jpg', 'rb')
}
data = {'denoise_steps': 20}

response = requests.post('http://your-server:8000/api/v1/tryon', files=files, data=data)
result = response.json()

# Decode base64 image
import base64
from PIL import Image
import io

image_data = base64.b64decode(result['result_image'])
image = Image.open(io.BytesIO(image_data))
image.show()
```

### Mobile - Android (Kotlin)

```kotlin
val client = OkHttpClient()
val requestBody = MultipartBody.Builder()
    .setType(MultipartBody.FORM)
    .addFormDataPart("person_image", "person.jpg",
        RequestBody.create(MediaType.parse("image/jpeg"), personFile))
    .addFormDataPart("garment_image", "garment.jpg",
        RequestBody.create(MediaType.parse("image/jpeg"), garmentFile))
    .addFormDataPart("denoise_steps", "20")  // Faster processing
    .build()

val request = Request.Builder()
    .url("http://your-server:8000/api/v1/tryon")
    .post(requestBody)
    .build()

val response = client.newCall(request).execute()
val json = JSONObject(response.body()?.string() ?: "")
val resultImageBase64 = json.getString("result_image")
// Decode base64 and display: Base64.decode(resultImageBase64, Base64.DEFAULT)
```

### Mobile - iOS (Swift)

```swift
let url = URL(string: "http://your-server:8000/api/v1/tryon")!
var request = URLRequest(url: url)
request.httpMethod = "POST"

let boundary = UUID().uuidString
request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

var body = Data()
// Add person_image
body.append("--\(boundary)\r\n".data(using: .utf8)!)
body.append("Content-Disposition: form-data; name=\"person_image\"; filename=\"person.jpg\"\r\n".data(using: .utf8)!)
body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
body.append(personImageData)
body.append("\r\n".data(using: .utf8)!)

// Add garment_image
body.append("--\(boundary)\r\n".data(using: .utf8)!)
body.append("Content-Disposition: form-data; name=\"garment_image\"; filename=\"garment.jpg\"\r\n".data(using: .utf8)!)
body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
body.append(garmentImageData)
body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)

request.httpBody = body

let task = URLSession.shared.dataTask(with: request) { data, response, error in
    if let data = data {
        let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        let resultImageBase64 = json?["result_image"] as? String
        // Decode: UIImage(data: Data(base64Encoded: resultImageBase64 ?? "") ?? Data())
    }
}
task.resume()
```

### Desktop Applications (C# / .NET)
```csharp
using System.Net.Http;
using System.Text.Json;

var client = new HttpClient();
var content = new MultipartFormDataContent();
content.Add(new ByteArrayContent(personImageBytes), "person_image", "person.jpg");
content.Add(new ByteArrayContent(garmentImageBytes), "garment_image", "garment.jpg");
content.Add(new StringContent("20"), "denoise_steps");

var response = await client.PostAsync("http://your-server:8000/api/v1/tryon", content);
var json = await response.Content.ReadAsStringAsync();
var result = JsonSerializer.Deserialize<Dictionary<string, object>>(json);

// Decode base64 image
var imageBytes = Convert.FromBase64String(result["result_image"].ToString());
```

### Server-Side Applications (Node.js)
```javascript
const FormData = require('form-data');
const axios = require('axios');
const fs = require('fs');

const form = new FormData();
form.append('person_image', fs.createReadStream('person.jpg'));
form.append('garment_image', fs.createReadStream('garment.jpg'));
form.append('denoise_steps', '20');

const response = await axios.post('http://your-server:8000/api/v1/tryon', form, {
    headers: form.getHeaders()
});

const resultImage = Buffer.from(response.data.result_image, 'base64');
fs.writeFileSync('result.png', resultImage);
```

### Other Platforms
The API works with any platform that supports HTTP requests:
- **Go**: Use `net/http` with multipart form data
- **Rust**: Use `reqwest` or `ureq` crates
- **PHP**: Use `curl` or `Guzzle`
- **Ruby**: Use `Net::HTTP` or `Faraday`
- **Java**: Use `HttpClient` or `OkHttp`
- **C++**: Use `libcurl` or `cpp-httplib`
- **Flutter/Dart**: Use `http` or `dio` packages
- **React Native**: Use `fetch` API
- **IoT Devices**: Any device with HTTP client capabilities

**Performance Tips:** Use `denoise_steps=20` for faster processing (15-20s) or `denoise_steps=30` for higher quality (20-25s). Resize images to max 1024x1024 before upload for best performance.

## Production Deployment

### Docker (Recommended)

```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3.10 python3-pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Run server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

```bash
# Optional: Configure CORS origins
ALLOWED_ORIGINS=https://your-mobile-app.com

# Optional: Configure logging level
LOG_LEVEL=INFO
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid input)
- `500`: Internal server error
- `503`: Service unavailable (models not loaded)
- `507`: Insufficient storage (GPU out of memory)

## Performance

- **Processing Time**: 15-25 seconds (optimized with denoise_steps=20)
- **Memory Usage**: 8-12GB GPU VRAM (with FP16 and optimizations)
- **Image Size**: Automatically resized to 768x1024
- **Optimizations**: Model compilation, FP16 precision, VAE slicing, memory management

## Security Considerations

1. **CORS**: Configure `allow_origins` in production
2. **Rate limiting**: Add rate limiting middleware
3. **Authentication**: Add API key authentication
4. **Input validation**: Already implemented via Pydantic
5. **File size limits**: Configure in FastAPI/Uvicorn

## Troubleshooting

### GPU Out of Memory

- Reduce `denoise_steps`
- Process smaller images
- Use CPU mode (slower but no GPU memory issues)

### Slow Processing

- Ensure GPU is being used (`/health` endpoint)
- Reduce `denoise_steps` for faster results
- Use smaller input images

## License

Same as IDM-VTON project (CC BY-NC-SA 4.0)

