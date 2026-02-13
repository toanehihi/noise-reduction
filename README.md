# Flask Noise Reduction Service

REST API service để nhận file .wav từ ESP32 và trả về audio đã khử tiếng ồn bằng DTLN model.

## 🏗️ Architecture

Service được thiết kế theo **layered architecture** để dễ dàng tái sử dụng:

```
noise-reduction-service/
├── app.py                    # Flask app entry point (Factory pattern)
├── config.py                 # Configuration management
├── api/                      # API Layer (HTTP endpoints)
│   ├── __init__.py
│   └── routes.py            # Flask routes/blueprints
├── services/                 # Service Layer (Business logic)
│   ├── __init__.py
│   └── noise_reduction.py   # DTLN service (standalone)
├── models/                   # Model files
│   └── DTLN_vivos_best.h5   # Trained DTLN model
├── requirements.txt
└── README.md
```

**Key Benefits:**
- ✅ **Service layer** có thể import và sử dụng độc lập (không cần Flask)
- ✅ **API layer** chỉ handle HTTP requests, delegate logic cho service
- ✅ Dễ test, dễ maintain, dễ scale
- ✅ Có thể reuse service trong bất kỳ application nào

## 🚀 Features

- ✅ RESTful API endpoint cho audio denoising
- ✅ **Standalone service** - có thể import và dùng trực tiếp
- ✅ CORS support cho ESP32 và web clients
- ✅ Xử lý file .wav lên đến 50MB
- ✅ Automatic cleanup của temporary files
- ✅ Health check endpoint
- ✅ Comprehensive error handling
- ✅ Logging đầy đủ

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.13+
- DTLN model (đã được include trong package tại `models/DTLN_vivos_best.h5`)

## 🔧 Installation

1. **Install dependencies:**
   ```bash
   cd noise-reduction-service
   pip install -r requirements.txt
   ```

2. **Configure environment (optional):**
   ```bash
   cp .env.example .env
   # Edit .env để thay đổi cấu hình nếu cần
   ```

3. **Model is ready:**
   - Model đã được đặt sẵn tại `models/DTLN_vivos_best.h5`
   - Không cần cấu hình thêm

## 🏃 Running the Service

### Development Mode

```bash
python app.py
```

Server sẽ start tại `http://0.0.0.0:5000`

### Production Mode

```bash
# Set environment variables
export FLASK_ENV=production
export FLASK_DEBUG=False
export SECRET_KEY=your-production-secret-key

# Run with production WSGI server (recommended)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 📡 API Endpoints

### 1. Health Check

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_info": {
    "status": "loaded",
    "model_path": "../models_DTLN_vivos/DTLN_vivos_best.h5",
    "sample_rate": 16000
  }
}
```

### 2. Denoise Audio

**Endpoint:** `POST /denoise`

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Form field `file` chứa .wav file

**Response:**
- Success (200): Denoised audio file (.wav)
- Error (400/413/500): JSON error message

**cURL Example:**
```bash
curl -X POST \
  -F "file=@noisy_audio.wav" \
  http://localhost:5000/denoise \
  --output denoised_audio.wav
```

**Python Example:**
```python
import requests

url = "http://localhost:5000/denoise"
files = {'file': open('noisy_audio.wav', 'rb')}

response = requests.post(url, files=files)

if response.status_code == 200:
    with open('denoised_audio.wav', 'wb') as f:
        f.write(response.content)
    print("Denoising successful!")
else:
    print(f"Error: {response.json()}")
```

## 🤖 ESP32 Integration

### Arduino/ESP32 Example (HTTP Client)

```cpp
#include <WiFi.h>
#include <HTTPClient.h>
#include <SD.h>

const char* serverUrl = "http://YOUR_SERVER_IP:5000/denoise";

void uploadAndDenoise(const char* inputFile, const char* outputFile) {
    HTTPClient http;
    
    // Read file from SD card
    File file = SD.open(inputFile, FILE_READ);
    if (!file) {
        Serial.println("Failed to open file");
        return;
    }
    
    // Prepare multipart form data
    http.begin(serverUrl);
    http.addHeader("Content-Type", "multipart/form-data");
    
    // Upload file
    int httpCode = http.POST(/* file data */);
    
    if (httpCode == 200) {
        // Save denoised audio
        File outFile = SD.open(outputFile, FILE_WRITE);
        http.writeToStream(&outFile);
        outFile.close();
        Serial.println("Denoising complete!");
    } else {
        Serial.printf("Error: %d\n", httpCode);
    }
    
    http.end();
    file.close();
}
```

### Postman Testing

1. Method: `POST`
2. URL: `http://localhost:5000/denoise`
3. Body > form-data
4. Key: `file` (type: File)
5. Value: Select your .wav file
6. Send request
7. Save response as .wav file

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_ENV` | `development` | Environment mode |
| `FLASK_HOST` | `0.0.0.0` | Server bind address |
| `FLASK_PORT` | `5000` | Server port |
| `FLASK_DEBUG` | `True` | Debug mode |
| `MODEL_PATH` | `../models_DTLN_vivos/DTLN_vivos_best.h5` | Path to model weights |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |
| `SECRET_KEY` | `dev-secret-key` | Flask secret key |

### File Upload Limits

- Maximum file size: 50MB
- Allowed extensions: `.wav`
- Sample rate: 16kHz (recommended)

## 📊 Audio Requirements

- **Format:** WAV
- **Sample Rate:** 16kHz (recommended, model trained on 16kHz)
- **Channels:** Mono (stereo will be auto-converted)
- **Bit Depth:** Any (will be converted to float32)

## 🐛 Troubleshooting

### Model not loading
```
Error: Model file not found
```
**Solution:** Verify `MODEL_PATH` trong `.env` hoặc `config.py`

### Memory errors
```
Error: OOM when allocating tensor
```
**Solution:** 
- Giảm MAX_CONTENT_LENGTH trong config
- Xử lý files nhỏ hơn
- Tăng RAM của server

### Sample rate mismatch
```
Warning: Audio sample rate doesn't match expected rate
```
**Solution:** Resample audio về 16kHz trước khi upload, hoặc service sẽ xử lý nhưng quality có thể giảm

## 📝 Logging

Logs được output ra console với format:
```
2026-02-13 00:08:00 - app - INFO - Processing audio: noisy.wav
```

Để save logs ra file:
```bash
python app.py 2>&1 | tee service.log
```

## 🔒 Security Notes

- Trong production, set `SECRET_KEY` mạnh và unique
- Giới hạn CORS origins về specific domains
- Sử dụng HTTPS trong production
- Rate limiting nên được thêm vào cho production use

## 📄 License

Same as DTLN project

## 🤝 Support

For issues or questions, liên hệ project maintainer.
