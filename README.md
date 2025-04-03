# Fake_AI_Image_Detection

# AntifakePrompt Detection API

## Overview
The **AntifakePrompt Detection API** is a FastAPI-based web service for detecting fake images using prompt-tuned vision-language models. It takes an image as input and predicts whether the image is real or fake.

## Features
- 🚀 **Fast Inference** using optimized **4-bit quantization**
- 🖼️ **Supports JPEG & PNG images**
- 🔥 **CORS enabled** for cross-origin requests
- 🏧 **Scalable & Docker-ready**
- 🔍 **Batch processing support (coming soon)**

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/AntifakePrompt-API.git
cd AntifakePrompt-API
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Start the API Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

#### **1. Health Check**
```http
GET /health
```
📌 **Response:**
```json
{
  "status": "OK",
  "model_loaded": true
}
```

#### **2. Fake Image Detection**
```http
POST /detect
```
📌 **Request:**
- **Form Data**: Image file (`.jpg` or `.png`)

📌 **cURL Example:**
```bash
curl -X POST "http://localhost:8000/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

📌 **Response:**
```json
{
  "is_real": false,
  "prediction": "Fake",
  "processing_time": 0.345
}
```

## Project Structure
```
AntifakePrompt-API/
│-- main.py                # FastAPI Application
│-- model.py               # Model Inference Code
│-- checkpoints/           # Model Checkpoints
```

## Deployment

### Run with Docker
```bash
docker build -t antifake-api .
docker run -p 8000:8000 antifake-api
```

### Run with Gunicorn for Production
```bash
uvicorn antifake_api:app --host 0.0.0.0 --port 8080
```

## License
This project is licensed under the **MIT License**.

## Contributors
- **Your Name** - [GitHub](https://github.com/your-profile)

---

For further queries, raise an issue on GitHub! 🚀

