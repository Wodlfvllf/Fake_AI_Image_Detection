from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import aiofiles
import os
import time
from model import AntifakePrompt

app = FastAPI(
    title="AntifakePrompt Detection API",
    description="API for detecting fake images using vision-language models",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Load the model at startup
model = AntifakePrompt(
    finetuned_path="./weights/ckpt/COCO_150k_SD3_SD2IP_lama.pth",
    quant_4bit=True,
    device_map="auto"
)

class DetectionResult(BaseModel):
    is_real: bool
    prediction: str
    processing_time: float

@app.post("/detect", response_model=DetectionResult)
async def detect_fake_image(file: UploadFile = File(...)):
    """Detect if an uploaded image is real or fake."""
    start_time = time.time()

    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG/PNG allowed")
        
        # Save temp file
        temp_path = f"temp_{file.filename}"
        async with aiofiles.open(temp_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        # Run inference
        result = model.predict(temp_path)

        # Cleanup
        os.remove(temp_path)

        return {
            "is_real": result,
            "prediction": "Real" if result else "Fake",
            "processing_time": round(time.time() - start_time, 3)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
