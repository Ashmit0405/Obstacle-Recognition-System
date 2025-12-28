from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import traceback
import uvicorn
from setup import process_base64_image

app = FastAPI(
    title="YOLO Open-Vocab Inference API",
    version="1.0"
)

class ImageRequest(BaseModel):
    data: str
    width: int
    height: int
    x: int
    y: int
    timestamp: int

class Detection(BaseModel):
    label: str
    depth_m: float
    direction: str
    angle_deg: float
    confidence: float

class DetectionResponse(BaseModel):
    detections: List[Detection]

@app.post("/detect", response_model=DetectionResponse)
def detect_objects(payload: ImageRequest):
    try:
        detections = process_base64_image(payload.data)
        print(detections)
        return {"detections": detections}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Internal inference error"
        )
PORT = 8000
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT
    )