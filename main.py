import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import base64
import cv2
import numpy as np
import torch
import onnxruntime as ort
from ultralytics import YOLO
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import traceback
import uvicorn

torch.set_grad_enabled(False)
torch.set_num_threads(1)
yolo = None
depth_session = None
def load_models():
    global yolo, depth_session

    if yolo is None:
        yolo = YOLO(
            "yoloe-11s-seg-pf.pt",
            verbose=False
        )

    if depth_session is None:
        depth_session = ort.InferenceSession(
            "./model.onnx",
            providers=["CPUExecutionProvider"]
        )
def decode_base64_image(b64_string: str):
    if "," in b64_string and b64_string.startswith("data:image"):
        b64_string = b64_string.split(",", 1)[1]

    b64_string = re.sub(r'[^A-Za-z0-9+/=]', '', b64_string)
    img_bytes = base64.b64decode(b64_string)

    img_array = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("Invalid image")

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def compute_angle(x_center, image_width, fov_deg=70.0):
    x_norm = (x_center - image_width / 2) / (image_width / 2)
    return x_norm * (fov_deg / 2)

def classify_direction(angle):
    if angle < -25:
        return "hard left"
    elif angle < -10:
        return "soft left"
    elif angle <= 10:
        return "center"
    elif angle <= 25:
        return "soft right"
    else:
        return "hard right"
def run_depth(frame_rgb):
    H, W = frame_rgb.shape[:2]

    img = cv2.resize(frame_rgb, (256, 256))
    inp = img.astype(np.float32) / 255.0
    inp = inp.transpose(2, 0, 1)[None]

    depth_256 = depth_session.run(
        None, {"pixel_values": inp}
    )[0][0, 0]

    return cv2.resize(depth_256, (W, H))
def detect_wall_from_segmentation(masks, depth, H, W):
    if not masks:
        return False, None

    cx1, cx2 = int(0.3 * W), int(0.7 * W)
    cy1 = int(0.3 * H)

    for mask in masks:
        roi = mask[cy1:H, cx1:cx2]
        if roi.mean() < 0.25:
            continue

        wall_depth = np.median(depth[mask])
        if np.isfinite(wall_depth) and 0.3 < wall_depth < 6.0:
            return True, float(wall_depth)

    return False, None
def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (areaA + areaB - inter)

def deduplicate_boxes(dets, thresh=0.5):
    dets.sort(key=lambda x: x["confidence"], reverse=True)
    out = []
    for d in dets:
        if all(iou(d["xyxy"], k["xyxy"]) < thresh for k in out):
            out.append(d)
    return out
def find_best_match(reference_xyxy, detections):
    best_iou = 0.0
    best_det = None

    for det in detections:
        curr_iou = iou(reference_xyxy, det["xyxy"])
        if curr_iou > best_iou:
            best_iou = curr_iou
            best_det = det

    return best_det
def process_base64_image(b64):
    frame = decode_base64_image(b64)
    H, W = frame.shape[:2]

    depth = run_depth(frame)
    results = yolo(frame, imgsz=416, conf=0.3, iou=0.5, device="cpu")[0]
    detections = []
    masks = []
    if results.masks is not None:
        for m in results.masks.data:
            m = m.cpu().numpy().astype(np.uint8)
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            masks.append(m.astype(bool))

    wall_present, wall_dist = detect_wall_from_segmentation(masks, depth, H, W)

    if wall_present:
        detections.append({
            "label": "wall",
            "xyxy": [int(0.3 * W), int(0.3 * H), int(0.7 * W), H],
            "depth_m": wall_dist,
            "direction": "center",
            "angle_deg": 0.0,
            "confidence": 1.0
        })

    objs = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        d = float(np.median(depth[y1:y2, x1:x2]))
        angle = compute_angle((x1 + x2) / 2, W)

        objs.append({
            "label": results.names[int(box.cls[0])],
            "xyxy": [x1, y1, x2, y2],
            "depth_m": d,
            "direction": classify_direction(angle),
            "angle_deg": angle,
            "confidence": conf
        })

    detections.extend(deduplicate_boxes(objs))
    return detections
app = FastAPI(title="YOLO Open-Vocab Inference API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.on_event("startup")
def startup():
    load_models()

class ImageRequest(BaseModel):
    data: str
    isannoted: bool=False
    xyxy: List[float] | None=None
    timestamp: int

class Detection(BaseModel):
    label: str
    depth_m: float
    direction: str
    angle_deg: float
    confidence: float
    xyxy: List[int]

class DetectionResponse(BaseModel):
    detections: List[Detection]
    destination: Detection | None=None
    timestamp: int
@app.post("/detect", response_model=DetectionResponse)
def detect(payload: ImageRequest):
    try:
        detections = process_base64_image(payload.data)

        destination = None
        if payload.isannoted and payload.xyxy:
            destination = find_best_match(payload.xyxy, detections)

        return {
            "detections": detections,
            "destination": destination,
	    "timestamp": payload.timestamp
        }

    except ValueError as e:
        raise HTTPException(400, str(e))

    except Exception:
        traceback.print_exc()
        raise HTTPException(500, "Inference error")
PORT = 8000
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT
    )