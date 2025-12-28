import base64
import cv2
import numpy as np
import torch
import onnxruntime as ort
from ultralytics import YOLO
import re
torch.set_grad_enabled(False)
yolo = YOLO("yoloe-11s-seg-pf.pt")
depth_session = ort.InferenceSession(
    "/content/model.onnx", #iski jagah "./model.onnx"
    providers=["CUDAExecutionProvider"]
)
#def safe_base64_decode(b64_string: str) -> bytes:
#    b64_string = re.sub(r'[^A-Za-z0-9+/=]', '', b64_string)
#    return base64.b64decode(b64_string)
def decode_base64_image(b64_string: str):
    if not isinstance(b64_string, str):
        raise TypeError("Input must be a base64 string")

    if "," in b64_string and b64_string.strip().startswith("data:image"):
        b64_string = b64_string.split(",", 1)[1]

    b64_string = re.sub(r'[^A-Za-z0-9+/=]', '', b64_string)
    try:
        img_bytes = base64.b64decode(b64_string, validate=True)
    except Exception as e:
        raise ValueError(f"Base64 decode failed: {e}")

    img_array = np.frombuffer(img_bytes, np.uint8)
    if img_array.size == 0:
        raise ValueError("Decoded byte array is empty")

    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("OpenCV failed to decode image (invalid image bytes)")

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

    img = cv2.resize(frame_rgb, (512, 512))
    inp = img.astype(np.float32) / 255.0
    inp = inp.transpose(2, 0, 1)[None]

    depth_512 = depth_session.run(
        None, {"pixel_values": inp}
    )[0][0, 0]

    return cv2.resize(depth_512, (W, H))
def detect_wall_from_segmentation(masks, depth, H, W):
    if not masks:
        return False, None

    cx1, cx2 = int(0.3 * W), int(0.7 * W)
    cy1 = int(0.3 * H)

    for mask in masks:
        if mask.shape != depth.shape:
            continue

        roi = mask[cy1:H, cx1:cx2]
        coverage = roi.sum() / roi.size
        if coverage < 0.25:
            continue

        wall_depth = np.median(depth[mask])
        if np.isfinite(wall_depth) and 0.3 < wall_depth < 6.0:
            return True, float(wall_depth)

    return False, None
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter / (areaA + areaB - inter)
def deduplicate_boxes(detections, iou_thresh=0.5):
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    final = []

    for det in detections:
        keep = True
        for kept in final:
            if iou(det["xyxy"], kept["xyxy"]) >= iou_thresh:
                keep = False
                break
        if keep:
            final.append(det)

    return final
def process_base64_image(b64_image):
    frame_rgb = decode_base64_image(b64_image)
    H, W = frame_rgb.shape[:2]

    depth = run_depth(frame_rgb)
    results = yolo(frame_rgb, conf=0.3, iou=0.5)[0]

    detections = []

    masks = []
    if results.masks is not None:
      for m in results.masks.data:
          m = m.cpu().numpy().astype(np.uint8)
          m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
          masks.append(m.astype(bool))

    wall_present, wall_dist = detect_wall_from_segmentation(
        masks, depth, H, W
    )

    if wall_present:
        detections.append({
            "label": "wall",
            "xyxy": [int(0.3*W), int(0.3*H), int(0.7*W), H],
            "depth_m": wall_dist,
            "direction": "center",
            "angle_deg": 0.0,
            "confidence": 1.0
        })
    object_dets = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        d = float(np.median(depth[y1:y2, x1:x2]))
        x_center = (x1 + x2) / 2
        angle = compute_angle(x_center, W)

        object_dets.append({
            "label": results.names[int(box.cls[0])],
            "xyxy": [x1, y1, x2, y2],
            "depth_m": d,
            "direction": classify_direction(angle),
            "angle_deg": angle,
            "confidence": conf
        })
    object_dets = deduplicate_boxes(object_dets, iou_thresh=0.5)

    detections.extend(object_dets)
    return detections
