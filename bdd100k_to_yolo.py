import json, os, shutil
from PIL import Image

# Safety-critical classes
CLASSES = [
    "person",
    "bicycle",
    "motorcycle",
    "car",
    "bus",
    "truck",
    "traffic sign"
]
class_to_id = {c: i for i, c in enumerate(CLASSES)}

IMG_DIR = "/root/.cache/kagglehub/datasets/solesensei/solesensei_bdd100k/versions/2/bdd100k/bdd100k/images/100k/train"
ANN_FILE = "/root/.cache/kagglehub/datasets/solesensei/solesensei_bdd100k/versions/2/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"

OUT_IMG = "safety_dataset/images/train"
OUT_LBL = "safety_dataset/labels/train"
os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

with open(ANN_FILE) as f:
    data = json.load(f)

for img in data:
    img_name = img["name"]
    img_path = os.path.join(IMG_DIR, img_name)

    if not os.path.exists(img_path):
        continue

    # ✅ FIX: read image size from file
    with Image.open(img_path) as im:
        w, h = im.size

    labels_out = []

    for obj in img.get("labels", []):
        if "box2d" not in obj:
            continue

        cat = obj["category"]
        if cat not in class_to_id:
            continue

        box = obj["box2d"]
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

        xc = (x1 + x2) / 2 / w
        yc = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        labels_out.append(
            f"{class_to_id[cat]} {xc} {yc} {bw} {bh}"
        )

    if labels_out:
        shutil.copy(img_path, os.path.join(OUT_IMG, img_name))
        with open(
            os.path.join(OUT_LBL, img_name.replace(".jpg", ".txt")), "w"
        ) as f:
            f.write("\n".join(labels_out))

print("✅ BDD100K train split converted to YOLO format (FIXED)")