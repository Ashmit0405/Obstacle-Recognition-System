import json, os, shutil
from PIL import Image

CLASSES = [
    "person","bicycle","motorcycle","car","bus","truck",
    "traffic sign","stairs","chair","bench","door","pole"
]

# COCO category name → our class id
COCO_MAP = {
    "person": 0,
    "bicycle": 1,
    "motorcycle": 2,
    "car": 3,
    "bus": 4,
    "truck": 5,
    "chair": 8,
    "bench": 9,
    "door": 10,
    "stop sign": 6,
    "traffic light": 6
}

COCO_IMG = "coco/train2017"
COCO_ANN = "coco/annotations/instances_train2017.json"

OUT_IMG = "safety_dataset/images/train"
OUT_LBL = "safety_dataset/labels/train"

with open(COCO_ANN) as f:
    coco = json.load(f)

cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
img_map = {img["id"]: img for img in coco["images"]}

img_to_anns = {}
for ann in coco["annotations"]:
    name = cat_id_to_name[ann["category_id"]]
    if name in COCO_MAP:
        img_to_anns.setdefault(ann["image_id"], []).append(ann)

for img_id, anns in img_to_anns.items():
    img = img_map[img_id]
    src = os.path.join(COCO_IMG, img["file_name"])
    if not os.path.exists(src):
        continue

    w, h = img["width"], img["height"]
    labels = []

    for ann in anns:
        x, y, bw, bh = ann["bbox"]
        cls = COCO_MAP[cat_id_to_name[ann["category_id"]]]

        xc = (x + bw/2) / w
        yc = (y + bh/2) / h
        bw /= w
        bh /= h

        labels.append(f"{cls} {xc} {yc} {bw} {bh}")

    if labels:
        shutil.copy(src, os.path.join(OUT_IMG, img["file_name"]))
        with open(
            os.path.join(OUT_LBL, img["file_name"].replace(".jpg",".txt")), "w"
        ) as f:
            f.write("\n".join(labels))

print("✅ COCO merged into safety_dataset")