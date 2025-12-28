import json, os, shutil
CLASS_MAP = {
    "table": 80,
    "cupboard": 81,
    "wardrobe": 82,
    "bed_edge": 83,
    "slippers": 84,
    "shoes": 85,
    "mat": 86,
    "cable_on_floor": 87,
    "extension_board": 88,
    "ceiling_fan": 89,
    "tube_light": 90,
    "hanging_wire": 91,
    "ac_indoor": 92,
    "television": 93,
    "speaker": 94,
    "router": 95,
    "office_chair": 96,
    "conference_table": 97,
    "cubicle_partition": 98,
    "whiteboard": 99,
    "projector_screen": 100,
    "server_rack": 101,
    "glass_partition": 102,
    "building": 103,
    "compound_wall": 104,
    "fence": 105,
    "garbage_bin": 106,
    "garbage_pile": 107,
    "electric_box": 108,
    "street_vendor_cart": 109,
    "door": 110,
    "window": 111,

    # ---------- Navigation / Hazards ----------
    "stairs": 112,
    "ladder": 113,

    # ---------- Lighting / Head-level ----------
    "lamp": 114,
    "light_bulb": 115,
}

IMG_DIR = "/kaggle/input/solesensei_bdd100k/bdd100k/bdd100k/images/100k/train"
LBL_JSON = "/kaggle/input/solesensei_bdd100k/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"

OUT_IMG = "processed/images"
OUT_LBL = "processed/labels"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

# BDD → your unified class names
BDD_TO_NAME = {
    "building": "building",
    "wall": "compound_wall",
    "fence": "fence",
    "pole": "electric_box",
    "traffic sign": "garbage_bin"
}

def main():
    with open(LBL_JSON) as f:
        data = json.load(f)

    for item in data:
        img_name = item["name"]
        img_path = os.path.join(IMG_DIR, img_name)

        if not os.path.exists(img_path):
            continue

        labels = []
        for lbl in item.get("labels", []):
            cat = lbl.get("category")
            if cat not in BDD_TO_NAME:
                continue

            cname = BDD_TO_NAME[cat]
            if cname not in CLASS_MAP:
                continue

            box = lbl.get("box2d")
            if not box:
                continue

            x1, y1 = box["x1"], box["y1"]
            x2, y2 = box["x2"], box["y2"]

            # Load image once
            img = cv2.imread(img_path)
            h, w = img.shape[:2]

            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            labels.append(
                f"{CLASS_MAP[cname]} {xc} {yc} {bw} {bh}"
            )

        if labels:
            shutil.copy(img_path, os.path.join(OUT_IMG, img_name))
            with open(
                os.path.join(OUT_LBL, img_name.replace(".jpg", ".txt")),
                "w"
            ) as f:
                f.write("\n".join(labels))

if __name__ == "__main__":
    import cv2
    main()
