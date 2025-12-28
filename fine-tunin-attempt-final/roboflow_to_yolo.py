import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import os, shutil, yaml
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

ROBO_ROOT = "raw/roboflow"

OUT_IMG = "processed/images"
OUT_LBL = "processed/labels"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

# ---------------------------
# Load Roboflow data.yaml
# ---------------------------
with open(os.path.join(ROBO_ROOT, "data.yaml")) as f:
    data = yaml.safe_load(f)

# Roboflow id -> name
ID_TO_NAME = {i: n for i, n in enumerate(data["names"])}

# Map Roboflow class name -> YOUR unified name
ROBO_TO_UNIFIED = {
    "stairs": "stairs",
    "ramp": "ramp",
    "garbage": "garbage_pile",
    "fan": "ceiling_fan",
    "shoe": "shoes",
    "slipper": "slippers",
    "office-chair": "office_chair",
    "extension-board": "extension_board",
    "ladder": "ladder"
}

def process_split(split):
    img_dir = os.path.join(ROBO_ROOT, split, "images")
    lbl_dir = os.path.join(ROBO_ROOT, split, "labels")

    if not os.path.exists(lbl_dir):
        return

    for lbl in os.listdir(lbl_dir):
        img_name = lbl.replace(".txt", ".jpg")
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, lbl)

        if not os.path.exists(img_path):
            continue

        new_lines = []

        with open(lbl_path) as f:
            for line in f:
                parts=line.strip().split()
                if len(parts) < 5:
                  continue
                cid, x, y, w, h = parts[:5]
                cid = int(cid)

                robo_name = ID_TO_NAME.get(cid)
                if robo_name not in ROBO_TO_UNIFIED:
                    continue

                unified_name = ROBO_TO_UNIFIED[robo_name]
                if unified_name not in CLASS_MAP:
                    continue

                new_lines.append(
                    f"{CLASS_MAP[unified_name]} {x} {y} {w} {h}"
                )

        if new_lines:
            shutil.copy(img_path, os.path.join(OUT_IMG, img_name))
            with open(os.path.join(OUT_LBL, lbl), "w") as f:
                f.write("\n".join(new_lines))

def main():
    for split in ["train", "valid", "test"]:
        process_split(split)

if __name__ == "__main__":
    main()
