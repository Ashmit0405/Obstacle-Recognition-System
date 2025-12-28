import os, shutil
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

SRC_IMG = "/root/.cache/kagglehub/datasets/sayedmohamed1/doors-detection/versions/1/train/images"
SRC_LBL = "/root/.cache/kagglehub/datasets/sayedmohamed1/doors-detection/versions/1/train/labels"

OUT_IMG = "processed/images"
OUT_LBL = "processed/labels"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

# Kaggle dataset has only one class: door = 0
OLD_ID_TO_NAME = {0: "door"}

def main():
    for lbl in os.listdir(SRC_LBL):
        lbl_path = os.path.join(SRC_LBL, lbl)
        img_name = lbl.replace(".txt", ".jpg")
        img_path = os.path.join(SRC_IMG, img_name)

        if not os.path.exists(img_path):
            continue

        new_lines = []
        with open(lbl_path) as f:
            for line in f:
                cid, x, y, w, h = line.strip().split()
                cname = OLD_ID_TO_NAME.get(int(cid))
                if cname not in CLASS_MAP:
                    continue
                new_lines.append(
                    f"{CLASS_MAP[cname]} {x} {y} {w} {h}"
                )

        if new_lines:
            shutil.copy(img_path, os.path.join(OUT_IMG, img_name))
            with open(os.path.join(OUT_LBL, lbl), "w") as f:
                f.write("\n".join(new_lines))

if __name__ == "__main__":
    main()
