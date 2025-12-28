import cv2, os
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

ADE_LABEL_MAP = {
    15: "table",
    17: "bed_edge",
    24: "wardrobe",
    31: "ceiling_fan",
    62: "whiteboard",
}

IMG_DIR = "raw/ade20k/ADEChallengeData2016/images/training"
MASK_DIR = "raw/ade20k/ADEChallengeData2016/annotations/training"
OUT_IMG = "processed/images"
OUT_LBL = "processed/labels"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

def main():
    for img in os.listdir(IMG_DIR):
        name = img.replace(".jpg", "")
        mask_path = f"{MASK_DIR}/{name}.png"
        if not os.path.exists(mask_path):
            continue

        img_path = f"{IMG_DIR}/{img}"
        image = cv2.imread(img_path)
        h, w = image.shape[:2]
        mask = cv2.imread(mask_path, 0)

        labels = []
        for ade_id, cname in ADE_LABEL_MAP.items():
            if cname not in CLASS_MAP:
                continue
            cls_id = CLASS_MAP[cname]
            binary = (mask == ade_id).astype("uint8") * 255
            cnts,_ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                x,y,bw,bh = cv2.boundingRect(c)
                labels.append(
                    f"{cls_id} {(x+bw/2)/w} {(y+bh/2)/h} {bw/w} {bh/h}"
                )

        if labels:
            cv2.imwrite(f"{OUT_IMG}/{img}", image)
            with open(f"{OUT_LBL}/{name}.txt", "w") as f:
                f.write("\n".join(labels))

if __name__ == "__main__":
    main()
