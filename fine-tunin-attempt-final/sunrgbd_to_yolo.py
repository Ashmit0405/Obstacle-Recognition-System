import os, shutil, cv2
import scipy.io
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

META = "raw/sunrgbd/SUNRGBDMeta2DBB_v2.mat"
IMG_ROOT = "raw/sunrgbd"
ESS_IMG = "raw/sunrgbd/essential/images"

OUT_IMG = "processed/images"
OUT_LBL = "processed/labels"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

# SUN RGB-D label → your unified name
SUN_TO_NAME = {
    "table": "table",
    "desk": "conference_table",
    "chair": "office_chair",
    "cabinet": "cupboard",
    "wardrobe": "wardrobe",
    "whiteboard": "whiteboard",
    "monitor": "television",
    "tv": "television",
    "speaker": "speaker",
    "router": "router"
}
def resolve_sunrgbd_path(meta_path):
    """
    Convert SUN RGB-D absolute metadata path
    to local raw/sunrgbd path
    """
    idx = meta_path.find("SUNRGBD/")
    if idx == -1:
        return None

    relative = meta_path[idx + len("SUNRGBD/"):]
    return os.path.join("raw/sunrgbd", relative)


def main():
    meta = scipy.io.loadmat(META)["SUNRGBDMeta2DBB"][0]

    valid_imgs = set(os.listdir(ESS_IMG))

    for m in meta:
        img_path = m["rgbpath"][0]
        img_name = os.path.basename(img_path)

        if img_name not in valid_imgs:
            continue

        full_img = resolve_sunrgbd_path(img_path)
        if full_img is None or not os.path.exists(full_img):
          continue

        img = cv2.imread(full_img)
        if img is None:
          continue

        h, w = img.shape[:2]
        labels = []

        for obj in m["objects"][0][0]:
            cname = obj["classname"][0]
            if cname not in SUN_TO_NAME:
                continue

            final_name = SUN_TO_NAME[cname]
            if final_name not in CLASS_MAP:
                continue

            x1, y1, x2, y2 = obj["bbox"][0]
            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            labels.append(
                f"{CLASS_MAP[final_name]} {xc} {yc} {bw} {bh}"
            )

        if labels:
            shutil.copy(full_img, f"{OUT_IMG}/{img_name}")
            with open(f"{OUT_LBL}/{img_name.replace('.jpg','.txt')}", "w") as f:
                f.write("\n".join(labels))

if __name__ == "__main__":
    main()
