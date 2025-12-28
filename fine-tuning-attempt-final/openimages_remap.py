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
import os, shutil

SRC_IMG = "raw/openimages/images/val"
SRC_LBL = "raw/openimages/labels/val"
OUT_IMG = "processed/images"
OUT_LBL = "processed/labels"

OLD_ID_TO_NAME = {
    0: "shoes",
    1: "slippers",
    2: "ceiling_fan",
    3: "garbage_bin",
}

def main():
    for lbl in os.listdir(SRC_LBL):
        lines = open(f"{SRC_LBL}/{lbl}").read().strip().splitlines()
        new_lines = []

        for l in lines:
            cid,x,y,w,h = l.split()
            cname = OLD_ID_TO_NAME.get(int(cid))
            if cname not in CLASS_MAP:
                continue
            new_lines.append(f"{CLASS_MAP[cname]} {x} {y} {w} {h}")

        if new_lines:
            shutil.copy(f"{SRC_IMG}/{lbl.replace('.txt','.jpg')}", OUT_IMG)
            with open(f"{OUT_LBL}/{lbl}", "w") as f:
                f.write("\n".join(new_lines))

if __name__ == "__main__":
    main()
