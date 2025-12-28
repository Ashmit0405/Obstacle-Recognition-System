import os, shutil

RF_IMG = "stairs-1/train/images"
RF_LBL = "stairs-1/train/labels"

OUT_IMG = "safety_dataset/images/train"
OUT_LBL = "safety_dataset/labels/train"

STAIRS_CLASS_ID = 7 

for lbl in os.listdir(RF_LBL):
    img = lbl.replace(".txt", ".jpg")

    src_img = os.path.join(RF_IMG, img)
    src_lbl = os.path.join(RF_LBL, lbl)

    if not os.path.exists(src_img):
        continue

    shutil.copy(src_img, os.path.join(OUT_IMG, img))

    with open(src_lbl) as f:
        lines = f.readlines()

    new_lines = []
    for l in lines:
        parts = l.strip().split()
        parts[0] = str(STAIRS_CLASS_ID)
        new_lines.append(" ".join(parts))

    with open(os.path.join(OUT_LBL, lbl), "w") as f:
        f.write("\n".join(new_lines))

print("Roboflow hazards merged")