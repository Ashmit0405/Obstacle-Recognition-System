import os
import random
import shutil

IMG_TRAIN = "safety_dataset/images/train"
LBL_TRAIN = "safety_dataset/labels/train"

IMG_VAL = "safety_dataset/images/val"
LBL_VAL = "safety_dataset/labels/val"

os.makedirs(IMG_VAL, exist_ok=True)
os.makedirs(LBL_VAL, exist_ok=True)

images = [f for f in os.listdir(IMG_TRAIN) if f.endswith(".jpg")]

random.shuffle(images)
val_count = int(0.2 * len(images))

val_images = images[:val_count]

for img in val_images:
    lbl = img.replace(".jpg", ".txt")

    src_img = os.path.join(IMG_TRAIN, img)
    src_lbl = os.path.join(LBL_TRAIN, lbl)

    if not os.path.exists(src_lbl):
        continue

    shutil.move(src_img, os.path.join(IMG_VAL, img))
    shutil.move(src_lbl, os.path.join(LBL_VAL, lbl))

print(f"Split complete: {val_count} images moved to validation set")