import os, shutil

IMG_DIR = "safety_dataset/images/train"
LBL_DIR = "safety_dataset/labels/train"

TARGET_CLASS = "7" 
MULTIPLIER = 4    

images = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]

count = 0
for img in images:
    lbl = img.replace(".jpg", ".txt")
    lbl_path = os.path.join(LBL_DIR, lbl)

    if not os.path.exists(lbl_path):
        continue

    with open(lbl_path) as f:
        if any(line.startswith(TARGET_CLASS + " ") for line in f):
            for i in range(MULTIPLIER):
                new_img = img.replace(".jpg", f"_stairs{i}.jpg")
                new_lbl = lbl.replace(".txt", f"_stairs{i}.txt")

                shutil.copy(
                    os.path.join(IMG_DIR, img),
                    os.path.join(IMG_DIR, new_img)
                )
                shutil.copy(
                    lbl_path,
                    os.path.join(LBL_DIR, new_lbl)
                )
                count += 1

print(f"Oversampled stairs: {count} new samples added")