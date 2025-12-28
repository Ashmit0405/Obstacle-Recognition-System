import os, shutil, random

imgs = os.listdir("processed/images")
random.shuffle(imgs)

n = len(imgs)
train = imgs[:int(0.7*n)]
val   = imgs[int(0.7*n):int(0.9*n)]
test  = imgs[int(0.9*n):]

def move(files, split):
    os.makedirs(f"dataset/images/{split}", exist_ok=True)
    os.makedirs(f"dataset/labels/{split}", exist_ok=True)
    for img in files:
        shutil.move(f"processed/images/{img}", f"dataset/images/{split}/{img}")
        shutil.move(
            f"processed/labels/{img.replace('.jpg','.txt')}",
            f"dataset/labels/{split}/{img.replace('.jpg','.txt')}"
        )

move(train,"train")
move(val,"val")
move(test,"test")
