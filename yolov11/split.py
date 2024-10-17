import os
import shutil
import random

def split_train(train_dir, val_dir, val_ratio=0.2):
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    images = [f for f in os.listdir(train_dir) if f.endswith(".jpg")]  # 이미지 파일만 선택

    # val_ratio에 따라 train 데이터를 섞어서 나누기
    random.shuffle(images)
    val_size = int(len(images) * val_ratio)

    val_images = images[:val_size]
    train_images = images[val_size:]

    print(f"Train set size: {len(train_images)}")
    print(f"Validation set size: {len(val_images)}")

    # Validation 파일 이동
    for img in val_images:
        img_path = os.path.join(train_dir, img)
        lbl_path = os.path.join(train_dir, img.replace(".jpg", ".txt"))

        shutil.move(img_path, val_dir)  # 이미지 이동
        if os.path.exists(lbl_path):
            shutil.move(lbl_path, val_dir)  # 라벨 파일이 존재하면 같이 이동

# 경로 설정
train_dir = "/data/ephemeral/home/dataset/train"
val_dir = "/data/ephemeral/home/dataset/val"

# train 데이터셋을 8:2로 train/val로 나누기
split_train(train_dir, val_dir, val_ratio=0.2)
