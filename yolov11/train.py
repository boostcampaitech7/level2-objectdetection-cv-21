import os
import wandb
from wandb.integration.ultralytics import add_wandb_callback
from ultralytics import YOLO
from convert import convert_yolo
from split import split_dataset

# Define paths
original_image_dir = "/data/ephemeral/home/dataset/train"  # 원본 이미지가 있는 디렉토리
train_split_dir = "/data/ephemeral/home/dataset/train_split"  # 학습 이미지가 저장될 디렉토리
val_split_dir = "/data/ephemeral/home/dataset/val_split"      # 검증 이미지가 저장될 디렉토리
train_json_path = "/data/ephemeral/home/dataset/train.json"   # COCO 형식 JSON 파일 경로
data_yaml_path = "/data/ephemeral/home/github/yolov11/cfg/data.yaml"  # data.yaml 파일 경로
model_path = "yolo11x.pt"  # YOLOv11x 모델 가중치 경로

# Step 1: Convert COCO format to YOLO format
print("Converting COCO data to YOLO format...")
convert_yolo(train_json_path, original_image_dir)
print("Conversion to YOLO format completed.")

# Step 2: Split dataset into train and val
print("Splitting dataset into train and val...")
split_dataset(original_image_dir, train_split_dir, val_split_dir)
print("Dataset split completed.")

# Step 3: Initialize WandB
wandb.init(project="Object Detection")

# Load YOLO model
model = YOLO(model_path)

# Add WandB callback (for mAP50 visualization)
add_wandb_callback(model)

# Step 4: Train YOLO model
results = model.train(
    data=data_yaml_path,   # data.yaml 파일 경로
    epochs=50,
    imgsz=512,
    batch=2,
    amp=True, # Mixed Precision Training
    accumulate=2 # Gradient Accumulation
)

# Step 5: Finish WandB session
wandb.finish()

print("Training completed.")