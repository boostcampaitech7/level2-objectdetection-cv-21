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

# 유효한 클래스 리스트 설정 (0부터 9까지 10개 클래스)
valid_classes = list(range(10))

# Step 1: Convert COCO format to YOLO format with valid class filtering
print("Converting COCO data to YOLO format...")
convert_yolo(train_json_path, original_image_dir, valid_classes)
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
    amp=True,  # Mixed Precision Training
)

# Step 5: Filter WandB predictions and ground-truth
def filter_wandb_predictions(predictions, valid_classes):
    """유효한 클래스만 남도록 WandB 예측 필터링"""
    filtered_predictions = {}
    for image_id, pred_string in predictions.items():
        filtered_pred = []
        pred_list = pred_string.strip().split(" ")
        for i in range(0, len(pred_list), 6):
            values = pred_list[i:i+6]
            if len(values) != 6:
                continue
            cls = int(values[0])
            if cls in valid_classes:
                filtered_pred.append(" ".join(values))
        filtered_predictions[image_id] = " ".join(filtered_pred)
    return filtered_predictions

def filter_ground_truth(ground_truth, valid_classes):
    """유효한 클래스만 남도록 ground-truth 필터링"""
    filtered_ground_truth = {}
    for image_id, gt_string in ground_truth.items():
        filtered_gt = []
        gt_list = gt_string.strip().split(" ")
        for i in range(0, len(gt_list), 6):
            values = gt_list[i:i+6]
            if len(values) != 6:
                continue
            cls = int(values[0])
            if cls in valid_classes:
                filtered_gt.append(" ".join(values))
        filtered_ground_truth[image_id] = " ".join(filtered_gt)
    return filtered_ground_truth

# WandB 업로드 전에 필터링 적용
if 'predictions' in results:
    filtered_predictions = filter_wandb_predictions(results['predictions'], valid_classes)
    wandb.log({"filtered_predictions": filtered_predictions})

if 'ground_truth' in results:
    filtered_ground_truth = filter_ground_truth(results['ground_truth'], valid_classes)
    wandb.log({"filtered_ground_truth": filtered_ground_truth})

# Step 6: Finish WandB session
wandb.finish()

print("Training completed.")