import os
import wandb
from wandb.integration.ultralytics import add_wandb_callback
from ultralytics import YOLO

# Define paths
train_split_dir = "/data/ephemeral/home/dataset/train_split"  # 학습 이미지가 저장된 디렉토리
val_split_dir = "/data/ephemeral/home/dataset/val_split"      # 검증 이미지가 저장된 디렉토리
data_yaml_path = "/data/ephemeral/home/github/yolov11/cfg/data.yaml"  # data.yaml 파일 경로
model_path = "yolo11x.pt"  # YOLOv11x 모델 가중치 경로

# data.yaml 파일이 올바르게 설정되어 있는지 확인합니다.
# 예시 data.yaml 파일:
# train: /data/ephemeral/home/dataset/train_split
# val: /data/ephemeral/home/dataset/val_split
# nc: 10  # 클래스 수
# names: ['General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']

# Step 1: W&B 초기화
wandb.init(project="Object Detection")

# YOLO 모델 로드
model = YOLO(model_path)

# W&B 콜백 추가 (mAP50 시각화)
add_wandb_callback(model)

# Step 2: YOLO 모델 학습 실행
results = model.train(
    data=data_yaml_path,   # 수정된 data.yaml 파일 경로
    epochs=50,
    imgsz=512,
    batch=4,
    name='train_run', # 실험 이름 설정
    project='Object Detection' # 프로젝트 이름 설정
)

# Step 3: W&B 세션 종료
wandb.finish()

print("Training completed.")