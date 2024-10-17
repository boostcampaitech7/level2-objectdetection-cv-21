import os
import wandb
from wandb.integration.ultralytics import add_wandb_callback
from ultralytics import YOLO

# 1. 데이터 변환 및 증강 모듈 불러오기 (convert.py와 augmentation.py의 함수 사용)
from convert import convert_yolo  # convert.py에서 convert_yolo 함수 가져오기
from augmentation import augment_and_save  # augmentation.py에서 augment_and_save 함수 가져오기

# 2. 데이터 변환 및 증강 실행

# 경로 설정
train_json_path = "/data/ephemeral/home/dataset/train.json"
train_image_dir = "/data/ephemeral/home/dataset/train"
train_label_output_dir = "/data/ephemeral/home/dataset/labels/train"

# Convert COCO to YOLO format (train.json -> YOLO labels)
print("COCO 데이터를 YOLO 형식으로 변환 중...")
convert_yolo(train_json_path, train_label_output_dir)

# Augment the data and save the results
augmented_image_dir = "/data/ephemeral/home/dataset/train_augmented"
if not os.path.exists(augmented_image_dir):
    os.makedirs(augmented_image_dir)

print("데이터 증강 중...")
augment_and_save(
    train_image_dir, 
    train_label_output_dir, 
    augmented_image_dir,  # 증강된 파일 저장 경로
    "yolo11x.pt", 
    blur_ratio=50
)

# 3. YOLO 모델 학습 실행
# W&B 초기화
wandb.init(project="Object Detection")

# YOLO 모델 로드
model = YOLO('yolo11x.pt')

# W&B 콜백 추가 (mAP50 시각화)
add_wandb_callback(model)

# 데이터 경로 설정 (data.yaml 파일에 원본과 증강된 이미지 경로 추가)
results = model.train(data='/data/ephemeral/home/github/yolov11/cfg/data.yaml', 
                      epochs=50, 
                      imgsz=512, 
                      batch=16)

# 학습 완료 후 W&B 세션 종료
wandb.finish()