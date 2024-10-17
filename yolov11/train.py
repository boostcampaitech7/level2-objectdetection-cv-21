import os
import wandb
from wandb.integration.ultralytics import add_wandb_callback
from ultralytics import YOLO
from convert import convert_yolo  # convert.py에서 convert_yolo 함수 가져오기
from augmentation import augment_and_save  # augmentation.py에서 augment_and_save 함수 가져오기
from split import split_dataset  # split.py에서 데이터셋 분리 함수 가져오기

# 1. 데이터 변환 및 증강 실행
# 경로 설정
train_json_path = "/data/ephemeral/home/dataset/train.json"
train_image_dir = "/data/ephemeral/home/dataset/train"
train_label_output_dir = "/data/ephemeral/home/dataset/labels/train"

# 2. COCO 형식의 train.json을 YOLO 형식으로 변환
print("COCO 데이터를 YOLO 형식으로 변환 중...")
convert_yolo(train_json_path, train_label_output_dir)

# 3. 증강 데이터 생성 (원본과 동일 폴더에 증강 이미지 저장)
print("데이터 증강 중...")
augment_and_save(
    train_image_dir, 
    train_label_output_dir, 
    train_image_dir,  # 원본과 동일 경로에 증강된 이미지 저장
    "yolo11x.pt", 
    blur_ratio=50,
    class_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10개의 클래스만 증강
)

# 4. 데이터셋 분리 작업 수행 (train_split.json, val_split.json으로 분리)
print("데이터셋 분리 중...")
split_dataset("/data/ephemeral/home/dataset/", test_size=0.2, random_state=42, train_file="train_split.json", val_file="val_split.json")

# 5. YOLO 모델 학습 실행
# W&B 초기화
wandb.init(project="Object Detection")

# YOLO 모델 로드
model = YOLO('yolo11x.pt')

# W&B 콜백 추가 (mAP50 시각화)
add_wandb_callback(model)

# 데이터 경로 설정 (train_split.json, val_split.json 사용)
results = model.train(data='/data/ephemeral/home/github/yolov11/cfg/data.yaml', 
                      epochs=50, 
                      imgsz=512, 
                      batch=16)

# 학습 완료 후 W&B 세션 종료
wandb.finish()