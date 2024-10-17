import os
import wandb
from wandb.integration.ultralytics import add_wandb_callback
from ultralytics import YOLO

# 1. 데이터 변환, 증강 및 분할 모듈 불러오기
from convert import convert_yolo  # convert.py에서 convert_yolo 함수 가져오기
from augmentation import augment_and_save  # augmentation.py에서 augment_and_save 함수 가져오기

# 데이터셋 분할을 위해 split.py를 직접 실행
print("데이터셋을 train/val로 분할 중...")
os.system("python split.py")  # split.py를 실행하여 train.json을 train2.json, val2.json으로 분할

# 경로 설정
train_json_path = "/data/ephemeral/home/dataset/train2.json"  # 분할된 train.json 사용
val_json_path = "/data/ephemeral/home/dataset/val2.json"  # 분할된 val.json 사용
train_image_dir = "/data/ephemeral/home/dataset/train"
train_label_output_dir = "/data/ephemeral/home/dataset/labels/train"

# Convert COCO to YOLO format (train2.json -> YOLO labels)
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

# 학습 경로 설정 (train2.json과 val2.json을 사용)
results = model.train(
    data='/data/ephemeral/home/github/yolov11/cfg/data.yaml',  # data.yaml 파일에 train/val 경로가 있어야 합니다.
    epochs=50, 
    imgsz=512, 
    batch=16
)

# 학습 완료 후 W&B 세션 종료
wandb.finish()