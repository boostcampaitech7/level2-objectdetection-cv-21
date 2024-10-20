import os
import wandb
from wandb.integration.ultralytics import add_wandb_callback
from ultralytics import YOLO
from convert import convert_yolo  # convert.py에서 convert_yolo 함수 가져오기
from augmentation import augment_and_save  # augmentation.py에서 augment_and_save 함수 가져오기
from split import split_dataset  # split.py에서 데이터셋 분리 함수 가져오기
from utils import check_and_adjust_dimensions  # utils.py에서 함수 불러오기

# 1. 데이터 변환 및 증강 실행
# 경로 설정
train_json_path = "../../dataset/train.json"
train_image_dir = "../../dataset/train"
train_label_output_dir = "../../dataset/labels/train"
augmented_dir = "../../dataset/train_aug"
model_path = "yolo11x.pt"

# 2. COCO 형식의 train.json을 YOLO 형식으로 변환
print("COCO 데이터를 YOLO 형식으로 변환 중...")
convert_yolo(train_json_path, train_label_output_dir)

# 3. 증강 데이터 생성 (증강된 이미지를 별도 디렉토리에 저장)
if not os.path.exists(augmented_dir):
    os.makedirs(augmented_dir)

print("데이터 증강 중...")
augment_and_save(
    image_dir=train_image_dir,
    label_dir=train_label_output_dir,
    output_dir="../../dataset/train_aug/",
    json_output_path="../../dataset/train_aug.json",
    model_path="yolo11x.pt",
    blur_ratio=50,
    class_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
)

# 4. 데이터셋 분리 작업 수행
print("데이터셋 분리 중...")
# split_dataset("/data/ephemeral/home/dataset/", test_size=0.2, random_state=42, train_file="train_split.json", val_file="val_split.json")

# 5. YOLO 모델 학습 실행
wandb.init(project="Object Detection")
model = YOLO(model_path)
add_wandb_callback(model)

# 데이터 경로 설정 (train_split.json, val_split.json 사용)
try:
    for batch in data_loader:  # 데이터 로더로부터 배치 데이터 로드
        batch["cls"] = check_and_adjust_dimensions(batch["cls"])

    results = model.train(data='/yolov11/cfg/data.yaml', 
                          epochs=50, 
                          imgsz=512, 
                          batch=16)
except IndexError as e:
    print(f"IndexError 발생: {e}")

wandb.finish()