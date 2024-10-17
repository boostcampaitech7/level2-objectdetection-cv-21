import wandb
from wandb.integration.ultralytics import add_wandb_callback
from ultralytics import YOLO

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