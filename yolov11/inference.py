import os
import csv
from ultralytics import YOLO

# 학습된 모델 경로 설정
model_path = "/data/ephemeral/home/github/yolov11/runs/detect/train/weights/best.pt"

# 모델 로드
model = YOLO(model_path)

# 테스트 이미지 경로 설정
test_dir = "/data/ephemeral/home/dataset/test"  # 테스트 이미지가 있는 디렉토리

# 예측 실행
results = model.predict(source=test_dir, conf=0.25)

# 결과 저장 및 출력
submission_file = "submission_yolo11x.csv"
with open(submission_file, mode="w") as file:
    writer = csv.writer(file)
    writer.writerow(["image_id", "PredictionString"])
    
    for result in results:
        img_id = os.path.splitext(result.path.split('/')[-1])[0]
        prediction_string = ""
        
        for pred in result.pred:
            cls, conf, xmin, ymin, xmax, ymax = pred
            prediction_string += f"{int(cls)} {conf} {xmin} {ymin} {xmax} {ymax} "
        
        writer.writerow([img_id, prediction_string.strip()])

print(f"Submission file saved as: {submission_file}")