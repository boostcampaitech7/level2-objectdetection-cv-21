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
with open(submission_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["image_id", "PredictionString"])  # 헤더 작성
    
    for result in results:
        img_id = os.path.splitext(os.path.basename(result.path))[0]  # 이미지 파일명에서 확장자 제거
        prediction_string = ""
        
        for pred in result.boxes:
            cls = int(pred.cls)  # 클래스 ID
            conf = float(pred.conf)  # 신뢰도
            xmin, ymin, xmax, ymax = map(float, pred.xyxy)  # 바운딩 박스 좌표 (Pascal VOC 형식)
            
            # Pascal VOC 포맷에 맞게 (label, score, xmin, ymin, xmax, ymax) 형식으로 저장
            prediction_string += f"{cls} {conf} {xmin} {ymin} {xmax} {ymax} "
        
        # PredictionString이 비어 있으면 빈 문자열로 저장
        writer.writerow([img_id, prediction_string.strip() if prediction_string else ""])

print(f"Submission file saved as: {submission_file}")