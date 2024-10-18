import csv
import os

# 예측 실행
results = model.predict(source="test", conf=0.25)

# 모델명을 기반으로 submission 파일명 생성
model_name = "yolo11x"
submission_file = f"submission_{model_name}.csv"

# submission_{model_name}.csv 생성
with open(submission_file, mode="w") as file:
    writer = csv.writer(file)
    writer.writerow(["image_id", "PredictionString"])
    
    for result in results:
        img_id = os.path.splitext(result.path.split('/')[-1])[0]
        prediction_string = ""
        
        for pred in result.pred:
            cls, conf, xmin, ymin, xmax, ymax = pred
            # Confidence 스코어 자리수 제한 제거
            prediction_string += f"{int(cls)} {conf} {xmin} {ymin} {xmax} {ymax} "
        
        writer.writerow([img_id, prediction_string.strip()])

print(f"Submission file saved as: {submission_file}")