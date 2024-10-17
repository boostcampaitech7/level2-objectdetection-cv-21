import csv

# 예측 실행
results = model.predict(source="test", conf=0.25)

# submission.csv 생성
with open("submission.csv", mode="w") as file:
    writer = csv.writer(file)
    writer.writerow(["image_id", "PredictionString"])
    
    for result in results:
        img_id = os.path.splitext(result.path.split('/')[-1])[0]
        prediction_string = ""
        
        for pred in result.pred:
            cls, conf, xmin, ymin, xmax, ymax = pred
            prediction_string += f"{int(cls)} {conf:.4f} {xmin} {ymin} {xmax} {ymax} "
        
        writer.writerow([img_id, prediction_string.strip()])