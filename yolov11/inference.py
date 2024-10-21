import os
import json
import csv
from ultralytics import YOLO


def convert_yolo(coco_json, image_dir):
    """
    COCO 형식의 JSON 파일을 YOLO 형식의 .txt 라벨 파일로 변환하는 함수
    :param coco_json: COCO JSON 파일 경로
    :param image_dir: 이미지와 .txt 파일이 저장될 동일한 디렉토리
    """
    # COCO JSON 파일 로드
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)

    # test 디렉토리에 있는 jpg 파일 목록 가져오기
    image_files = {os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.jpg')}

    # 각 이미지에 대해 YOLO 형식의 텍스트 파일 생성
    for img in coco_data['images']:
        img_id = img['id']
        img_file = os.path.splitext(img['file_name'])[0]  # 확장자 제거한 파일명
        width = img['width']
        height = img['height']

        # 해당 이미지가 test 디렉토리에 있을 때만 변환
        if img_file not in image_files:
            continue

        # .txt 파일 생성 경로 설정
        label_file = img_file + '.txt'
        label_path = os.path.join(image_dir, label_file)

        # .txt 파일을 저장할 디렉토리가 있는지 확인
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        # 해당 이미지에 대한 라벨 파일 생성
        with open(label_path, 'w') as label_f:
            for ann in coco_data['annotations']:
                if ann['image_id'] == img_id:
                    category_id = ann['category_id']  # COCO에서는 1부터 시작하므로 그대로 사용
                    if category_id < 1:
                        continue

                    category_id -= 1  # YOLO 형식에서는 클래스 ID가 0부터 시작

                    bbox = ann['bbox']
                    if bbox[2] <= 0 or bbox[3] <= 0:  # 너비나 높이가 0 이하인 경우 필터링
                        continue

                    # YOLO 형식으로 변환 (중심 좌표 및 너비, 높이를 정규화)
                    x_center = (bbox[0] + bbox[2] / 2) / width
                    y_center = (bbox[1] + bbox[3] / 2) / height
                    w = bbox[2] / width
                    h = bbox[3] / height

                    # YOLO 형식으로 라벨 파일 작성
                    label_f.write(f"{category_id} {x_center} {y_center} {w} {h}\n")

    print(f"YOLO 형식으로 변환된 라벨이 {image_dir}에 저장되었습니다.")

# 학습된 모델 경로 설정
model_path = "/data/ephemeral/home/github/yolov11/runs/detect/train/weights/best.pt"

# 모델 로드
model = YOLO(model_path)

# 테스트 이미지 경로 설정
test_dir = "/data/ephemeral/home/dataset/test"  # 테스트 이미지가 있는 디렉토리
coco_json = "/data/ephemeral/home/dataset/test.json"  # COCO JSON 파일 경로

# COCO 형식의 JSON을 YOLO 형식의 라벨로 변환
convert_yolo(coco_json, test_dir)

# 예측 실행
results = model.predict(source=test_dir, conf=0.25)

# 결과 저장 및 출력
submission_file = "submission_yolo11x.csv"
with open(submission_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["PredictionString", "image_id"])  # 헤더 순서를 바꿔서 작성
    
    for result in results:
        img_id = os.path.splitext(os.path.basename(result.path))[0]  # 이미지 파일명에서 확장자 제거
        prediction_string = ""
        
        for pred in result.boxes:
            cls = int(pred.cls)  # 클래스 ID
            conf = float(pred.conf)  # 신뢰도
            xmin, ymin, xmax, ymax = map(float, pred.xyxy[0].cpu().numpy())  # 바운딩 박스 좌표
            
            # Pascal VOC 포맷에 맞게 (label, score, xmin, ymin, xmax, ymax) 형식으로 저장
            prediction_string += f"{cls} {conf:.6f} {xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f} "
        
        # PredictionString이 비어 있으면 빈 문자열로 저장
        writer.writerow([prediction_string.strip() if prediction_string else "", img_id])

print(f"Submission file saved as: {submission_file}")