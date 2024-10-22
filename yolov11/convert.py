import os
import json
import wandb

def convert_yolo(coco_json, image_dir, valid_classes):
    """
    COCO 형식의 JSON 파일을 YOLO 형식의 .txt 라벨 파일로 변환하는 함수
    :param coco_json: COCO JSON 파일 경로
    :param image_dir: 이미지와 .txt 파일이 저장될 동일한 디렉토리
    :param valid_classes: 유효한 클래스 ID 리스트
    """
    # COCO JSON 파일 로드
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)

    # 각 이미지에 대해 YOLO 형식의 텍스트 파일 생성
    for img in coco_data['images']:
        img_id = img['id']
        img_file = img['file_name']
        width = img['width']
        height = img['height']

        # 이미지 디렉토리에서 파일 이름 추출
        label_file = os.path.splitext(os.path.basename(img_file))[0] + '.txt'
        label_path = os.path.join(image_dir, label_file)

        # .txt 파일을 저장할 디렉토리가 있는지 확인
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        # 해당 이미지에 대한 라벨 파일 생성
        with open(label_path, 'w') as label_f:
            for ann in coco_data['annotations']:
                if ann['image_id'] == img_id:
                    category_id = ann['category_id']
                    if category_id not in valid_classes:
                        # 유효한 클래스 ID에 포함되지 않는 경우 건너뜀
                        continue

                    # YOLO 형식으로 변환 (중심 좌표 및 너비, 높이를 정규화)
                    bbox = ann['bbox']
                    if bbox[2] <= 0 or bbox[3] <= 0:
                        continue

                    x_center = (bbox[0] + bbox[2] / 2) / width
                    y_center = (bbox[1] + bbox[3] / 2) / height
                    w = bbox[2] / width
                    h = bbox[3] / height

                    # YOLO 형식으로 라벨 파일 작성
                    label_f.write(f"{category_id} {x_center} {y_center} {w} {h}\n")

    print(f"YOLO 형식으로 변환된 라벨이 {image_dir}에 저장되었습니다.")

def filter_predictions(predictions, valid_classes):
    """유효한 클래스만 남도록 예측 필터링"""
    filtered_predictions = {}
    for image_id, pred_string in predictions.items():
        filtered_pred = []
        pred_list = pred_string.strip().split(" ")
        for i in range(0, len(pred_list), 6):
            values = pred_list[i:i+6]
            if len(values) != 6:
                continue
            cls = int(values[0])
            if cls in valid_classes:
                filtered_pred.append(" ".join(values))

        filtered_predictions[image_id] = " ".join(filtered_pred)
    return filtered_predictions

# Sample predictions dictionary for testing
predictions = {
    "image1": "0 0.9 100 100 200 200 11 0.8 150 150 250 250",
    "image2": "1 0.85 50 50 150 150"
}

# WandB에 업로드하기 전에 필터링
valid_classes = list(range(10))  # 0부터 9까지 10개의 클래스 ID
filtered_predictions = filter_predictions(predictions, valid_classes)

# WandB Vision Table 설정
wandb.init(project="Object Detection")
table = wandb.Table(columns=["image_id", "PredictionString"])

for image_id, pred_str in filtered_predictions.items():
    table.add_data(image_id, pred_str)

wandb.log({"Filtered Predictions": table})

# Finish WandB session
wandb.finish()

print("Filtered predictions have been logged to WandB.")