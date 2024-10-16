import json
import os

def convert_coco_to_yolo(json_path, img_folder):
    with open(json_path) as f:
        data = json.load(f)

    for img in data['images']:
        img_id = img['id']
        img_file = img['file_name']
        width = img['width']
        height = img['height']

        # 이미지와 동일한 이름의 .txt 파일 생성
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(img_folder, label_file)

        with open(label_path, 'w') as label_f:
            for ann in data['annotations']:
                if ann['image_id'] == img_id:
                    category_id = ann['category_id'] - 1  # 클래스 ID가 1부터 시작한다고 가정
                    bbox = ann['bbox']

                    # YOLO 형식으로 변환 (중심 좌표 및 너비, 높이를 정규화)
                    x_center = (bbox[0] + bbox[2] / 2) / width
                    y_center = (bbox[1] + bbox[3] / 2) / height
                    w = bbox[2] / width
                    h = bbox[3] / height

                    label_f.write(f"{category_id} {x_center} {y_center} {w} {h}\n")

# 학습 및 테스트 데이터를 YOLO 형식으로 변환
convert_coco_to_yolo('dataset/train.json', 'dataset/train')
convert_coco_to_yolo('dataset/test.json', 'dataset/test')