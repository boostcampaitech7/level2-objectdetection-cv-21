import json
import os

def convert_yolo(coco_json, image_folder):
    """
    COCO 형식의 JSON 파일을 YOLO 형식의 .txt 파일로 변환하는 함수입니다.
    :param coco_json: COCO 형식의 JSON 파일 경로
    :param image_folder: 이미지가 저장된 폴더 경로 (라벨 파일도 이 폴더에 저장)
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

        # YOLO 라벨 파일을 저장할 경로 설정 (이미지 파일과 같은 폴더에 저장)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(image_folder, label_file)

        # 해당 이미지에 대한 라벨 파일 생성
        with open(label_path, 'w') as label_f:
            for ann in coco_data['annotations']:
                if ann['image_id'] == img_id:
                    category_id = ann['category_id'] - 1  # YOLO에서는 클래스 ID가 0부터 시작
                    if category_id < 0:
                        print(f"Skipping annotation with invalid category_id: {category_id}")
                        continue

                    bbox = ann['bbox']
                    if bbox[2] <= 0 or bbox[3] <= 0:  # 너비나 높이가 0 이하인 경우 필터링
                        print(f"Skipping annotation with invalid bbox: {bbox}")
                        continue

                    # YOLO 형식으로 변환 (중심 좌표 및 너비, 높이를 정규화)
                    x_center = (bbox[0] + bbox[2] / 2) / width
                    y_center = (bbox[1] + bbox[3] / 2) / height
                    w = bbox[2] / width
                    h = bbox[3] / height

                    # YOLO 형식으로 라벨 파일 작성
                    label_f.write(f"{category_id} {x_center} {y_center} {w} {h}\n")

    print(f"YOLO 형식으로 변환된 라벨이 {image_folder}에 저장되었습니다.")