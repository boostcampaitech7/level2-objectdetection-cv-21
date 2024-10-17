import json
import os

def convert_yolo(coco_json, label_output_folder):
    # 라벨 저장할 폴더가 존재하지 않으면 생성
    if not os.path.exists(label_output_folder):
        os.makedirs(label_output_folder)

    # COCO JSON 파일 로드
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)

    # 각 이미지에 대해 YOLO 형식의 텍스트 파일 생성
    for img in coco_data['images']:
        img_id = img['id']
        img_file = img['file_name']
        width = img['width']
        height = img['height']

        # YOLO 라벨 파일을 저장할 경로 설정 (label_output_folder 바로 아래에 저장)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(label_output_folder, label_file)

        # 디렉토리 확인 및 생성 (파일 저장 전 경로가 없을 경우 생성)
        label_dir = os.path.dirname(label_path)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        # 해당 이미지에 대한 라벨 파일 생성
        with open(label_path, 'w') as label_f:
            for ann in coco_data['annotations']:
                if ann['image_id'] == img_id:
                    category_id = ann['category_id'] - 1  # YOLO에서는 클래스 ID가 0부터 시작
                    if category_id >= 0:  # 클래스 ID가 0 이상인 경우에만 저장
                        bbox = ann['bbox']

                        # YOLO 형식으로 변환 (중심 좌표 및 너비, 높이를 정규화)
                        x_center = (bbox[0] + bbox[2] / 2) / width
                        y_center = (bbox[1] + bbox[3] / 2) / height
                        w = bbox[2] / width
                        h = bbox[3] / height

                        # YOLO 형식으로 라벨 파일 작성
                        label_f.write(f"{category_id} {x_center} {y_center} {w} {h}\n")

# 학습 및 테스트 데이터를 YOLO 형식으로 변환
train_json_path = "/data/ephemeral/home/dataset/train.json"
train_label_output_dir = "/data/ephemeral/home/dataset/labels/train"

# 최종 변환된 json 파일명 변경
train_aug_json_path = "/data/ephemeral/home/dataset/train_aug.json"

# 변환 실행
convert_yolo(train_json_path, train_label_output_dir)

# 변환된 파일은 train_aug.json 또는 적절한 이름으로 저장
print(f"YOLO 형식으로 변환된 라벨이 {train_label_output_dir}에 저장되었습니다. 변환된 JSON 파일은 {train_aug_json_path}에 저장됩니다.")