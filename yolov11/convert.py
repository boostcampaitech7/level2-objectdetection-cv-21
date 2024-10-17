import json
import os

def convert_yolo(coco_json, output_dir, img_dir):
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)
    
    # 각 이미지에 대해 YOLO 형식의 텍스트 파일 생성
    for img in coco_data['images']:
        img_id = img['id']
        img_filename = img['file_name']
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
        
        with open(os.path.join(output_dir, f"{os.path.splitext(img_filename)[0]}.txt"), 'w') as file:
            for ann in annotations:
                category_id = ann['category_id'] - 1  # YOLO에서는 클래스 ID가 0부터 시작
                bbox = ann['bbox']
                x_center = (bbox[0] + bbox[2] / 2) / img['width']
                y_center = (bbox[1] + bbox[3] / 2) / img['height']
                width = bbox[2] / img['width']
                height = bbox[3] / img['height']
                file.write(f"{category_id} {x_center} {y_center} {width} {height}\n")

# COCO 형식의 train.json 파일을 YOLO 형식으로 변환
convert_yolo("/data/ephemeral/home/dataset/train.json", "yolo_labels", "train")