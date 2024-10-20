import json
import argparse
import pandas as pd
from datetime import datetime
import os
from pathlib import Path
from typing import Dict, List

def parse_prediction_string(prediction_string: str) -> List[Dict]:
    """예측 문자열을 파싱하여 바운딩 박스 정보로 변환"""
    boxes = []
    if isinstance(prediction_string, str):
        predictions = prediction_string.strip().split()
        for i in range(0, len(predictions), 6):
            if i + 6 <= len(predictions):
                box = {
                    'category_id': int(predictions[i]),
                    'confidence': float(predictions[i + 1]),
                    'bbox': [
                        float(predictions[i + 2]),  # x_min
                        float(predictions[i + 3]),  # y_min
                        float(predictions[i + 4]) - float(predictions[i + 2]),  # width
                        float(predictions[i + 5]) - float(predictions[i + 3])   # height
                    ]
                }
                boxes.append(box)
    return boxes

def convert_to_coco_format(csv_path: str, confidence_threshold: float = 0.5) -> Dict:
    """CSV 예측 결과를 COCO 형식으로 변환"""

    # COCO 형식 기본 구조
    coco_format = {
        "info": {
            "year": 2021,
            "version": "1.0",
            "description": "Recycle Trash",
            "contributor": "Upstage",
            "url": None,
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [{
            "id": 0,
            "name": "CC BY 4.0",
            "url": "https://creativecommons.org/licenses/by/4.0/deed.ast"
        }],
        "images": [],
        "annotations": []
    }

    # CSV 파일 읽기
    df = pd.read_csv(csv_path)
    annotation_id = 0

    # 각 이미지에 대해 처리
    for idx, row in df.iterrows():
        # 이미지 정보 추가
        image_info = {
            "width": 1024,  # 고정된 이미지 크기
            "height": 1024,
            "file_name": row['image_id'],
            "license": 0,
            "flickr_url": None,
            "coco_url": None,
            "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "id": int(row['image_id'].split('/')[-1].split('.')[0])  # 파일명에서 ID 추출
        }
        coco_format["images"].append(image_info)

        # 예측 결과 파싱 및 어노테이션 추가
        boxes = parse_prediction_string(row['PredictionString'])
        for box in boxes:
            if box['confidence'] >= confidence_threshold:
                annotation = {
                    "image_id": image_info["id"],
                    "category_id": box['category_id'],
                    "area": box['bbox'][2] * box['bbox'][3],
                    "bbox": [
                        round(box['bbox'][0], 1),  # x_min
                        round(box['bbox'][1], 1),  # y_min
                        round(box['bbox'][2], 1),  # width
                        round(box['bbox'][3], 1)   # height
                    ],
                    "iscrowd": 0,
                    "id": annotation_id,
                    "segmentation": [[0, 0, 0, 0, 0, 0, 0, 0]]
                }
                coco_format["annotations"].append(annotation)
                annotation_id += 1

    return coco_format

def save_coco_json(coco_data: Dict, base_dir: str = "predictions") -> str:
    """COCO 형식 데이터를 JSON 파일로 저장"""
    # 저장 디렉토리 생성
    os.makedirs(base_dir, exist_ok=True)

    # 파일명 생성 (타임스탬프 포함)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"predictions_coco_{current_time}.json"

    # 전체 경로 생성
    output_path = os.path.join(base_dir, filename)

    # JSON 파일 저장
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    return output_path

def main():
    parser = argparse.ArgumentParser(description="Convert CSV predictions to COCO format")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file containing predictions")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="Confidence threshold for including predictions")
    parser.add_argument("--output_path", type=str, default="predictions", help="Base directory for saving the output JSON file")
    args = parser.parse_args()

    # COCO 형식으로 변환
    coco_data = convert_to_coco_format(
        csv_path = args.csv_path,
        confidence_threshold = args.confidence_threshold
    )

    # 결과 저장
    output_path = save_coco_json(coco_data)
    print(f"파일이 저장되었습니다: {output_path}")

    # 통계 출력
    print(f"\n총 이미지 수: {len(coco_data['images'])}")
    print(f"총 어노테이션 수: {len(coco_data['annotations'])}")

    # 클래스별 분포 확인
    class_dist = {}
    for ann in coco_data['annotations']:
        class_id = ann['category_id']
        class_dist[class_id] = class_dist.get(class_id, 0) + 1
    print("\n클래스별 분포:")
    for class_id, count in sorted(class_dist.items()):
        print(f"Class {class_id}: {count}")

if __name__ == "__main__":
    main()
