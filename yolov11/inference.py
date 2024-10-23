import os
import csv
import wandb
from convert import convert_yolo  # convert.py 파일에서 convert_yolo 함수 가져오기
from ultralytics import RTDETR

def download_best_checkpoint(wandb_path: str, output_dir: str) -> str:
    """wandb에서 best checkpoint를 다운로드하고 파일 경로를 반환합니다."""
    # run = wandb.init()
    # artifact = run.use_artifact(wandb_path, type='model')
    # artifact_dir = artifact.download() #wandb 안쓰면 체크포인트 경로 쓰기
    
    # best checkpoint 파일 검색
    checkpoint_path = '/data/ephemeral/home/github/yolov11/runs/detect/train/weights/best.pt'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found in artifact.")
    
    # 지정된 출력 디렉토리로 이동
    os.makedirs(output_dir, exist_ok=True)
    final_checkpoint_path = os.path.join(output_dir, 'rtdetr_best.pt')
    os.rename(checkpoint_path, final_checkpoint_path)
    
    return final_checkpoint_path

def main():
    # COCO 형식의 JSON을 YOLO 형식의 라벨로 변환
    coco_json = "/data/ephemeral/home/dataset/test.json"  # COCO JSON 파일 경로
    test_dir = "/data/ephemeral/home/dataset/test"  # 테스트 이미지가 있는 디렉토리
    valid_classes = list(range(10))  # 0부터 9까지의 클래스 설정

    convert_yolo(coco_json, test_dir, valid_classes)  # COCO 형식의 JSON 파일을 YOLO 형식으로 변환
    valid_classes = list(range(10))  # 0부터 9까지의 클래스 설정

    convert_yolo(coco_json, test_dir, valid_classes)  # COCO 형식의 JSON 파일을 YOLO 형식으로 변환

    # wandb에서 best checkpoint 다운로드
    wandb_path = 'your-wandb-entity/your-project/your-artifact:latest'  # wandb artifact 경로
    output_dir = '/data/ephemeral/home/github/yolov11/runs/detect'  # checkpoint를 저장할 디렉토리
    model_path = download_best_checkpoint(wandb_path, output_dir)

    # 모델 로드
    model = RTDETR(model_path)

    # 클래스 매핑 (data.yaml에 있는 순서대로 매핑)
    class_names = ['General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']

    # 예측 실행
    results = model.predict(source=test_dir, conf=0.25)

    # 결과 저장 및 출력
    submission_file = "submission_yolo11x.csv"
    with open(submission_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["PredictionString", "image_id"])  # 헤더 순서 변경

        for result in results:
            img_id = os.path.join("test", os.path.basename(result.path))  # 이미지 파일명에 디렉토리 포함
            img_id = os.path.join("test", os.path.basename(result.path))  # 이미지 파일명에 디렉토리 포함
            prediction_list = []

            for pred in result.boxes:
                cls = int(pred.cls)  # 예측된 클래스 ID
                cls = int(pred.cls)  # 예측된 클래스 ID
                conf = float(pred.conf)  # 신뢰도

                # 클래스 ID가 유효한 범위인지 확인
                if cls < 0 or cls >= len(class_names):
                    print(f"잘못된 클래스 ID: {cls}")
                    continue  # 잘못된 클래스 ID 건너뜀

                # 예측 결과가 있는지 확인하고 변환
                if pred.xywh is not None and len(pred.xywh) > 0:
                    x_center, y_center, width, height = map(float, pred.xywh[0].cpu().numpy())  # YOLO 형식 좌표

                # 클래스 ID가 유효한 범위인지 확인
                if cls < 0 or cls >= len(class_names):
                    print(f"잘못된 클래스 ID: {cls}")
                    continue  # 잘못된 클래스 ID 건너뜀

                # 예측 결과가 있는지 확인하고 변환
                if pred.xywh is not None and len(pred.xywh) > 0:
                    x_center, y_center, width, height = map(float, pred.xywh[0].cpu().numpy())  # YOLO 형식 좌표

                    # YOLO 형식을 Pascal VOC 형식으로 변환
                    xmin = x_center - (width / 2)
                    ymin = y_center - (height / 2)
                    xmax = x_center + (width / 2)
                    ymax = y_center + (height / 2)

                    # 변환된 좌표를 prediction_list에 추가
                    prediction_list.append((cls, conf, xmin, ymin, xmax, ymax))

            # 예측 결과가 있는지 확인한 후에 처리
            if prediction_list:
                # 클래스 ID 오름차순으로 정렬
                prediction_list.sort(key=lambda x: x[0])
            # 예측 결과가 있는지 확인한 후에 처리
            if prediction_list:
                # 클래스 ID 오름차순으로 정렬
                prediction_list.sort(key=lambda x: x[0])

                # 정렬된 결과를 PredictionString 형식으로 변환
                prediction_string = " ".join(
                    f"{cls} {conf:.6f} {xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f}"
                    for cls, conf, xmin, ymin, xmax, ymax in prediction_list
                )
            else:
                # 예측 결과가 없을 경우 빈 문자열로 설정
                prediction_string = ""

            # PredictionString을 CSV 파일에 저장
            writer.writerow([prediction_string.strip(), img_id])
            # PredictionString을 CSV 파일에 저장
            writer.writerow([prediction_string.strip(), img_id])

        print(f"Submission file saved as: {submission_file}")
        print(f"Submission file saved as: {submission_file}")


if __name__ == "__main__":
    main()