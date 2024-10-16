import argparse
import cv2
from ultralytics import YOLO

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='models/yolo11x.pt', help='모델 가중치 파일 경로')
    parser.add_argument('--img-path', type=str, required=True, help='입력 이미지 파일 경로')
    parser.add_argument('--output-path', type=str, default='output.jpg', help='결과 이미지 저장 경로')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='탐지 신뢰도 임계값')
    opt = parser.parse_args()
    return opt

def predict(opt):
    # YOLO 모델 불러오기
    model = YOLO(opt.weights)

    # 이미지 불러오기
    img = cv2.imread(opt.img_path)

    # 예측 수행
    results = model(img, conf=opt.conf_thres)

    # 탐지 결과 이미지 저장
    results.save(opt.output_path)
    print(f"결과가 {opt.output_path}에 저장되었습니다.")

if __name__ == "__main__":
    opt = parse_opt()
    predict(opt)