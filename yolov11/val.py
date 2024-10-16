import argparse
from ultralytics import YOLO

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='config/yolov11_config.yaml', help='데이터셋 및 설정 파일 경로')
    parser.add_argument('--weights', type=str, default='models/yolo11x.pt', help='모델 가중치 파일 경로')
    parser.add_argument('--img-size', type=int, default=640, help='이미지 크기')
    opt = parser.parse_args()
    return opt

def validate(opt):
    # YOLO 모델 불러오기
    model = YOLO(opt.weights)

    # 검증 수행
    metrics = model.val(data=opt.data, imgsz=opt.img_size)
    print(metrics)

if __name__ == "__main__":
    opt = parse_opt()
    validate(opt)