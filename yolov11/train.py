import argparse
from ultralytics import YOLO

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='config/yolov11_config.yaml', help='데이터셋 및 설정 파일 경로')
    parser.add_argument('--weights', type=str, default='models/yolo11x.pt', help='모델 가중치 파일 경로')
    parser.add_argument('--epochs', type=int, default=50, help='학습 에포크 수')
    parser.add_argument('--img-size', type=int, default=640, help='이미지 크기')
    parser.add_argument('--batch-size', type=int, default=16, help='배치 크기')
    parser.add_argument('--resume', action='store_true', help='중단된 학습 이어하기')
    parser.add_argument('--device', default=0, help='사용할 GPU 또는 CPU 설정')
    opt = parser.parse_args()
    return opt

def train(opt):
    # YOLO 모델 불러오기
    model = YOLO(opt.weights)
    
    # 모델 학습
    model.train(
        data=opt.data, 
        epochs=opt.epochs, 
        imgsz=opt.img_size, 
        batch=opt.batch_size, 
        resume=opt.resume,  # 중단된 학습 이어하기
        device=opt.device   # 사용할 GPU 또는 CPU
    )

if __name__ == "__main__":
    opt = parse_opt()
    train(opt)