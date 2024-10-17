import os
import shutil
import cv2
from ultralytics import YOLO, solutions
from ultralytics.utils.plotting import Annotator, colors

def augment_and_save(image_dir, label_dir, output_dir, model_path, blur_ratio=50, class_indices=None):
    """이미지에 객체 카운팅 및 블러링을 적용하고 증강된 이미지를 같은 디렉토리에 저장합니다."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 모델 로드
    model = YOLO(model_path)

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, image_name.replace('.jpg', '.txt'))  # 라벨 파일 경로
        img = cv2.imread(image_path)
        assert img is not None, f"이미지를 읽을 수 없습니다: {image_path}"

        # 객체 탐지 수행 (특정 클래스만 필터링)
        results = model.predict(img, show=False, classes=class_indices)  # classes 옵션 추가
        boxes = results[0].boxes.xyxy.cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()

        annotator = Annotator(img, line_width=2, example=model.names)

        # 각 객체에 대해 블러 처리 및 카운팅 수행
        for box, cls in zip(boxes, clss):
            annotator.box_label(box, color=colors(int(cls), True), label=model.names[int(cls)])

            # 객체 블러 처리 (데이터 증강 효과)
            obj = img[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
            blur_obj = cv2.blur(obj, (blur_ratio, blur_ratio))
            img[int(box[1]): int(box[3]), int(box[0]): int(box[2])] = blur_obj

        # 증강된 이미지 파일명 변경 (예: 원본이 image.jpg인 경우 image_aug.jpg로 저장)
        augmented_image_name = image_name.replace('.jpg', '_aug.jpg')
        output_image_path = os.path.join(output_dir, augmented_image_name)
        cv2.imwrite(output_image_path, img)

        # 라벨 파일 복사 및 파일명 변경
        if os.path.exists(label_path):
            augmented_label_name = image_name.replace('.jpg', '_aug.txt')
            output_label_path = os.path.join(output_dir, augmented_label_name)
            shutil.copy(label_path, output_label_path)
        else:
            print(f"라벨 파일을 찾을 수 없습니다: {label_path}")

    print(f"증강된 이미지와 라벨 파일이 {output_dir}에 저장되었습니다.")

# 증강 실행 (10개의 클래스만 처리되도록 class_indices 설정)
augment_and_save(
    "/data/ephemeral/home/dataset/train/",
    "/data/ephemeral/home/dataset/train/",  # 원본 라벨과 동일 경로
    "/data/ephemeral/home/dataset/train/",  # 증강된 파일도 동일 경로
    "yolo11x.pt",
    blur_ratio=50,
    class_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10개의 클래스만 사용
)