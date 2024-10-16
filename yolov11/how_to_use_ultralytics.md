#### 1. 필요한 라이브러리 설치
```bash
pip install opencv-python ultralytics
```


#### 2. COCO 형식 주석을 YOLO 형식으로 변환
COCO 형식의 `train.json` 및 `test.json` 파일을 YOLO 형식의 `.txt` 파일로 변환합니다.
(COCO 형식에서 클래스 ID는 1부터 시작하기 때문에 YOLO에서 사용하는 0부터 시작하는 방식으로 맞추기 위해 -1을 해줍니다.)

명령어
```bash
python /data/ephemeral/home/cv-21/yolov11/utils/convert_json_to_yolo.py
```

![](https://i.imgur.com/pjlwy87.png)

![](https://i.imgur.com/bsMaaPI.png)


YOLO 형식, 각 이미지마다 `.txt` 파일이 생성되었습니다.

데이터 출력 확인은 변환된 `.txt` 파일에 실제로 클래스 ID와 바운딩 박스 정보가 제대로 저장되었는지 확인하면 됩니다.

변환이 제대로 완료되면 `.txt` 파일 예시입니다. `0 0.5 0.5 0.2 0.3 1 0.4 0.6 0.1 0.1`
각 줄은 개별 객체의 정보를 나타내며, 형식은 `[class_id] [x_center] [y_center] [width] [height]` 입니다.
- `x_center`, `y_center`: 바운딩 박스의 중심 좌표 (이미지 크기에 정규화된 값, 즉 0~1 사이의 값)
- `width`, `height`: 바운딩 박스의 너비와 높이 (역시 이미지 크기에 정규화된 값)

위와 같은 포맷으로 클래스 ID와 바운딩 박스 정보가 저장되어야 모델이 정상적으로 학습할 수 있습니다.

`<class_id> <x_center> <y_center> <width> <height>`

명령어
```bash
python train.py --data config/dataset.yaml --weights models/yolo11x.pt --epochs 50 --img-size 512 --batch-size 16
```
`YOLO`와 `MMDetection`의 구조가 다르기 때문에, `inference.py`를 YOLO에 맞게 수정할 필요가 있습니다. YOLOv11을 기반으로 inference 및 CSV 파일 생성을 위한 코드를 작성하려면, 모델을 불러오고, 테스트 데이터를 처리한 후, 결과를 CSV 파일로 출력하는 부분을 구성해야 합니다.

MMDetection에서는 `single_gpu_test`와 같은 유틸리티 함수가 제공되지만, YOLO에서는 직접 모델을 사용하여 예측값을 추론하고 결과를 파일로 저장하는 방식을 사용해야 합니다. YOLO의 경우 보통 `detect.py`나 비슷한 스크립트를 사용하여 추론을 수행합니다.

아래는 YOLOv11의 추론 결과를 CSV 파일로 저장하는 Python 코드 예시입니다.

### 3. `detect.py` (추론 결과 CSV로 저장)
YOLOv11을 사용해 추론을 수행하고 그 결과를 CSV 파일로 저장

- `detect_and_save_csv`: 이 함수는 모델을 불러오고 이미지를 로드한 후 추론을 수행하고 결과를 CSV 파일로 저장
- 결과 형식 결과는 이미지 파일 이름, 클래스, 신뢰도, 중심 좌표(x, y) 및 객체의 너비, 높이를 포함하는 CSV 파일로 저장
- 파라미터
    - `weights`: 추론에 사용할 YOLO 모델의 가중치 파일 경로.
    - `source`: 추론할 이미지나 비디오 파일의 경로.
    - `img_size`: 이미지 크기.
    - `conf_thres`: 객체 검출의 confidence threshold.
    - `iou_thres`: NMS(Non-Maximum Suppression)를 적용할 때의 IoU threshold.
    - `save_csv_path`: 결과를 저장할 CSV 파일 경로.

명령어
```bash
python detect.py --weights weights/best.pt --source data/test --save-csv-path output.csv
```