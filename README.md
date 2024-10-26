# [AI Tech 7th CV21]
![](https://i.imgur.com/4UZzVAP.png) 
## content
- [Overview](#Overview)
- [Member](#Member)
- [Collaboration](#Collaboration)   
- [Approach](#Approach)
- [EDA](#EDA)
- [Wrap Reports](#Wrap Reports)
- [File Tree](#filetree)
- [Usage](#Code)


<br></br>
## Overview <a id = 'Overview'></a>
![first](https://github.com/user-attachments/assets/74c70c01-c7c7-4240-9c73-64c78e1b246a)

다양한 종류의 쓰레기(일반 쓰레기, 플라스틱, 종이, 유리 등 총 10가지)를 포함한 이미지 데이터셋을 활용했다. 데이터셋은 COCO 포맷의 바운딩 박스 정보(좌표, 카테고리)를 포함하고 있어, 학습 시 정확한 쓰레기 위치와 종류를 인식하도록 모델을 훈련시켰다. 모델의 출력값은 바운딩 박스 좌표, 카테고리, 그리고 신뢰도 점수(score)를 포함하며, 이 값을 기반으로 평가가 이루어졌다.

<br></br>
## Member <a id = 'Member'></a>

|김성주|김보현|윤남규|정수현|김한얼|허민석|
|:--:|
|![week8](https://github.com/user-attachments/assets/3d0a21fb-9210-41b6-aba3-fc9588ea8f60)|

<br></br>
## Collaboration <a id = 'Collaboration'></a>
![image](https://github.com/user-attachments/assets/057afead-a676-4f4b-9979-b5da50cef14b)
<br></br>

## Approach <a id = 'Approach'></a>
### Architecture
- Faster R-CNN
- Cascade R-CNN
- YOLO v11x
- YOLO v11n
- YOLO World
- RT-DETR
- ContralModel(Diffusion Model)
- DiffusionDet
- SWIN

### Backbone model
- ResNet(TorchVision, Faster R-CNN)
- Swin Transformer(MMDetection)
- One Stage detector(Ultralytics)

<br></br>

## EDA <a id = 'Result'></a>
![labels](https://github.com/user-attachments/assets/f3e551bd-1b4e-41ee-ae34-78d37ffd363c)

<br></br>
## Wrap Reports <a id = 'Wrap Reports'></a>

<br></br>

## File Tree <a id = 'filetree'></a>
```
level2-objectdetection-cv-21
|
|── tools
|    |- pseudo_labeling
|    |- check_image.py
|    |- ensemble.py
|    └─ mAP_from_csv.py
|
|── mmdetection(v2)
|    |- configs
|    |- inference.py
|    |- inference_on_val_set.py
|    |- testTTA.py
|    └─ trainer.py
|
|── mmdetection(v3)
|    |- configs
|    .
|    .
|    └─ tools
|        |- train.py
|        └─ test.py
|
|── yolov11
|    |- cfg
|    |- augmentation.py
|    |- convert.py
|    |- inference.py
|    |- split.py
|    |- streamlit.py
|    └─ train.py
|
└── README.md
```

<br></br>
## Usage <a id = 'Code'></a>

### Package install

### Model Train & Inference
- [Yolov8](./docs/Yolov8.md)
- [MMDetection(v2)](./docs/MMDetection(v2).md)
- [MMDetection(v3)](./docs/MMDetection(v3).md)

### Extra Tools
- [Tools](./docs/Tools.md)
    1. Annotation Analysis (EDA)
    2. Clean Supervisely
    3. Stratified Group K Fold
    4. Submission Visualization

### [WBF ensemble](./docs/wbf_ensemble.md)
