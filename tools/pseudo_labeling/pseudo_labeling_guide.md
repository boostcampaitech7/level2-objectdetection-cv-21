# Introduction
- "pseudo_labeling.py"는 Inference를 불러와 confidence pseudo_labeling 작업을 진행하고, 새로운 inference file을 생성합니다.
- 이 파일을 작동시키기 위해서는 "inference.py"를 이용해서 불러온 inference file이 있어야만 합니다.
- 아래는 작동 원리, 사용에 필요한 인자, 그리고 사용예시를 설명합니다.
 
# About Confidence Pseudo Labeling
- Object Detection model이 예측한 BBox의 confidence(존재 가능성) 값이 0.35(threshold)보다 낮은 경우에, 이를 inference file에서 제거하는 작업을 수행합니다.
- 이는 존재 가능성이 낮은 BBox를 제거함으로써, 모델이 보다 confident BBox에만 집중하도록 돕습니다.
- CSV 파일 전체를 순회하는 작업을 수행하므로, 시간 복잡도는 O(n)과 같습니다.

# --input_dir
Pseudo labeling 작업을 수행할 inference file의 경로입니다.

# --output_path
Pseudo labeling 작업을 수행한 inference file의 저장 경로입니다.


# 사용 방법
'--input_dir'와 '--output_path'의 기본설정은 'None'입니다. 때문에 입력하지 않으면 코드는 작동하지 않습니다.

사용 예시입니다.

```python pseudo_labeling.py --input_dir test.csv --output_path ./dataset```

# 추가로 볼 만한 문서

* [inference guilde](inference_guide.md)
* [Pseudo Labeling Introduction](Pseudo_Labeling_Introduction.md)
* [Seminar : ](http://dmqm.korea.ac.kr/activity/seminar/402)
