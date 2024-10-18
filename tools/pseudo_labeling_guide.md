- "pseudo_labeling.py"는 Inference를 불러와 pseudo_labeling 작업을 진행하고, 새로운 inference file을 생성합니다.
- 이 파일을 작동시키기 위해서는 "inference.py"를 이용해서 불러온 inference file이 있어야만 합니다.
- 아래는 사용에 필요한 인자와 사용예시를 설명합니다.

# --input_dir
Pseudo labeling 작업을 수행할 inference file의 경로입니다.

# --output_path
Pseudo labeling 작업을 수행한 inference file의 저장 경로입니다.


# 사용 방법
'--input_dir'와 '--input_dir'의 기본설정은 'None'입니다. 때문에 입력하지 않으면 코드는 작동하지 않습니다.

사용 예시입니다.

```python pseudo_labeling.py --input_dir test.csv --output_path ./dataset```

# 추가로 볼 만한 문서

* [tmux를 사용한 백그라운드 트레이닝](using_tmux_for_background_training.md)
* [inference guilde.md](https://github.com/boostcampaitech7/level2-objectdetection-cv-21/blob/main/docs/inference_guide.md)
* [Pseudo Labeling Introduction](https://github.com/boostcampaitech7/level2-objectdetection-cv-21/blob/feat-76/Pseudo-Labeling/tools/Pseudo_Labeling_Introduction.md)
