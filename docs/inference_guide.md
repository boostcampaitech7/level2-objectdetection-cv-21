# 시작하기

inferece.py는 Wandb서버에 저장된 checkpoint를 불러와 Inference를 진행합니다. 아래는 사용에 필요한 인자와 사용예시를 설명합니다.

## --model_config
학습에 사용된 모델 config를 입력합니다.
default는 'test'이며 trainer에서와 같이 'test_config'에서 '_config'를 빼고 입력합니다.

## --wandb_path
wandb에서 불러올 모델 경로를 설정합니다.
* 사용을 원하는 프로젝트에서 Artifacts - model - '불러올 모델 선택' - Usage를 들어갑니다.
* 3번째 줄 `artifact = run.use_artifact('cv-21/Object Detection/run_testv1_model:v0', type='model')` 에서 `'cv-21/Object Detection/run_testv1_model:v0'`부분을 복사하여 입력합니다.
* 이때 반드시 따옴표(' ')를 함께 복사해주세요.

## --epoch_number
`--wandb_path`경로에 저장된 체크포인트 .pth 파일의 epoch 번호를 입력합니다.
* 사용을 원하는 프로젝트에서 Artifacts - model - '불러올 모델 선택' - Files 에서 epoch 번호를 확인해주세요.
* 이때 epoch뒤에 숫자만 입력합니다.

### 사용 방법
'--wandb_path'와 '--epoch_number'의 기본설정은 'None'입니다. 때문에 입력하지 않으면 코드는 작동하지 않습니다.

사용 예시입니다.

```python inference.py --model_config test --wandb_path 'cv-21/test/run_avv4188o_model:v1' --epoch_number 20```

## 추가로 볼 만한 문서

* [tmux를 사용한 백그라운드 트레이닝](using_tmux_for_background_training.md)
