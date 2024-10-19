# 시작하기

trainer_custom.py는 기존의 trainer.py와 다르게 다양한 인자들을 입력받아 Customizing option을 제공합니다. 아래는 사용에 필요한 인자와 사용예시를 설명합니다.

## (Optional) --max_epoch
학습을 진행할 최대 epoch 수를 설정합니다. 기본값은 25 입니다.
* 이때 epoch뒤에 숫자만 입력합니다.

## (Required) --model_name
학습을 진행할 model name을 입력합니다. 사용 가능한 모델의 이름들은 "config"에서 참고해주세요.
* 사용 예시 : "cascade_rcnn_config.py" -> "cascade_rcnn" 입력

## (Optional) --inf_path
학습에 진행할 inference file의 경로입니다. 만약 pseudo labeling 등이 적용된 새로운 inference file을 학습에 이용하고 싶은 경우에, 이를 지정할 수 있습니다.

## (Optional) --pretrained_artifact
WanDB에서 불러올 checkpoint(artifact)경로를 지정합니다.
* 사용을 원하는 프로젝트에서 Artifacts - model - '불러올 모델 선택' - Usage를 들어갑니다.
* 3번째 줄 `artifact = run.use_artifact('cv-21/Object Detection/run_testv1_model:v0', type='model')` 에서 `'cv-21/Object Detection/run_testv1_model:v0'`부분을 복사하여 입력합니다.
* 이때 반드시 따옴표(' ')를 함께 복사해주세요.


### 사용 방법
'--wandb_path'와 '--epoch_number'의 기본설정은 'None'입니다. 때문에 입력하지 않으면 코드는 작동하지 않습니다.

사용 예시입니다.

```python inference.py --model_config test --wandb_path 'cv-21/test/run_avv4188o_model:v1' --epoch_number 20```

## 추가로 볼 만한 문서

* [tmux를 사용한 백그라운드 트레이닝](using_tmux_for_background_training.md)
