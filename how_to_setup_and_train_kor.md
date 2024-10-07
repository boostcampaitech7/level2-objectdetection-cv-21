# 시작하기
===============

이 프로젝트는 Python 환경에 특정 종속성을 설치해야 합니다. 다음 단계를 따라 환경을 설정하고 `train.py` 파일을 실행합니다.

## 1단계: 필요한 종속성 설치
먼저, 필요한 종속성을 설치해야 합니다.

### Boostcamp 서버 사용자
이미 설치된 CUDA, PyTorch 및 Torchvision이 있으므로 이러한 종속성을 설치할 필요가 없습니다. 그러나 이 프로젝트에 일부 종속성을 설치해야 합니다. 다음 스텝을 따라하십시오.
1. ```apt-get update -y&&apt-get install -y libgl1-mesa-glx&&apt-get install -y libglib2.0-0&&apt-get install wget -y&&cd ~&&wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000325/data/20240902115340/code.tar.gz&&wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000325/data/data.tar.gz&&tar -zxvf data.tar.gz&&tar --strip-components=1 -xzf code.tar.gz```
2. requirements.txt의 마지막 줄을 지운다.
3. ```pip install -r requirements.txt&&mim install mmcv-full==1.7.0&&apt install git-all -y&&pip install lightning==2.1 torch==1.12.1&&pip install "ray[data,train,tune,serve]" wandb&&pip install protobuf==3.19.6&&apt-get install tmux -y&&cd ~/mmdetection&&pip install -v -e .cd /home&&mkdir (자기 이름)&&cd (자기 이름)&&mkdir proj2&&cd proj2```

이 명령어는 MMDetection, Detectron2, WandB 등의 필요한 종속성을 설치합니다. **새로운 Conda 환경을 Boostcamp 서버에서 만들지 마세요. CUDA는 새 환경에서 설치할 수 없습니다.**

## 2단계: Weights & Biases 설정
Weights & Biases 계정을 설정합니다. Weights & Biases 설정 방법에 대한 자세한 내용은 [여기](https://docs.wandb.ai/ko/quickstart)를 참조하세요.

## 3단계: trainer.py 파일 실행
종속성이 설치되면 Python을 사용하여 각 패키지 별 `trainer.py` 파일을 실행할 수 있습니다.

### 기본 설정
인수를 지정하지 않으면 스크립트는 기본 설정을 사용합니다.

```python trainer.py```

이 명령어는 기본 설정으로 모델을 훈련합니다.

### 사용자 정의 설정
대신 인수를 지정하여 기본 설정을 재정의할 수도 있습니다.

```python train.py --model-name <모델_이름> --num_gpus <GPU_개수> --smoke_test```

## 추가로 볼 만한 문서

* [tmux를 사용한 백그라운드 트레이닝](using_tmux_for_background_training.md)
