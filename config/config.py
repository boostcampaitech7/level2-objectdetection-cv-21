# 필요한 라이브러리를 가져옵니다
from ray import tune


class ModelConfig:
    """
    이 클래스는 모델과 관련된 설정을 저장합니다.
    """

    def __init__(self):
        # 사용하는 모델의 유형입니다 (예: ResNet18, ResNet50 등)
        self.model_name = "ResNet18"  # 기본 모델
        # 모델이 미리 학습되어 있는지 여부입니다
        self.pretrained = True
        # 모델에서 경로를 줄일 확률
        self.drop_path_rate = 0.0
        # 모델의 레이어 수 (현재 사용하지 않음)
        # self.num_layers = ~
        # 모델의 어텐션 헤드의 수 (현재 사용하지 않음)
        # self.num_heads = ~


class TrainingConfig:
    """
    이 클래스는 모델의 학습과 관련된 설정을 저장합니다.
    """

    def __init__(self):
        # 하이퍼파라미터 튜닝에 사용할 배치 크기 목록입니다
        self.batch_size = tune.choice([32, 64]) # nvidia-smi로 gpu 사용량을 관찰하면서 조정해야함
        # 하이퍼파라미터 튜닝에 사용할 학습률 범위입니다
        self.lr = tune.loguniform(1e-5, 2e-4) # 너무 작으면 학습 진행이 안되며, 너무 크면 수렴이 느림. 
        # 하이퍼파라미터 튜닝에 사용할 가중치 감소 범위입니다
        self.weight_decay = tune.loguniform(0.001, 0.1)

        # 학습률 스케줄러와 관련된 파라미터입니다
        # 스케줄러 이름
        self.sched = 'cosine' # cosine, step
        # 워밍업 학습률
        self.warmup_lr = 1e-6
        # 학습률을 워밍업하는 에폭 수
        self.warmup_epochs = 5

        # 믹스업 관련 파라미터
        # 믹스업
        self.mixup = 0.8
        # 컷믹스
        self.cutmix = 1.0
        # 믹스업 확률
        self.mixup_prob = 1.0
        # 믹스업과 컷믹스 동시 사용시 전환될 확률
        self.mixup_switch_prob = 0.5
        
        # 학습 Loss 관련 파라미터
        # Label smoothing 파라미터
        self.smoothing = 0.1
        
        # 자동 혼합 정밀도 사용
        self.use_amp = True  


class DatasetConfig:
    """
    이 클래스는 데이터셋과 관련된 설정을 저장합니다.
    """

    def __init__(self):
        # 데이터셋 경로입니다
        self.data_path = "/data/ephemeral/home/data/"
        # 데이터 변환 유형입니다 (torchvision, alubmentations, autoaugment)
        self.transform_type = "albumentations"
        # 이미지 크기입니다
        self.input_size = 224
        # 데이터 로딩에 사용할 워커의 수입니다
        self.num_workers = 3

        # Augment 관련 파라미터
        # AutoAugment 정책 파라미터
        self.aa = 'rand-m9-mstd0.5-inc1'

        


class ExperimentConfig:
    """
    실험 관련 설정입니다.

    이 클래스는 실험과 관련된 설정을 저장합니다.
    """

    def __init__(self):
        # 실험 결과를 저장할 디렉토리입니다
        self.save_dir = "/data/ephemeral/home/logs/"
        # 학습에 사용할 GPU의 수입니다
        self.num_gpus = 1 # ddp가 불가능한 관계로 항상 1 고정
        # 스케줄링에 사용할 워커의 수입니다
        self.num_workers = 3 # 여러 hparam 조합을 관찰할 목적이라면 3 또는 6을 추천함(메모리 부족시 3)
        # 하이퍼파라미터 튜닝에 사용할 트라이의 수입니다
        self.num_samples = 20 # 적을수록 실험이 빨리 끝남. 서버 사양이 좋을 경우 높은 게 좋음
        # 분산 데이터 병렬처리 (DDP)를 사용할지 여부입니다
        self.ddp = False

        # ASHA 스케줄러의 설정입니다
        # 학습의 최대 에폭입니다
        self.max_epochs = 100 # Fine-tuning시 50으로도 충분할 가능성이 높음. 수렴이 느릴 경우 80~100
        # 고려할 최소 에폭입니다
        self.grace_period = 10
        # 각 브래킷의 트라이의 수 감소 비율입니다
        self.reduction_factor = 3
        # 사용할 브래킷의 수입니다
        self.brackets = 2

        # 테스트에 사용할 체크포인트 경로입니다(필요 시 사용)
        self.checkpoint_path = None

        # 앙상블 관련
        # 다수의 모델 체크포인트가 들어있는 폴더 경로(한 폴더에 전부 들어있어야 함)
        self.ensemble_path = None  
        # 앙상블 메써드(uniform_soup, greedy_soup, ensemble_predict)
        self.ensemble_method = 'ensemble_predict'



class Config:
    """
    메인 설정 클래스입니다.

    이 클래스는 실험의 전체 설정을 저장합니다.
    """

    def __init__(self):
        # 설정 클래스를 초기화합니다
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.dataset = DatasetConfig()
        self.experiment = ExperimentConfig()

        # 하이퍼파라미터 튜닝의 검색 공간을 정의합니다
        self.search_space = vars(self.training)

    def update_from_args(self, args):
        """
        커맨드 라인 args로 config를 재귀함수를 사용하여 업데이트합니다.

        인수:
        args: 명령 줄 인수입니다.
        """

        def update_config(obj, args):
            """
            설정을 업데이트합니다.
            """
            for key, value in vars(obj).items():
                # 객체에 dict 속성이 존재하는 경우 이를 사용
                if hasattr(value, "__dict__"):
                    update_config(value, args)
                else:
                    # 속성에 커맨드라인 인자가 있는 경우 모든 속성 value를 업데이트
                    for arg_key, arg_value in vars(args).items():
                        if key == arg_key.replace("-", "_") and arg_value is not None:
                            setattr(obj, key, arg_value)

        update_config(self, args)
