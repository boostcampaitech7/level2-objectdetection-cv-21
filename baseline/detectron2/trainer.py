import os
import copy
import uuid
import datetime

import torch
import detectron2
import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import setup_logger
setup_logger()

import wandb, yaml

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader, build_detection_train_loader

# 상수 값들: 학습 전체에서 고정으로 쓰이며, 변하지 않음
DATA_DIR = '/data/ephemeral/home/dataset/'
OUTPUT_DIR = '/data/ephemeral/home/output/detectron2/'
TRAIN_JSON = 'train2.json'
VAL_JSON = 'val2.json'
TEST_JSON  = 'test.json'
# MODEL_YAML = 'Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv'->nvcc 컴파일러가 없으면 사용 불가
# MODEL_YAML = 'COCO-Detection/faster_rcnn_R_101_FPN_3x'->기존 코드
MODEL_YAML = 'PascalVOC-Detection/faster_rcnn_R_50_C4'
MODEL_NAME = MODEL_YAML.split('/')[-1]

# Register Dataset
def register_dataset():
    try:
        register_coco_instances('coco_trash_train', {}, os.path.join(DATA_DIR, TRAIN_JSON), DATA_DIR)
    except AssertionError:
        pass

    try:
        register_coco_instances('coco_trash_test', {}, os.path.join(DATA_DIR, VAL_JSON), DATA_DIR)
    except AssertionError:
        pass

    MetadataCatalog.get('coco_trash_train').thing_classes = ["General trash", "Paper", "Paper pack", "Metal", 
                                                            "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

def load_and_fix_config():
    # config 불러오기
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(f'{MODEL_YAML}.yaml'))

    # config 수정하기
    cfg.DATASETS.TRAIN = ('coco_trash_train',)
    cfg.DATASETS.TEST = ('coco_trash_test',)

    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f'{MODEL_YAML}.yaml')

    cfg.SOLVER.IMS_PER_BATCH = 32 # Batch size
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 5 # 100 for smoke test, 15000 is approximately 6.15 epochs.
    cfg.SOLVER.STEPS = (6000,9000)
    cfg.SOLVER.GAMMA = 0.005
    cfg.SOLVER.CHECKPOINT_PERIOD = 3000

    # AMP 사용 여부 확인
    cfg.SOLVER.AMP.ENABLED = True

    # 시간+랜덤 코드 5자리로 결과 폴더명 생성
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    random_code = str(uuid.uuid4())[:5]
    experiment_name = f"{timestamp}_{random_code}"
    experiment_dir = os.path.join(OUTPUT_DIR, experiment_name)
    cfg.OUTPUT_DIR = experiment_dir

    if 'ROI_KEYPOINT_HEAD' in cfg.MODEL:
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 0

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10

    # cfg.TEST.EVAL_PERIOD = 3906//cfg.SOLVER.IMS_PER_BATCH
    cfg.TEST.EVAL_PERIOD = cfg.SOLVER.MAX_ITER

    return cfg



# mapper - input data를 어떤 형식으로 return할지 (따라서 augmnentation 등 데이터 전처리 포함 됨)
def MyMapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')
    
    transform_list = [
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3)
    ]
    
    image, transforms = T.apply_transform_gens(transform_list, image)
    
    dataset_dict['image'] = torch.as_tensor(image.transpose(2,0,1).astype('float32'))
    
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop('annotations')
        if obj.get('iscrowd', 0) == 0
    ]
    
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict['instances'] = utils.filter_empty_instances(instances)
    
    return dataset_dict


# trainer - DefaultTrainer를 상속
class MyTrainer(DefaultTrainer):
    
    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        return build_detection_train_loader(
        cfg, mapper = MyMapper, sampler = sampler
        )
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            eval_folder = os.path.join(cfg.OUTPUT_DIR, "output_eval")
            os.makedirs(eval_folder, exist_ok = True)
            output_folder = eval_folder
            
        return COCOEvaluator(dataset_name, cfg, False, output_folder)
    

class WandbLoggerHook(HookBase):
    def __init__(self, project=None, log_period=100, config=None):
        self.project = project
        self.log_period = log_period
        self.iter = 0
        self.config = config
        self._setup_wandb()

    def _setup_wandb(self):
        # Initialize WandB project
        wandb.init(project=self.project, config=self.config)

    def before_train(self):
        self.iter = 0

    def after_step(self):
        self.iter += 1
        if self.iter % self.log_period == 0:
            storage = get_event_storage()
            for k, v in storage.latest_with_smoothing_hint(20).items():
                wandb.log({f"{k}": v[0]}, step=storage.iter)
            # metrics = self.trainer._trainer_storage.get_metrics()
            # metrics = {k: np.mean(v) for k, v in metrics.items()}
            # wandb.log(metrics)
            
    def after_train(self):
        wandb.finish()

def main():
    register_dataset()
    cfg = load_and_fix_config()

    cfg_wandb = yaml.safe_load(cfg.dump())
    
    wandb_logger = WandbLoggerHook(project=MODEL_NAME, log_period=cfg.TEST.EVAL_PERIOD, config=cfg_wandb)


    # train
    os.makedirs(cfg.OUTPUT_DIR, exist_ok = True)

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.register_hooks([wandb_logger])
    trainer.train()

if __name__=="__main__":
    main()