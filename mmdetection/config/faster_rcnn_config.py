# faster_rcnn_config.py
import os

from .base_config import BaseConfig
from mmcv import Config


class faster_rcnn_config(BaseConfig):
    def __init__(self):
        super().__init__()
        self.config_dir = '/data/ephemeral/home/mmdetection/configs/faster_rcnn/faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco.py'
        self.model_name = os.path.basename(self.config_dir).split('.')[0]
        try:
            self.cfg = Config.fromfile(self.config_dir)
        except Exception as e:
            raise RuntimeError(f"설정 파일을 불러오는 데 실패했습니다: {str(e)}")

    def build_config(self):
        self.cfg = self.setup_config(self.cfg, self.model_name)
        # dataset config 수정
        self.cfg.data.train.classes = self.classes
        self.cfg.data.train.dataset.classes = self.classes
        self.cfg.data.train.dataset.img_prefix = self.data_dir
        self.cfg.data.train.dataset.ann_file = self.data_dir + 'train2.json' # train json 정보
        self.cfg.data.train.dataset.pipeline[2]['img_scale'] = (512,512) # Resize
        
        
        self.cfg.data.val.classes = self.classes
        self.cfg.data.val.img_prefix = self.data_dir
        self.cfg.data.val.ann_file = self.data_dir + 'val2.json' # val json 정보
        self.cfg.data.val.pipeline[1]['img_scale'] = (512,512) # Resize

        self.cfg.data.test.classes = self.classes
        self.cfg.data.test.img_prefix = self.data_dir
        self.cfg.data.test.ann_file = self.data_dir + 'test.json' # test json 정보
        self.cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize

        # print(self.cfg.data)
        # exit()

        self.cfg.data.samples_per_gpu = 16

        self.cfg.model.roi_head.bbox_head.num_classes = self.num_classes
        # 학습 설정
        self.cfg.runner.max_epochs = 1 # 1 only when smoke-test, otherwise 12 or bigger
        
        return self.cfg


# {'type': 'FasterRCNN', 
#  'backbone': {'type': 'ResNeXt', 'depth': 101, 'num_stages': 4, 'out_indices': (0, 1, 2, 3), 'frozen_stages': 1, 'norm_cfg': {'type': 'BN', 'requires_grad': True}, 'norm_eval': True, 'style': 'pytorch', 'init_cfg': {'type': 'Pretrained', 'checkpoint': 'open-mmlab://resnext101_64x4d'}, 'groups': 64, 'base_width': 4}, 
#  'neck': {'type': 'FPN', 'in_channels': [256, 512, 1024, 2048], 'out_channels': 256, 'num_outs': 5}, 
#  'rpn_head': {'type': 'RPNHead', 'in_channels': 256, 'feat_channels': 256, 'anchor_generator': {'type': 'AnchorGenerator', 'scales': [8], 'ratios': [0.5, 1.0, 2.0], 'strides': [4, 8, 16, 32, 64]}, 'bbox_coder': {'type': 'DeltaXYWHBBoxCoder', 'target_means': [0.0, 0.0, 0.0, 0.0], 'target_stds': [1.0, 1.0, 1.0, 1.0]}, 'loss_cls': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'loss_weight': 1.0}, 'loss_bbox': {'type': 'L1Loss', 'loss_weight': 1.0}}, 
#  'roi_head': {'type': 'StandardRoIHead', 
#               'bbox_roi_extractor': {'type': 'SingleRoIExtractor', 'roi_layer': {'type': 'RoIAlign', 'output_size': 7, 'sampling_ratio': 0}, 'out_channels': 256, 'featmap_strides': [4, 8, 16, 32]}, 
#               'bbox_head': {'type': 'Shared2FCBBoxHead', 'in_channels': 256, 'fc_out_channels': 1024, 'roi_feat_size': 7, 'num_classes': 80, 'bbox_coder': {'type': 'DeltaXYWHBBoxCoder', 'target_means': [0.0, 0.0, 0.0, 0.0], 'target_stds': [0.1, 0.1, 0.2, 0.2]}, 'reg_class_agnostic': False, 'loss_cls': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0}, 'loss_bbox': {'type': 'L1Loss', 'loss_weight': 1.0}}}, 'train_cfg': {'rpn': {'assigner': {'type': 'MaxIoUAssigner', 'pos_iou_thr': 0.7, 'neg_iou_thr': 0.3, 'min_pos_iou': 0.3, 'match_low_quality': True, 'ignore_iof_thr': -1}, 'sampler': {'type': 'RandomSampler', 'num': 256, 'pos_fraction': 0.5, 'neg_pos_ub': -1, 'add_gt_as_proposals': False}, 'allowed_border': -1, 'pos_weight': -1, 'debug': False}, 'rpn_proposal': {'nms_pre': 2000, 'max_per_img': 1000, 'nms': {'type': 'nms', 'iou_threshold': 0.7}, 'min_bbox_size': 0}, 'rcnn': {'assigner': {'type': 'MaxIoUAssigner', 'pos_iou_thr': 0.5, 'neg_iou_thr': 0.5, 'min_pos_iou': 0.5, 'match_low_quality': False, 'ignore_iof_thr': -1}, 'sampler': {'type': 'RandomSampler', 'num': 512, 'pos_fraction': 0.25, 'neg_pos_ub': -1, 'add_gt_as_proposals': True}, 'pos_weight': -1, 'debug': False}}, 'test_cfg': {'rpn': {'nms_pre': 1000, 'max_per_img': 1000, 'nms': {'type': 'nms', 'iou_threshold': 0.7}, 'min_bbox_size': 0}, 'rcnn': {'score_thr': 0.05, 'nms': {'type': 'nms', 'iou_threshold': 0.5}, 'max_per_img': 100}}}




