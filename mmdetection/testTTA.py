from mmengine.config import Config, ConfigDict

# add below code on condfig files
# tta_model = dict(
#     type='DetTTAModel',
#     tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))

# tta_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=None),
#     dict(
#         type='TestTimeAug',
#         transforms=[
#             [dict(type='Resize', scale=(1333, 800), keep_ratio=True)],
#             [dict(type='RandomFlip', prob=1.), dict(type='RandomFlip', prob=0.)],
#             [dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction'))]
#         ])
# ]

config_file = 'path/to/your/config.py'
checkpoint_file = 'path/to/your/checkpoint.pth'

cfg = Config.fromfile(config_file)
cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

model = init_detector(cfg, checkpoint_file, device='cuda:0')