def update_config_with_dataset(cfg, dataset_dir: str, data_root: str) -> None:
    """Update configuration with split dataset settings from dataset directory."""
    # JSON 파일 경로 설정
    train_json = os.path.join(dataset_dir, 'train2.json')
    val_json = os.path.join(dataset_dir, 'val2.json')
    test_json = os.path.join(dataset_dir, 'test.json')

    # 기본 데이터셋 설정
    cfg.data_root = data_root

    # Train 데이터셋 설정
    if os.path.exists(train_json):
        cfg.data.train.update({
            'type': 'CocoDataset',
            'ann_file': train_json,
            'img_prefix': data_root,
            'pipeline': [
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(type='Normalize'),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ]
        })

    # Validation 데이터셋 설정
    if os.path.exists(val_json):
        cfg.data.val.update({
            'type': 'CocoDataset',
            'ann_file': val_json,
            'img_prefix': data_root,
            'pipeline': [
                dict(type='LoadImageFromFile'),
                dict(type='MultiScaleFlipAug',
                    img_scale=(1024, 1024),
                    flip=False,
                    transforms=[
                        dict(type='Resize', keep_ratio=True),
                        dict(type='RandomFlip'),
                        dict(type='Normalize'),
                        dict(type='Pad', size_divisor=32),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ])
            ]
        })

    # Test 데이터셋 설정
    if os.path.exists(test_json):
        cfg.data.test.update({
            'type': 'CocoDataset',
            'ann_file': test_json,
            'img_prefix': data_root,
            'pipeline': [
                dict(type='LoadImageFromFile'),
                dict(type='MultiScaleFlipAug',
                    img_scale=(1024, 1024),
                    flip=False,
                    transforms=[
                        dict(type='Resize', keep_ratio=True),
                        dict(type='RandomFlip'),
                        dict(type='Normalize'),
                        dict(type='Pad', size_divisor=32),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ])
            ]
        })

def main() -> None:
    """Parse arguments and run training/inference."""
    parser = argparse.ArgumentParser(description='PyTorch Object Detection Training/Inference')
    parser.add_argument('--model_config', type=str, default='test', help='config name without _config')
    parser.add_argument('--wandb_path', type=str, default=None, help='wandb artifact path')
    parser.add_argument('--epoch_number', type=int, default=None, help='epoch number for inference')
    parser.add_argument('--dataset_dir', type=str, required=True,
                       help='directory containing train.json, val.json, and test.json') # ".json" 파일들의 경로를 입력해주세요.
    parser.add_argument('--data_root', type=str, required=True, help='root path to image directory') # Image 파일들의 경로를 입력해주세요.
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], required=True,
                       help='whether to train or run inference')
    args = parser.parse_args()

    # Create base configuration
    cfg, model_name, output_dir = create_config(args.model_config)

    # Update configuration with dataset settings
    update_config_with_dataset(
        cfg,
        args.dataset_dir,
        args.data_root
    )

    if args.mode == 'train':
        # Training mode
        train_model(cfg, args.model_config)
    else:
        # Inference mode
        cfg.data.test.test_mode = True
        cfg.model.train_cfg = None

        if args.wandb_path:
            run = wandb.init()
            artifact = run.use_artifact(args.wandb_path, type='model')
            artifact_dir = artifact.download(path_prefix=f'epoch_{args.epoch_number}.pth')
            cfg.work_dir = artifact_dir

        inference(cfg, args.epoch_number, args.model_config)

        if args.wandb_path:
            shutil.rmtree(cfg.work_dir)

if __name__ == "__main__":
    main()
