import matplotlib.pyplot as plt
import numpy as np
from mmcv import Config
from mmdet.datasets import build_dataloader, build_dataset
from mmcv.parallel import collate, scatter
import os
import sys
sys.path.append('/data/ephemeral/home/github/proj2/level2-objectdetection-cv-21/mmdetection')

from config.test_config import test_config


# Config에서 데이터셋과 파이프라인을 불러옵니다.
config_obj = test_config(max_epochs=25)  # Adjust parameters if needed
cfg = config_obj.build_config()
dataset = build_dataset(cfg.data.train)

# DataLoader 생성
data_loader = build_dataloader(
    dataset,
    samples_per_gpu=1,  # 한 번에 하나의 이미지만 로드
    workers_per_gpu=2,  # 워커 개수 설정
    dist=False,
    shuffle=False)

# 저장할 폴더 설정 (없으면 생성)
save_dir = './augmented_images'
os.makedirs(save_dir, exist_ok=True)

# 데이터 불러오기 및 시각화
for i, data in enumerate(data_loader):
    if i >= 5:  # 처음 5개 이미지만 확인
        break
    
    # 이미지와 어노테이션 데이터 추출
    img_tensor = data['img'].data[0][0]  # Tensor에서 이미지를 꺼냄
    img = img_tensor.permute(1, 2, 0).numpy()  # 채널 순서를 (H, W, C)로 변경

    # 이미지를 normalize 반대로 변환 (이미지를 원래의 색으로 변환)
    mean = np.array(cfg.img_norm_cfg['mean'], dtype=np.float32)
    std = np.array(cfg.img_norm_cfg['std'], dtype=np.float32)
    img = img * std + mean  # normalize 역연산
    img = img.astype(np.uint8)

    # 새로운 figure 생성
    plt.figure(figsize=(8, 8))
    
    # 이미지를 화면에 시각화
    plt.imshow(img)
    
    # 축을 제거
    plt.axis('off')

    # 이미지 파일로 저장
    file_path = os.path.join(save_dir, f"augmented_image_{i+1}.png")
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)  # 이미지 저장
    plt.close()  # 메모리 낭비 방지를 위해 플롯 닫기

    print(f"Image {i+1} saved at {file_path}")