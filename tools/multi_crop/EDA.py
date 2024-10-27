import matplotlib
matplotlib.use('Agg')  # Non-interactive 백엔드 설정
import matplotlib.pyplot as plt
import cv2
import json
import os

# JSON 파일 경로와 이미지 파일 경로
json_path = '/data/ephemeral/home/dataset/subimages/train_SR_4images.json'
image_dir = '/data/ephemeral/home/dataset/subimages/'

# JSON 파일 읽기
with open(json_path, 'r') as file:
    data = json.load(file)

# 이미지별로 bbox를 그리기 (처음 4장만)
fig, axes = plt.subplots(2, 2, figsize=(12, 12))  # 2x2 subplot
axes = axes.ravel()  # 쉽게 반복문을 돌리기 위해 1D 배열로 변환

for i, img_info in enumerate(data['images'][:4]):  # 처음 4개 이미지만
    image_id = img_info['id']
    file_name = img_info['file_name']
    
    # 이미지 읽기
    image_path = os.path.join(image_dir, file_name)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Image {file_name} not found.")
        continue
    
    # BGR 이미지를 RGB로 변환 (OpenCV는 기본적으로 BGR 포맷을 사용하므로)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 해당 이미지의 annotation 정보 불러오기
    for ann in data['annotations']:
        if ann['image_id'] == image_id:
            # bounding box 좌표 (x_min, y_min, width, height)
            x_min, y_min, width, height = ann['bbox']
            x_max = x_min + width
            y_max = y_min + height
            
            # bounding box 그리기 (빨간색, 두께 2)
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
    
    # Plot 그리기
    axes[i].imshow(image)
    axes[i].set_title(f"Image: {file_name}")
    axes[i].axis('off')

# 전체 레이아웃 조정 및 출력
plt.tight_layout()
plt.savefig('/data/ephemeral/home/dataset/subimages/plot.png')  # 파일로 저장
print("Plot saved as plot.png")