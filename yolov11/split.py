import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(image_dir, train_dir, val_dir, test_size=0.2, random_state=42):
    """
    이미지와 대응하는 .txt 라벨 파일을 train과 val 폴더로 나누는 함수.
    :param image_dir: 원본 이미지와 라벨 파일이 있는 디렉토리
    :param train_dir: train 이미지와 라벨 파일이 저장될 디렉토리
    :param val_dir: val 이미지와 라벨 파일이 저장될 디렉토리
    :param test_size: 검증 세트의 비율 (기본값은 0.2)
    :param random_state: 데이터 분리 시 랜덤 시드 값
    """
    # 이미 분할된 폴더가 있는 경우, 분할 작업을 건너뜁니다.
    if os.path.exists(train_dir) and os.listdir(train_dir) and os.path.exists(val_dir) and os.listdir(val_dir):
        print(f"{train_dir} 및 {val_dir}에 데이터가 이미 분할되어 있습니다. 분할을 건너뜁니다.")
        return

    # 원본 이미지 리스트 가져오기 (확장자에 관계없이 모든 이미지 파일 포함)
    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    
    # 이미지 파일이 없는 경우 예외 처리
    if len(images) == 0:
        raise ValueError(f"이미지 파일이 {image_dir}에 없습니다. 경로를 확인하세요.")

    # Train/Val 분할
    train_images, val_images = train_test_split(images, test_size=test_size, random_state=random_state)
    
    # 디렉토리 생성
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 파일 이동 (이미지와 동일한 이름의 .txt 파일도 이동)
    for img in train_images:
        # 이미지 이동
        shutil.move(os.path.join(image_dir, img), os.path.join(train_dir, img))
        # 대응하는 .txt 파일이 있으면 같이 이동
        label_file = os.path.splitext(img)[0] + '.txt'
        label_path = os.path.join(image_dir, label_file)
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(train_dir, label_file))
    
    for img in val_images:
        # 이미지 이동
        shutil.move(os.path.join(image_dir, img), os.path.join(val_dir, img))
        # 대응하는 .txt 파일이 있으면 같이 이동
        label_file = os.path.splitext(img)[0] + '.txt'
        label_path = os.path.join(image_dir, label_file)
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(val_dir, label_file))
    
    print(f"Train 이미지 수: {len(train_images)}, Val 이미지 수: {len(val_images)}")

# split.py 파일이 직접 실행될 때 테스트 코드 실행
if __name__ == "__main__":
    original_image_dir = "/data/ephemeral/home/dataset/train"
    train_output_dir = "/data/ephemeral/home/dataset/train_split"
    val_output_dir = "/data/ephemeral/home/dataset/val_split"
    split_dataset(original_image_dir, train_output_dir, val_output_dir)