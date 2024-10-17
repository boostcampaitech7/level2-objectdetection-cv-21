import os
import json
from sklearn.model_selection import train_test_split

def split_dataset(data_dir, test_size=0.2, random_state=42):
    """
    COCO 형식의 데이터를 train/val으로 분리하여 train2.json과 val2.json 파일로 저장하는 함수
    :param data_dir: COCO 형식의 원본 JSON 파일이 있는 디렉토리 경로
    :param test_size: 검증 데이터셋의 비율 (0~1 사이 값)
    :param random_state: 데이터 분리 시 사용되는 난수 시드 값
    """
    train_json_path = os.path.join(data_dir, 'train.json')
    
    with open(train_json_path) as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']

    # Create a dictionary to map image IDs to their corresponding annotations
    image_annotations = {}
    for annotation in annotations:
        annotation['segmentation'] = [[0, 0, 0, 0, 0, 0, 0, 0]]  # segmentation이 필요 없을 경우 기본값 설정
        image_annotations.setdefault(annotation['image_id'], []).append(annotation)

    # Split the images and their corresponding annotations
    train_images, val_images = train_test_split(images, test_size=test_size, random_state=random_state)

    train_annotations = []
    for image in train_images:
        train_annotations.extend(image_annotations[image['id']])
    
    val_annotations = []
    for image in val_images:
        val_annotations.extend(image_annotations[image['id']])

    # Create the training and validation JSON objects
    train_data = {
        'info': data['info'],
        'licenses': data['licenses'],
        'images': train_images,
        'annotations': train_annotations,
        'categories': data['categories']
    }
    
    val_data = {
        'info': data['info'],
        'licenses': data['licenses'],
        'images': val_images,
        'annotations': val_annotations,
        'categories': data['categories']
    }

    # Remove existing files if they exist
    if os.path.exists(os.path.join(data_dir, 'train2.json')):
        os.remove(os.path.join(data_dir, 'train2.json'))
        print("Removed the existing train2.json")
    
    if os.path.exists(os.path.join(data_dir, 'val2.json')):
        os.remove(os.path.join(data_dir, 'val2.json'))
        print("Removed the existing val2.json")

    # Save the training and validation JSON objects to file
    with open(os.path.join(data_dir, 'train2.json'), 'w') as f:
        json.dump(train_data, f)
        print("Successfully created the train2.json!")
    
    with open(os.path.join(data_dir, 'val2.json'), 'w') as f:
        json.dump(val_data, f)
        print("Successfully created the val2.json!")
