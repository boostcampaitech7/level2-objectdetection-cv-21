import json
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os

def load_coco_data(coco_file):
    """COCO 형식의 JSON 파일을 로드하는 함수"""
    with open(coco_file, "r") as f:
        return json.load(f)

def plot_category_distribution(coco_data):
    """카테고리별 객체 분포를 시각화하는 함수"""
    st.header("Object Distribution by Category")
    category_counts = {cat['name']: 0 for cat in coco_data['categories']}
    
    for ann in coco_data['annotations']:
        category_id = ann['category_id']
        category_name = next(cat['name'] for cat in coco_data['categories'] if cat['id'] == category_id)
        category_counts[category_name] += 1

    plt.figure(figsize=(10, 5))
    plt.bar(category_counts.keys(), category_counts.values(), color="skyblue")
    plt.xticks(rotation=90)
    plt.title("Object Distribution by Category")
    st.pyplot(plt)

def show_image_with_annotations(coco_data, image_dir):
    """선택된 이미지 및 해당 어노테이션을 표시하는 함수"""
    st.header("Sample Image with Annotations")
    
    selected_image_id = st.selectbox("Select Image ID", [img['id'] for img in coco_data['images']])
    selected_image = next(img for img in coco_data['images'] if img['id'] == selected_image_id)

    img_path = os.path.join(image_dir, selected_image['file_name'])
    image = Image.open(img_path)
    
    # 이미지를 그릴 객체 준비 (바운딩 박스 포함)
    draw = ImageDraw.Draw(image)
    
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == selected_image_id]
    st.write(f"Total Annotations for this Image: {len(annotations)}")
    
    for ann in annotations:
        category_name = next(cat['name'] for cat in coco_data['categories'] if cat['id'] == ann['category_id'])
        bbox = ann['bbox']
        draw.rectangle(
            [(bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3])], 
            outline="red", width=3
        )
        draw.text((bbox[0], bbox[1]), category_name, fill="white")

    st.image(image, caption=f"Image ID: {selected_image['id']}")

# Streamlit 레이아웃 설정
st.title("COCO Dataset EDA with Streamlit")

# 데이터 로드
coco_data = load_coco_data("train.json")

# 데이터셋 통계 출력
st.header("Dataset Statistics")
st.write(f"Total Images: {len(coco_data['images'])}")
st.write(f"Total Annotations: {len(coco_data['annotations'])}")
st.write(f"Total Categories: {len(coco_data['categories'])}")

# 카테고리별 객체 분포 시각화
plot_category_distribution(coco_data)

# 이미지 및 어노테이션 표시
show_image_with_annotations(coco_data, "train")

# Streamlit 앱 실행 명령어
# streamlit run streamlit.py