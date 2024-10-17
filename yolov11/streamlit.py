import json
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import os

# COCO 형식의 학습 데이터 파일 불러오기
with open("train.json", "r") as f:
    coco_data = json.load(f)

# Streamlit 레이아웃 설정
st.title("COCO Dataset EDA with Streamlit")

# 데이터셋 통계 출력
st.header("Dataset Statistics")
st.write(f"Total Images: {len(coco_data['images'])}")
st.write(f"Total Annotations: {len(coco_data['annotations'])}")
st.write(f"Total Categories: {len(coco_data['categories'])}")

# 클래스별 객체 수 시각화
st.header("Object Distribution by Category")
category_counts = {cat['name']: 0 for cat in coco_data['categories']}
for ann in coco_data['annotations']:
    category_id = ann['category_id']
    category_name = next(cat['name'] for cat in coco_data['categories'] if cat['id'] == category_id)
    category_counts[category_name] += 1

plt.figure(figsize=(10, 5))
plt.bar(category_counts.keys(), category_counts.values())
plt.xticks(rotation=90)
plt.title("Object Distribution by Category")
st.pyplot(plt)

# 샘플 이미지와 그에 대한 어노테이션 시각화
st.header("Sample Image with Annotations")
selected_image_id = st.selectbox("Select Image ID", [img['id'] for img in coco_data['images']])
selected_image = next(img for img in coco_data['images'] if img['id'] == selected_image_id)

img_path = os.path.join("train", selected_image['file_name'])
image = Image.open(img_path)
st.image(image, caption=selected_image['file_name'])

st.write(f"Image ID: {selected_image['id']}, Width: {selected_image['width']}, Height: {selected_image['height']}")

annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == selected_image_id]
st.write(f"Total Annotations for this Image: {len(annotations)}")
for ann in annotations:
    category_name = next(cat['name'] for cat in coco_data['categories'] if cat['id'] == ann['category_id'])
    st.write(f"Category: {category_name}, BBox: {ann['bbox']}")

# Streamlit 앱 실행 명령어
# streamlit run <filename.py>