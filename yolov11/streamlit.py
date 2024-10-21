import csv
import json
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os

def load_predictions(prediction_file):
    """예측 파일을 로드하는 함수"""
    predictions = {}
    with open(prediction_file, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # 헤더 스킵
        for row in reader:
            image_id = row[0]
            prediction_string = row[1]
            predictions[image_id] = prediction_string
    return predictions

def show_predictions(predictions, image_dir):
    """예측된 바운딩 박스를 이미지 위에 표시하는 함수"""
    st.header("Predicted Annotations")
    
    selected_image_id = st.selectbox("Select Image ID", list(predictions.keys()))
    prediction_string = predictions[selected_image_id]
    
    img_path = os.path.join(image_dir, f"{selected_image_id}.jpg")
    image = Image.open(img_path)
    draw = ImageDraw.Draw(image)
    
    # 예측 문자열이 비어 있지 않은 경우에만 바운딩 박스 그리기
    if prediction_string.strip():
        for pred in prediction_string.strip().split(" "):
            # 각 예측 항목은 여섯 개의 값(label, score, xmin, ymin, xmax, ymax)이어야 함
            values = pred.split()
            if len(values) != 6:
                continue  # 값이 6개가 아닐 경우 해당 항목을 건너뜀
            
            cls, conf, xmin, ymin, xmax, ymax = map(float, values)
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="blue", width=2)
            draw.text((xmin, ymin), f"{int(cls)}: {conf:.2f}", fill="blue")
    
    st.image(image, caption=f"Image ID: {selected_image_id}")

# Streamlit 애플리케이션 실행
st.title("Prediction Results")

# 예측 결과 로드
prediction_file = "submission_yolo11x.csv"
predictions = load_predictions(prediction_file)

# 이미지 디렉토리 설정
image_dir = "/data/ephemeral/home/dataset/test"

# 예측 결과 시각화
show_predictions(predictions, image_dir)