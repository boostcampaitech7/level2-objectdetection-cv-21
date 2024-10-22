import csv
import json
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os

def load_predictions(prediction_file):
    """Load prediction results from a CSV file."""
    predictions = {}
    with open(prediction_file, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            image_id = row[0]
            prediction_string = row[1]
            predictions[image_id] = prediction_string
    return predictions

def show_predictions(predictions, image_dir, class_labels):
    """Display predicted bounding boxes on the images."""
    st.header("Predicted Annotations")
    
    selected_image_id = st.selectbox("Select Image ID", list(predictions.keys()))
    prediction_string = predictions[selected_image_id]
    
    img_path = os.path.join(image_dir, f"{selected_image_id}.jpg")
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Font for drawing text
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    # Draw predicted bounding boxes
    if prediction_string.strip():
        predictions = prediction_string.strip().split(" ")
        for i in range(0, len(predictions), 6):
            # Each prediction consists of (label, score, xmin, ymin, xmax, ymax)
            cls, conf, xmin, ymin, xmax, ymax = map(float, predictions[i:i+6])
            label_name = class_labels[int(cls)]
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="blue", width=2)
            draw.text((xmin, ymin), f"{label_name}: {conf:.2f}", fill="blue", font=font)

    st.image(image, caption=f"Image ID: {selected_image_id}")

    # Display a bar chart for class label distributions
    st.subheader("Class Distribution")
    fig, ax = plt.subplots()
    ax.barh(class_labels, [predictions.count(f"{i}") for i in range(len(class_labels))], color="skyblue")
    ax.set_xlabel("Instances")
    ax.set_ylabel("Class")
    st.pyplot(fig)

# Streamlit app configuration
st.title("Object Detection - Prediction Viewer")

# Load prediction results and class labels
prediction_file = "submission_yolo11x.csv"
predictions = load_predictions(prediction_file)
class_labels = ["General trash", "Paper", "Paper pack", "Metal", "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

# Image directory configuration
image_dir = "/data/ephemeral/home/dataset/test"

# Display prediction results with Streamlit
show_predictions(predictions, image_dir, class_labels)